"""Backend-dispatching launch: picks a model, frees the port, starts the
right backend, waits for it, warms it up, syncs opencode.json, then hands
off to `opencode`.
"""
import json
import os
import subprocess
import sys
import time

from . import classify, gguf_meta, installer, locking, manifest, opencode_sync, prereqs, specs
from .backends import get_backend

PORT = int(os.environ.get("YOJIT_PORT", "8080"))
PORT_ENV_VAR = "YOJIT_PORT"
OPENCODE_PROVIDER = "local"  # matches the provider key opencode_sync.py writes

# A launch waits up to 60 * 2s for health, then warms up, so this lock is held far
# longer than an ordinary state mutation. Everything else uses the 10s default.
LAUNCH_LOCK_TIMEOUT = 180.0

SERVER_RECORD_NAME = "server.json"


class PortOwnedByAnotherProcess(RuntimeError):
    """The port is held by a process yojit did not start."""


def server_record_path():
    return locking.state_dir() / SERVER_RECORD_NAME


def read_server_record() -> dict:
    """The server yojit last started: {pid, port, model, context, output, started_at}.
    Empty when nothing is recorded, or when the file is unreadable."""
    try:
        return json.loads(server_record_path().read_text())
    except Exception:
        return {}


def _record_server(pid: int, model_id: str, context: int, output: int) -> None:
    record = {
        "pid": int(pid),
        "port": PORT,
        "model": model_id,
        "context": int(context),
        "output": int(output),
        "started_at": manifest.now_iso(),
    }
    with locking.state_lock():
        locking.atomic_write_text(server_record_path(), json.dumps(record, indent=2) + "\n")


def clear_server_record() -> None:
    with locking.state_lock():
        try:
            server_record_path().unlink()
        except OSError:
            pass


def pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _port_pid(port: int) -> int | None:
    try:
        out = subprocess.check_output(["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"], text=True).strip()
        return int(out.splitlines()[0]) if out else None
    except Exception:
        return None


def _free_port(port: int) -> None:
    """Frees `port` if -- and only if -- the listener is a server yojit started.

    Raises PortOwnedByAnotherProcess rather than killing an unidentified process:
    `lsof` cannot tell us the listener is ours, and port 8080 is a port other
    things legitimately use. The old behaviour was `kill` on whatever held it.
    """
    pid = _port_pid(port)
    if not pid:
        return
    recorded = read_server_record()
    if recorded.get("pid") == pid and recorded.get("port") == port:
        print(f"Port {port} is in use by yojit's own server PID {pid} -- stopping it...")
        subprocess.run(["kill", str(pid)])
        time.sleep(2)
        clear_server_record()
        return
    raise PortOwnedByAnotherProcess(
        f"Port {port} is in use by PID {pid}, which yojit did not start.\n"
        f"Refusing to kill it. Stop it yourself, or run with {PORT_ENV_VAR}=<free port>."
    )


def _tier_label(entry: dict) -> str:
    return "RISKY" if entry.get("tier") == "high" else "safe"


def _print_model_menu(models: dict, ids: list[str], default_index: int | None) -> None:
    print("\nRISKY = enough RAM to have a real OOM crash risk on this machine, not just a guess.")
    print("\nWhich model do you want to serve? (sorted safest-first)")
    for i, model_id in enumerate(ids, 1):
        entry = models[model_id]
        marker = " (current default)" if i == default_index else ""
        print(f"  {i}) {model_id}{marker}  [{entry.get('backend')}, {entry.get('size_gb')} GB, {_tier_label(entry)}]")


def _read_model_choice(ids: list[str], default_index: int | None) -> str:
    prompt = f"Enter a number (1-{len(ids)})"
    prompt += f", or press Enter for {default_index}: " if default_index else ": "
    while True:
        choice = input(prompt).strip()
        if choice == "" and default_index:
            choice = str(default_index)
        if choice.isdigit() and 1 <= int(choice) <= len(ids):
            return ids[int(choice) - 1]
        print("Invalid choice.")


def pick_model_interactive() -> str | None:
    """Lists installed models safest-first, with a confirmation gate on RISKY picks."""
    models = manifest.list_models()
    if not models:
        return None
    default = manifest.get_default()
    ids = sorted(models.keys(), key=lambda m: (_tier_label(models[m]) == "RISKY", models[m].get("size_gb", 0)))
    default_index = ids.index(default) + 1 if default in ids else None
    _print_model_menu(models, ids, default_index)

    while True:
        model_id = _read_model_choice(ids, default_index)
        if _tier_label(models[model_id]) != "RISKY":
            return model_id
        confirm = input("This model uses enough RAM to risk a crash. Continue anyway? (y/N): ").strip().lower()
        if confirm == "y":
            return model_id


def _attempt_launch(model_id: str) -> tuple[bool, int | None]:
    """Returns (success, pid). On success the server stays running; on failure nothing is left running."""
    entry = manifest.get_model(model_id)
    if not entry:
        print(f"{model_id} is not installed.")
        return False, None

    backend = get_backend(entry["backend"])
    backend.ensure_installed()
    model_path = manifest.models_root() / entry["store_path"]

    # Always offline: the model is already installed locally, and going
    # online here can hang on tokenizer/config revalidation for no benefit.
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("Running fully offline from local cache (model is already installed).")

    # Every launch parameter is computed fresh from this machine's actual specs, never a fixed constant.
    s = specs.detect()
    tuning = classify.compute_launch_tuning(entry.get("size_gb", 0.0), s.total_ram_gb, s.cpu_cores)

    # Spec-driven defaults, then anything set via `yojit config` overrides them.
    cfg = (gguf_meta.to_hf_style_config(gguf_meta.read_metadata(model_path)) if backend.name == "llamacpp"
           else classify.load_hf_config(model_path))
    requested_context = entry.get("context", 4096)
    plan = classify.resolve_kv_cache(cfg, backend.name, entry.get("size_gb", 0.0),
                                     s.total_ram_gb, requested_context)
    context = plan.context
    # Derived from the context we are actually launching with, so the budget can
    # never exceed a quarter of the window the live server really has.
    output_limit = classify.output_for_context(context)
    if not plan.fits:
        print(
            f"Context reduced {requested_context} -> {context} tokens: at "
            f"{plan.bytes_per_token} KV bytes/token the requested window needs "
            f"{plan.required_bytes / (1024 ** 3):.2f} GiB but only "
            f"{plan.headroom_bytes / (1024 ** 3):.2f} GiB of headroom is free. "
            f"Make this permanent with `yojit config {model_id} --context {context}`."
        )
    overrides = {**plan.overrides, **entry.get("overrides", {})}

    try:
        # One critical section for freeing the port, launching, and recording the
        # result. Two serve() runs interleaving here is exactly how a live server
        # gets killed out from under a session that was just told it is up.
        with locking.state_lock(timeout=LAUNCH_LOCK_TIMEOUT):
            _free_port(PORT)

            print(f"Starting {backend.name} backend for {model_id}...")
            proc = backend.launch(model_path, PORT, context, output_limit, tuning, overrides)

            print(f"Waiting for server to come up on port {PORT}...")
            for _ in range(60):
                if backend.health_check(PORT):
                    print(f">>> Server is UP on http://localhost:{PORT} <<<")
                    break
                time.sleep(2)
            else:
                print("Server never came up.")
                return False, None

            print("Sending a warm-up request so the model is loaded before you type anything...")
            # Must match the local path the server was launched with, not the manifest's repo-style model_id.
            backend.warm_up(PORT, str(model_path))

            if not backend.health_check(PORT):
                print("WARM-UP FAILED. The model most likely crashed (OOM?).")
                return False, None

            _record_server(proc.pid, model_id, context, output_limit)
    except PortOwnedByAnotherProcess as exc:
        print(str(exc))
        return False, None
    except locking.StateLockTimeout as exc:
        print(f"Another yojit invocation is mid-launch ({exc}). Try again shortly.")
        return False, None

    return True, proc.pid


def _serve_interactively() -> tuple[str, int]:
    """Picker with a retry loop: offer to remove a broken model and pick again."""
    while True:
        model_id = pick_model_interactive()
        if not model_id:
            print("No models installed. Run `yojit install <repo>` first.")
            sys.exit(1)
        ok, pid = _attempt_launch(model_id)
        if ok:
            # The default is committed together with opencode.json by serve(),
            # under one lock -- see the comment there.
            return model_id, pid
        remove_it = input(f"\n{model_id} failed to serve. Remove it from disk? (y/N): ").strip().lower()
        if remove_it == "y":
            print(installer.remove(model_id))
        print("Picking again...")


def _start_server(model_id: str | None) -> tuple[str, int, bool]:
    """Picks a model if needed and launches it. Returns (model_id, pid, picked)."""
    if not model_id:
        installed = manifest.list_models()
        # Leave it unset when there is a real choice to make: 0 installed reports
        # and exits below, >1 always shows the picker.
        if len(installed) == 1:
            model_id = next(iter(installed))
            return model_id, _launch_or_die(model_id), False
        model_id, pid = _serve_interactively()
        return model_id, pid, True
    # Explicit model: one attempt, no retry -- respect the caller's choice.
    return model_id, _launch_or_die(model_id), False


def _launch_or_die(model_id: str) -> int:
    ok, pid = _attempt_launch(model_id)
    if not ok:
        print(f"\nFailed to serve {model_id}. Not launching opencode against a dead server.")
        sys.exit(1)
    return pid


def _publish_handoff(model_id: str, pid: int, picked: bool) -> None:
    """Writes the default/model list and re-validates that the server is still ours.

    One critical section for both writes. The manifest's default slot and the
    "(running)" marker in opencode.json describe the same fact, and letting two
    processes interleave them is what leaves opencode advertising a model whose
    server is already gone (formal/tla/ServeLifecycle.tla, DefaultMatchesAdvertised).

    The re-validation has to happen in that same section, because the launch lock
    is released before this point: a second `yojit serve` may have replaced our
    server in the meantime. Handing a dead endpoint to opencode is the exact
    failure ServeLifecycle.tla reports as ReportedServerIsAlive.
    """
    record = read_server_record()
    running_limits = None
    if record.get("model") == model_id and record.get("pid") == pid:
        running_limits = (record.get("context"), record.get("output"))

    with locking.state_lock():
        if picked:
            manifest.set_default(model_id)  # a successful pick becomes the new default
        print(opencode_sync.sync(running_model=model_id, running_limits=running_limits))

        # formal/tla/ServeLifecycleRecord.tla: StatusNeverReportsForeign
        still_ours = read_server_record()
        if (still_ours.get("model") != model_id or still_ours.get("pid") != pid
                or not pid_alive(pid)):
            print(
                f"\nAnother yojit invocation replaced the server on port {PORT} while this one was "
                f"starting. Not launching opencode against a server that is no longer ours."
            )
            sys.exit(1)


def _open_opencode(model_id: str, pid: int, open_opencode: bool) -> None:
    entry = manifest.get_model(model_id)
    log_hint = f"~/.yojit/{entry.get('backend')}-server.log"
    print(f"\nServer running in the background (PID {pid}, log: {log_hint}).")

    if not open_opencode:
        return
    if not prereqs.ensure_opencode_installed():
        return

    print("Updating opencode...")
    subprocess.run(["opencode", "upgrade"])
    print("Launching opencode...\n")
    # Bind opencode's session to the model just started -- must match opencode_sync.py's key exactly.
    local_path = str(manifest.models_root() / entry["store_path"])
    subprocess.run(["opencode", "-m", f"{OPENCODE_PROVIDER}/{local_path}"])


def serve(model_id: str | None, open_opencode: bool = True) -> None:
    model_id, pid, picked = _start_server(model_id)
    _publish_handoff(model_id, pid, picked)
    _open_opencode(model_id, pid, open_opencode)
