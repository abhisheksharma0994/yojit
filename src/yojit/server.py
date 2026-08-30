"""Backend-dispatching launch: picks a model, frees the port, starts the
right backend, waits for it, warms it up, syncs opencode.json, then hands
off to `opencode`.
"""
import os
import subprocess
import sys
import time

from . import classify, gguf_meta, installer, manifest, opencode_sync, prereqs, specs
from .backends import get_backend

PORT = int(os.environ.get("YOJIT_PORT", "8080"))
OPENCODE_PROVIDER = "local"  # matches the provider key opencode_sync.py writes


def _port_pid(port: int) -> int | None:
    try:
        out = subprocess.check_output(["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"], text=True).strip()
        return int(out.splitlines()[0]) if out else None
    except Exception:
        return None


def _free_port(port: int) -> None:
    pid = _port_pid(port)
    if pid:
        print(f"Port {port} is in use by PID {pid} -- stopping it...")
        subprocess.run(["kill", str(pid)])
        time.sleep(2)


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

    _free_port(PORT)

    # Every launch parameter is computed fresh from this machine's actual specs, never a fixed constant.
    s = specs.detect()
    tuning = classify.compute_launch_tuning(entry.get("size_gb", 0.0), s.total_ram_gb, s.cpu_cores)

    # Spec-driven defaults, then anything set via `yojit config` overrides them.
    cfg = (gguf_meta.to_hf_style_config(gguf_meta.read_metadata(model_path)) if backend.name == "llamacpp"
           else classify.load_hf_config(model_path))
    overrides = {
        **classify.default_kv_cache_overrides(
            cfg, backend.name, entry.get("size_gb", 0.0), s.total_ram_gb, entry.get("context", 4096)),
        **entry.get("overrides", {}),
    }

    print(f"Starting {backend.name} backend for {model_id}...")
    proc = backend.launch(model_path, PORT, entry.get("context", 4096), entry.get("output", 1024), tuning, overrides)

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
            manifest.set_default(model_id)  # successful pick becomes the new default
            return model_id, pid
        remove_it = input(f"\n{model_id} failed to serve. Remove it from disk? (y/N): ").strip().lower()
        if remove_it == "y":
            print(installer.remove(model_id))
        print("Picking again...")


def serve(model_id: str | None, open_opencode: bool = True) -> None:
    if not model_id:
        installed = manifest.list_models()
        if len(installed) == 1:
            model_id = next(iter(installed))
        # else: leave unset -- 0 installed reports and exits below, >1 always shows the picker.

    if model_id:
        # Explicit model: one attempt, no retry -- respect the caller's choice.
        ok, pid = _attempt_launch(model_id)
        if not ok:
            print(f"\nFailed to serve {model_id}. Not launching opencode against a dead server.")
            sys.exit(1)
    else:
        model_id, pid = _serve_interactively()

    print(opencode_sync.sync(running_model=model_id))

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
