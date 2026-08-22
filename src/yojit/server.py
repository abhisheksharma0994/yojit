"""Backend-dispatching launch: picks a model (interactively if needed), frees
the port if something's already bound to it, starts the right backend, waits
for it to come up, warms it up, syncs opencode.json, then always hands off to
`opencode` -- this end-to-end flow is the entire point of the tool, not a
side effect of model management.

Mirrors every safety behavior proven out in the bash prototype tonight:
offline-by-default posture (only goes online if reachability is confirmed),
port-conflict auto-recovery, a real post-warmup health check (not just
"the HTTP call didn't throw"), a RISKY/safe picker with a confirmation gate,
a retry loop that re-prompts on failure with an offer to remove the broken
model, opencode self-upgrade before launch, and binding opencode's session to
the exact model just started via `-m`.
"""
import os
import subprocess
import sys
import time

import requests

from . import classify, installer, manifest, opencode_sync, prereqs, specs
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


def _internet_available() -> bool:
    try:
        requests.get("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def apply_offline_posture() -> bool:
    """Offline-by-default: the environment starts in offline mode, and is
    only switched to online if a real reachability check succeeds -- never
    the other way around. Returns True if online."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    if _internet_available():
        os.environ.pop("HF_HUB_OFFLINE", None)
        return True
    return False


def _tier_label(entry: dict) -> str:
    return "RISKY" if entry.get("tier") == "high" else "safe"


def pick_model_interactive() -> str | None:
    """Lists installed models sorted safest-first, with the same RISKY/safe
    labeling the bash prototype uses, and a confirmation gate on RISKY picks.
    The current default (if any) is pre-highlighted and accepted on blank
    input, so switching between installed models is always visible (this
    always runs when there's a real choice to make -- see serve() below) but
    accepting the existing default is still a single Enter press."""
    models = manifest.list_models()
    if not models:
        return None
    default = manifest.get_default()
    ids = sorted(models.keys(), key=lambda m: (_tier_label(models[m]) == "RISKY", models[m].get("size_gb", 0)))
    default_index = ids.index(default) + 1 if default in ids else None

    print("\nRISKY = uses enough RAM to have a real Metal-OOM crash risk on this")
    print("machine (see classify.py's MEDIUM_TIER_MAX_FRACTION), not just a guess.")
    print("\nWhich model do you want to serve? (sorted safest-first)")
    for i, model_id in enumerate(ids, 1):
        entry = models[model_id]
        marker = " (current default)" if i == default_index else ""
        print(f"  {i}) {model_id}{marker}  [{entry.get('backend')}, {entry.get('size_gb')} GB, {_tier_label(entry)}]")

    prompt = f"Enter a number (1-{len(ids)})"
    prompt += f", or press Enter for {default_index}: " if default_index else ": "

    while True:
        choice = input(prompt).strip()
        if choice == "" and default_index:
            choice = str(default_index)
        if not (choice.isdigit() and 1 <= int(choice) <= len(ids)):
            print("Invalid choice.")
            continue
        model_id = ids[int(choice) - 1]
        if _tier_label(models[model_id]) == "RISKY":
            confirm = input("This model uses enough RAM to risk a crash. Continue anyway? (y/N): ").strip().lower()
            if confirm != "y":
                continue
        return model_id


def _attempt_launch(model_id: str) -> tuple[bool, int | None]:
    """Returns (success, pid). On success the server is left running in the
    background; on failure nothing is left running."""
    entry = manifest.get_model(model_id)
    if not entry:
        print(f"{model_id} is not installed.")
        return False, None

    backend = get_backend(entry["backend"])
    backend.ensure_installed()
    model_path = manifest.models_root() / entry["store_path"]

    print("Checking internet connectivity...")
    online = apply_offline_posture()
    print("  -> Internet reachable." if online else "  -> No internet detected. Running fully offline from local cache.")

    _free_port(PORT)

    # Every launch parameter beyond context/output is computed fresh from
    # this machine's actual specs -- not fixed constants -- and recomputed
    # every launch rather than stored, so a hardware change (or moving to a
    # different machine) is picked up automatically.
    s = specs.detect()
    tuning = classify.compute_launch_tuning(entry.get("size_gb", 0.0), s.total_ram_gb, s.cpu_cores)

    print(f"Starting {backend.name} backend for {model_id}...")
    proc = backend.launch(model_path, PORT, entry.get("context", 4096), entry.get("output", 1024), tuning)

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
    backend.warm_up(PORT, model_id)

    # Actually verify the server survived its own warm-up, instead of assuming
    # success -- this exact gap let a crashed server get handed to opencode
    # during real testing.
    if not backend.health_check(PORT):
        print("WARM-UP FAILED. The model most likely crashed (Metal OOM?).")
        return False, None

    return True, proc.pid


def serve(model_id: str | None, open_opencode: bool = True) -> None:
    if not model_id:
        installed = manifest.list_models()
        if len(installed) == 1:
            # Nothing to choose between -- skip the picker entirely.
            model_id = next(iter(installed))
        # else: leave model_id unset. With 0 installed, the interactive
        # branch below reports that and exits. With >1 installed, it always
        # shows the picker (default pre-highlighted for a fast Enter-accept)
        # -- a real choice must always be visible, not silently skipped just
        # because a sticky default happens to be set.

    if model_id:
        # Explicit model (arg, or the sole installed model): one attempt, no
        # retry loop -- respect the caller's explicit choice instead of
        # second-guessing it.
        ok, pid = _attempt_launch(model_id)
        if not ok:
            print(f"\nFailed to serve {model_id}. Not launching opencode against a dead server.")
            sys.exit(1)
    else:
        # No model specified and more than one installed (or none at all):
        # interactive picker with a retry loop -- on failure, offer to
        # remove the broken model and pick again.
        while True:
            model_id = pick_model_interactive()
            if not model_id:
                print("No models installed. Run `yojit install <repo>` first.")
                sys.exit(1)
            ok, pid = _attempt_launch(model_id)
            if ok:
                # A successful interactive pick becomes the new default, so
                # the next bare `serve()` launches it directly without
                # re-prompting.
                manifest.set_default(model_id)
                break
            remove_it = input(f"\n{model_id} failed to serve. Remove it from disk? (y/N): ").strip().lower()
            if remove_it == "y":
                print(installer.remove(model_id))
            print("Picking again...")

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
    # Bind opencode's session to the exact model just started -- otherwise it
    # defaults to whatever model was last used in a previous session, which
    # is very likely NOT the one actually running right now.
    subprocess.run(["opencode", "-m", f"{OPENCODE_PROVIDER}/{model_id}"])
