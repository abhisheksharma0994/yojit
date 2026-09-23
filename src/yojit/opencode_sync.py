"""Writes the backend-neutral "local" provider block into opencode.json.
Both backends expose an OpenAI-compatible endpoint on the same local port."""
import json
import os
from pathlib import Path

from . import locking, manifest

CONFIG_PATH_DEFAULT = Path.home() / ".config" / "opencode" / "opencode.json"
CONFIG_PATH_ENV_VAR = "YOJIT_OPENCODE_CONFIG"
PORT = int(os.environ.get("YOJIT_PORT", "8080"))


def config_path() -> Path:
    """Overridable via YOJIT_OPENCODE_CONFIG -- lets tests point at a
    throwaway file instead of a developer's real opencode.json."""
    override = os.environ.get(CONFIG_PATH_ENV_VAR)
    return Path(override) if override else CONFIG_PATH_DEFAULT


def sync(running_model: str | None = None,
         running_limits: tuple[int, int] | None = None) -> str:
    """Writes every installed model into opencode.json's "local" provider.

    Holds the shared lock across the whole read-modify-write and writes
    atomically: two concurrent invocations used to discard each other's models
    (formal/tla/OpencodeSync.tla shows the interleaving).

    `running_limits` is the (context, output) the running model was *actually*
    launched with -- pass it whenever serve() had to shrink the context to fit
    the KV cache, so opencode is never told a bigger window than the live server
    is holding. Omitted, the manifest's stored values are used as before.
    """
    CONFIG_PATH = config_path()
    if not CONFIG_PATH.exists():
        return f"opencode.json not found at {CONFIG_PATH}, skipping sync"

    lines = []
    with locking.state_lock():
        config = json.loads(CONFIG_PATH.read_text())
        config.setdefault("provider", {})
        local_provider = config["provider"].setdefault("local", {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Local (yojit)",
            "options": {"baseURL": f"http://localhost:{PORT}/v1"},
        })

        models_obj = {}
        for model_id, entry in manifest.list_models().items():
            label = model_id
            if running_model and model_id == running_model:
                label = f"{model_id} (running)"
            # Keyed by the exact local path the backend launches with, not the
            # manifest's repo-style model_id -- the server matches "model" by
            # exact string equality against --model.
            key = str(manifest.models_root() / entry["store_path"])
            context = entry.get("context", 4096)
            output = entry.get("output", 1024)
            if running_limits and model_id == running_model:
                context, output = running_limits
            models_obj[key] = {
                "name": f"{label} ({entry.get('backend', 'local')})",
                "limit": {"context": context, "output": output},
            }
            marker = " <- currently running" if model_id == running_model else ""
            lines.append(
                f"  - {model_id}{marker}\n"
                f"      backend: {entry.get('backend')}, tier: {entry.get('tier')}, "
                f"weights: {entry.get('size_gb')} GB, context: {context}, "
                f"max output: {output}"
            )

        local_provider["models"] = models_obj
        locking.atomic_write_text(CONFIG_PATH, json.dumps(config, indent=2) + "\n")

    return f"Synced {len(lines)} model(s) into {CONFIG_PATH}:\n" + "\n".join(lines)
