"""Writes the backend-neutral "local" provider block into opencode.json.

Both backends (MLX via mlx_lm.server, llama.cpp via llama-server) expose an
OpenAI-compatible endpoint on the same local port, so opencode's config never
needs to know which backend is actually running.
"""
import json
import os
from pathlib import Path

from . import manifest

CONFIG_PATH_DEFAULT = Path.home() / ".config" / "opencode" / "opencode.json"
CONFIG_PATH_ENV_VAR = "YOJIT_OPENCODE_CONFIG"
PORT = int(os.environ.get("YOJIT_PORT", "8080"))


def config_path() -> Path:
    """Overridable via YOJIT_OPENCODE_CONFIG -- lets tests point at a
    throwaway file instead of a developer's real opencode.json."""
    override = os.environ.get(CONFIG_PATH_ENV_VAR)
    return Path(override) if override else CONFIG_PATH_DEFAULT


def sync(running_model: str | None = None) -> str:
    """Writes every installed model into opencode.json's "local" provider.
    Returns a human-readable summary string."""
    CONFIG_PATH = config_path()
    if not CONFIG_PATH.exists():
        return f"opencode.json not found at {CONFIG_PATH}, skipping sync"

    config = json.loads(CONFIG_PATH.read_text())
    config.setdefault("provider", {})
    local_provider = config["provider"].setdefault("local", {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Local (yojit)",
        "options": {"baseURL": f"http://localhost:{PORT}/v1"},
    })

    models_obj = {}
    lines = []
    for model_id, entry in manifest.list_models().items():
        label = model_id
        if running_model and model_id == running_model:
            label = f"{model_id} (running)"
        # Keyed by the exact local path the backend launches with (see
        # server.py's _attempt_launch), NOT the manifest's HF-repo-style
        # model_id -- mlx_lm.server matches the "model" field of every
        # request against a plain string equality check with whatever was
        # passed to --model at launch (ModelProvider.load in mlx_lm/server.py
        # only special-cases the literal sentinel "default_model", nothing
        # else). Keying by model_id here silently broke every actual chat
        # request from opencode: the mismatched string made mlx_lm.server
        # treat each request as "load a different model", which tries to
        # resolve model_id as a fresh HF Hub repo -- hanging when online,
        # failing outright when offline. The "name" field still shows the
        # friendly repo-style label in opencode's picker.
        key = str(manifest.models_root() / entry["store_path"])
        models_obj[key] = {
            "name": f"{label} ({entry.get('backend', 'local')})",
            "limit": {
                "context": entry.get("context", 4096),
                "output": entry.get("output", 1024),
            },
        }
        marker = " <- currently running" if model_id == running_model else ""
        lines.append(
            f"  - {model_id}{marker}\n"
            f"      backend: {entry.get('backend')}, tier: {entry.get('tier')}, "
            f"weights: {entry.get('size_gb')} GB, context: {entry.get('context')}, "
            f"max output: {entry.get('output')}"
        )

    local_provider["models"] = models_obj
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    summary = f"Synced {len(models_obj)} model(s) into {CONFIG_PATH}:\n" + "\n".join(lines)
    return summary
