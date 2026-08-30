"""Diagnostics: port conflicts, RAM headroom, opencode.json validity, prerequisite checks."""
import json
import shutil

from . import classify, manifest, mlx_env, opencode_sync, server, specs


def _check_port(port: int) -> str:
    pid = server._port_pid(port)
    if pid:
        return f"WARNING: port {port} is currently in use by PID {pid}"
    return f"OK: port {port} is free"


def _check_prerequisites(s: specs.Specs) -> list[str]:
    findings = []
    # mlx-vlm lives in yojit's own venv, never on PATH -- check via mlx_env, not shutil.which().
    if s.is_apple_silicon and not mlx_env.is_installed("mlx_vlm"):
        findings.append("WARNING: mlx-vlm not installed (needed for MLX models)")
    if not shutil.which("llama-server"):
        findings.append("INFO: llama-server not installed (needed for GGUF models)")
    if not shutil.which("opencode"):
        findings.append("WARNING: opencode not installed")
    return findings


def _check_opencode_json() -> str:
    config_path = opencode_sync.config_path()
    if not config_path.exists():
        return f"WARNING: opencode.json not found at {config_path}"
    try:
        json.loads(config_path.read_text())
        return "OK: opencode.json is valid JSON"
    except Exception as e:
        return f"ERROR: opencode.json is invalid JSON: {e}"


def _check_default_model(s: specs.Specs) -> str:
    default = manifest.get_default()
    if not default:
        return "INFO: no default model set yet"
    entry = manifest.get_model(default)
    if not entry:
        return f"WARNING: default model {default} has no manifest entry"
    fraction = entry.get("size_gb", 0) / s.total_ram_gb if s.total_ram_gb else 1
    if fraction > classify.MEDIUM_TIER_MAX_FRACTION:
        return (f"WARNING: default model {default} uses {round(fraction * 100)}% of RAM "
                f"as weights alone -- real OOM crash risk, not just a tight fit.")
    return f"OK: default model {default} fits comfortably ({entry.get('tier')} tier)"


def run() -> list[str]:
    s = specs.detect()
    findings = [f"Platform: {s.platform}, chip: {s.chip}, RAM: {round(s.total_ram_gb)} GB, "
                f"free disk: {round(s.free_disk_gb)} GB"]
    findings.append(_check_port(server.PORT))
    findings.extend(_check_prerequisites(s))
    findings.append(_check_opencode_json())
    findings.append(_check_default_model(s))
    return findings
