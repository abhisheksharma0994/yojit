"""Diagnostics: port conflicts, RAM headroom, opencode.json validity, prerequisite checks."""
import json
import shutil

from . import classify, manifest, opencode_sync, server, specs


def run() -> list[str]:
    findings = []
    s = specs.detect()
    findings.append(f"Platform: {s.platform}, chip: {s.chip}, RAM: {round(s.total_ram_gb)} GB, "
                     f"free disk: {round(s.free_disk_gb)} GB")

    pid = server._port_pid(server.PORT)
    if pid:
        findings.append(f"WARNING: port {server.PORT} is currently in use by PID {pid}")
    else:
        findings.append(f"OK: port {server.PORT} is free")

    if not shutil.which("mlx_lm.server") and s.is_apple_silicon:
        findings.append("WARNING: mlx-lm not installed (needed for MLX models)")
    if not shutil.which("llama-server"):
        findings.append("INFO: llama-server not installed (needed for GGUF models)")
    if not shutil.which("opencode"):
        findings.append("WARNING: opencode not installed")

    config_path = opencode_sync.config_path()
    if not config_path.exists():
        findings.append(f"WARNING: opencode.json not found at {config_path}")
    else:
        try:
            json.loads(config_path.read_text())
            findings.append("OK: opencode.json is valid JSON")
        except Exception as e:
            findings.append(f"ERROR: opencode.json is invalid JSON: {e}")

    default = manifest.get_default()
    if default:
        entry = manifest.get_model(default)
        if entry:
            fraction = entry.get("size_gb", 0) / s.total_ram_gb if s.total_ram_gb else 1
            if fraction > classify.MEDIUM_TIER_MAX_FRACTION:
                findings.append(
                    f"WARNING: default model {default} uses {round(fraction * 100)}% of RAM "
                    f"as weights alone -- real crash risk (Metal OOM), not just a tight fit. "
                    f"Verified crash threshold on similar hardware is ~50%+."
                )
            else:
                findings.append(f"OK: default model {default} fits comfortably ({entry.get('tier')} tier)")
    else:
        findings.append("INFO: no default model set yet")

    return findings
