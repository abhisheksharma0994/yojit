"""Shared prerequisite installation for things that aren't a model-serving
backend (see backends/mlx.py and backends/llamacpp.py for those) but that
`init` and `serve` both need available -- currently just opencode, the
client that talks to whichever backend is running.

This closes a real gap: the README promised "missing prerequisites are
installed on demand," but opencode's absence was only ever detected and
printed as a manual-install instruction, never actually installed.
"""
import platform
import shutil
import subprocess


def ensure_opencode_installed() -> bool:
    """Returns True if opencode is available (already present, or just
    installed successfully). Auto-install is currently only implemented for
    macOS with Homebrew, matching how it was installed during this project's
    own development -- other platforms get a clear manual-install pointer
    instead of a guessed package-manager command that might be wrong."""
    if shutil.which("opencode"):
        return True

    if platform.system() != "Darwin" or not shutil.which("brew"):
        print("opencode not found, and auto-install is only supported on macOS with Homebrew right now.")
        print("Install it manually from https://opencode.ai, then re-run this command.")
        return False

    print("opencode not found -- installing via Homebrew (brew install anomalyco/tap/opencode)...")
    subprocess.run(["brew", "install", "anomalyco/tap/opencode"], check=False)
    return bool(shutil.which("opencode"))
