"""Shared prerequisite installation for things that aren't a model-serving
backend but that `init` and `serve` both need -- currently just opencode.
"""
import platform
import shutil
import subprocess


def ensure_opencode_installed() -> bool:
    """Returns True if opencode is available. Auto-install is macOS/Homebrew only."""
    if shutil.which("opencode"):
        return True

    if platform.system() != "Darwin" or not shutil.which("brew"):
        print("opencode not found, and auto-install is only supported on macOS with Homebrew right now.")
        print("Install it manually from https://opencode.ai, then re-run this command.")
        return False

    print("opencode not found -- installing via Homebrew (brew install anomalyco/tap/opencode)...")
    subprocess.run(["brew", "install", "anomalyco/tap/opencode"], check=False)
    return bool(shutil.which("opencode"))
