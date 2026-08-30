"""Isolated venv for the MLX runtime (mlx-vlm), kept separate from system
Python so an install here can never break anything outside yojit.
"""
import subprocess
import sys
from pathlib import Path

VENV_DIR = Path.home() / ".yojit" / "venv"


def venv_python() -> Path:
    return VENV_DIR / "bin" / "python3"


def ensure_venv() -> None:
    if venv_python().exists():
        return
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    subprocess.run([str(venv_python()), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)


def is_installed(module: str) -> bool:
    ensure_venv()
    r = subprocess.run(
        [str(venv_python()), "-c", f"import {module}"],
        capture_output=True,
    )
    return r.returncode == 0


def pip_install(*packages: str, upgrade: bool = False) -> None:
    ensure_venv()
    cmd = [str(venv_python()), "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    subprocess.run(cmd, check=True)


def module_command(module: str, *args: str) -> list[str]:
    # Always the venv's own interpreter, never whatever's on PATH.
    # No ensure_venv() call -- pure path-building, safe to test without a real venv.
    return [str(venv_python()), "-m", module, *args]
