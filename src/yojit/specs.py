"""Hardware detection: RAM, chip, disk space, platform, CPU cores."""
import builtins
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class Specs:
    platform: str       # "darwin", "linux", "windows"
    is_apple_silicon: bool
    chip: str
    total_ram_gb: float
    free_disk_gb: float
    cpu_cores: int


def cpu_cores() -> int:
    """Total logical CPU count -- portable across chip generations, unlike sysctl core-tier names."""
    return os.cpu_count() or 4


def _mac_chip_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
    except Exception:
        return "unknown"


def total_ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(out) / (1024 ** 3)
        if system == "Linux":
            with builtins.open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass
    return 16.0  # conservative fallback


def free_disk_gb(path: str = "/") -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return 0.0


def detect() -> Specs:
    system = platform.system().lower()
    is_apple_silicon = system == "darwin" and platform.machine() == "arm64"
    chip = _mac_chip_name() if system == "darwin" else platform.processor() or "unknown"
    return Specs(
        platform=system,
        is_apple_silicon=is_apple_silicon,
        chip=chip,
        total_ram_gb=total_ram_gb(),
        free_disk_gb=free_disk_gb(),
        cpu_cores=cpu_cores(),
    )
