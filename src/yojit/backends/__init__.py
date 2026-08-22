from pathlib import Path

from .base import Backend
from .llamacpp import LlamaCppBackend
from .mlx import MLXBackend

REGISTRY: dict[str, Backend] = {
    "mlx": MLXBackend(),
    "llamacpp": LlamaCppBackend(),
}


def detect_backend(path: Path) -> Backend | None:
    for backend in REGISTRY.values():
        if backend.detect(path):
            return backend
    return None


def get_backend(name: str) -> Backend:
    if name not in REGISTRY:
        raise KeyError(f"Unknown backend: {name}")
    return REGISTRY[name]
