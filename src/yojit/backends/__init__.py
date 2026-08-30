from pathlib import Path

from .base import Backend
from .llamacpp import LlamaCppBackend
from .mlx_vlm import MLXVLMBackend

# Old "mlx" (mlx-lm) entries are migrated to "mlx_vlm" by manifest.load().
REGISTRY: dict[str, Backend] = {
    "mlx_vlm": MLXVLMBackend(),
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
