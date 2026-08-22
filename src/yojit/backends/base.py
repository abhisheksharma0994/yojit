"""Shared interface every backend (MLX, llama.cpp, ...) implements."""
from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    name: str  # "mlx" or "llamacpp"

    @abstractmethod
    def detect(self, path: Path) -> bool:
        """Can this backend serve the model at `path`?"""

    @abstractmethod
    def ensure_installed(self) -> None:
        """Install the runtime (mlx-lm / llama.cpp) if it's missing."""

    @abstractmethod
    def launch(self, model_path: Path, port: int, context: int, output_limit: int, tuning: dict):
        """Start the server as a background subprocess. Returns the Popen object.
        `tuning` is classify.compute_launch_tuning()'s output -- every other
        server parameter (prefill chunk size, KV-cache ceiling, threads, GPU
        layers, batch sizes, concurrency), computed from this machine's
        actual specs rather than fixed constants."""

    @abstractmethod
    def health_check(self, port: int) -> bool:
        """Is the server on `port` responding?"""

    @abstractmethod
    def warm_up(self, port: int, model_id: str) -> None:
        """Send a small request so the model is loaded into memory before the
        user's first real request."""
