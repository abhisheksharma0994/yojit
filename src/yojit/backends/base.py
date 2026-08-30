"""Shared interface every backend (MLX, llama.cpp, ...) implements."""
from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    name: str  # "mlx_vlm" or "llamacpp"

    @abstractmethod
    def detect(self, path: Path) -> bool:
        """Can this backend serve the model at `path`?"""

    @abstractmethod
    def ensure_installed(self) -> None:
        """Install the runtime if it's missing."""

    @abstractmethod
    def launch(self, model_path: Path, port: int, context: int, output_limit: int, tuning: dict,
               overrides: dict | None = None):
        """Starts the server as a background subprocess, returns the Popen object.
        `tuning` is machine-derived launch tuning; `overrides` are optional per-model user knobs."""

    @abstractmethod
    def health_check(self, port: int) -> bool:
        """Is the server on `port` responding?"""

    @abstractmethod
    def warm_up(self, port: int, model_id: str) -> None:
        """Send a small request so the model is loaded into memory before the
        user's first real request."""
