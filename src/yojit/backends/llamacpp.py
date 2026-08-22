"""llama.cpp backend: wraps llama-server with launch parameters computed
from this machine's actual specs (see classify.compute_launch_tuning), not
fixed constants.

Memory-safety flags here are llama.cpp's own equivalents to what was tuned
for MLX -- they still need a real stress-testing pass (large-prompt OOM
behavior on llama.cpp differs from MLX's Metal allocator) before being
trusted as blindly as the MLX flags, which were verified against real
crashes.
"""
import shutil
import subprocess
from pathlib import Path

import requests

from .base import Backend

LOG_PATH = Path.home() / ".yojit" / "llamacpp-server.log"


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def detect(self, path: Path) -> bool:
        return path.suffix == ".gguf" or (path.is_dir() and any(path.glob("*.gguf")))

    def ensure_installed(self) -> None:
        if shutil.which("llama-server"):
            return
        print("llama-server not found -- installing via Homebrew (brew install llama.cpp)...")
        subprocess.run(["brew", "install", "llama.cpp"], check=True)

    def launch(self, model_path: Path, port: int, context: int, output_limit: int, tuning: dict):
        gguf_file = model_path if model_path.suffix == ".gguf" else next(model_path.glob("*.gguf"))
        cmd = [
            "llama-server",
            "--model", str(gguf_file),
            "--port", str(port),
            "--ctx-size", str(context),
            "--n-predict", str(output_limit),
            "--mlock",  # avoid the model getting paged out mid-generation
            "--threads", str(tuning["threads"]),
            "--gpu-layers", str(tuning["ngl"]),
            "--batch-size", str(tuning["batch_size"]),
            "--ubatch-size", str(tuning["ubatch_size"]),
            # llama-server defaults to 4 parallel request slots, each
            # allocating its own KV cache sized to --ctx-size -- found via
            # real testing (server log showed "n_slots = 4"). classify.py's
            # RAM math assumes a single request's worth of KV cache, so
            # without this the real memory usage could be up to 4x what was
            # computed as "safe". Pin to one slot to match that assumption.
            "--parallel", "1",
        ]
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_PATH, "w")
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    def health_check(self, port: int) -> bool:
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def warm_up(self, port: int, model_id: str) -> None:
        try:
            requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json={"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
                timeout=300,
            )
        except Exception as e:
            print(f"Warm-up request failed (server may still be usable): {e}")
