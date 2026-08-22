"""MLX backend: wraps mlx_lm.server with launch parameters computed from this
machine's actual specs (see classify.compute_launch_tuning), not fixed
constants."""
import shutil
import subprocess
from pathlib import Path

import requests

from .base import Backend

LOG_PATH = Path.home() / ".yojit" / "mlx-server.log"


class MLXBackend(Backend):
    name = "mlx"

    def detect(self, path: Path) -> bool:
        return (path / "config.json").exists() and any(path.glob("*.safetensors"))

    def ensure_installed(self) -> None:
        if shutil.which("mlx_lm.server"):
            return
        print("mlx-lm not found -- installing (pip install --break-system-packages mlx-lm)...")
        subprocess.run(
            ["pip3", "install", "--break-system-packages", "--upgrade", "mlx-lm"],
            check=True,
        )

    def launch(self, model_path: Path, port: int, context: int, output_limit: int, tuning: dict):
        cmd = [
            "mlx_lm.server",
            "--model", str(model_path),
            "--port", str(port),
            "--prefill-step-size", str(tuning["prefill_step_size"]),
            "--prompt-cache-bytes", str(tuning["prompt_cache_bytes"]),
            "--decode-concurrency", str(tuning["decode_concurrency"]),
            "--prompt-concurrency", str(tuning["prompt_concurrency"]),
            # mlx_lm.server has no CLI flag to cap context length -- it
            # accepts whatever a client sends, up to the model's native max.
            # --max-tokens is the one enforceable limit it does offer, so the
            # computed "safe" output size is at least enforced here even
            # though context itself is only an advisory hint to opencode
            # (see opencode_sync.py / README's Safety model section).
            "--max-tokens", str(output_limit),
        ]
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_PATH, "w")
        # stdout/stderr go straight to a file, never to a PIPE we don't drain --
        # an unread PIPE fills its OS buffer and makes the child block on write().
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
