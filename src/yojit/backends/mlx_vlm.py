"""MLX-VLM backend: wraps mlx_vlm.server, the universal backend for every
MLX-format model -- vision or plain text, dense or MoE.
"""
import subprocess
from pathlib import Path

import requests

from .. import mlx_env
from .base import Backend

LOG_PATH = Path.home() / ".yojit" / "mlx-vlm-server.log"


class MLXVLMBackend(Backend):
    name = "mlx_vlm"

    def detect(self, path: Path) -> bool:
        # Any MLX-format model dir, vision or not -- mlx_vlm.server handles both.
        return (path / "config.json").exists() and any(path.glob("*.safetensors"))

    def ensure_installed(self) -> None:
        # jinja2 isn't a hard dependency of mlx-vlm/transformers, but chat-template
        # rendering fails at request time without it -- install it explicitly too.
        missing = [pkg for pkg, mod in (("mlx-vlm", "mlx_vlm"), ("jinja2", "jinja2"))
                   if not mlx_env.is_installed(mod)]
        if not missing:
            return
        print(f"Installing {', '.join(missing)} into yojit's isolated venv (~/.yojit/venv)...")
        mlx_env.pip_install(*missing)

    def launch(self, model_path: Path, port: int, context: int, output_limit: int, tuning: dict,
               overrides: dict | None = None):
        overrides = overrides or {}
        cmd = mlx_env.module_command(
            "mlx_vlm.server",
            "--model", str(model_path),
            "--port", str(port),
            "--prefill-step-size", str(tuning["prefill_step_size"]),
            "--max-tokens", str(output_limit),
            "--max-kv-size", str(context),  # real context cap
            "--max-num-seqs", str(overrides.get("max_concurrent_predictions", tuning["decode_concurrency"])),
        )
        if overrides.get("kv_cache_quant"):
            cmd += ["--kv-bits", str(overrides["kv_cache_quant"])]
        if overrides.get("kv_group_size"):
            cmd += ["--kv-group-size", str(overrides["kv_group_size"])]
        if overrides.get("quantized_kv_start") is not None:
            cmd += ["--quantized-kv-start", str(overrides["quantized_kv_start"])]
        # seed isn't a launch flag here -- it's per-request only.
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
