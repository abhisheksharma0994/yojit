from pathlib import Path

from yojit import classify
from yojit.backends import detect_backend, get_backend
from yojit.backends.llamacpp import LlamaCppBackend
from yojit.backends.mlx import MLXBackend


def _tuning(**overrides):
    """A representative tuning dict, as classify.compute_launch_tuning()
    would produce -- tests override only the fields they care about."""
    base = classify.compute_launch_tuning(weight_gb=8.0, ram_gb=24.0, cpu_cores=12)
    base.update(overrides)
    return base


def test_mlx_backend_detects_a_real_mlx_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"")
    assert MLXBackend().detect(model_dir) is True


def test_mlx_backend_rejects_a_dir_missing_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"")
    assert MLXBackend().detect(model_dir) is False


def test_llamacpp_backend_detects_a_gguf_file(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"")
    assert LlamaCppBackend().detect(gguf_file) is True


def test_llamacpp_backend_detects_a_dir_containing_gguf(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"")
    assert LlamaCppBackend().detect(model_dir) is True


def test_llamacpp_backend_rejects_non_gguf_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"")
    assert LlamaCppBackend().detect(model_dir) is False


def test_detect_backend_registry_finds_mlx(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"")
    backend = detect_backend(model_dir)
    assert backend is not None and backend.name == "mlx"


def test_detect_backend_registry_returns_none_for_unrecognized_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert detect_backend(model_dir) is None


def test_get_backend_by_name():
    assert get_backend("mlx").name == "mlx"
    assert get_backend("llamacpp").name == "llamacpp"
    try:
        get_backend("nonexistent")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_mlx_launch_uses_the_computed_tuning_values(tmp_path, mocker):
    """Regression: the flags that fixed the real Metal OOM crashes tonight
    (prefill chunking + a KV-cache ceiling) must always be present -- and
    now must reflect classify.compute_launch_tuning()'s per-machine values,
    not fixed constants."""
    mock_popen = mocker.patch("yojit.backends.mlx.subprocess.Popen")
    mocker.patch("builtins.open")  # log file handle
    tuning = _tuning(prefill_step_size=2048, prompt_cache_bytes=3_000_000_000,
                      decode_concurrency=1, prompt_concurrency=1)
    backend = MLXBackend()
    backend.launch(tmp_path / "model", port=8080, context=4096, output_limit=1024, tuning=tuning)

    args = mock_popen.call_args[0][0]
    assert "--prefill-step-size" in args and "2048" in args
    assert "--prompt-cache-bytes" in args and "3000000000" in args
    assert "--decode-concurrency" in args and "1" in args
    assert "--prompt-concurrency" in args and "1" in args


def test_mlx_launch_passes_output_limit_as_max_tokens(tmp_path, mocker):
    """Regression: output_limit was computed by classify.py but never
    actually enforced server-side for MLX (only reported to opencode as a
    hint) -- --max-tokens is the one enforceable cap mlx_lm.server offers."""
    mock_popen = mocker.patch("yojit.backends.mlx.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXBackend()
    backend.launch(tmp_path / "model", port=8080, context=4096, output_limit=2048, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert "--max-tokens" in args
    assert args[args.index("--max-tokens") + 1] == "2048"


def test_mlx_launch_never_uses_an_unread_pipe(tmp_path, mocker):
    """Regression: subprocess.Popen(stdout=PIPE) without a reader deadlocks
    the child once the OS pipe buffer fills -- this caused a real hang
    during testing. stdout must go to a file, never PIPE."""
    import subprocess as subprocess_module

    mock_popen = mocker.patch("yojit.backends.mlx.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXBackend()
    backend.launch(tmp_path / "model", port=8080, context=4096, output_limit=1024, tuning=_tuning())

    kwargs = mock_popen.call_args[1]
    assert kwargs.get("stdout") != subprocess_module.PIPE


def test_llamacpp_launch_uses_context_and_mlock(tmp_path, mocker):
    mock_popen = mocker.patch("yojit.backends.llamacpp.subprocess.Popen")
    mocker.patch("builtins.open")
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"")
    backend = LlamaCppBackend()
    backend.launch(gguf_file, port=8080, context=2048, output_limit=512, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert "--ctx-size" in args and "2048" in args


def test_llamacpp_launch_pins_to_a_single_parallel_slot(tmp_path, mocker):
    """Regression: llama-server defaults to 4 parallel slots, each with its
    own KV cache sized to --ctx-size -- discovered via real testing (server
    log showed n_slots=4). Without pinning to one slot, real memory usage
    could be ~4x what classify.py's RAM math assumed as safe."""
    mock_popen = mocker.patch("yojit.backends.llamacpp.subprocess.Popen")
    mocker.patch("builtins.open")
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"")
    backend = LlamaCppBackend()
    backend.launch(gguf_file, port=8080, context=2048, output_limit=512, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert "--parallel" in args
    assert args[args.index("--parallel") + 1] == "1"
    assert "--mlock" in args


def test_llamacpp_launch_uses_the_computed_tuning_values(tmp_path, mocker):
    """Regression: threads/gpu-layers/batch-size/ubatch-size were never set
    at all before -- llama-server ran on whatever its own defaults were,
    with zero awareness of this machine's actual CPU core count or RAM
    headroom."""
    mock_popen = mocker.patch("yojit.backends.llamacpp.subprocess.Popen")
    mocker.patch("builtins.open")
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"")
    tuning = _tuning(threads=11, ngl=999, batch_size=2048, ubatch_size=512)
    backend = LlamaCppBackend()
    backend.launch(gguf_file, port=8080, context=2048, output_limit=512, tuning=tuning)

    args = mock_popen.call_args[0][0]
    assert "--threads" in args and "11" in args
    assert "--gpu-layers" in args and "999" in args
    assert "--batch-size" in args and "2048" in args
    assert "--ubatch-size" in args and "512" in args
