import json

from yojit import classify, mlx_env
from yojit.backends import detect_backend, get_backend
from yojit.backends.llamacpp import LlamaCppBackend
from yojit.backends.mlx_vlm import MLXVLMBackend


def _tuning(**overrides):
    """A representative tuning dict, as classify.compute_launch_tuning()
    would produce -- tests override only the fields they care about."""
    base = classify.compute_launch_tuning(weight_gb=8.0, ram_gb=24.0, cpu_cores=12)
    base.update(overrides)
    return base


def test_llamacpp_ensure_installed_skips_when_already_present(mocker):
    mocker.patch("yojit.backends.llamacpp.shutil.which", return_value="/usr/bin/llama-server")
    mock_run = mocker.patch("yojit.backends.llamacpp.subprocess.run")
    LlamaCppBackend().ensure_installed()
    mock_run.assert_not_called()


def test_llamacpp_ensure_installed_installs_via_brew_when_missing(mocker):
    mocker.patch("yojit.backends.llamacpp.shutil.which", return_value=None)
    mock_run = mocker.patch("yojit.backends.llamacpp.subprocess.run")
    LlamaCppBackend().ensure_installed()
    mock_run.assert_called_once_with(["brew", "install", "llama.cpp"], check=True)


def test_llamacpp_health_check_true_on_200(mocker):
    mock_get = mocker.patch("yojit.backends.llamacpp.requests.get")
    mock_get.return_value.status_code = 200
    assert LlamaCppBackend().health_check(8080) is True


def test_llamacpp_health_check_false_on_non_200(mocker):
    mock_get = mocker.patch("yojit.backends.llamacpp.requests.get")
    mock_get.return_value.status_code = 500
    assert LlamaCppBackend().health_check(8080) is False


def test_llamacpp_health_check_false_on_exception(mocker):
    mocker.patch("yojit.backends.llamacpp.requests.get", side_effect=Exception("connection refused"))
    assert LlamaCppBackend().health_check(8080) is False


def test_llamacpp_warm_up_sends_expected_request(mocker):
    mock_post = mocker.patch("yojit.backends.llamacpp.requests.post")
    LlamaCppBackend().warm_up(8080, "/some/model/path")
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8080/v1/chat/completions"
    assert kwargs["json"]["model"] == "/some/model/path"


def test_llamacpp_warm_up_does_not_raise_on_failure(mocker, capsys):
    mocker.patch("yojit.backends.llamacpp.requests.post", side_effect=Exception("timed out"))
    LlamaCppBackend().warm_up(8080, "/some/model/path")  # must not raise
    assert "Warm-up request failed" in capsys.readouterr().out


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


def test_detect_backend_registry_finds_mlx_vlm_for_any_mlx_format_dir(tmp_path):
    """Any well-formed MLX model dir -- vision or plain text -- routes to mlx_vlm."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"")
    backend = detect_backend(model_dir)
    assert backend is not None and backend.name == "mlx_vlm"


def test_detect_backend_registry_returns_none_for_unrecognized_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert detect_backend(model_dir) is None


def test_get_backend_by_name():
    assert get_backend("mlx_vlm").name == "mlx_vlm"
    assert get_backend("llamacpp").name == "llamacpp"
    try:
        get_backend("nonexistent")
        assert False, "expected KeyError"
    except KeyError:
        pass


# --- Vision + plain-text detection: mlx_vlm serves both -------------------

def _vision_config_dir(tmp_path):
    model_dir = tmp_path / "vision-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vision_config": {"hidden_size": 1024},
        "image_token_id": 248056,
    }))
    (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


def _text_only_config_dir(tmp_path):
    model_dir = tmp_path / "text-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


def test_mlx_vlm_backend_accepts_a_vision_model_dir(tmp_path):
    assert MLXVLMBackend().detect(_vision_config_dir(tmp_path)) is True


def test_mlx_vlm_backend_accepts_a_text_only_model_dir_too(tmp_path):
    """mlx_vlm.server is the universal MLX backend, not vision-only."""
    assert MLXVLMBackend().detect(_text_only_config_dir(tmp_path)) is True


def test_mlx_vlm_backend_rejects_a_dir_missing_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"")
    assert MLXVLMBackend().detect(model_dir) is False


def test_detect_backend_routes_vision_model_to_mlx_vlm(tmp_path):
    backend = detect_backend(_vision_config_dir(tmp_path))
    assert backend is not None and backend.name == "mlx_vlm"


def test_detect_backend_routes_text_only_model_to_mlx_vlm_too(tmp_path):
    backend = detect_backend(_text_only_config_dir(tmp_path))
    assert backend is not None and backend.name == "mlx_vlm"


def test_mlx_vlm_ensure_installed_skips_when_already_present(mocker):
    mocker.patch.object(mlx_env, "is_installed", return_value=True)
    mock_pip_install = mocker.patch.object(mlx_env, "pip_install")
    MLXVLMBackend().ensure_installed()
    mock_pip_install.assert_not_called()


def test_mlx_vlm_ensure_installed_installs_when_missing(mocker):
    mocker.patch.object(mlx_env, "is_installed", return_value=False)
    mock_pip_install = mocker.patch.object(mlx_env, "pip_install")
    MLXVLMBackend().ensure_installed()
    mock_pip_install.assert_called_once_with("mlx-vlm")


def test_mlx_vlm_health_check_true_on_200(mocker):
    mock_get = mocker.patch("yojit.backends.mlx_vlm.requests.get")
    mock_get.return_value.status_code = 200
    assert MLXVLMBackend().health_check(8080) is True


def test_mlx_vlm_health_check_false_on_non_200(mocker):
    mock_get = mocker.patch("yojit.backends.mlx_vlm.requests.get")
    mock_get.return_value.status_code = 500
    assert MLXVLMBackend().health_check(8080) is False


def test_mlx_vlm_health_check_false_on_exception(mocker):
    mocker.patch("yojit.backends.mlx_vlm.requests.get", side_effect=Exception("connection refused"))
    assert MLXVLMBackend().health_check(8080) is False


def test_mlx_vlm_warm_up_sends_expected_request(mocker):
    mock_post = mocker.patch("yojit.backends.mlx_vlm.requests.post")
    MLXVLMBackend().warm_up(8080, "/some/model/path")
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8080/v1/chat/completions"
    assert kwargs["json"]["model"] == "/some/model/path"


def test_mlx_vlm_warm_up_does_not_raise_on_failure(mocker, capsys):
    mocker.patch("yojit.backends.mlx_vlm.requests.post", side_effect=Exception("timed out"))
    MLXVLMBackend().warm_up(8080, "/some/model/path")  # must not raise
    assert "Warm-up request failed" in capsys.readouterr().out


def test_mlx_vlm_launch_uses_venv_python_and_module_invocation(tmp_path, mocker):
    """The venv's own interpreter must always be used explicitly, never PATH resolution."""
    mock_popen = mocker.patch("yojit.backends.mlx_vlm.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXVLMBackend()
    backend.launch(tmp_path / "model", port=8080, context=16384, output_limit=1024, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert args[0] == str(mlx_env.venv_python())
    assert args[1] == "-m"
    assert args[2] == "mlx_vlm.server"
    assert "--max-kv-size" in args and "16384" in args


def test_mlx_vlm_launch_uses_the_computed_tuning_values(tmp_path, mocker):
    """Prefill chunking and concurrency cap must reflect per-machine tuning, not fixed constants."""
    mock_popen = mocker.patch("yojit.backends.mlx_vlm.subprocess.Popen")
    mocker.patch("builtins.open")
    tuning = _tuning(prefill_step_size=2048, decode_concurrency=1)
    backend = MLXVLMBackend()
    backend.launch(tmp_path / "model", port=8080, context=4096, output_limit=1024, tuning=tuning)

    args = mock_popen.call_args[0][0]
    assert "--prefill-step-size" in args and "2048" in args
    assert "--max-num-seqs" in args and "1" in args


def test_mlx_vlm_launch_passes_output_limit_as_max_tokens(tmp_path, mocker):
    mock_popen = mocker.patch("yojit.backends.mlx_vlm.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXVLMBackend()
    backend.launch(tmp_path / "model", port=8080, context=4096, output_limit=2048, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert "--max-tokens" in args
    assert args[args.index("--max-tokens") + 1] == "2048"


def test_mlx_vlm_launch_never_uses_an_unread_pipe(tmp_path, mocker):
    """stdout must go to a file, never PIPE -- an unread pipe deadlocks the child once its buffer fills."""
    import subprocess as subprocess_module

    mock_popen = mocker.patch("yojit.backends.mlx_vlm.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXVLMBackend()
    backend.launch(tmp_path / "model", port=8080, context=4096, output_limit=1024, tuning=_tuning())

    kwargs = mock_popen.call_args[1]
    assert kwargs.get("stdout") != subprocess_module.PIPE


def test_mlx_vlm_launch_applies_kv_cache_quant_overrides(tmp_path, mocker):
    mock_popen = mocker.patch("yojit.backends.mlx_vlm.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXVLMBackend()
    backend.launch(tmp_path / "model", port=8080, context=16384, output_limit=1024, tuning=_tuning(),
                    overrides={"kv_cache_quant": 8, "kv_group_size": 64, "quantized_kv_start": 5000,
                               "max_concurrent_predictions": 2})

    args = mock_popen.call_args[0][0]
    assert "--kv-bits" in args and "8" in args
    assert "--kv-group-size" in args and "64" in args
    assert "--quantized-kv-start" in args and "5000" in args
    assert "--max-num-seqs" in args and "2" in args


def test_mlx_vlm_launch_omits_kv_quant_flags_when_not_overridden(tmp_path, mocker):
    mock_popen = mocker.patch("yojit.backends.mlx_vlm.subprocess.Popen")
    mocker.patch("builtins.open")
    backend = MLXVLMBackend()
    backend.launch(tmp_path / "model", port=8080, context=16384, output_limit=1024, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert "--kv-bits" not in args
    assert "--kv-group-size" not in args
    assert "--quantized-kv-start" not in args


def test_llamacpp_launch_applies_seed_and_kv_cache_quant_overrides(tmp_path, mocker):
    mock_popen = mocker.patch("yojit.backends.llamacpp.subprocess.Popen")
    mocker.patch("builtins.open")
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"")
    backend = LlamaCppBackend()
    backend.launch(gguf_file, port=8080, context=2048, output_limit=512, tuning=_tuning(),
                    overrides={"seed": 42, "kv_cache_quant": "q8_0", "max_concurrent_predictions": 2})

    args = mock_popen.call_args[0][0]
    assert "--seed" in args and "42" in args
    assert "--cache-type-k" in args and "q8_0" in args[args.index("--cache-type-k") + 1]
    assert "--cache-type-v" in args and "q8_0" in args[args.index("--cache-type-v") + 1]
    assert args[args.index("--parallel") + 1] == "2"


def test_llamacpp_launch_defaults_to_random_seed_and_single_slot(tmp_path, mocker):
    mock_popen = mocker.patch("yojit.backends.llamacpp.subprocess.Popen")
    mocker.patch("builtins.open")
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"")
    backend = LlamaCppBackend()
    backend.launch(gguf_file, port=8080, context=2048, output_limit=512, tuning=_tuning())

    args = mock_popen.call_args[0][0]
    assert args[args.index("--seed") + 1] == "-1"
    assert args[args.index("--parallel") + 1] == "1"
    assert "--cache-type-k" not in args


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
    """llama-server defaults to 4 parallel slots, each with its own KV cache -- must pin to 1."""
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
    """threads/gpu-layers/batch-size/ubatch-size must reflect this machine's real specs."""
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
