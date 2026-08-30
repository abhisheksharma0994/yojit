from yojit import hf_explore


def test_search_builds_expected_params_and_returns_json(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = [{"id": "org/model-a"}]
    mock_get = mocker.patch("yojit.hf_explore.requests.get", return_value=mock_response)

    result = hf_explore.search(query="qwen", limit=10)

    assert result == [{"id": "org/model-a"}]
    mock_response.raise_for_status.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == hf_explore.API
    assert kwargs["params"] == {
        "sort": "downloads", "direction": "-1", "limit": 10, "author": hf_explore.DEFAULT_AUTHOR, "search": "qwen",
    }
    assert kwargs["timeout"] == 15


def test_search_uses_documented_default_limit_when_unspecified(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = []
    mock_get = mocker.patch("yojit.hf_explore.requests.get", return_value=mock_response)

    hf_explore.search()

    assert mock_get.call_args.kwargs["params"]["limit"] == 50


def test_search_omits_author_when_explicitly_none(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = []
    mock_get = mocker.patch("yojit.hf_explore.requests.get", return_value=mock_response)

    hf_explore.search(query=None, author=None)

    assert "author" not in mock_get.call_args.kwargs["params"]
    assert "search" not in mock_get.call_args.kwargs["params"]


def test_repo_files_hits_expected_url_and_returns_json(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = [{"path": "config.json"}]
    mock_get = mocker.patch("yojit.hf_explore.requests.get", return_value=mock_response)

    result = hf_explore.repo_files("org/model-a")

    assert result == [{"path": "config.json"}]
    mock_response.raise_for_status.assert_called_once()
    assert mock_get.call_args[0][0] == f"{hf_explore.API}/org/model-a/tree/main"


def test_detect_backend_from_files_recognizes_mlx_repo():
    files = [{"path": "config.json"}, {"path": "model.safetensors"}, {"path": "tokenizer.json"}]
    assert hf_explore.detect_backend_from_files(files) == "mlx"


def test_detect_backend_from_files_recognizes_gguf_repo():
    files = [{"path": "model-Q4_K_M.gguf"}, {"path": "README.md"}]
    assert hf_explore.detect_backend_from_files(files) == "llamacpp"


def test_detect_backend_from_files_returns_none_for_unrecognized_repo():
    files = [{"path": "README.md"}, {"path": "config.yaml"}]
    assert hf_explore.detect_backend_from_files(files) is None


def test_detect_backend_from_files_requires_both_safetensors_and_config_json():
    """Neither file alone is enough evidence of an mlx repo."""
    assert hf_explore.detect_backend_from_files([{"path": "model.safetensors"}]) is None
    assert hf_explore.detect_backend_from_files([{"path": "config.json"}]) is None


def test_detect_backend_prefers_mlx_when_repo_has_both_formats():
    # Multi-format repos are rare but possible; mlx takes priority since
    # that's the default backend on Apple Silicon (the common case).
    files = [{"path": "config.json"}, {"path": "model.safetensors"}, {"path": "model.gguf"}]
    assert hf_explore.detect_backend_from_files(files) == "mlx"


def test_bit_width_from_name_extracts_trailing_bit_suffix():
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-4bit") == 4
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-8bit") == 8
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-6BIT") == 6  # case-insensitive


def test_bit_width_from_name_extracts_hyphenated_bare_subfolder_names():
    """Regression: real multi-quant repos use bare "N-bit" subfolder names
    (digit-hyphen-bit), a different convention from full repo names'
    "-Nbit" suffix (hyphen-digit-bit, no hyphen before "bit"). Both must
    parse correctly since _resolve_mlx_install uses this for both."""
    assert hf_explore.bit_width_from_name("4-bit") == 4
    assert hf_explore.bit_width_from_name("8-bit") == 8
    assert hf_explore.bit_width_from_name("6-BIT") == 6


def test_bit_width_from_name_returns_none_for_non_bit_suffixed_names():
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-bf16") is None
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-OptiQ-4bit") == 4


def _gguf(path, size_gb):
    return {"path": path, "size": int(size_gb * 1024 ** 3)}


def test_pick_best_gguf_file_prefers_largest_that_still_fits_comfortably():
    """Regression: a real user's `yojit install` picked whatever
    .gguf file happened to be listed first (alphabetically), completely
    ignoring RAM. GGUF quant selection must use the same 'largest that still
    fits' principle as MLX bit selection."""
    files = [
        _gguf("model-Q2_K.gguf", 8.0),    # 33%, low tier, fits
        _gguf("model-Q4_K_M.gguf", 11.0),  # 46%, medium tier, fits
        _gguf("model-Q8_0.gguf", 20.0),    # 83%, doesn't fit
    ]
    best = hf_explore.pick_best_gguf_file(files, ram_gb=24.0)
    assert best["path"] == "model-Q4_K_M.gguf"  # largest that still fits, not the smallest


def test_pick_best_gguf_file_returns_none_when_nothing_fits():
    files = [_gguf("model-Q8_0.gguf", 20.0)]
    assert hf_explore.pick_best_gguf_file(files, ram_gb=24.0) is None


def test_pick_best_gguf_file_excludes_high_tier_even_if_it_technically_fits_at_all():
    """A file can fit in RAM at all yet still be too risky (high tier) --
    both conditions must hold, not just one."""
    files = [_gguf("model-Q6_K.gguf", 15.0)]  # 62.5% of RAM: fits_at_all True, but high tier
    assert hf_explore.pick_best_gguf_file(files, ram_gb=24.0) is None


def test_pick_best_gguf_file_ignores_non_gguf_and_sizeless_files():
    files = [
        {"path": "README.md", "size": 100},
        {"path": "model.gguf"},  # no size field
        _gguf("model-Q4_0.gguf", 5.0),
    ]
    best = hf_explore.pick_best_gguf_file(files, ram_gb=24.0)
    assert best["path"] == "model-Q4_0.gguf"


def _chat(downloads=0, tags=None):
    return {"pipeline_tag": "text-generation", "downloads": downloads, "tags": tags or []}


def test_rank_candidates_sorts_by_downloads_descending():
    raw = [
        {"id": "org/a", **_chat(100)},
        {"id": "org/b", **_chat(5000)},
        {"id": "org/c", **_chat(900)},
    ]
    ranked = hf_explore.rank_candidates(raw, ram_gb=24.0)
    assert [c["id"] for c in ranked] == ["org/b", "org/c", "org/a"]


def test_rank_candidates_excludes_non_chat_pipeline_tags():
    """Regression for a real bug: an unfiltered ranking surfaced an
    automatic-speech-recognition model as opencode's suggested chat model."""
    raw = [
        {"id": "org/asr-model", "pipeline_tag": "automatic-speech-recognition", "downloads": 999999, "tags": []},
        {"id": "org/embedding-model", "pipeline_tag": "feature-extraction", "downloads": 500000, "tags": []},
        {"id": "org/chat-model", **_chat(100)},
    ]
    ranked = hf_explore.rank_candidates(raw, ram_gb=24.0)
    assert [c["id"] for c in ranked] == ["org/chat-model"]


def test_rank_candidates_excludes_repos_with_no_declared_pipeline_tag():
    raw = [{"id": "org/unknown", "downloads": 999999, "tags": []}]
    ranked = hf_explore.rank_candidates(raw, ram_gb=24.0)
    assert ranked == []


def test_rank_candidates_includes_vision_language_models():
    raw = [{"id": "org/vlm", "pipeline_tag": "image-text-to-text", "downloads": 100, "tags": []}]
    ranked = hf_explore.rank_candidates(raw, ram_gb=24.0)
    assert [c["id"] for c in ranked] == ["org/vlm"]


def test_rank_candidates_boosts_agentic_tagged_models_above_higher_download_counts():
    raw = [
        {"id": "org/popular-no-tag", **_chat(downloads=100000, tags=["conversational"])},
        {"id": "org/agentic-tagged", **_chat(downloads=100, tags=["qwen3", "agentic", "conversational"])},
    ]
    ranked = hf_explore.rank_candidates(raw, ram_gb=24.0)
    assert [c["id"] for c in ranked] == ["org/agentic-tagged", "org/popular-no-tag"]


def test_has_agentic_signal_matches_known_keyword_variants():
    assert hf_explore.has_agentic_signal(_chat(tags=["Agentic-256k"]))
    assert hf_explore.has_agentic_signal(_chat(tags=["tool-calling"]))
    assert hf_explore.has_agentic_signal(_chat(tags=["function-calling"]))
    assert not hf_explore.has_agentic_signal(_chat(tags=["conversational", "qwen3"]))
