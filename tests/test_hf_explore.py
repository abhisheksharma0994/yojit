from yojit import hf_explore


def test_detect_backend_from_files_recognizes_mlx_repo():
    files = [{"path": "config.json"}, {"path": "model.safetensors"}, {"path": "tokenizer.json"}]
    assert hf_explore.detect_backend_from_files(files) == "mlx"


def test_detect_backend_from_files_recognizes_gguf_repo():
    files = [{"path": "model-Q4_K_M.gguf"}, {"path": "README.md"}]
    assert hf_explore.detect_backend_from_files(files) == "llamacpp"


def test_detect_backend_from_files_returns_none_for_unrecognized_repo():
    files = [{"path": "README.md"}, {"path": "config.yaml"}]
    assert hf_explore.detect_backend_from_files(files) is None


def test_detect_backend_prefers_mlx_when_repo_has_both_formats():
    # Multi-format repos are rare but possible; mlx takes priority since
    # that's the default backend on Apple Silicon (the common case).
    files = [{"path": "config.json"}, {"path": "model.safetensors"}, {"path": "model.gguf"}]
    assert hf_explore.detect_backend_from_files(files) == "mlx"


def test_bit_width_from_name_extracts_trailing_bit_suffix():
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-4bit") == 4
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-8bit") == 8
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-6BIT") == 6  # case-insensitive


def test_bit_width_from_name_returns_none_for_non_bit_suffixed_names():
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-bf16") is None
    assert hf_explore.bit_width_from_name("mlx-community/Qwen3.6-27B-OptiQ-4bit") == 4


def test_sibling_bit_variants_matches_same_base_model_family():
    candidates = [
        {"id": "mlx-community/Qwen3.6-27B-4bit"},
        {"id": "mlx-community/Qwen3.6-27B-6bit"},
        {"id": "mlx-community/Qwen3.6-27B-8bit"},
        {"id": "mlx-community/Qwen3.6-35B-A3B-4bit"},  # different base model
    ]
    siblings = hf_explore.sibling_bit_variants("mlx-community/Qwen3.6-27B-4bit", candidates)
    ids = {c["id"] for c in siblings}
    assert ids == {
        "mlx-community/Qwen3.6-27B-4bit",
        "mlx-community/Qwen3.6-27B-6bit",
        "mlx-community/Qwen3.6-27B-8bit",
    }


def test_pick_best_fit_prefers_largest_bit_width_that_still_fits_comfortably():
    """Regression for the 'don't assume smallest is best' lesson: given a
    choice, pick_best_fit must choose the LARGEST bit-width that still lands
    in low/medium tier, not the tightest-possible fit."""
    ram_gb = 24.0
    variants = [
        {"id": "org/model-4bit"},
        {"id": "org/model-6bit"},
        {"id": "org/model-8bit"},
    ]
    sizes = {
        "org/model-4bit": 8.0,   # 33% -> low, fits
        "org/model-6bit": 11.0,  # 46% -> medium, fits
        "org/model-8bit": 20.0,  # 83% -> high, does NOT fit safely
    }
    best = hf_explore.pick_best_fit(variants, ram_gb, sizes)
    assert best["id"] == "org/model-6bit"  # largest that still fits comfortably


def test_pick_best_fit_returns_none_when_nothing_fits():
    ram_gb = 24.0
    variants = [{"id": "org/model-8bit"}]
    sizes = {"org/model-8bit": 20.0}  # 83%, doesn't fit
    assert hf_explore.pick_best_fit(variants, ram_gb, sizes) is None


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
