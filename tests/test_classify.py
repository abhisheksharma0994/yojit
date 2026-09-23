"""Regression fixtures from two real model architectures (a hybrid
mamba/linear-attention model and a dense-ish MoE model), numbers verified
against real server behavior -- re-verify against a real run before adjusting."""
import pytest

from yojit import classify

RAM_GB = 24.0


def test_weight_size_gb_from_dir_sums_safetensors(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"0" * 1024)
    (tmp_path / "b.safetensors").write_bytes(b"0" * 1024)
    assert classify.weight_size_gb_from_dir(tmp_path) == 2048 / (1024 ** 3)


def test_weight_size_gb_from_dir_falls_back_to_all_files_when_no_safetensors(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"0" * 1024)
    assert classify.weight_size_gb_from_dir(tmp_path) == 1024 / (1024 ** 3)


def test_weight_size_gb_from_dir_only_counts_safetensors_when_present(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"0" * 1024)
    (tmp_path / "readme.txt").write_bytes(b"0" * 5000)  # must be ignored, not swept into the fallback sum
    assert classify.weight_size_gb_from_dir(tmp_path) == 1024 / (1024 ** 3)


def test_weight_size_gb_from_dir_returns_zero_for_missing_dir(tmp_path):
    assert classify.weight_size_gb_from_dir(tmp_path / "does-not-exist") == 0.0
    assert classify.weight_size_gb_from_dir(None) == 0.0


def test_weight_size_gb_from_file_returns_real_size(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"0" * 2048)
    assert classify.weight_size_gb_from_file(f) == 2048 / (1024 ** 3)


def test_weight_size_gb_from_file_returns_zero_for_missing_file(tmp_path):
    assert classify.weight_size_gb_from_file(tmp_path / "does-not-exist.gguf") == 0.0
    assert classify.weight_size_gb_from_file(None) == 0.0


def test_load_hf_config_returns_empty_dict_when_missing(tmp_path):
    assert classify.load_hf_config(tmp_path) == {}


def test_load_hf_config_returns_empty_dict_on_invalid_json(tmp_path):
    (tmp_path / "config.json").write_text("{not valid json")
    assert classify.load_hf_config(tmp_path) == {}


def test_load_hf_config_parses_real_json(tmp_path):
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 32}')
    assert classify.load_hf_config(tmp_path) == {"num_hidden_layers": 32}


def test_resource_tier_treats_zero_or_negative_ram_as_high_risk():
    assert classify.resource_tier(5.0, 0.0) == "high"
    assert classify.resource_tier(5.0, -1.0) == "high"


def test_resource_tier_zero_ram_short_circuit_does_not_extend_to_small_positive_ram():
    assert classify.resource_tier(0.1, 0.5) == "low"  # fraction 0.2, well under low-tier threshold


def test_resource_tier_tier_boundaries_are_inclusive():
    assert classify.resource_tier(classify.LOW_TIER_MAX_FRACTION * 24.0, 24.0) == "low"
    assert classify.resource_tier(classify.MEDIUM_TIER_MAX_FRACTION * 24.0, 24.0) == "medium"

BONSAI_CONFIG = {
    "max_position_embeddings": 262144,
    "num_hidden_layers": 64,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
}

QWEN38_UNCENSORED_CONFIG = {
    "max_position_embeddings": 262144,
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
    "hidden_size": 5120,
    "num_attention_heads": 24,
}


def test_bonsai_2bit_context():
    context, output = classify.estimate_limits_from_config(BONSAI_CONFIG, weight_gb=7.9, ram_gb=RAM_GB)
    assert context == 32768, f"expected 32768, got {context}"
    assert output == 4096, f"expected 4096, got {output}"


def test_qwen38_uncensored_4bit_context():
    context, output = classify.estimate_limits_from_config(QWEN38_UNCENSORED_CONFIG, weight_gb=15.0, ram_gb=RAM_GB)
    assert context == 4096, f"expected 4096, got {context}"
    assert output == 1024, f"expected 1024, got {output}"


def test_native_ctx_is_a_hard_ceiling_even_when_below_min_context():
    """A model's native max must never be inflated past its real limit, even below MIN_CONTEXT."""
    tiny_config = {
        "max_position_embeddings": 2048,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
    }
    context, output = classify.estimate_limits_from_config(tiny_config, weight_gb=0.6, ram_gb=RAM_GB)
    assert context == 2048, f"expected native cap of 2048, got {context}"


def test_resource_tier_boundaries():
    # Boundaries are evidence-based: models at/above 50% of RAM risk OOM regardless of context size.
    assert classify.resource_tier(5.0, 24.0) == "low"        # 21%
    assert classify.resource_tier(10.0, 24.0) == "medium"     # 42%
    assert classify.resource_tier(15.0, 24.0) == "high"       # 62.5%
    assert classify.resource_tier(20.0, 24.0) == "high"       # 79%


def test_compute_launch_tuning_scales_prefill_step_size_with_headroom():
    """More headroom -> bigger prefill chunks -> faster prefill."""
    tight = classify.compute_launch_tuning(weight_gb=15.0, ram_gb=24.0, cpu_cores=8)  # ~1GB headroom
    generous = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=64.0, cpu_cores=8)  # ~52GB headroom
    assert tight["prefill_step_size"] == 512
    assert generous["prefill_step_size"] == 8192


def test_compute_launch_tuning_pins_concurrency_to_one_regardless_of_headroom():
    """No concurrency knob scales up, even with abundant headroom."""
    tuning = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=128.0, cpu_cores=16)
    assert tuning["decode_concurrency"] == 1
    assert tuning["prompt_concurrency"] == 1


def test_compute_launch_tuning_prompt_cache_bytes_scales_with_headroom_not_fixed():
    """Must scale with real headroom, within sane min/max bounds, not a fixed constant."""
    tight = classify.compute_launch_tuning(weight_gb=15.0, ram_gb=24.0, cpu_cores=8)
    generous = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=64.0, cpu_cores=8)
    assert tight["prompt_cache_bytes"] < generous["prompt_cache_bytes"]
    # never below the floor or above the ceiling, regardless of headroom
    assert classify._PROMPT_CACHE_BYTES_MIN_GB * 1024 ** 3 <= tight["prompt_cache_bytes"]
    assert generous["prompt_cache_bytes"] <= classify._PROMPT_CACHE_BYTES_MAX_GB * 1024 ** 3


def test_compute_launch_tuning_threads_leaves_one_core_for_the_os():
    tuning = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=24.0, cpu_cores=8)
    assert tuning["threads"] == 7


def test_compute_launch_tuning_threads_never_goes_below_one():
    tuning = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=24.0, cpu_cores=1)
    assert tuning["threads"] == 1


def test_compute_launch_tuning_forces_full_gpu_offload():
    tuning = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=24.0, cpu_cores=8)
    assert tuning["ngl"] == 999


def test_compute_launch_tuning_batch_and_ubatch_scale_together_with_prefill_step_size():
    tight = classify.compute_launch_tuning(weight_gb=15.0, ram_gb=24.0, cpu_cores=8)
    generous = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=64.0, cpu_cores=8)
    assert tight["batch_size"] < generous["batch_size"]
    assert tight["ubatch_size"] < generous["ubatch_size"]


def test_effective_layers_prefers_tcfg_over_cfg():
    assert classify._effective_layers({"num_hidden_layers": 10}, {"num_hidden_layers": 99}) == 10


def test_effective_layers_falls_back_to_cfg_when_tcfg_missing():
    assert classify._effective_layers({}, {"num_hidden_layers": 20}) == 20


def test_effective_layers_defaults_to_32_when_neither_has_it():
    assert classify._effective_layers({}, {}) == 32


def test_effective_layers_counts_only_full_attention_layers_in_hybrid_models():
    tcfg = {"num_hidden_layers": 6, "layer_types": ["linear", "full", "linear", "full", "linear", "full"]}
    assert classify._effective_layers(tcfg, {}) == 3


def test_effective_layers_falls_back_to_num_layers_if_no_layer_is_full_attention():
    tcfg = {"num_hidden_layers": 4, "layer_types": ["linear", "linear", "linear", "linear"]}
    assert classify._effective_layers(tcfg, {}) == 4


def test_resolved_head_dim_prefers_tcfg_over_cfg():
    assert classify._resolved_head_dim({"head_dim": 64}, {"head_dim": 999}) == 64


def test_resolved_head_dim_falls_back_to_cfg_head_dim():
    assert classify._resolved_head_dim({}, {"head_dim": 80}) == 80


def test_resolved_head_dim_computed_from_hidden_size_over_heads_when_no_head_dim():
    assert classify._resolved_head_dim({"hidden_size": 4096, "num_attention_heads": 32}, {}) == 128


def test_resolved_head_dim_falls_back_to_cfg_hidden_size_and_heads():
    assert classify._resolved_head_dim({}, {"hidden_size": 2048, "num_attention_heads": 16}) == 128


def test_resolved_head_dim_defaults_when_nothing_declared():
    assert classify._resolved_head_dim({}, {}) == 4096 / 32


def test_fits_at_all():
    assert classify.fits_at_all(15.0, 24.0) is True
    assert classify.fits_at_all(27.0, 24.0) is False  # weights alone exceed total RAM


def test_fits_at_all_boundary_leaves_no_room_for_os_is_a_fail():
    ram = 20.0
    assert classify.fits_at_all(ram - classify.RESERVED_OS_GB, ram) is False


def test_maple_preview_moe_2bit_produces_garbage_is_not_this_modules_job():
    """A model can pass every fit check here and still produce broken output --
    classify.py can only guarantee memory fit, not quantization quality."""
    community_2bit_weight_gb = 10.1
    assert classify.resource_tier(community_2bit_weight_gb, 24.0) == "medium"
    assert classify.fits_at_all(community_2bit_weight_gb, 24.0) is True


# --- default_kv_cache_overrides: hand-derived expected values, not copied from a prior run ---
# 32 layers, 8 KV heads, head_dim 128 -> bytes_per_token_fp16 = 2*32*8*128*2 = 131072.
_KV_TEST_CFG = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
_KV_TEST_BYTES_PER_TOKEN_FP16 = 2 * 32 * 8 * 128 * 2


def test_default_kv_cache_overrides_empty_for_an_unsupported_backend_name():
    assert classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx", weight_gb=20.0, ram_gb=24.0, context=16384) == {}


def test_default_kv_cache_overrides_empty_when_unquantized_already_fits():
    result = classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx_vlm", weight_gb=5.0, ram_gb=24.0, context=16384)
    assert result == {}


def test_default_kv_cache_overrides_picks_8bit_when_that_fits_but_not_fp16(mocker):
    """Must pick the highest-precision bit-width that actually fits, not jump straight to the most aggressive."""
    result = classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx_vlm", weight_gb=10.0, ram_gb=24.0, context=16384)
    assert result["kv_cache_quant"] == "8"
    assert result["quantized_kv_start"] == int(
        classify.headroom_bytes(10.0, 24.0) / _KV_TEST_BYTES_PER_TOKEN_FP16)


def test_default_kv_cache_overrides_falls_back_to_4bit_when_even_8bit_does_not_fit():
    result = classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=24.0, context=16384)
    assert result["kv_cache_quant"] == "4"
    assert 0 <= result["quantized_kv_start"] <= 16384


def test_default_kv_cache_overrides_uses_llamacpp_cache_type_names():
    result = classify.default_kv_cache_overrides(_KV_TEST_CFG, "llamacpp", weight_gb=10.0, ram_gb=24.0, context=16384)
    assert result == {"kv_cache_quant": "q8_0"}


def test_default_kv_cache_overrides_scales_with_more_ram_not_a_fixed_tier():
    tight = classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=24.0, context=16384)
    roomy = classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=128.0, context=16384)
    assert tight != {}
    assert roomy == {}


# --- output budget: a strict quarter, with no MIN_OUTPUT floor ----------------
# These pin the fix for the case the old floor got wrong: a context below
# MIN_CONTEXT used to be handed MIN_OUTPUT (1024 = MIN_CONTEXT / 4) regardless of
# how small the window actually was, so a 2048 window got half of itself as the
# output budget. formal/lean/Yojit/Limits.lean states both formulas, and proves
# the old one exceeds a quarter for every input below MIN_CONTEXT.

def test_output_for_context_is_a_strict_quarter():
    assert classify.output_for_context(4096) == 1024
    assert classify.output_for_context(16384) == 4096
    assert classify.output_for_context(32768) == 4096  # capped
    assert classify.output_for_context(262144) == 4096  # still capped


def test_output_for_context_below_min_context_is_not_floored_at_1024():
    """The regression: 2048 // 4 = 512, not the old 1024."""
    assert classify.output_for_context(2048) == 512
    assert classify.output_for_context(1024) == 256


def test_output_for_context_never_exceeds_a_quarter_of_the_window():
    for context in list(range(4, 4096, 7)) + [4096, 8192, 20480, 65536, 100000]:
        output = classify.output_for_context(context)
        assert 4 * output <= context, f"context {context} got output {output}, more than a quarter"


def test_output_for_context_is_never_zero_and_never_exceeds_the_window():
    for context in range(1, 200):
        output = classify.output_for_context(context)
        assert output >= 1
        assert output <= context


def test_estimate_limits_uses_the_same_quarter_rule_for_a_tiny_native_context():
    """End to end through estimate_limits_from_config: MIN_CONTEXT stays a floor
    on the memory estimate but not on the model's own native ceiling, and the
    output that comes back is a quarter of the context it reports."""
    tiny = {"max_position_embeddings": 2048, "num_hidden_layers": 22,
            "num_attention_heads": 32, "num_key_value_heads": 4, "hidden_size": 2048}
    context, output = classify.estimate_limits_from_config(tiny, weight_gb=0.6, ram_gb=RAM_GB)
    assert context == 2048
    assert output == classify.output_for_context(2048) == 512


# --- resolve_kv_cache: the plan reports a context it can actually hold -------

def test_resolve_kv_cache_needs_no_override_when_fp16_fits():
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx_vlm", weight_gb=5.0, ram_gb=24.0, context=16384)
    assert plan.fits is True
    assert plan.overrides == {}
    assert plan.context == 16384


def test_resolve_kv_cache_shrinks_the_context_when_even_4bit_does_not_fit():
    """The project's own test parameters. 16384 tokens at the 4-bit width needs
    536870912 bytes against 268435456 of headroom; the plan stops claiming it fits
    and reports the 8192 tokens that do."""
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=24.0, context=16384)
    assert plan.fits is False
    assert plan.context == 8192
    assert plan.bytes_per_token == 32768
    assert plan.required_bytes == 16384 * 32768  # what was asked for
    assert plan.headroom_bytes == 268435456
    assert plan.context * plan.bytes_per_token <= plan.headroom_bytes  # what was granted


def test_resolve_kv_cache_start_index_stays_inside_the_window_it_reports():
    """quantized_kv_start is an index into the cache, so it must not point past
    the window the plan reports (formal/lean/Yojit/Kv.lean: kvStartClamped_le_context)."""
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=24.0, context=16384)
    assert plan.overrides["quantized_kv_start"] <= plan.context


def test_resolve_kv_cache_rounds_a_shrunk_context_down_to_4096():
    """Headroom 1.25 GiB gives 10240 tokens at the 4-bit width; the reported
    context rounds down to 8192 so every reported context is a 4096 multiple."""
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=24.25, context=16384)
    assert plan.fits is False
    assert plan.context == 8192


def test_resolve_kv_cache_keeps_the_context_when_only_the_precision_has_to_drop():
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx_vlm", weight_gb=10.0, ram_gb=24.0, context=16384)
    assert plan.fits is True
    assert plan.context == 16384
    assert plan.overrides["kv_cache_quant"] == "8"


def test_resolve_kv_cache_llamacpp_reports_cache_type_names():
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "llamacpp", weight_gb=15.0, ram_gb=24.0, context=16384)
    assert plan.overrides == {"kv_cache_quant": "q4_0"}
    assert plan.context == 8192


def test_resolve_kv_cache_passes_an_unsupported_backend_through_untouched():
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx", weight_gb=20.0, ram_gb=24.0, context=16384)
    assert plan == classify.KvPlan(context=16384, overrides={}, fits=True,
                                   bytes_per_token=0, required_bytes=0, headroom_bytes=0)


def test_default_kv_cache_overrides_is_exactly_the_plans_overrides():
    """The old entry point stays a thin wrapper, so the two cannot disagree."""
    plan = classify.resolve_kv_cache(_KV_TEST_CFG, "mlx_vlm", weight_gb=15.0, ram_gb=24.0, context=16384)
    assert classify.default_kv_cache_overrides(_KV_TEST_CFG, "mlx_vlm", 15.0, 24.0, 16384) == plan.overrides


# --- kv_fit_limits: install-time application of the same rule ----------------

def test_kv_fit_limits_leaves_a_fitting_config_alone():
    assert classify.kv_fit_limits(_KV_TEST_CFG, "mlx_vlm", 5.0, 24.0, 16384, 4096) == (16384, 4096)


def test_kv_fit_limits_recomputes_the_output_for_the_shrunk_context():
    """Not just the context: the output budget has to follow it down, or the
    manifest records a pair that no longer satisfies the quarter rule."""
    context, output = classify.kv_fit_limits(_KV_TEST_CFG, "mlx_vlm", 15.0, 24.0, 16384, 1024)
    assert (context, output) == (8192, 2048)
    assert 4 * output <= context


# --- the estimate and the fit check share one headroom budget ----------------
# Regression for a break the daily e2e job caught. The fit check used a 0.1 GiB
# headroom floor while the estimate used 1.0 GiB, so on a machine smaller than
# weights + RESERVED_OS_GB -- a 7 GB CI runner serving a 0.6 GB model -- the
# estimate chose 20480 tokens and the check rejected it, installing and launching
# an 8192 window. opencode's own request needs 9620 (7572 prompt + 2048 output),
# so the server answered 400 to every prompt. Real config of
# mlx-community/LFM2.5-1.2B-Instruct-4bit: 16 layers, 6 of them full attention,
# 8 KV heads, hidden 2048 / 32 heads -> head_dim 64.
_LFM25_CONFIG = {
    "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
    "hidden_size": 2048, "max_position_embeddings": 128000,
    "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                    "full_attention", "conv", "full_attention", "conv"],
}
_LFM25_BYTES_PER_TOKEN_FP16 = 2 * 6 * 8 * 64 * 2  # only the full-attention layers pay KV


@pytest.mark.parametrize("ram_gb", [7.0, 8.0, 8.6])
def test_the_fit_check_accepts_the_context_the_estimate_chose(ram_gb):
    """Where both floors bind, the estimate and the check must still agree -- and
    the window must stay big enough for a real client prompt (9620 tokens)."""
    context, _ = classify.estimate_limits_from_config(_LFM25_CONFIG, 0.6, ram_gb)
    plan = classify.resolve_kv_cache(_LFM25_CONFIG, "mlx_vlm", 0.6, ram_gb, context)
    assert classify._kv_bytes_per_token_fp16(_LFM25_CONFIG) == _LFM25_BYTES_PER_TOKEN_FP16
    assert context == 20480
    assert plan.fits is True
    assert plan.context == context
    assert plan.context > 9620


@pytest.mark.parametrize("ram_gb", [6.0, 7.0, 8.0, 12.0, 18.0, 24.0, 64.0])
def test_the_plan_only_shrinks_a_context_memory_never_claimed(ram_gb):
    """Feeding the plan the estimate's own context may only shrink it below
    MIN_CONTEXT -- the case where MIN_CONTEXT, not memory, set the context. Any
    other shrink means the two budgets have drifted apart again.

    Both configs matter: the 0.1 GiB floor only diverged from the estimate on a
    machine whose RAM does not cover the weights plus the OS reservation, and
    whether that shrink stays above MIN_CONTEXT depends on the model's KV width."""
    for cfg in (_KV_TEST_CFG, _LFM25_CONFIG):
        for weight_gb in (0.6, 4.0, 9.0, 15.0, 30.0):
            context, _ = classify.estimate_limits_from_config(cfg, weight_gb, ram_gb)
            plan = classify.resolve_kv_cache(cfg, "mlx_vlm", weight_gb, ram_gb, context)
            assert plan.fits or plan.context < classify.MIN_CONTEXT, (
                f"weight={weight_gb} ram={ram_gb}: the estimate chose {context} tokens "
                f"but the plan shrank it to {plan.context}"
            )
