"""Regression fixtures from real OOM debugging tonight: Bonsai-27B (2-bit,
hybrid mamba/linear-attention) and Qwen3.8-Uncensored (4-bit, dense-ish) on a
24GB Mac. These exact numbers were verified against real mlx_lm.server
crashes/successes -- if this test ever needs updating, re-verify against a
real server run, don't just adjust the fixture to match new code."""
from yojit import classify

RAM_GB = 24.0

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
    """A small model's own native max must never be inflated past its real
    architectural limit, even if that's below MIN_CONTEXT -- this was a real
    bug found while testing GGUF models (TinyLlama's native 2048 was getting
    bumped to 4096)."""
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
    # Boundaries are evidence-based, not theoretical: every model at/above
    # 50% of RAM crashed with Metal OOM tonight regardless of context size
    # (15GB/24GB=62.5% and 19GB/24GB=79% both crashed); Bonsai at 33% was
    # rock-solid. See classify.py's MEDIUM_TIER_MAX_FRACTION comment.
    assert classify.resource_tier(5.0, 24.0) == "low"        # 21%
    assert classify.resource_tier(10.0, 24.0) == "medium"     # 42%
    assert classify.resource_tier(15.0, 24.0) == "high"       # 62.5% -- this exact case crashed tonight
    assert classify.resource_tier(20.0, 24.0) == "high"       # 79% -- this exact case crashed tonight (harder/faster)


def test_compute_launch_tuning_scales_prefill_step_size_with_headroom():
    """More headroom -> bigger prefill chunks -> faster prefill, per real
    benchmarking (mlx-lm/lmstudio-js discussions): ~8192 is the sweet spot
    with headroom to spare, not higher (16384 regresses due to allocation
    overhead), and tight headroom keeps the original conservative 512."""
    tight = classify.compute_launch_tuning(weight_gb=15.0, ram_gb=24.0, cpu_cores=8)  # ~1GB headroom
    generous = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=64.0, cpu_cores=8)  # ~52GB headroom
    assert tight["prefill_step_size"] == 512
    assert generous["prefill_step_size"] == 8192


def test_compute_launch_tuning_pins_concurrency_to_one_regardless_of_headroom():
    """No concurrency knob is scaled up even with abundant headroom -- until
    concurrent-request memory accounting is modeled explicitly, single-
    request is the only verified-safe configuration for both backends."""
    tuning = classify.compute_launch_tuning(weight_gb=4.0, ram_gb=128.0, cpu_cores=16)
    assert tuning["decode_concurrency"] == 1
    assert tuning["prompt_concurrency"] == 1


def test_compute_launch_tuning_prompt_cache_bytes_scales_with_headroom_not_fixed():
    """Regression: the old flat 5GB constant wasn't actually protective on a
    tight machine (could exceed the entire headroom) and under-used
    available memory on a generous one. Must scale with real headroom,
    within sane min/max bounds."""
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


def test_fits_at_all():
    assert classify.fits_at_all(15.0, 24.0) is True
    assert classify.fits_at_all(27.0, 24.0) is False  # weights alone exceed total RAM


def test_maple_preview_moe_2bit_produces_garbage_is_not_this_modules_job():
    """Documents a real finding: connorbillen's community 2-bit re-quant of
    Qwen3.6-35B-A3B (10.1GB, 42% -- passes every check in this module) still
    produced completely broken/repetitive output in live testing. classify.py
    can only ever guarantee a model *fits in memory* -- it has no way to
    predict quantization-induced output quality. This test exists so nobody
    "fixes" classify.py to try to catch that class of failure; it can't."""
    community_2bit_weight_gb = 10.1
    assert classify.resource_tier(community_2bit_weight_gb, 24.0) == "medium"
    assert classify.fits_at_all(community_2bit_weight_gb, 24.0) is True
    # ^ both checks pass; the model was still unusable. Quality verification
    # requires an actual generation test (see backends warm_up / doctor
    # "verified" flag), not a memory-fit calculation.
