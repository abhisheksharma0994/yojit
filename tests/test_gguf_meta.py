"""gguf_meta.to_hf_style_config() translates GGUF header fields into the
HF-config-shaped dict classify.py already knows how to read.

Regression coverage for a real bug found via a live opencode report: a
hybrid mamba/linear-attention GGUF model (Qwen3.5/3.8's "qwen35" arch) was
computing a far-too-small context, causing an infinite compaction loop in
opencode. Root cause: GGUF stores hybrid-architecture info as a single
`{arch}.full_attention_interval` integer, not a per-layer layer_types list
like HF's config.json -- the translation never read it, so classify.py
treated every layer as full-attention (~4x KV-cache overestimate). A second,
compounding bug: `{arch}.attention.key_length` (the real head_dim) was never
read either, so it fell back to an incorrect hidden_size/num_attention_heads
approximation.
"""
from yojit import classify, gguf_meta


def test_to_hf_style_config_synthesizes_layer_types_from_full_attention_interval():
    """This is the exact real metadata from the GGUF file that triggered the
    live bug report (unsloth/Qwen3.8-27B-GGUF, IQ3_S)."""
    meta = {
        "general.architecture": "qwen35",
        "qwen35.context_length": 262144,
        "qwen35.block_count": 65,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.full_attention_interval": 4,
    }
    cfg = gguf_meta.to_hf_style_config(meta)

    assert len(cfg["layer_types"]) == 65
    full_count = sum(1 for lt in cfg["layer_types"] if lt == "full_attention")
    assert full_count == 16  # every 4th of 65 layers, not all 65
    assert cfg["head_dim"] == 256  # real declared value, not hidden_size/heads (5120/24 != 256)


def test_to_hf_style_config_omits_layer_types_when_not_a_hybrid_architecture():
    """A normal (non-hybrid) architecture has no full_attention_interval key
    at all -- layer_types must be None, not an incorrectly-synthesized list,
    so classify.py falls back to treating every layer as full-attention
    (correct for a real dense transformer)."""
    meta = {
        "general.architecture": "llama",
        "llama.context_length": 2048,
        "llama.block_count": 22,
        "llama.embedding_length": 2048,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 4,
    }
    cfg = gguf_meta.to_hf_style_config(meta)
    assert cfg["layer_types"] is None
    assert cfg["head_dim"] is None  # not declared -- classify.py derives it


def test_to_hf_style_config_returns_empty_dict_without_an_architecture_field():
    assert gguf_meta.to_hf_style_config({}) == {}


def test_hybrid_gguf_context_is_roughly_4x_larger_than_treating_every_layer_as_full_attention():
    """End-to-end regression: verifies the actual context computation
    improves by the expected ~4x factor once the hybrid architecture is
    correctly recognized -- not just that layer_types looks right in
    isolation."""
    meta = {
        "general.architecture": "qwen35",
        "qwen35.context_length": 262144,
        "qwen35.block_count": 65,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.full_attention_interval": 4,
    }
    ram_gb = 24.0
    weight_gb = 11.2

    hybrid_cfg = gguf_meta.to_hf_style_config(meta)
    hybrid_context, _ = classify.estimate_limits_from_config(hybrid_cfg, weight_gb, ram_gb)

    # Simulate the old buggy behavior: same config, minus the hybrid signal.
    naive_meta = dict(meta)
    del naive_meta["qwen35.full_attention_interval"]
    naive_cfg = gguf_meta.to_hf_style_config(naive_meta)
    naive_context, _ = classify.estimate_limits_from_config(naive_cfg, weight_gb, ram_gb)

    assert hybrid_context > naive_context * 3  # roughly 4x, allowing for 4096-rounding
