"""Resource-fit tier + context/output limit math, shared by MLX and llama.cpp."""
import json
from pathlib import Path

RESERVED_OS_GB = 8.0   # left for the OS and other apps
SAFETY_FACTOR = 0.25   # stay well under raw headroom; prefill also spikes memory transiently
MIN_CONTEXT = 4096
MAX_CONTEXT_HARD_CAP = 65536
MAX_OUTPUT_HARD_CAP = 4096
MIN_OUTPUT = 1024
CONTEXT_ROUND_TO = 4096

# Fraction of total RAM consumed by weights alone. Empirically, models above
# ~50% of RAM risk a Metal OOM crash regardless of context size.
LOW_TIER_MAX_FRACTION = 0.35
MEDIUM_TIER_MAX_FRACTION = 0.50
# Above MEDIUM_TIER_MAX_FRACTION is "high" (RISKY) tier.


def weight_size_gb_from_dir(leaf_dir: Path) -> float:
    """Sum of .safetensors (or all files, as fallback) in a model directory."""
    if not leaf_dir or not leaf_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in leaf_dir.rglob("*.safetensors"))
    if total == 0:
        total = sum(f.stat().st_size for f in leaf_dir.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def weight_size_gb_from_file(gguf_path: Path) -> float:
    if not gguf_path or not gguf_path.exists():
        return 0.0
    return gguf_path.stat().st_size / (1024 ** 3)


def load_hf_config(leaf_dir: Path) -> dict:
    cfg_path = leaf_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def resource_tier(weight_gb: float, ram_gb: float) -> str:
    if ram_gb <= 0:
        return "high"
    fraction = weight_gb / ram_gb
    if fraction <= LOW_TIER_MAX_FRACTION:
        return "low"
    if fraction <= MEDIUM_TIER_MAX_FRACTION:
        return "medium"
    return "high"


def fits_at_all(weight_gb: float, ram_gb: float) -> bool:
    """Weights alone must leave room for the OS -- otherwise it plain won't load."""
    return weight_gb + RESERVED_OS_GB < ram_gb


def _effective_layers(tcfg: dict, cfg: dict) -> int:
    """Hybrid architectures only pay KV-cache cost on full-attention layers."""
    num_layers = tcfg.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 32
    layer_types = tcfg.get("layer_types") or cfg.get("layer_types")
    if not layer_types:
        return num_layers
    return sum(1 for lt in layer_types if "full" in lt) or num_layers


def _resolved_head_dim(tcfg: dict, cfg: dict) -> float:
    head_dim = tcfg.get("head_dim") or cfg.get("head_dim")
    if head_dim:
        return head_dim
    hidden = tcfg.get("hidden_size") or cfg.get("hidden_size") or 4096
    heads = tcfg.get("num_attention_heads") or cfg.get("num_attention_heads") or 32
    return hidden / heads


def _kv_bytes_per_token_fp16(cfg: dict) -> float:
    """Bytes of KV cache per token at full precision, from an HF-style config dict."""
    tcfg = cfg.get("text_config", cfg)
    effective_layers = _effective_layers(tcfg, cfg)
    kv_heads = (tcfg.get("num_key_value_heads") or tcfg.get("num_attention_heads")
                or cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 8)
    head_dim = _resolved_head_dim(tcfg, cfg)
    return 2 * effective_layers * kv_heads * head_dim * 2  # K+V, fp16


def estimate_limits_from_config(cfg: dict, weight_gb: float, ram_gb: float):
    """Core KV-cache-aware context/output estimate from an HF-style config dict."""
    tcfg = cfg.get("text_config", cfg)

    native_ctx = tcfg.get("max_position_embeddings") or cfg.get("max_position_embeddings") or 32768
    kv_bytes_per_token = _kv_bytes_per_token_fp16(cfg)

    headroom_gb = max(ram_gb - weight_gb - RESERVED_OS_GB, 1.0)
    headroom_bytes = headroom_gb * (1024 ** 3) * SAFETY_FACTOR

    max_ctx_by_mem = int(headroom_bytes / kv_bytes_per_token) if kv_bytes_per_token > 0 else native_ctx

    # Memory floor never overrides the model's own native context ceiling.
    context = min(native_ctx, MAX_CONTEXT_HARD_CAP, max(MIN_CONTEXT, max_ctx_by_mem))
    if context >= CONTEXT_ROUND_TO:
        context = (context // CONTEXT_ROUND_TO) * CONTEXT_ROUND_TO

    output = max(MIN_OUTPUT, min(context // 4, MAX_OUTPUT_HARD_CAP))
    return int(context), int(output)


_KV_QUANT_BIT_OPTIONS = (16, 8, 4)  # 16 = unquantized, no override needed
_LLAMACPP_CACHE_TYPE_BY_BITS = {8: "q8_0", 4: "q4_0"}


def default_kv_cache_overrides(cfg: dict, backend_name: str, weight_gb: float, ram_gb: float, context: int) -> dict:
    """Picks the highest-precision KV-cache bit-width that still fits this
    model's context in this machine's real headroom. `yojit config` can
    always override the result."""
    if backend_name not in ("mlx_vlm", "llamacpp"):
        return {}

    bytes_per_token_fp16 = _kv_bytes_per_token_fp16(cfg)
    headroom_gb = max(ram_gb - weight_gb - RESERVED_OS_GB, 0.1)
    headroom_bytes = headroom_gb * (1024 ** 3) * SAFETY_FACTOR

    for bits in _KV_QUANT_BIT_OPTIONS:
        bytes_per_token = bytes_per_token_fp16 * (bits / 16)
        fits = bytes_per_token <= 0 or context * bytes_per_token <= headroom_bytes
        if fits or bits == _KV_QUANT_BIT_OPTIONS[-1]:
            if bits == _KV_QUANT_BIT_OPTIONS[0]:
                return {}
            if backend_name == "llamacpp":
                return {"kv_cache_quant": _LLAMACPP_CACHE_TYPE_BY_BITS.get(bits, "q4_0")}
            # Derive where quantization should start from the real headroom, not a fixed constant.
            start = int(headroom_bytes / bytes_per_token_fp16) if bytes_per_token_fp16 > 0 else 0
            return {"kv_cache_quant": str(bits), "quantized_kv_start": max(0, min(start, context))}


def classify_mlx_model(leaf_dir: Path, ram_gb: float):
    """Returns (tier, context, output, weight_gb) for an MLX model directory."""
    weight_gb = weight_size_gb_from_dir(leaf_dir)
    cfg = load_hf_config(leaf_dir)
    context, output = estimate_limits_from_config(cfg, weight_gb, ram_gb)
    tier = resource_tier(weight_gb, ram_gb)
    return tier, context, output, round(weight_gb, 1)


def classify_gguf_model(gguf_path: Path, ram_gb: float, arch_hints: dict | None = None):
    """Returns (tier, context, output, weight_gb) for a single GGUF file.
    arch_hints (from the GGUF header) map onto the same keys as an HF config."""
    weight_gb = weight_size_gb_from_file(gguf_path)
    cfg = arch_hints or {}
    context, output = estimate_limits_from_config(cfg, weight_gb, ram_gb)
    tier = resource_tier(weight_gb, ram_gb)
    return tier, context, output, round(weight_gb, 1)


# Headroom -> (prefill-step-size for MLX, batch-size/ubatch-size for llama.cpp).
_HEADROOM_TIER_GB = (2.0, 4.0, 8.0, 16.0)
_MLX_PREFILL_STEP_SIZES = (512, 1024, 2048, 4096, 8192)
_LLAMACPP_BATCH_SIZES = (512, 1024, 2048, 2048, 4096)
_LLAMACPP_UBATCH_SIZES = (256, 512, 512, 1024, 2048)

# Fraction of headroom reserved for MLX's KV-cache ceiling (--prompt-cache-bytes).
_PROMPT_CACHE_HEADROOM_FRACTION = 0.4
_PROMPT_CACHE_BYTES_MIN_GB = 0.5
_PROMPT_CACHE_BYTES_MAX_GB = 8.0


def _headroom_tier_index(headroom_gb: float) -> int:
    for i, ceiling in enumerate(_HEADROOM_TIER_GB):
        if headroom_gb < ceiling:
            return i
    return len(_HEADROOM_TIER_GB)


def compute_launch_tuning(weight_gb: float, ram_gb: float, cpu_cores: int) -> dict:
    """Every launch parameter, computed fresh from this machine's actual
    specs rather than fixed constants. Concurrency stays pinned to 1 until
    concurrent-request memory accounting is modeled explicitly."""
    headroom_gb = max(ram_gb - weight_gb - RESERVED_OS_GB, 1.0)
    tier = _headroom_tier_index(headroom_gb)

    prompt_cache_gb = min(
        _PROMPT_CACHE_BYTES_MAX_GB,
        max(_PROMPT_CACHE_BYTES_MIN_GB, headroom_gb * _PROMPT_CACHE_HEADROOM_FRACTION),
    )

    return {
        # MLX
        "prefill_step_size": _MLX_PREFILL_STEP_SIZES[tier],
        "prompt_cache_bytes": int(prompt_cache_gb * 1024 ** 3),
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        # llama.cpp
        "threads": max(1, cpu_cores - 1),
        "ngl": 999,  # full GPU offload
        "batch_size": _LLAMACPP_BATCH_SIZES[tier],
        "ubatch_size": _LLAMACPP_UBATCH_SIZES[tier],
    }
