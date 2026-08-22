"""Resource-fit tier + context/output limit math.

This is backend-independent: a model's KV-cache cost is a property of its
architecture (layers, hidden size, hybrid layer types), not of which runtime
(MLX / llama.cpp) ends up serving it. Only the source of the architecture
metadata differs (HF config.json for MLX repos, GGUF header fields for GGUF
files) -- both funnel into the same estimate_limits() math below.
"""
import json
from pathlib import Path

RESERVED_OS_GB = 8.0        # left for the OS + browser/other apps (observed real-world usage)
SAFETY_FACTOR = 0.25        # KV cache isn't the only consumer -- prefill activations also
                            # spike memory transiently, so stay well under raw headroom
MIN_CONTEXT = 4096
MAX_CONTEXT_HARD_CAP = 65536
MAX_OUTPUT_HARD_CAP = 4096
MIN_OUTPUT = 1024

# Tier boundaries as a fraction of total RAM consumed by weights alone.
# Community guidance (multiple independent sources) converges on keeping MLX
# model weights under ~60% of unified memory, with more conservative voices
# recommending 50%. Real crash data gathered tonight on a 24GB Mac confirms
# this empirically: every model at or above 62.5% (15GB/24GB, 19GB/24GB)
# crashed with a Metal out-of-memory error despite prefill/KV-cache tuning;
# the one model well under 50% (Bonsai, 7.9GB/24GB = 33%) was rock-solid
# across every stress test. MEDIUM_TIER_MAX_FRACTION is therefore set at the
# conservative end (0.50) rather than the community's upper bound (0.60) --
# the crashes we saw were about weight footprint alone, independent of how
# small a context was requested, so tuning context down does NOT rescue a
# model above this line.
LOW_TIER_MAX_FRACTION = 0.35
MEDIUM_TIER_MAX_FRACTION = 0.50
# Anything above MEDIUM_TIER_MAX_FRACTION is "high" (== RISKY in the CLI's
# own language) -- real crash risk, not just a tight context. Anything above
# ~0.9 additionally fails fits_at_all() below (weights alone leave no room
# for the OS at all).


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


def estimate_limits_from_config(cfg: dict, weight_gb: float, ram_gb: float):
    """Core KV-cache-aware context/output estimate. `cfg` is an HF-style
    config dict (works whether it came from config.json or was translated
    from GGUF metadata by the llama.cpp backend)."""
    tcfg = cfg.get("text_config", cfg)

    native_ctx = tcfg.get("max_position_embeddings") or cfg.get("max_position_embeddings") or 32768
    num_layers = tcfg.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 32
    layer_types = tcfg.get("layer_types") or cfg.get("layer_types")

    if layer_types:
        # Hybrid architectures (e.g. mamba/linear-attention mixes) only pay
        # KV-cache cost on their full-attention layers.
        effective_layers = sum(1 for lt in layer_types if "full" in lt)
        if effective_layers == 0:
            effective_layers = num_layers
    else:
        effective_layers = num_layers

    kv_heads = (tcfg.get("num_key_value_heads") or tcfg.get("num_attention_heads")
                or cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 8)
    head_dim = tcfg.get("head_dim") or cfg.get("head_dim")
    if not head_dim:
        hidden = tcfg.get("hidden_size") or cfg.get("hidden_size") or 4096
        heads = tcfg.get("num_attention_heads") or cfg.get("num_attention_heads") or 32
        head_dim = hidden / heads

    kv_bytes_per_token = 2 * effective_layers * kv_heads * head_dim * 2  # K+V, fp16

    headroom_gb = max(ram_gb - weight_gb - RESERVED_OS_GB, 1.0)
    headroom_bytes = headroom_gb * (1024 ** 3) * SAFETY_FACTOR

    max_ctx_by_mem = int(headroom_bytes / kv_bytes_per_token) if kv_bytes_per_token > 0 else native_ctx

    # MIN_CONTEXT is a floor on the memory-derived estimate only (never propose
    # an absurdly tiny context just because headroom is tight) -- it must NOT
    # override the model's own native architectural ceiling, which stays a
    # hard cap applied last via min().
    context = min(native_ctx, MAX_CONTEXT_HARD_CAP, max(MIN_CONTEXT, max_ctx_by_mem))
    if context >= 4096:
        context = (context // 4096) * 4096

    output = max(MIN_OUTPUT, min(context // 4, MAX_OUTPUT_HARD_CAP))
    return int(context), int(output)


def classify_mlx_model(leaf_dir: Path, ram_gb: float):
    """Returns (tier, context, output, weight_gb) for an MLX model directory."""
    weight_gb = weight_size_gb_from_dir(leaf_dir)
    cfg = load_hf_config(leaf_dir)
    context, output = estimate_limits_from_config(cfg, weight_gb, ram_gb)
    tier = resource_tier(weight_gb, ram_gb)
    return tier, context, output, round(weight_gb, 1)


def classify_gguf_model(gguf_path: Path, ram_gb: float, arch_hints: dict | None = None):
    """Returns (tier, context, output, weight_gb) for a single GGUF file.

    arch_hints, when available (parsed from the GGUF header by the llama.cpp
    backend), maps onto the same HF-style keys estimate_limits_from_config
    expects. Falls back to a generic dense-transformer guess otherwise --
    less accurate, but keeps the classifier usable before GGUF-header parsing
    is implemented.
    """
    weight_gb = weight_size_gb_from_file(gguf_path)
    cfg = arch_hints or {}
    context, output = estimate_limits_from_config(cfg, weight_gb, ram_gb)
    tier = resource_tier(weight_gb, ram_gb)
    return tier, context, output, round(weight_gb, 1)


# Headroom -> (prefill-step-size for MLX, batch-size, ubatch-size for
# llama.cpp) lookup, in ascending headroom order. Grounded in real benchmark
# findings, not guessed: raising MLX's prefill-step-size from 512 toward
# ~8192 measurably speeds up prefill on long prompts when memory allows it
# (up to ~1.5x), while 16384 regresses due to allocation/kernel-launch
# overhead -- so 8192 is the ceiling, not higher. llama.cpp's --ubatch-size
# is the equivalent lever there; 512 is its own safe default, and raising it
# toward 1024-2048 with headroom to spare improves prefill throughput on
# Apple Silicon by the same logic. Tight headroom keeps every value at each
# tool's own conservative default instead of guessing higher.
_HEADROOM_TIER_GB = (2.0, 4.0, 8.0, 16.0)  # 5 tiers: <2, <4, <8, <16, >=16
_MLX_PREFILL_STEP_SIZES = (512, 1024, 2048, 4096, 8192)
_LLAMACPP_BATCH_SIZES = (512, 1024, 2048, 2048, 4096)
_LLAMACPP_UBATCH_SIZES = (256, 512, 512, 1024, 2048)

# Fraction of headroom reserved for the KV-cache ceiling (MLX's
# --prompt-cache-bytes), replacing the previous flat 5GB constant -- which
# was a real gap: on a tight-headroom machine, a flat 5GB ceiling isn't
# actually protective (it can exceed the entire available headroom), and on
# a generous machine it needlessly under-uses what's available.
_PROMPT_CACHE_HEADROOM_FRACTION = 0.4
_PROMPT_CACHE_BYTES_MIN_GB = 0.5
_PROMPT_CACHE_BYTES_MAX_GB = 8.0


def _headroom_tier_index(headroom_gb: float) -> int:
    for i, ceiling in enumerate(_HEADROOM_TIER_GB):
        if headroom_gb < ceiling:
            return i
    return len(_HEADROOM_TIER_GB)


def compute_launch_tuning(weight_gb: float, ram_gb: float, cpu_cores: int) -> dict:
    """Every parameter passed to mlx_lm.server / llama-server, computed from
    this machine's actual specs -- not fixed constants. Recomputed fresh at
    every launch (like tier/context elsewhere in this module) rather than
    stored in the manifest, so moving to different hardware or a RAM change
    is picked up automatically without re-installing anything.

    Concurrency knobs (decode/prompt-concurrency for MLX, --parallel for
    llama.cpp) are pinned to 1 everywhere, not scaled -- a real bug found
    tonight showed llama-server's default of 4 parallel slots each
    allocating their own KV cache can silently multiply real memory usage
    past what this same math assumes is safe. Until concurrent-request
    memory accounting is modeled explicitly, single-request is the only
    verified-safe configuration for both backends.
    """
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
        "threads": max(1, cpu_cores - 1),  # leave one core free for the OS
        "ngl": 999,  # offload every layer to GPU -- standard on unified-memory Apple Silicon
        "batch_size": _LLAMACPP_BATCH_SIZES[tier],
        "ubatch_size": _LLAMACPP_UBATCH_SIZES[tier],
    }
