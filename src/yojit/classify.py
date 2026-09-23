"""Resource-fit tier + context/output limit math, shared by MLX and llama.cpp."""
import json
from dataclasses import dataclass
from pathlib import Path

RESERVED_OS_GB = 8.0   # left for the OS and other apps
SAFETY_FACTOR = 0.25   # stay well under raw headroom; prefill also spikes memory transiently
# Named rather than inlined so formal/lean can mirror it: a bare `max(..., 1.0)`
# in the middle of an expression is a number the Lean model has no handle on, and
# this floor is what keeps a tight machine from being budgeted at zero.
#
# There is exactly ONE headroom floor. `resolve_kv_cache` used to carry its own
# (0.1 GiB), smaller than the estimate's (1.0 GiB) -- so on a machine where
# `ram - weight - RESERVED_OS_GB` is negative (a 7 GB Mac serving a 0.6 GB model,
# say) the estimate chose a context under one budget and the fit check rejected it
# under the other, shrinking every install to the smaller one. See
# headroom_bytes(), the single budget both paths now share.
MIN_HEADROOM_GB = 1.0
MIN_CONTEXT = 4096
MAX_CONTEXT_HARD_CAP = 65536
MAX_OUTPUT_HARD_CAP = 4096
# Fallback budget for a manifest entry that predates the quarter rule. This is
# deliberately no longer a *floor* on the computed value: MIN_OUTPUT is a quarter
# of MIN_CONTEXT, so for any context below MIN_CONTEXT a floor of 1024 hands the
# client a budget bigger than the window it can use (a 2048 window got 1024, half).
# See output_for_context().
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


def headroom_gb(weight_gb: float, ram_gb: float) -> float:
    """RAM left for caches once the weights and the OS reservation are accounted
    for, floored so a tight machine is not budgeted at zero.

    Mirror of `headroomUnits` in formal/lean/Yojit/Limits.lean, and the only place
    this expression may live. Everything derived from headroom -- the KV-cache
    allowance and the launch tuning alike -- reads it from here, so a changed floor
    cannot leave one caller on the old one."""
    return max(ram_gb - weight_gb - RESERVED_OS_GB, MIN_HEADROOM_GB)


def headroom_bytes(weight_gb: float, ram_gb: float) -> float:
    """Bytes this launch may spend on the KV cache: `headroom_gb` times the safety
    factor. `estimate_limits_from_config` sizes a context against it and
    `resolve_kv_cache` checks that context against it -- one budget, two decisions.
    Two copies of this arithmetic is what let a low-RAM machine be sized by one and
    then rejected by the other."""
    return headroom_gb(weight_gb, ram_gb) * (1024 ** 3) * SAFETY_FACTOR


def estimate_limits_from_config(cfg: dict, weight_gb: float, ram_gb: float):
    """Core KV-cache-aware context/output estimate from an HF-style config dict."""
    tcfg = cfg.get("text_config", cfg)

    native_ctx = tcfg.get("max_position_embeddings") or cfg.get("max_position_embeddings") or 32768
    kv_bytes_per_token = _kv_bytes_per_token_fp16(cfg)

    headroom = headroom_bytes(weight_gb, ram_gb)
    max_ctx_by_mem = int(headroom / kv_bytes_per_token) if kv_bytes_per_token > 0 else native_ctx

    # Memory floor never overrides the model's own native context ceiling.
    context = min(native_ctx, MAX_CONTEXT_HARD_CAP, max(MIN_CONTEXT, max_ctx_by_mem))
    if context >= CONTEXT_ROUND_TO:
        context = (context // CONTEXT_ROUND_TO) * CONTEXT_ROUND_TO

    return int(context), output_for_context(context)


def output_for_context(context: int) -> int:
    """Output budget for a context window: a strict quarter, capped.

    Never more than a quarter of the window, and never zero. Source of truth for
    both the stored manifest value and every serve-time recomputation, so the two
    can never disagree about the rule."""
    return max(1, min(context // 4, MAX_OUTPUT_HARD_CAP))


_KV_QUANT_BIT_OPTIONS = (16, 8, 4)  # 16 = unquantized, no override needed
_LLAMACPP_CACHE_TYPE_BY_BITS = {8: "q8_0", 4: "q4_0"}
_KV_BACKENDS = ("mlx_vlm", "llamacpp")


@dataclass(frozen=True)
class KvPlan:
    """A resolved KV-cache plan for one (model, machine, requested context).

    `fits` is the load-bearing field: it separates "this is the most precise
    width that fits" from "nothing fits and the context had to come down". The
    previous contract returned a bare flags dict, so a caller could not tell
    those apart -- a 2x overshoot looked exactly like a perfect fit.
    """
    context: int            # effective context: <= requested, smaller only when nothing fit
    overrides: dict         # launch flags for the chosen width
    fits: bool              # True when the requested context needed no shrink
    bytes_per_token: int    # KV bytes per token at the chosen width
    required_bytes: int     # requested_context * bytes_per_token
    headroom_bytes: int     # what was available to spend


def _pick_kv_bits(context: int, bytes_per_token_fp16: float, headroom_bytes: float) -> int:
    """Highest precision from `_KV_QUANT_BIT_OPTIONS` whose cache fits `context`,
    falling back to the lowest option when nothing fits."""
    for candidate in _KV_QUANT_BIT_OPTIONS:
        if context * bytes_per_token_fp16 * (candidate / 16) <= headroom_bytes:
            return candidate
    return _KV_QUANT_BIT_OPTIONS[-1]


def _fit_context(context: int, bytes_per_token: float, headroom_bytes: float) -> tuple[int, bool]:
    """(effective context, whether the requested one fits). When it does not, the
    effective context is what the chosen width can hold, rounded down to a
    CONTEXT_ROUND_TO multiple. Reporting that pair is the whole point of KvPlan."""
    max_tokens = int(headroom_bytes / bytes_per_token) if bytes_per_token > 0 else context
    if context <= max_tokens:
        return int(context), True
    effective = max(1, min(int(context), max_tokens))
    if effective >= CONTEXT_ROUND_TO:
        effective = (effective // CONTEXT_ROUND_TO) * CONTEXT_ROUND_TO
    return int(effective), False


def _kv_overrides(backend_name: str, bits: int, headroom: float,
                  bytes_per_token_fp16: float, effective_context: int) -> dict:
    if backend_name == "llamacpp":
        return {"kv_cache_quant": _LLAMACPP_CACHE_TYPE_BY_BITS.get(bits, "q4_0")}
    # Where quantization should begin: the token index at which the *unquantized*
    # cache would exhaust headroom. `quantized_kv_start` indexes into the cache, so
    # it must not point past the window we are actually launching with.
    #
    # The clamp is kept deliberately, and the reason is worth stating precisely.
    # Reaching this branch guarantees start < the *requested* context, but not
    # start < effective_context: when the shrink fires, the window is
    # round_to_4096(min(context, max_tokens)), and rounding *down* is applied to a
    # bound derived from different arithmetic. The clamp makes the invariant local
    # instead of resting on a truncation argument about max_tokens/start.
    # (formal/lean/Yojit/Kv.lean: kvStartClamped_le_context states the guarantee;
    #  kvStart_lt_requested_context_of_not_fp16_fits is why it usually changes
    #  nothing.)
    start = int(headroom / bytes_per_token_fp16) if bytes_per_token_fp16 > 0 else 0
    return {"kv_cache_quant": str(bits),
            "quantized_kv_start": max(0, min(start, effective_context))}


def resolve_kv_cache(cfg: dict, backend_name: str, weight_gb: float, ram_gb: float,
                     context: int) -> KvPlan:
    """Picks the highest-precision KV-cache width that fits `context` in this
    machine's real headroom, shrinking `context` when even the lowest width does
    not fit. `yojit config` can still override any returned flag."""
    if backend_name not in _KV_BACKENDS:
        return KvPlan(context=context, overrides={}, fits=True, bytes_per_token=0,
                      required_bytes=0, headroom_bytes=0)

    bytes_per_token_fp16 = _kv_bytes_per_token_fp16(cfg)
    headroom = headroom_bytes(weight_gb, ram_gb)

    # Unquantized already fits: no override, no shrink.
    if bytes_per_token_fp16 <= 0 or context * bytes_per_token_fp16 <= headroom:
        return KvPlan(context=int(context), overrides={}, fits=True,
                      bytes_per_token=int(bytes_per_token_fp16),
                      required_bytes=int(context * bytes_per_token_fp16),
                      headroom_bytes=int(headroom))

    bits = _pick_kv_bits(context, bytes_per_token_fp16, headroom)
    bytes_per_token = bytes_per_token_fp16 * (bits / 16)
    effective_context, fits = _fit_context(context, bytes_per_token, headroom)

    return KvPlan(
        context=effective_context,
        overrides=_kv_overrides(backend_name, bits, headroom, bytes_per_token_fp16,
                                effective_context),
        fits=fits,
        bytes_per_token=int(bytes_per_token),
        required_bytes=int(context * bytes_per_token),
        headroom_bytes=int(headroom),
    )


def default_kv_cache_overrides(cfg: dict, backend_name: str, weight_gb: float, ram_gb: float,
                               context: int) -> dict:
    """Launch flags only. Kept for callers that do not need the effective context;
    anything that actually launches a server should use resolve_kv_cache() instead,
    so the shrunk context reaches both the backend and opencode.json."""
    return resolve_kv_cache(cfg, backend_name, weight_gb, ram_gb, context).overrides


def kv_fit_limits(cfg: dict, backend_name: str, weight_gb: float, ram_gb: float,
                  context: int, output: int) -> tuple[int, int]:
    """Shrinks (context, output) to what the chosen KV-cache width can actually
    hold here. Applied at install time *and* at serve time so the manifest never
    records a context this machine cannot serve, and a machine whose RAM changed
    since install still gets a servable one."""
    plan = resolve_kv_cache(cfg, backend_name, weight_gb, ram_gb, context)
    if plan.fits:
        return context, output
    return plan.context, output_for_context(plan.context)


def classify_mlx_model(leaf_dir: Path, ram_gb: float):
    """Returns (tier, context, output, weight_gb) for an MLX model directory."""
    weight_gb = weight_size_gb_from_dir(leaf_dir)
    cfg = load_hf_config(leaf_dir)
    context, output = estimate_limits_from_config(cfg, weight_gb, ram_gb)
    context, output = kv_fit_limits(cfg, "mlx_vlm", weight_gb, ram_gb, context, output)
    tier = resource_tier(weight_gb, ram_gb)
    return tier, context, output, round(weight_gb, 1)


def classify_gguf_model(gguf_path: Path, ram_gb: float, arch_hints: dict | None = None):
    """Returns (tier, context, output, weight_gb) for a single GGUF file.
    arch_hints (from the GGUF header) map onto the same keys as an HF config."""
    weight_gb = weight_size_gb_from_file(gguf_path)
    cfg = arch_hints or {}
    context, output = estimate_limits_from_config(cfg, weight_gb, ram_gb)
    context, output = kv_fit_limits(cfg, "llamacpp", weight_gb, ram_gb, context, output)
    tier = resource_tier(weight_gb, ram_gb)
    return tier, context, output, round(weight_gb, 1)


# Headroom -> (prefill-step-size for MLX, batch-size/ubatch-size for llama.cpp).
_HEADROOM_TIER_GB = (2.0, 4.0, 8.0, 16.0)
_MLX_PREFILL_STEP_SIZES = (512, 1024, 2048, 4096, 8192)
_LLAMACPP_BATCH_SIZES = (512, 1024, 2048, 2048, 4096)
_LLAMACPP_UBATCH_SIZES = (256, 512, 512, 1024, 2048)


def _headroom_tier_index(headroom_gb: float) -> int:
    for i, ceiling in enumerate(_HEADROOM_TIER_GB):
        if headroom_gb < ceiling:
            return i
    return len(_HEADROOM_TIER_GB)


def compute_launch_tuning(weight_gb: float, ram_gb: float, cpu_cores: int) -> dict:
    """Every launch parameter, computed fresh from this machine's actual
    specs rather than fixed constants. Concurrency stays pinned to 1 until
    concurrent-request memory accounting is modeled explicitly.

    Every key here is read by a backend. A `prompt_cache_bytes` budget used to be
    computed here for `--prompt-cache-bytes`, a flag mlx_vlm's server has never
    had (absent in 0.6.17 and 0.7.2), so it sized a ceiling that nothing could
    apply -- and at 0.4 of raw headroom with a 0.5 GiB floor it would have been
    the one budget not obeying SAFETY_FACTOR. Removed rather than left as a knob
    that looks implemented."""
    headroom = headroom_gb(weight_gb, ram_gb)
    tier = _headroom_tier_index(headroom)

    return {
        # MLX
        "prefill_step_size": _MLX_PREFILL_STEP_SIZES[tier],
        "decode_concurrency": 1,
        # llama.cpp
        "threads": max(1, cpu_cores - 1),
        "ngl": 999,  # full GPU offload
        "batch_size": _LLAMACPP_BATCH_SIZES[tier],
        "ubatch_size": _LLAMACPP_UBATCH_SIZES[tier],
    }
