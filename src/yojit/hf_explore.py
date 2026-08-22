"""Search Hugging Face, rank by downloads, detect backend, and pick the best
bit/quant variant that still fits comfortably (not just the tightest fit)."""
import re

import requests

from . import classify

API = "https://huggingface.co/api/models"
DEFAULT_AUTHOR = "mlx-community"


def search(query: str | None = None, author: str | None = DEFAULT_AUTHOR, limit: int = 50):
    params = {"sort": "downloads", "direction": "-1", "limit": limit}
    if author:
        params["author"] = author
    if query:
        params["search"] = query
    r = requests.get(API, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def repo_files(repo_id: str) -> list[dict]:
    r = requests.get(f"{API}/{repo_id}/tree/main", timeout=15)
    r.raise_for_status()
    return r.json()


def detect_backend_from_files(files: list[dict]) -> str | None:
    names = [f.get("path", "") for f in files]
    if any(n.endswith(".safetensors") for n in names) and any(n == "config.json" for n in names):
        return "mlx"
    if any(n.endswith(".gguf") for n in names):
        return "llamacpp"
    return None


BIT_SUFFIX_RE = re.compile(r"-(\d+)bit$", re.IGNORECASE)


def bit_width_from_name(repo_id: str) -> int | None:
    m = BIT_SUFFIX_RE.search(repo_id)
    return int(m.group(1)) if m else None


def sibling_bit_variants(repo_id: str, candidates: list[dict]) -> list[dict]:
    """Given a chosen repo, find other bit-width variants of the same base
    model family (same name with the -Nbit suffix swapped)."""
    base = BIT_SUFFIX_RE.sub("", repo_id)
    return [c for c in candidates if BIT_SUFFIX_RE.sub("", c["id"]) == base]


def pick_best_fit(variants: list[dict], ram_gb: float, sizes_gb: dict[str, float]):
    """Among sibling bit-variants, pick the LARGEST bit-width that still lands
    in low/medium tier -- comfortable fit, not the tightest possible one."""
    scored = []
    for v in variants:
        size = sizes_gb.get(v["id"])
        if size is None:
            continue
        tier = classify.resource_tier(size, ram_gb)
        if tier in ("low", "medium") and classify.fits_at_all(size, ram_gb):
            bits = bit_width_from_name(v["id"]) or 0
            scored.append((bits, v))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def pick_best_gguf_file(files: list[dict], ram_gb: float) -> dict | None:
    """Among all .gguf files in a repo, pick the LARGEST (highest precision)
    one that still lands in low/medium tier -- the same "don't assume
    smallest is best" principle as pick_best_fit() above, but keyed on
    actual file size directly rather than a parsed bit-width label. GGUF
    quant-name conventions (Q4_K_M, Q5_0, IQ3_XXS, ...) vary too much across
    converters to parse reliably; a GGUF file's listed size is a single,
    unsharded, always-accurate number, so there's no need to guess."""
    candidates = [f for f in files if f.get("path", "").endswith(".gguf") and f.get("size")]
    fitting = []
    for f in candidates:
        size_gb = f["size"] / (1024 ** 3)
        tier = classify.resource_tier(size_gb, ram_gb)
        if tier in ("low", "medium") and classify.fits_at_all(size_gb, ram_gb):
            fitting.append((size_gb, f))
    if not fitting:
        return None
    fitting.sort(key=lambda x: x[0], reverse=True)
    return fitting[0][1]


# mlx-community (and HF generally) hosts every kind of MLX-format model --
# speech-to-text, embeddings, vision-only -- not just chat LLMs. Filtering by
# file format alone (detect_backend_from_files) can't tell them apart; HF's
# own pipeline_tag field can. Found via a real bug: an unfiltered download
# ranking surfaced an automatic-speech-recognition model as opencode's
# suggested "chat" model.
CHAT_COMPATIBLE_PIPELINE_TAGS = {
    "text-generation",
    "text2text-generation",
    "conversational",
    "image-text-to-text",  # vision-language models still serve chat completions
}


def is_chat_model(candidate: dict) -> bool:
    """True if this repo is actually usable as a chat/completions model.
    A missing pipeline_tag is treated as unknown, not chat-compatible --
    a repo with no declared task should not be assumed safe to serve."""
    return candidate.get("pipeline_tag") in CHAT_COMPATIBLE_PIPELINE_TAGS


# HF has no first-class "supports tool-calling" field -- most models that DO
# support it (e.g. Qwen3 via its Hermes-style format) don't tag it at all,
# so this can only be a soft ranking boost, never a hard filter: excluding
# every repo that lacks the tag would drop most genuinely tool-capable
# models. This checks the repo's own self-declared tags for these keywords
# -- it is not a hardcoded model/family list, and stays generic across
# whatever repos exist now or get published later.
AGENTIC_TAG_KEYWORDS = ("agent", "agentic", "tool-use", "tool-calling", "function-calling")


def has_agentic_signal(candidate: dict) -> bool:
    tags = [t.lower() for t in candidate.get("tags", [])]
    return any(any(kw in tag for kw in AGENTIC_TAG_KEYWORDS) for tag in tags)


def rank_candidates(raw_results: list[dict], ram_gb: float) -> list[dict]:
    """Filters out non-chat models (ASR, embeddings, vision-only encoders,
    etc.) via pipeline_tag, then sorts the rest with models carrying an
    agentic/tool-use tag boosted to the front, ties broken by downloads."""
    chat_only = [m for m in raw_results if is_chat_model(m)]
    return sorted(
        chat_only,
        key=lambda m: (has_agentic_signal(m), m.get("downloads", 0)),
        reverse=True,
    )
