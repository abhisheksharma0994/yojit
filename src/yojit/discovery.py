"""Rebuilds the low/medium/high x mlx/llamacpp symlink tree from the manifest.

Tier is relative to *this machine's* RAM, so it's never baked into where a
model's actual files live (store/<backend>/...) -- only the symlink tree,
which this function deletes and recreates from scratch every time, is
recomputed. Moving to different hardware (or a RAM upgrade) just means
re-running this; no re-download needed.
"""
import shutil
from pathlib import Path

from . import manifest

TIERS = ("low", "medium", "high")
BACKENDS = ("mlx", "llamacpp")


def rebuild_tier_symlinks() -> None:
    root = manifest.models_root()
    for tier in TIERS:
        tier_dir = root / tier
        if tier_dir.exists():
            shutil.rmtree(tier_dir)
        for backend in BACKENDS:
            (tier_dir / backend).mkdir(parents=True, exist_ok=True)

    for model_id, entry in manifest.list_models().items():
        tier = entry.get("tier", "high")
        backend = entry.get("backend", "mlx")
        store_path = root / entry["store_path"]
        if not store_path.exists():
            continue
        link_dir = root / tier / backend
        link_name = link_dir / store_path.name
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        try:
            link_name.symlink_to(store_path)
        except OSError:
            pass  # symlinks unsupported (rare) -- browsing convenience only, not fatal
