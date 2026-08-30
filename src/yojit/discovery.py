"""Rebuilds the low/medium/high x backend symlink tree from the manifest.
Tier depends on this machine's RAM, so it's recomputed fresh each time, never baked into the actual file layout."""
import shutil

from . import manifest

TIERS = ("low", "medium", "high")
BACKENDS = ("mlx_vlm", "llamacpp")


def _recreate_tier_dirs(root) -> None:
    for tier in TIERS:
        tier_dir = root / tier
        if tier_dir.exists():
            shutil.rmtree(tier_dir)
        for backend in BACKENDS:
            (tier_dir / backend).mkdir(parents=True, exist_ok=True)


def _link_model(root, tier: str, backend: str, store_path) -> None:
    link_dir = root / tier / backend
    link_dir.mkdir(parents=True, exist_ok=True)  # covers any backend not in BACKENDS too
    link_name = link_dir / store_path.name
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    try:
        link_name.symlink_to(store_path)
    except OSError:
        pass  # symlinks unsupported (rare) -- browsing convenience only, not fatal


def rebuild_tier_symlinks() -> None:
    root = manifest.models_root()
    _recreate_tier_dirs(root)
    for entry in manifest.list_models().values():
        store_path = root / entry["store_path"]
        if store_path.exists():
            _link_model(root, entry.get("tier", "high"), entry.get("backend", "mlx_vlm"), store_path)
