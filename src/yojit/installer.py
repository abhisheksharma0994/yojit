"""Downloads models straight into store/<backend>/, bypassing HF's cache-blob
layer, so multi-quant repos only pull the one variant we actually want."""
from pathlib import Path

from huggingface_hub import snapshot_download

from . import classify, discovery, gguf_meta, hf_explore, manifest, opencode_sync


def _safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def install_mlx(repo_id: str, subfolder: str | None, ram_gb: float) -> str:
    root = manifest.store_root() / "mlx"
    root.mkdir(parents=True, exist_ok=True)
    dest_name = _safe_name(repo_id) if not subfolder else f"{_safe_name(repo_id)}__{subfolder}"
    dest = root / dest_name

    allow_patterns = [f"{subfolder}/*"] if subfolder else None
    snapshot_download(repo_id, allow_patterns=allow_patterns, local_dir=str(dest))
    leaf_dir = dest / subfolder if subfolder else dest

    tier, context, output, weight_gb = classify.classify_mlx_model(leaf_dir, ram_gb)
    bits = hf_explore.bit_width_from_name(repo_id) or hf_explore.bit_width_from_name(subfolder or "")

    entry = {
        "backend": "mlx_vlm",
        "store_path": str(leaf_dir.relative_to(manifest.models_root())),
        "bits": bits,
        "size_gb": weight_gb,
        "tier": tier,
        "context": context,
        "output": output,
        "source_repo": repo_id,
        "verified": False,
    }
    manifest.add_model(repo_id if not subfolder else f"{repo_id}:{subfolder}", entry)
    discovery.rebuild_tier_symlinks()
    opencode_sync.sync()
    return f"Installed {repo_id} ({weight_gb} GB, tier={tier}, context={context})"


def install_gguf(repo_id: str, filename: str, ram_gb: float) -> str:
    from huggingface_hub import hf_hub_download

    root = manifest.store_root() / "llamacpp"
    root.mkdir(parents=True, exist_ok=True)
    dest_name = f"{_safe_name(repo_id)}__{filename}"

    downloaded = hf_hub_download(repo_id, filename, local_dir=str(root))
    final_path = root / dest_name
    Path(downloaded).rename(final_path)

    arch_hints = gguf_meta.to_hf_style_config(gguf_meta.read_metadata(final_path))
    tier, context, output, weight_gb = classify.classify_gguf_model(final_path, ram_gb, arch_hints)
    quant_match = filename.rsplit(".", 1)[0].split("-")[-1]

    model_id = f"{repo_id}:{quant_match}"
    entry = {
        "backend": "llamacpp",
        "store_path": str(final_path.relative_to(manifest.models_root())),
        "quant": quant_match,
        "size_gb": weight_gb,
        "tier": tier,
        "context": context,
        "output": output,
        "source_repo": repo_id,
        "verified": False,
    }
    manifest.add_model(model_id, entry)
    discovery.rebuild_tier_symlinks()
    opencode_sync.sync()
    return f"Installed {model_id} ({weight_gb} GB, tier={tier}, context={context})"


def remove(model_id: str) -> str:
    entry = manifest.get_model(model_id)
    if not entry:
        return f"{model_id} is not installed"
    store_path = manifest.models_root() / entry["store_path"]
    if store_path.is_dir():
        import shutil
        shutil.rmtree(store_path, ignore_errors=True)
    elif store_path.exists():
        store_path.unlink()
    manifest.remove_model(model_id)
    discovery.rebuild_tier_symlinks()
    opencode_sync.sync()
    return f"Removed {model_id}"
