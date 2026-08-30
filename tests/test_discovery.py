from pathlib import Path

from yojit import discovery, manifest


def _make_fake_store_entry(root, backend, name):
    store_dir = manifest.store_root() / backend / name
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def test_rebuild_creates_all_tier_backend_folders_even_when_empty(models_root):
    discovery.rebuild_tier_symlinks()
    for tier in discovery.TIERS:
        for backend in discovery.BACKENDS:
            assert (manifest.models_root() / tier / backend).is_dir()


def test_rebuild_symlinks_installed_model_into_its_tier_and_backend(models_root):
    store_dir = _make_fake_store_entry(models_root, "mlx_vlm", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx_vlm",
        "store_path": f"store/mlx_vlm/{store_dir.name}",
        "tier": "low",
    })
    discovery.rebuild_tier_symlinks()

    link = manifest.models_root() / "low" / "mlx_vlm" / store_dir.name
    assert link.is_symlink()
    assert link.resolve() == store_dir.resolve()
    # Model must NOT appear in a tier/backend combo it doesn't belong to
    assert not (manifest.models_root() / "medium" / "mlx_vlm" / store_dir.name).exists()
    assert not (manifest.models_root() / "low" / "llamacpp" / store_dir.name).exists()


def test_rebuild_is_idempotent_and_drops_stale_entries(models_root):
    store_dir = _make_fake_store_entry(models_root, "mlx_vlm", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx_vlm", "store_path": f"store/mlx_vlm/{store_dir.name}", "tier": "low",
    })
    discovery.rebuild_tier_symlinks()
    manifest.remove_model("org/model-a")
    discovery.rebuild_tier_symlinks()

    link = manifest.models_root() / "low" / "mlx_vlm" / store_dir.name
    assert not link.exists()


def test_rebuild_skips_manifest_entries_whose_files_are_missing(models_root):
    # Manifest says this model exists, but its store_path was never created --
    # rebuild must not crash, just skip it.
    manifest.add_model("org/ghost-model", {
        "backend": "mlx_vlm", "store_path": "store/mlx_vlm/does-not-exist", "tier": "low",
    })
    discovery.rebuild_tier_symlinks()  # must not raise
    assert not any((manifest.models_root() / "low" / "mlx_vlm").iterdir())


def test_rebuild_moves_model_to_new_tier_when_reclassified(models_root):
    """Simulates re-running discovery after moving to different hardware (or
    after a RAM change) -- the same model can land in a different tier
    without re-downloading anything."""
    store_dir = _make_fake_store_entry(models_root, "mlx_vlm", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx_vlm", "store_path": f"store/mlx_vlm/{store_dir.name}", "tier": "low",
    })
    discovery.rebuild_tier_symlinks()
    assert (manifest.models_root() / "low" / "mlx_vlm" / store_dir.name).is_symlink()

    # Re-classify to "high" (as if re-run on a machine with less RAM)
    data = manifest.load()
    data["models"]["org/model-a"]["tier"] = "high"
    manifest.save(data)
    discovery.rebuild_tier_symlinks()

    assert not (manifest.models_root() / "low" / "mlx_vlm" / store_dir.name).exists()
    assert (manifest.models_root() / "high" / "mlx_vlm" / store_dir.name).is_symlink()


def test_rebuild_replaces_a_stale_regular_file_at_the_link_location(models_root):
    """If something (e.g. leftover from a bug) already occupies the link
    path as a real file rather than a symlink, rebuild must replace it."""
    store_dir = _make_fake_store_entry(models_root, "mlx_vlm", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx_vlm", "store_path": f"store/mlx_vlm/{store_dir.name}", "tier": "low",
    })
    stale_path = manifest.models_root() / "low" / "mlx_vlm" / store_dir.name
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("not a symlink")

    discovery.rebuild_tier_symlinks()

    assert stale_path.is_symlink()
    assert stale_path.resolve() == store_dir.resolve()


def test_rebuild_tolerates_a_platform_that_cannot_create_symlinks(models_root, mocker):
    store_dir = _make_fake_store_entry(models_root, "mlx_vlm", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx_vlm", "store_path": f"store/mlx_vlm/{store_dir.name}", "tier": "low",
    })
    mocker.patch.object(Path, "symlink_to", side_effect=OSError("symlinks not supported"))

    discovery.rebuild_tier_symlinks()  # must not raise
