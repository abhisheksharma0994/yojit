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
    store_dir = _make_fake_store_entry(models_root, "mlx", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx",
        "store_path": f"store/mlx/{store_dir.name}",
        "tier": "low",
    })
    discovery.rebuild_tier_symlinks()

    link = manifest.models_root() / "low" / "mlx" / store_dir.name
    assert link.is_symlink()
    assert link.resolve() == store_dir.resolve()
    # Model must NOT appear in a tier/backend combo it doesn't belong to
    assert not (manifest.models_root() / "medium" / "mlx" / store_dir.name).exists()
    assert not (manifest.models_root() / "low" / "llamacpp" / store_dir.name).exists()


def test_rebuild_is_idempotent_and_drops_stale_entries(models_root):
    store_dir = _make_fake_store_entry(models_root, "mlx", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx", "store_path": f"store/mlx/{store_dir.name}", "tier": "low",
    })
    discovery.rebuild_tier_symlinks()
    manifest.remove_model("org/model-a")
    discovery.rebuild_tier_symlinks()

    link = manifest.models_root() / "low" / "mlx" / store_dir.name
    assert not link.exists()


def test_rebuild_skips_manifest_entries_whose_files_are_missing(models_root):
    # Manifest says this model exists, but its store_path was never created --
    # rebuild must not crash, just skip it.
    manifest.add_model("org/ghost-model", {
        "backend": "mlx", "store_path": "store/mlx/does-not-exist", "tier": "low",
    })
    discovery.rebuild_tier_symlinks()  # must not raise
    assert not any((manifest.models_root() / "low" / "mlx").iterdir())


def test_rebuild_moves_model_to_new_tier_when_reclassified(models_root):
    """Simulates re-running discovery after moving to different hardware (or
    after a RAM change) -- the same model can land in a different tier
    without re-downloading anything."""
    store_dir = _make_fake_store_entry(models_root, "mlx", "org__model-a")
    manifest.add_model("org/model-a", {
        "backend": "mlx", "store_path": f"store/mlx/{store_dir.name}", "tier": "low",
    })
    discovery.rebuild_tier_symlinks()
    assert (manifest.models_root() / "low" / "mlx" / store_dir.name).is_symlink()

    # Re-classify to "high" (as if re-run on a machine with less RAM)
    data = manifest.load()
    data["models"]["org/model-a"]["tier"] = "high"
    manifest.save(data)
    discovery.rebuild_tier_symlinks()

    assert not (manifest.models_root() / "low" / "mlx" / store_dir.name).exists()
    assert (manifest.models_root() / "high" / "mlx" / store_dir.name).is_symlink()
