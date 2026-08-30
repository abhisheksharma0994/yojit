import json
from pathlib import Path

import pytest

from yojit import manifest


def test_models_root_respects_env_override(models_root):
    assert manifest.models_root() == models_root


def test_models_root_defaults_to_repo_local_models_dir_in_a_dev_checkout(monkeypatch):
    """No env var set: if this package is running from a real source
    checkout (pyproject.toml present), models_root must resolve to
    <repo>/models -- never silently fall back anywhere else."""
    monkeypatch.delenv(manifest.MODELS_ROOT_ENV_VAR, raising=False)
    repo_root = manifest._repo_root_if_dev_checkout()
    assert repo_root is not None, "this test must itself run from a dev checkout"
    assert manifest.models_root() == repo_root / "models"


def test_repo_root_if_dev_checkout_resolves_two_levels_up_from_this_file():
    """src/yojit/manifest.py -> parents[2] is the repo root (2 levels up)."""
    expected = Path(manifest.__file__).resolve().parents[2]
    assert manifest._repo_root_if_dev_checkout() == expected


def test_models_root_raises_a_clear_error_when_not_a_dev_checkout_and_no_override(monkeypatch):
    """Simulates a real pip/pipx install of a built package: no repo folder
    exists at runtime. There must be NO silent fallback location -- just a
    clear, actionable error telling the user to set the env var."""
    monkeypatch.delenv(manifest.MODELS_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(manifest, "_repo_root_if_dev_checkout", lambda: None)
    with pytest.raises(RuntimeError, match=manifest.MODELS_ROOT_ENV_VAR):
        manifest.models_root()


def test_load_on_missing_manifest_returns_empty_shape(models_root):
    data = manifest.load()
    assert data == {"schema_version": 1, "default_model": None, "models": {}}


def test_add_model_creates_manifest_and_sets_default(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "size_gb": 5.0})
    data = manifest.load()
    assert "org/model-a" in data["models"]
    assert data["default_model"] == "org/model-a"
    # add_model must fill in defaults for fields callers don't provide
    assert "added_at" in data["models"]["org/model-a"]
    assert data["models"]["org/model-a"]["verified"] is False


def test_add_model_does_not_override_existing_default(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    manifest.add_model("org/model-b", {"backend": "mlx", "store_path": "store/mlx/b"})
    assert manifest.get_default() == "org/model-a"


def test_remove_model_reassigns_default_to_a_survivor(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    manifest.add_model("org/model-b", {"backend": "mlx", "store_path": "store/mlx/b"})
    manifest.remove_model("org/model-a")
    assert manifest.get_default() == "org/model-b"
    assert manifest.get_model("org/model-a") is None


def test_remove_model_clears_default_when_no_models_remain(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    manifest.remove_model("org/model-a")
    assert manifest.get_default() is None
    assert manifest.list_models() == {}


def test_remove_nonexistent_model_is_a_safe_noop(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    entry = manifest.remove_model("org/does-not-exist")
    assert entry is None
    assert manifest.get_default() == "org/model-a"  # untouched


def test_set_default_rejects_uninstalled_model(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    with pytest.raises(KeyError, match="org/not-installed"):
        manifest.set_default("org/not-installed")


def test_set_default_switches_between_installed_models(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    manifest.add_model("org/model-b", {"backend": "mlx", "store_path": "store/mlx/b"})
    manifest.set_default("org/model-b")
    assert manifest.get_default() == "org/model-b"


def test_manifest_survives_corrupt_json_by_returning_empty_shape(models_root):
    manifest.models_root().mkdir(parents=True, exist_ok=True)
    manifest.manifest_path().write_text("{not valid json")
    data = manifest.load()
    assert data == {"schema_version": 1, "default_model": None, "models": {}}


def test_update_overrides_rejects_uninstalled_model(models_root):
    with pytest.raises(KeyError, match="org/not-installed"):
        manifest.update_overrides("org/not-installed", seed=42)


def test_update_overrides_writes_only_given_fields(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a"})
    result = manifest.update_overrides("org/model-a", seed=42, kv_cache_quant=None)
    assert result == {"seed": 42}
    assert manifest.get_model("org/model-a")["overrides"] == {"seed": 42}


def test_update_overrides_merges_without_clobbering_prior_fields(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a"})
    manifest.update_overrides("org/model-a", seed=42)
    result = manifest.update_overrides("org/model-a", kv_cache_quant=8)
    assert result == {"seed": 42, "kv_cache_quant": 8}


def test_update_overrides_can_replace_a_previously_set_field(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a"})
    manifest.update_overrides("org/model-a", seed=42)
    result = manifest.update_overrides("org/model-a", seed=7)
    assert result == {"seed": 7}


# --- Retired-backend migration ---------------------------------------------
# The plain mlx-lm-only backend ("mlx") was removed once mlx_vlm.server was
# verified to serve text-only models correctly too. Any manifest already on
# disk with the old name must keep working transparently -- never a
# KeyError from backends.get_backend("mlx"), which no longer exists.

def test_load_migrates_a_retired_backend_name_transparently(models_root):
    manifest.models_root().mkdir(parents=True, exist_ok=True)
    manifest.manifest_path().write_text(json.dumps({
        "schema_version": 1,
        "default_model": "org/old-model",
        "models": {"org/old-model": {"backend": "mlx", "store_path": "store/mlx/old-model"}},
    }))

    data = manifest.load()

    assert data["models"]["org/old-model"]["backend"] == "mlx_vlm"


def _write_manifest_without_models_key(models_root):
    manifest.models_root().mkdir(parents=True, exist_ok=True)
    manifest.manifest_path().write_text(json.dumps({"schema_version": 1, "default_model": None}))


def test_load_tolerates_a_manifest_with_no_models_key_at_all(models_root):
    _write_manifest_without_models_key(models_root)
    manifest.load()  # must not raise while migrating retired backend names


def test_get_model_and_list_models_default_to_empty_when_models_key_missing(models_root):
    _write_manifest_without_models_key(models_root)
    assert manifest.get_model("anything") is None
    assert manifest.list_models() == {}


def test_set_default_and_update_overrides_raise_key_error_when_models_key_missing(models_root):
    _write_manifest_without_models_key(models_root)
    with pytest.raises(KeyError):
        manifest.set_default("org/model-a")
    with pytest.raises(KeyError):
        manifest.update_overrides("org/model-a", seed=1)


def test_remove_model_is_a_safe_noop_when_models_key_missing(models_root):
    _write_manifest_without_models_key(models_root)
    assert manifest.remove_model("org/model-a") is None


def test_add_model_creates_models_key_when_manifest_lacks_it(models_root):
    _write_manifest_without_models_key(models_root)
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a"})
    assert "org/model-a" in manifest.list_models()


def test_now_iso_matches_the_documented_utc_format():
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest.now_iso())


def test_load_leaves_current_backend_names_untouched(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a"})
    manifest.add_model("org/model-b", {"backend": "llamacpp", "store_path": "store/llamacpp/b.gguf"})

    data = manifest.load()

    assert data["models"]["org/model-a"]["backend"] == "mlx_vlm"
    assert data["models"]["org/model-b"]["backend"] == "llamacpp"
