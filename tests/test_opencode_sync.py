import json

from yojit import manifest, opencode_sync


def test_sync_reports_missing_config_without_crashing(models_root, tmp_path, monkeypatch):
    monkeypatch.setenv("YOJIT_OPENCODE_CONFIG", str(tmp_path / "does-not-exist.json"))
    result = opencode_sync.sync()
    assert "not found" in result


def test_sync_writes_local_provider_with_installed_models(models_root, opencode_config):
    manifest.add_model("org/model-a", {
        "backend": "mlx", "store_path": "store/mlx/a", "size_gb": 5.0,
        "tier": "low", "context": 8192, "output": 2048,
    })
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    assert "local" in config["provider"]
    assert config["provider"]["local"]["options"]["baseURL"] == f"http://localhost:{opencode_sync.PORT}/v1"
    models = config["provider"]["local"]["models"]
    assert "org/model-a" in models
    assert models["org/model-a"]["limit"] == {"context": 8192, "output": 2048}


def test_sync_marks_the_running_model(models_root, opencode_config):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    opencode_sync.sync(running_model="org/model-a")
    config = json.loads(opencode_config.read_text())
    assert "(running)" in config["provider"]["local"]["models"]["org/model-a"]["name"]


def test_sync_fully_replaces_model_list_dropping_removed_models(models_root, opencode_config):
    """This is the specific behavior the user relies on: a model deleted
    outside the tool (or via `remove`) must disappear from opencode.json on
    the next sync, not linger as a stale entry."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    assert "org/model-a" in config["provider"]["local"]["models"]

    manifest.remove_model("org/model-a")
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    assert config["provider"]["local"]["models"] == {}


def test_sync_preserves_other_provider_blocks(models_root, opencode_config):
    """Regression: syncing must not clobber unrelated provider config the
    user has (e.g. a cloud provider, or another local backend)."""
    config = json.loads(opencode_config.read_text())
    config["provider"]["other-provider"] = {"npm": "@ai-sdk/openai", "name": "Other"}
    opencode_config.write_text(json.dumps(config))

    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    assert "other-provider" in config["provider"]
    assert config["provider"]["other-provider"]["name"] == "Other"
