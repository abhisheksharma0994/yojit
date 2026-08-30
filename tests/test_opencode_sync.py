import json

from yojit import manifest, opencode_sync


def test_sync_reports_missing_config_without_crashing(models_root, tmp_path, monkeypatch):
    monkeypatch.setenv("YOJIT_OPENCODE_CONFIG", str(tmp_path / "does-not-exist.json"))
    result = opencode_sync.sync()
    assert "not found" in result


def test_sync_creates_the_provider_block_when_the_config_lacks_one(tmp_path, monkeypatch, models_root):
    config_file = tmp_path / "opencode.json"
    config_file.write_text(json.dumps({"$schema": "https://opencode.ai/config.json"}))
    monkeypatch.setenv("YOJIT_OPENCODE_CONFIG", str(config_file))

    opencode_sync.sync()

    config = json.loads(config_file.read_text())
    assert config["provider"]["local"] == {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Local (yojit)",
        "options": {"baseURL": f"http://localhost:{opencode_sync.PORT}/v1"},
        "models": {},
    }


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
    # Keyed by the exact local path the backend launches with (matches
    # mlx_lm.server's plain string-equality model matching), not the
    # manifest's HF-repo-style model_id -- see opencode_sync.py's comment.
    key = str(manifest.models_root() / "store/mlx/a")
    assert key in models
    assert "org/model-a" in models[key]["name"]
    assert models[key]["limit"] == {"context": 8192, "output": 2048}


def test_sync_marks_the_running_model(models_root, opencode_config):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    opencode_sync.sync(running_model="org/model-a")
    config = json.loads(opencode_config.read_text())
    key = str(manifest.models_root() / "store/mlx/a")
    assert "(running)" in config["provider"]["local"]["models"][key]["name"]


def test_sync_does_not_mark_other_models_when_a_different_model_is_running(models_root, opencode_config):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    manifest.add_model("org/model-b", {"backend": "mlx", "store_path": "store/mlx/b"})
    opencode_sync.sync(running_model="org/model-b")
    config = json.loads(opencode_config.read_text())
    key_a = str(manifest.models_root() / "store/mlx/a")
    assert "(running)" not in config["provider"]["local"]["models"][key_a]["name"]


def test_sync_defaults_missing_fields_on_a_sparse_manifest_entry(models_root, opencode_config):
    manifest.add_model("org/bare", {"store_path": "store/mlx/bare"})  # no backend, context, or output

    opencode_sync.sync()

    config = json.loads(opencode_config.read_text())
    key = str(manifest.models_root() / "store/mlx/bare")
    model = config["provider"]["local"]["models"][key]
    assert "(local)" in model["name"]
    assert model["limit"] == {"context": 4096, "output": 1024}


def test_sync_fully_replaces_model_list_dropping_removed_models(models_root, opencode_config):
    """This is the specific behavior the user relies on: a model deleted
    outside the tool (or via `remove`) must disappear from opencode.json on
    the next sync, not linger as a stale entry."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a"})
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    key = str(manifest.models_root() / "store/mlx/a")
    assert key in config["provider"]["local"]["models"]

    manifest.remove_model("org/model-a")
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    assert config["provider"]["local"]["models"] == {}


def test_sync_summary_text_reports_each_model_and_marks_the_running_one(models_root, opencode_config):
    manifest.add_model("org/model-a", {
        "backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low", "size_gb": 5.0, "context": 8192,
        "output": 2048,
    })
    manifest.add_model("org/model-b", {"backend": "llamacpp", "store_path": "store/llamacpp/b", "tier": "medium"})

    summary = opencode_sync.sync(running_model="org/model-a")

    assert "Synced 2 model(s)" in summary
    assert "org/model-a <- currently running" in summary
    assert "org/model-b\n" in summary  # no marker for the model that isn't running
    assert "backend: mlx_vlm, tier: low, weights: 5.0 GB, context: 8192, max output: 2048" in summary


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
