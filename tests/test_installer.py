import json

from yojit import installer, manifest


def _fake_snapshot_download(repo_id, allow_patterns, local_dir):
    """Mimics huggingface_hub.snapshot_download's real effect closely enough
    for classify.py to read something real: creates config.json +
    a safetensors file at the requested location."""
    from pathlib import Path
    dest = Path(local_dir)
    leaf = dest / allow_patterns[0].split("/")[0] if allow_patterns else dest
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "config.json").write_text(json.dumps({
        "max_position_embeddings": 32768,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
    }))
    (leaf / "model.safetensors").write_bytes(b"0" * (1024 * 1024))  # 1MB fake weight
    return str(dest)


def test_install_mlx_writes_manifest_entry_and_syncs(models_root, opencode_config, mocker):
    mocker.patch("yojit.installer.snapshot_download", side_effect=_fake_snapshot_download)

    result = installer.install_mlx("org/fake-model-4bit", None, ram_gb=24.0)

    assert "Installed org/fake-model-4bit" in result
    entry = manifest.get_model("org/fake-model-4bit")
    assert entry is not None
    # mlx_vlm, not mlx: verified live to serve plain text-only models
    # correctly too, plus it's the only one with real KV-cache-quant/
    # context-enforcement flags -- see backends/mlx_vlm.py's docstring.
    assert entry["backend"] == "mlx_vlm"
    assert entry["bits"] == 4  # parsed from the "-4bit" suffix
    assert entry["source_repo"] == "org/fake-model-4bit"
    assert entry["verified"] is False

    # discovery + opencode sync side effects must have run
    assert ((manifest.models_root() / "low" / "mlx_vlm").exists()
            or (manifest.models_root() / "medium" / "mlx_vlm").exists())
    config = json.loads(opencode_config.read_text())
    # Keyed by the local path the backend actually launches with, not the
    # HF-repo-style model_id -- see opencode_sync.py's comment.
    key = str(manifest.models_root() / entry["store_path"])
    assert key in config["provider"]["local"]["models"]
    assert "org/fake-model-4bit" in config["provider"]["local"]["models"][key]["name"]


def test_install_mlx_with_subfolder_uses_composite_model_id(models_root, opencode_config, mocker):
    """A multi-quant repo's subfolder install must be addressable as repo:subfolder."""
    mocker.patch("yojit.installer.snapshot_download", side_effect=_fake_snapshot_download)

    installer.install_mlx("org/multi-quant-repo", "4-bit", ram_gb=24.0)

    entry = manifest.get_model("org/multi-quant-repo:4-bit")
    assert entry is not None
    assert entry["store_path"].endswith("4-bit")


def test_install_gguf_writes_manifest_entry(models_root, opencode_config, mocker, tmp_path):
    def fake_hf_hub_download(repo_id, filename, local_dir):
        from pathlib import Path
        # Simulate the real cache-then-symlink shape: file lands at some path
        # under local_dir, not necessarily named exactly `filename`.
        p = Path(local_dir) / filename
        p.write_bytes(b"GGUF" + b"0" * 1024)  # fake header + body, not a valid GGUF but enough for the rename step
        return str(p)

    # install_gguf imports hf_hub_download locally (inside the function) at
    # call time, so the patch target is huggingface_hub itself, not installer.
    mocker.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download)

    result = installer.install_gguf("org/gguf-repo", "model-Q4_K_M.gguf", ram_gb=24.0)

    assert "Installed org/gguf-repo:Q4_K_M" in result
    entry = manifest.get_model("org/gguf-repo:Q4_K_M")
    assert entry is not None
    assert entry["backend"] == "llamacpp"
    assert entry["quant"] == "Q4_K_M"


def test_remove_deletes_store_files_and_manifest_entry_and_syncs(models_root, opencode_config, mocker):
    mocker.patch("yojit.installer.snapshot_download", side_effect=_fake_snapshot_download)
    installer.install_mlx("org/fake-model-4bit", None, ram_gb=24.0)
    entry = manifest.get_model("org/fake-model-4bit")
    store_path = manifest.models_root() / entry["store_path"]
    assert store_path.exists()

    result = installer.remove("org/fake-model-4bit")

    assert "Removed" in result
    assert not store_path.exists()
    assert manifest.get_model("org/fake-model-4bit") is None
    config = json.loads(opencode_config.read_text())
    assert config["provider"]["local"]["models"] == {}


def test_remove_nonexistent_model_reports_cleanly(models_root, opencode_config):
    result = installer.remove("org/never-installed")
    assert "not installed" in result


def test_remove_deletes_a_single_gguf_file(models_root, opencode_config, mocker):
    def fake_hf_hub_download(repo_id, filename, local_dir):
        from pathlib import Path
        p = Path(local_dir) / filename
        p.write_bytes(b"GGUF" + b"0" * 1024)
        return str(p)

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download)
    installer.install_gguf("org/gguf-repo", "model-Q4_K_M.gguf", ram_gb=24.0)
    entry = manifest.get_model("org/gguf-repo:Q4_K_M")
    store_path = manifest.models_root() / entry["store_path"]
    assert store_path.is_file()

    result = installer.remove("org/gguf-repo:Q4_K_M")

    assert "Removed" in result
    assert not store_path.exists()
    assert manifest.get_model("org/gguf-repo:Q4_K_M") is None
