"""Shared fixtures. Every test that touches "disk" or "config" must go
through these -- nothing in this suite is allowed to read or write a
developer's real ~/Models or ~/.config/opencode/opencode.json."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    """Points manifest.models_root() at a throwaway directory."""
    root = tmp_path / "yojit-home"
    monkeypatch.setenv("YOJIT_HOME", str(root))
    return root


@pytest.fixture
def opencode_config(tmp_path, monkeypatch):
    """Points opencode_sync.config_path() at a throwaway file, pre-seeded
    with the minimal valid shape sync() expects to find."""
    config_file = tmp_path / "opencode.json"
    config_file.write_text(json.dumps({"$schema": "https://opencode.ai/config.json", "provider": {}}))
    monkeypatch.setenv("YOJIT_OPENCODE_CONFIG", str(config_file))
    return config_file


@pytest.fixture
def no_network(monkeypatch):
    """Makes any real network call fail loudly instead of hanging or hitting
    the real internet -- tests that need network behavior mock requests
    explicitly instead of relying on this fixture succeeding silently."""
    import requests

    def _blocked(*args, **kwargs):
        raise RuntimeError("network access attempted in a unit test")

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
