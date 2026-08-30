import json

import pytest

from yojit import manifest, opencode_sync, server


class FakeBackend:
    """Stands in for a real Backend so tests never spawn a real process or
    hit a real port."""
    name = "mlx"

    def __init__(self, health_sequence):
        # health_sequence is popped from the front on each health_check() call
        self._health_sequence = list(health_sequence)
        self.ensure_installed_called = False
        self.warm_up_called = False
        self.warm_up_called_with = None
        self.launch_called_with = None

    def ensure_installed(self):
        self.ensure_installed_called = True

    def launch(self, model_path, port, context, output_limit, tuning, overrides=None):
        self.launch_called_with = (model_path, port, context, output_limit, tuning, overrides)

        class FakeProc:
            pid = 4242
        return FakeProc()

    def health_check(self, port):
        if self._health_sequence:
            return self._health_sequence.pop(0)
        return True

    def warm_up(self, port, model_id):
        self.warm_up_called = True
        self.warm_up_called_with = model_id


# --- opencode.json <-> backend launch contract -----------------------------

def test_serve_passes_manifest_overrides_through_to_backend_launch(
    models_root, opencode_config, monkeypatch
):
    """Per-model overrides set via `yojit config` must reach backend.launch()."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    manifest.update_overrides("org/model-a", seed=42, kv_cache_quant=8)
    fake = FakeBackend(health_sequence=[True])
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    server.serve("org/model-a", open_opencode=False)

    overrides_passed = fake.launch_called_with[5]
    assert overrides_passed == {"seed": 42, "kv_cache_quant": 8}


def test_serve_passes_empty_overrides_when_none_ever_configured(
    models_root, opencode_config, monkeypatch
):
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    fake = FakeBackend(health_sequence=[True])
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    server.serve("org/model-a", open_opencode=False)

    assert fake.launch_called_with[5] == {}


def test_serve_applies_hardware_derived_defaults_for_a_never_configured_model(
    models_root, opencode_config, monkeypatch
):
    """A model with no `yojit config` overrides must still get a real,
    machine-derived configuration, not an empty {}."""
    import yojit.specs as specs_module
    fake_specs = specs_module.Specs(platform="darwin", is_apple_silicon=True, chip="test",
                                     total_ram_gb=24.0, free_disk_gb=100.0, cpu_cores=8)
    monkeypatch.setattr(server.specs, "detect", lambda: fake_specs)
    fake = FakeBackend(health_sequence=[True])
    fake.name = "mlx_vlm"
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "high",
                                        "size_gb": 15.0, "context": 16384})
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    server.serve("org/model-a", open_opencode=False)

    from yojit import classify
    expected = classify.default_kv_cache_overrides({}, "mlx_vlm", 15.0, 24.0, 16384)
    assert fake.launch_called_with[5] == expected
    assert expected != {}  # sanity: this scenario (15GB/24GB) must actually need quantization


def test_serve_explicit_config_overrides_win_over_hardware_derived_defaults(
    models_root, opencode_config, monkeypatch
):
    import yojit.specs as specs_module
    fake_specs = specs_module.Specs(platform="darwin", is_apple_silicon=True, chip="test",
                                     total_ram_gb=24.0, free_disk_gb=100.0, cpu_cores=8)
    monkeypatch.setattr(server.specs, "detect", lambda: fake_specs)
    fake = FakeBackend(health_sequence=[True])
    fake.name = "mlx_vlm"
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "high",
                                        "size_gb": 15.0, "context": 16384})
    manifest.update_overrides("org/model-a", kv_cache_quant="4")  # user explicitly wants 4-bit
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    server.serve("org/model-a", open_opencode=False)

    overrides = fake.launch_called_with[5]
    assert overrides["kv_cache_quant"] == "4"
    assert "quantized_kv_start" in overrides  # the untouched default field survives the merge


def test_warm_up_uses_the_local_path_not_the_manifest_model_id(
    models_root, opencode_config, monkeypatch
):
    """warm_up() must use the local path, not the manifest's repo-style model_id --
    the server matches "model" by exact string equality against --model."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    fake = FakeBackend(health_sequence=[True])
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    server.serve("org/model-a", open_opencode=False)

    launched_model_path = str(fake.launch_called_with[0])
    assert fake.warm_up_called_with == launched_model_path
    assert fake.warm_up_called_with != "org/model-a"


def test_opencode_json_model_key_matches_what_the_backend_was_launched_with(
    models_root, opencode_config, monkeypatch
):
    """opencode.json's model key must match exactly what the backend was launched with,
    not the manifest's repo-style model_id."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    fake = FakeBackend(health_sequence=[True])
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    server.serve("org/model-a", open_opencode=False)

    launched_model_path = str(fake.launch_called_with[0])
    config = json.loads(opencode_config.read_text())
    opencode_keys = list(config["provider"]["local"]["models"].keys())

    assert launched_model_path in opencode_keys, (
        f"backend was launched with {launched_model_path!r} but opencode.json's "
        f"model keys are {opencode_keys!r} -- opencode would send a 'model' "
        f"field the running server doesn't recognize"
    )


def test_opencode_sync_key_format_matches_backends_model_path_computation(models_root, opencode_config):
    """opencode_sync.py must key by exactly manifest.models_root() / entry["store_path"],
    matching what each backend uses for --model."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    expected_key = str(manifest.models_root() / "store/mlx/a")
    assert expected_key in config["provider"]["local"]["models"]


# --- RISKY/safe picker with confirmation gate ------------------------------

def test_pick_model_interactive_sorts_safe_before_risky(models_root, monkeypatch, capsys):
    manifest.add_model("org/risky", {"backend": "mlx", "tier": "high", "size_gb": 20.0})
    manifest.add_model("org/safe", {"backend": "mlx", "tier": "low", "size_gb": 5.0})

    inputs = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chosen = server.pick_model_interactive()

    out = capsys.readouterr().out
    safe_line_pos = out.index("org/safe")
    risky_line_pos = out.index("org/risky")
    assert safe_line_pos < risky_line_pos  # safe listed first
    assert chosen == "org/safe"  # option "1" is the safe one


def test_pick_model_interactive_requires_confirmation_for_risky_pick(models_root, monkeypatch):
    manifest.add_model("org/risky", {"backend": "mlx", "tier": "high", "size_gb": 20.0})
    manifest.add_model("org/safe", {"backend": "mlx", "tier": "low", "size_gb": 5.0})
    # Sorted safe-first: "1"=org/safe, "2"=org/risky.
    # User picks the risky one ("2"), declines ("n"), then picks the safe one ("1").
    inputs = iter(["2", "n", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chosen = server.pick_model_interactive()
    assert chosen == "org/safe"  # after declining the risky one, safe is picked


def test_pick_model_interactive_accepts_risky_pick_on_confirmation(models_root, monkeypatch):
    manifest.add_model("org/risky", {"backend": "mlx", "tier": "high", "size_gb": 20.0})
    inputs = iter(["1", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chosen = server.pick_model_interactive()
    assert chosen == "org/risky"


def test_pick_model_interactive_returns_none_when_nothing_installed(models_root):
    assert server.pick_model_interactive() is None


def test_pick_model_interactive_pre_highlights_current_default(models_root, monkeypatch, capsys):
    manifest.add_model("org/a", {"backend": "mlx", "tier": "low", "size_gb": 5.0})
    manifest.add_model("org/b", {"backend": "mlx", "tier": "low", "size_gb": 5.0})
    manifest.set_default("org/b")

    monkeypatch.setattr("builtins.input", lambda _: "1")  # pick org/a regardless
    server.pick_model_interactive()

    out = capsys.readouterr().out
    assert "org/b (current default)" in out


def test_pick_model_interactive_blank_input_accepts_the_current_default(models_root, monkeypatch):
    manifest.add_model("org/a", {"backend": "mlx", "tier": "low", "size_gb": 5.0})
    manifest.add_model("org/b", {"backend": "mlx", "tier": "low", "size_gb": 5.0})
    manifest.set_default("org/b")

    monkeypatch.setattr("builtins.input", lambda _: "")  # blank = accept default
    chosen = server.pick_model_interactive()

    assert chosen == "org/b"


# --- serve(): picker always shown when there's a real choice ---------------

def test_serve_skips_the_picker_when_exactly_one_model_is_installed(models_root, opencode_config, monkeypatch):
    manifest.add_model("org/only-one", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    monkeypatch.setattr(server, "_attempt_launch", lambda model_id: (True, 4242))
    mock_pick = MockCounter()
    monkeypatch.setattr(server, "pick_model_interactive", mock_pick)

    server.serve(None, open_opencode=False)

    assert mock_pick.calls == 0  # nothing to choose between -- picker never invoked


def test_serve_always_shows_the_picker_when_multiple_models_installed_even_with_a_default_set(
    models_root, opencode_config, monkeypatch
):
    """Bare serve() must always show the picker when multiple models exist, even with a default set."""
    manifest.add_model("org/a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    manifest.add_model("org/b", {"backend": "mlx", "store_path": "store/mlx/b", "tier": "low"})
    manifest.set_default("org/a")  # a default IS set

    monkeypatch.setattr(server, "_attempt_launch", lambda model_id: (True, 4242))
    mock_pick = MockCounter(return_value="org/b")
    monkeypatch.setattr(server, "pick_model_interactive", mock_pick)

    server.serve(None, open_opencode=False)

    assert mock_pick.calls == 1  # picker WAS shown despite the default being set


class MockCounter:
    def __init__(self, return_value=None):
        self.calls = 0
        self.return_value = return_value

    def __call__(self):
        self.calls += 1
        return self.return_value


# --- retry-on-failure loop with removal offer ------------------------------

def test_serve_with_explicit_model_does_not_retry_on_failure(models_root, opencode_config, monkeypatch):
    """An explicit model (arg or default) gets ONE attempt -- failure exits,
    it never silently falls back to a different model the caller didn't ask for."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    fake = FakeBackend(health_sequence=[True, False])  # comes up, then fails warm-up check
    monkeypatch.setattr(server, "get_backend", lambda name: fake)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    with pytest.raises(SystemExit):
        server.serve("org/model-a", open_opencode=False)


def test_serve_interactive_retry_loop_offers_removal_on_failure(models_root, opencode_config, monkeypatch):
    manifest.add_model("org/broken", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/broken",
                                       "tier": "low", "size_gb": 5.0})
    manifest.add_model("org/good", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/good",
                                     "tier": "low", "size_gb": 5.0})
    # Clear the default so serve(None) takes the interactive-picker-with-retry
    # path -- with a default set, serve(None) correctly uses it directly via
    # the single-attempt path instead (that's the documented, intended split).
    data = manifest.load()
    data["default_model"] = None
    manifest.save(data)

    call_count = {"n": 0}

    def fake_attempt_launch(model_id):
        call_count["n"] += 1
        if model_id == "org/broken":
            return False, None
        return True, 4242

    monkeypatch.setattr(server, "_attempt_launch", fake_attempt_launch)

    pick_sequence = iter(["org/broken", "org/good"])
    monkeypatch.setattr(server, "pick_model_interactive", lambda: next(pick_sequence))

    remove_calls = []

    def fake_remove(model_id):
        remove_calls.append(model_id)
        manifest.remove_model(model_id)  # real side effect, so default reassignment is genuine
        return f"Removed {model_id}"

    monkeypatch.setattr(server.installer, "remove", fake_remove)

    input_sequence = iter(["y"])  # agree to remove the broken model
    monkeypatch.setattr("builtins.input", lambda _: next(input_sequence))

    server.serve(None, open_opencode=False)

    assert remove_calls == ["org/broken"]
    assert call_count["n"] == 2  # tried broken, then good
    assert manifest.get_default() == "org/good"  # broken one is gone from the manifest too
