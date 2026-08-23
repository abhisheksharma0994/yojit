import json
import os

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
        self.launch_called_with = None

    def ensure_installed(self):
        self.ensure_installed_called = True

    def launch(self, model_path, port, context, output_limit, tuning):
        self.launch_called_with = (model_path, port, context, output_limit, tuning)

        class FakeProc:
            pid = 4242
        return FakeProc()

    def health_check(self, port):
        if self._health_sequence:
            return self._health_sequence.pop(0)
        return True

    def warm_up(self, port, model_id):
        self.warm_up_called = True


# --- opencode.json <-> backend launch contract -----------------------------

def test_opencode_json_model_key_matches_what_the_backend_was_launched_with(
    models_root, opencode_config, monkeypatch
):
    """Regression for a real bug: opencode.json used to be keyed by the
    manifest's HF-repo-style model_id, while the backend was launched with a
    local filesystem path -- two different strings that were supposed to
    refer to the same running model but never matched. mlx_lm.server matches
    each request's "model" field against exactly what it was launched with
    (plain string equality, no alias resolution beyond one internal
    sentinel), so every chat request from opencode silently looked like "load
    a different model" and tried to resolve model_id as a fresh Hugging Face
    repo -- hanging when online, failing outright when offline. Neither
    test_opencode_sync.py (tests sync() in isolation) nor the rest of this
    file (tests serve()'s orchestration logic) ever cross-checked the two
    against each other, so the mismatch shipped silently. This test exists
    specifically to close that gap: it asserts equality between the two
    live values, not two independently-hardcoded expectations that could
    both be wrong the same way."""
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
    """Narrower unit-level version of the same contract, independent of
    serve()'s orchestration: opencode_sync.py must key by exactly
    manifest.models_root() / entry["store_path"] -- the same expression
    backends/mlx.py and backends/llamacpp.py use for their --model
    argument (see backends/mlx.py:33, backends/llamacpp.py:38)."""
    manifest.add_model("org/model-a", {"backend": "mlx", "store_path": "store/mlx/a", "tier": "low"})
    opencode_sync.sync()
    config = json.loads(opencode_config.read_text())
    expected_key = str(manifest.models_root() / "store/mlx/a")
    assert expected_key in config["provider"]["local"]["models"]


# --- offline-by-default posture -------------------------------------------

def test_offline_posture_defaults_to_offline_before_checking(monkeypatch):
    """The environment must be set to offline FIRST, and only flipped to
    online after a real reachability check succeeds -- never the reverse."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(server, "_internet_available", lambda: True)
    online = server.apply_offline_posture()
    assert online is True
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_posture_stays_offline_when_unreachable(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(server, "_internet_available", lambda: False)
    online = server.apply_offline_posture()
    assert online is False
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_offline_posture_sets_offline_env_var_before_the_network_check_runs(monkeypatch):
    """Regression: if the reachability check itself throws or hangs, the
    environment must already be in the safe (offline) state -- not left
    however it was before this function ran."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen_env_during_check = {}

    def flaky_check():
        seen_env_during_check["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        raise RuntimeError("network blew up")

    monkeypatch.setattr(server, "_internet_available", flaky_check)
    with pytest.raises(RuntimeError):
        server.apply_offline_posture()
    assert seen_env_during_check["HF_HUB_OFFLINE"] == "1"


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
    """Regression for a real UX complaint: a sticky default silently skipped
    the picker even when multiple models existed, so the user never got a
    chance to choose. Bare serve() must always show the picker whenever
    there's a real choice, regardless of whether a default is set."""
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
    monkeypatch.setattr(server, "apply_offline_posture", lambda: True)
    monkeypatch.setattr(server, "_free_port", lambda port: None)

    with pytest.raises(SystemExit):
        server.serve("org/model-a", open_opencode=False)


def test_serve_interactive_retry_loop_offers_removal_on_failure(models_root, opencode_config, monkeypatch):
    manifest.add_model("org/broken", {"backend": "mlx", "store_path": "store/mlx/broken", "tier": "low", "size_gb": 5.0})
    manifest.add_model("org/good", {"backend": "mlx", "store_path": "store/mlx/good", "tier": "low", "size_gb": 5.0})
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
