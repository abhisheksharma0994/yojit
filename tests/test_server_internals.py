"""Lower-level server.py pieces not already covered by test_server.py's
higher-level serve()/pick_model_interactive() tests."""
import pytest

from yojit import manifest, server


def test_port_pid_returns_none_on_lsof_failure(mocker):
    mocker.patch.object(server.subprocess, "check_output", side_effect=Exception("lsof not found"))
    assert server._port_pid(8080) is None


def test_port_pid_parses_the_first_pid(mocker):
    mocker.patch.object(server.subprocess, "check_output", return_value="4242\n5555\n")
    assert server._port_pid(8080) == 4242


def test_free_port_kills_the_occupying_process(mocker):
    mocker.patch.object(server, "_port_pid", return_value=4242)
    mock_run = mocker.patch.object(server.subprocess, "run")
    mocker.patch.object(server.time, "sleep")
    server._free_port(8080)
    mock_run.assert_called_once_with(["kill", "4242"])


def test_free_port_does_nothing_when_port_is_free(mocker):
    mocker.patch.object(server, "_port_pid", return_value=None)
    mock_run = mocker.patch.object(server.subprocess, "run")
    server._free_port(8080)
    mock_run.assert_not_called()


def test_read_model_choice_reprompts_on_invalid_input(monkeypatch, capsys):
    inputs = iter(["bogus", "99", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = server._read_model_choice(["org/only"], None)
    assert result == "org/only"
    assert capsys.readouterr().out.count("Invalid choice.") == 2


def test_attempt_launch_reports_uninstalled_model(models_root):
    ok, pid = server._attempt_launch("org/never-installed")
    assert ok is False
    assert pid is None


def test_attempt_launch_fails_when_server_never_comes_up(models_root, mocker):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low"})

    class FakeBackend:
        name = "mlx_vlm"

        def ensure_installed(self):
            pass

        def launch(self, *a, **kw):
            return mocker.Mock(pid=1)

        def health_check(self, port):
            return False

    mocker.patch.object(server, "get_backend", return_value=FakeBackend())
    mocker.patch.object(server, "_free_port")
    mocker.patch.object(server.time, "sleep")

    ok, pid = server._attempt_launch("org/a")
    assert ok is False
    assert pid is None


def test_serve_interactively_exits_when_nothing_installed(models_root):
    with pytest.raises(SystemExit):
        server._serve_interactively()


def test_serve_opens_opencode_when_prereqs_and_default_flag(models_root, opencode_config, mocker):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low"})

    class FakeBackend:
        name = "mlx_vlm"

        def ensure_installed(self):
            pass

        def launch(self, *a, **kw):
            return mocker.Mock(pid=4242)

        def health_check(self, port):
            return True

        def warm_up(self, port, model_id):
            pass

    mocker.patch.object(server, "get_backend", return_value=FakeBackend())
    mocker.patch.object(server, "_free_port")
    mocker.patch.object(server.prereqs, "ensure_opencode_installed", return_value=True)
    mock_run = mocker.patch.object(server.subprocess, "run")

    server.serve("org/a", open_opencode=True)

    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["opencode", "upgrade"] in calls
    assert any(c[0] == "opencode" and c[1] == "-m" for c in calls)


def test_serve_skips_opencode_launch_when_prereqs_missing(models_root, opencode_config, mocker):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low"})

    class FakeBackend:
        name = "mlx_vlm"

        def ensure_installed(self):
            pass

        def launch(self, *a, **kw):
            return mocker.Mock(pid=4242)

        def health_check(self, port):
            return True

        def warm_up(self, port, model_id):
            pass

    mocker.patch.object(server, "get_backend", return_value=FakeBackend())
    mocker.patch.object(server, "_free_port")
    mocker.patch.object(server.prereqs, "ensure_opencode_installed", return_value=False)
    mock_run = mocker.patch.object(server.subprocess, "run")

    server.serve("org/a", open_opencode=True)

    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["opencode", "upgrade"] not in calls
    assert not any(isinstance(c, list) and c[:1] == ["opencode"] and "-m" in c for c in calls)
