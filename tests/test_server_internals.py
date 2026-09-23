"""Lower-level server.py pieces not already covered by test_server.py's
higher-level serve()/pick_model_interactive() tests."""
import os

import pytest

from yojit import locking, manifest, server

# The real implementation, captured before the autouse fixture below replaces the
# attribute on the module. Tests of pid_alive itself must not go through the stub.
real_pid_alive = server.pid_alive


@pytest.fixture(autouse=True)
def _fake_pid_is_alive(monkeypatch):
    """The fake backends here return pid 4242, which is not a real process; the
    liveness half of serve()'s re-validation is stubbed so the tests can focus on
    the record itself."""
    monkeypatch.setattr(server, "pid_alive", lambda pid: True)


def test_port_pid_returns_none_on_lsof_failure(mocker):
    mocker.patch.object(server.subprocess, "check_output", side_effect=Exception("lsof not found"))
    assert server._port_pid(8080) is None


def test_port_pid_parses_the_first_pid(mocker):
    mocker.patch.object(server.subprocess, "check_output", return_value="4242\n5555\n")
    assert server._port_pid(8080) == 4242


def test_free_port_stops_yojits_own_recorded_server(models_root, mocker):
    server._record_server(4242, "org/a", 4096, 1024)
    mocker.patch.object(server, "_port_pid", return_value=4242)
    mock_run = mocker.patch.object(server.subprocess, "run")
    mocker.patch.object(server.time, "sleep")
    server._free_port(8080)
    mock_run.assert_called_once_with(["kill", "4242"])
    assert server.read_server_record() == {}, "the stale record must not outlive the server it names"


def test_free_port_refuses_to_kill_a_foreign_listener(models_root, mocker):
    """No record names this PID, so it is not ours to kill -- the old code
    killed whatever held the port, which is how you lose someone's other server."""
    mocker.patch.object(server, "_port_pid", return_value=4242)
    mock_run = mocker.patch.object(server.subprocess, "run")
    with pytest.raises(server.PortOwnedByAnotherProcess):
        server._free_port(8080)
    mock_run.assert_not_called()


def test_free_port_ignores_a_record_for_a_different_port(models_root, mocker):
    server._record_server(4242, "org/a", 4096, 1024)  # recorded on server.PORT, not 9999
    mocker.patch.object(server, "_port_pid", return_value=4242)
    mock_run = mocker.patch.object(server.subprocess, "run")
    with pytest.raises(server.PortOwnedByAnotherProcess):
        server._free_port(9999)
    mock_run.assert_not_called()


def test_free_port_does_nothing_when_port_is_free(mocker):
    mocker.patch.object(server, "_port_pid", return_value=None)
    mock_run = mocker.patch.object(server.subprocess, "run")
    server._free_port(8080)
    mock_run.assert_not_called()


def test_recorded_limits_are_the_ones_actually_launched_with(models_root, opencode_config, mocker):
    """The record must describe the launch plan, not the stored entry -- otherwise
    `yojit status` reports a context the live server never had."""
    manifest.add_model("org/a", {
        "backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low",
        "context": 4096, "output": 1024, "size_gb": 0.0,
    })
    launched = {}

    class FakeBackend:
        name = "mlx_vlm"

        def ensure_installed(self):
            pass

        def launch(self, path, port, context, output, tuning, overrides):
            launched["context"] = context
            launched["output"] = output
            return mocker.Mock(pid=4242)

        def health_check(self, port):
            return True

        def warm_up(self, port, model_id):
            pass

    mocker.patch.object(server, "get_backend", return_value=FakeBackend())
    mocker.patch.object(server, "_free_port")
    mocker.patch.object(server.specs, "detect", return_value=server.specs.Specs(
        platform="darwin", is_apple_silicon=True, chip="M1", total_ram_gb=0.0,
        free_disk_gb=100.0, cpu_cores=4))

    ok, _ = server._attempt_launch("org/a")

    assert ok is True
    record = server.read_server_record()
    assert record["pid"] == 4242
    assert record["context"] == launched["context"]
    assert record["output"] == launched["output"]


def test_read_server_record_returns_empty_for_a_corrupt_file(models_root, mocker):
    """A truncated or hand-edited server.json must degrade to "nothing recorded"
    rather than taking down every command that consults it."""
    path = server.server_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert server.read_server_record() == {}


def test_clear_server_record_is_a_noop_when_there_is_nothing_to_clear(models_root):
    server.clear_server_record()
    assert server.read_server_record() == {}


def test_pid_alive_rejects_values_that_are_not_pids():
    assert real_pid_alive(None) is False
    assert real_pid_alive("not-a-pid") is False
    assert real_pid_alive(0) is False
    assert real_pid_alive(-5) is False
    assert real_pid_alive(os.getpid()) is True


def test_pid_alive_is_false_for_a_process_that_does_not_exist():
    assert real_pid_alive(999999999) is False


def test_attempt_launch_gives_up_when_the_port_belongs_to_someone_else(models_root, opencode_config, mocker):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low"})

    class FakeBackend:
        name = "mlx_vlm"

        def ensure_installed(self):
            pass

    mocker.patch.object(server, "get_backend", return_value=FakeBackend())
    mocker.patch.object(server, "_free_port",
                        side_effect=server.PortOwnedByAnotherProcess("port 8080 is not ours"))

    ok, pid = server._attempt_launch("org/a")
    assert ok is False
    assert pid is None


def test_attempt_launch_gives_up_when_another_invocation_is_mid_launch(models_root, opencode_config, mocker):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "tier": "low"})

    class FakeBackend:
        name = "mlx_vlm"

        def ensure_installed(self):
            pass

    mocker.patch.object(server, "get_backend", return_value=FakeBackend())
    mocker.patch.object(server, "_free_port", side_effect=locking.StateLockTimeout("held by PID 7"))

    ok, pid = server._attempt_launch("org/a")
    assert ok is False
    assert pid is None


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
