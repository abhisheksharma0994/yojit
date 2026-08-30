"""Coverage for the simple cmd_* dispatch functions in cli.py -- each one
mostly delegates to another module, so these tests confirm the right call
happens with the right arguments and output."""
import pytest

from yojit import cli, manifest
from yojit.specs import Specs


def _specs(**overrides):
    base = dict(platform="darwin", is_apple_silicon=True, chip="Apple Silicon",
                total_ram_gb=24.0, free_disk_gb=100.0, cpu_cores=8)
    base.update(overrides)
    return Specs(**base)


def _args(**kwargs):
    class Args:
        pass
    a = Args()
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def test_cmd_install_resolves_with_detected_ram(mocker):
    mocker.patch.object(cli.specs, "detect", return_value=_specs(total_ram_gb=32.0))
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    cli.cmd_install(_args(repo="org/repo", bits=4, file=None))
    mock_resolve.assert_called_once_with("org/repo", 4, None, 32.0)


def test_cmd_explore_reports_no_results(mocker, capsys):
    mocker.patch.object(cli.specs, "detect", return_value=_specs())
    mocker.patch.object(cli.hf_explore, "search", return_value=[])
    mocker.patch.object(cli.hf_explore, "rank_candidates", return_value=[])
    cli.cmd_explore(_args(query=None, limit=50, bits=None))
    assert "No results." in capsys.readouterr().out


def test_cmd_explore_installs_the_chosen_result(mocker, monkeypatch):
    mocker.patch.object(cli.specs, "detect", return_value=_specs())
    mocker.patch.object(cli.hf_explore, "search", return_value=[{"id": "org/a"}])
    mocker.patch.object(cli.hf_explore, "rank_candidates", return_value=[{"id": "org/a"}, {"id": "org/b"}])
    monkeypatch.setattr("builtins.input", lambda _: "2")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    cli.cmd_explore(_args(query=None, limit=50, bits=None))
    mock_resolve.assert_called_once_with("org/b", None, None, 24.0)


def test_cmd_explore_cancels_on_invalid_choice(mocker, monkeypatch, capsys):
    mocker.patch.object(cli.specs, "detect", return_value=_specs())
    mocker.patch.object(cli.hf_explore, "search", return_value=[{"id": "org/a"}])
    mocker.patch.object(cli.hf_explore, "rank_candidates", return_value=[{"id": "org/a"}])
    monkeypatch.setattr("builtins.input", lambda _: "not-a-number")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    cli.cmd_explore(_args(query=None, limit=50, bits=None))
    mock_resolve.assert_not_called()
    assert "Cancelled." in capsys.readouterr().out


def test_cmd_list_reports_no_models(models_root, capsys):
    cli.cmd_list(_args())
    assert "No models installed." in capsys.readouterr().out


def test_cmd_list_shows_installed_models_and_marks_default(models_root, capsys):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a",
                                  "tier": "low", "size_gb": 5.0, "context": 8192, "output": 2048})
    out = capsys.readouterr()  # clear
    cli.cmd_list(_args())
    out = capsys.readouterr().out
    assert "org/a (default)" in out
    assert "backend=mlx_vlm" in out


def test_cmd_use_sets_default_and_syncs(models_root, opencode_config):
    manifest.add_model("org/a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a"})
    manifest.add_model("org/b", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/b"})
    cli.cmd_use(_args(model="org/b"))
    assert manifest.get_default() == "org/b"


def test_cmd_serve_forwards_model_and_no_open_flag(mocker):
    mock_serve = mocker.patch.object(cli.server, "serve")
    cli.cmd_serve(_args(model="org/a", no_open=True))
    mock_serve.assert_called_once_with("org/a", open_opencode=False)


def test_cmd_stop_reports_when_nothing_running(mocker, capsys):
    mocker.patch.object(cli.server, "_port_pid", return_value=None)
    cli.cmd_stop(_args())
    assert "No server running." in capsys.readouterr().out


def test_cmd_stop_kills_the_running_pid(mocker, capsys):
    mocker.patch.object(cli.server, "_port_pid", return_value=4242)
    mock_run = mocker.patch.object(cli.subprocess, "run")
    cli.cmd_stop(_args())
    mock_run.assert_called_once_with(["kill", "4242"])
    assert "Stopped PID 4242" in capsys.readouterr().out


def test_cmd_status_reports_running(mocker, capsys):
    mocker.patch.object(cli.server, "_port_pid", return_value=4242)
    cli.cmd_status(_args())
    assert "PID 4242" in capsys.readouterr().out


def test_cmd_status_reports_not_running(mocker, capsys):
    mocker.patch.object(cli.server, "_port_pid", return_value=None)
    cli.cmd_status(_args())
    assert "No server running." in capsys.readouterr().out


def test_cmd_remove_prints_installer_result(mocker, capsys):
    mocker.patch.object(cli.installer, "remove", return_value="Removed org/a")
    cli.cmd_remove(_args(model="org/a"))
    assert "Removed org/a" in capsys.readouterr().out


def test_cmd_sync_prints_sync_result(mocker, capsys):
    mocker.patch.object(cli.opencode_sync, "sync", return_value="Synced 0 model(s)")
    cli.cmd_sync(_args())
    assert "Synced 0 model(s)" in capsys.readouterr().out


def test_cmd_doctor_prints_every_finding(mocker, capsys):
    mocker.patch.object(cli.doctor, "run", return_value=["OK: one", "WARNING: two"])
    cli.cmd_doctor(_args())
    out = capsys.readouterr().out
    assert "OK: one" in out and "WARNING: two" in out


def test_cmd_upgrade_upgrades_mlx_vlm_when_venv_exists(mocker):
    mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.mlx_env, "venv_python", return_value=mocker.MagicMock(exists=lambda: True))
    mock_pip_install = mocker.patch.object(cli.mlx_env, "pip_install")
    mocker.patch.object(cli.shutil, "which", return_value=None)
    mocker.patch.object(cli.manifest, "list_models", return_value={})
    cli.cmd_upgrade(_args())
    mock_pip_install.assert_called_once_with("mlx-vlm", upgrade=True)


def test_cmd_upgrade_skips_mlx_vlm_when_venv_missing(mocker):
    mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.mlx_env, "venv_python", return_value=mocker.MagicMock(exists=lambda: False))
    mock_pip_install = mocker.patch.object(cli.mlx_env, "pip_install")
    mocker.patch.object(cli.shutil, "which", return_value=None)
    mocker.patch.object(cli.manifest, "list_models", return_value={})
    cli.cmd_upgrade(_args())
    mock_pip_install.assert_not_called()


def test_cmd_upgrade_upgrades_llama_cpp_and_opencode_when_present(mocker):
    mock_run = mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.mlx_env, "venv_python", return_value=mocker.MagicMock(exists=lambda: False))
    mocker.patch.object(cli.shutil, "which", return_value="/usr/bin/fake")
    mocker.patch.object(cli.manifest, "list_models", return_value={})
    cli.cmd_upgrade(_args())
    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["brew", "upgrade", "llama.cpp"] in calls
    assert ["opencode", "upgrade"] in calls


def test_cmd_upgrade_reports_each_installed_model(mocker, capsys):
    mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.mlx_env, "venv_python", return_value=mocker.MagicMock(exists=lambda: False))
    mocker.patch.object(cli.shutil, "which", return_value=None)
    mocker.patch.object(cli.manifest, "list_models",
                         return_value={"org/a": {"source_repo": "org/a"}})
    cli.cmd_upgrade(_args())
    assert "org/a: checked" in capsys.readouterr().out


def test_main_exits_without_a_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
