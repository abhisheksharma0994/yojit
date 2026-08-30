"""cmd_upgrade must treat both backends symmetrically. mlx_env.pip_install()
is mocked explicitly since it uses its own subprocess call, separate from cli.subprocess.run."""
import argparse

from yojit import cli


def _args():
    return argparse.Namespace()


def test_upgrade_covers_both_backends_when_both_are_installed(mocker):
    mocker.patch.object(cli.shutil, "which", return_value="/usr/bin/fake")
    mocker.patch.object(cli.mlx_env, "venv_python", return_value=mocker.MagicMock(exists=lambda: True))
    mock_pip_install = mocker.patch.object(cli.mlx_env, "pip_install")
    mock_run = mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.manifest, "list_models", return_value={})

    cli.cmd_upgrade(_args())

    mock_pip_install.assert_any_call("mlx-vlm", upgrade=True)
    all_calls = [call.args[0] for call in mock_run.call_args_list]
    assert any("llama.cpp" in call for call in all_calls), "llama.cpp upgrade must run -- this was missing before"


def test_upgrade_skips_mlx_vlm_when_the_venv_does_not_exist(mocker):
    def fake_which(name):
        return None if name == "llama-server" else "/usr/bin/fake"

    mocker.patch.object(cli.shutil, "which", side_effect=fake_which)
    mocker.patch.object(cli.mlx_env, "venv_python", return_value=mocker.MagicMock(exists=lambda: False))
    mock_pip_install = mocker.patch.object(cli.mlx_env, "pip_install")
    mock_run = mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.manifest, "list_models", return_value={})

    cli.cmd_upgrade(_args())

    mock_pip_install.assert_not_called()
    all_calls = [call.args[0] for call in mock_run.call_args_list]
    assert not any("llama.cpp" in call for call in all_calls)
