"""cmd_upgrade must treat both backends symmetrically -- whatever upgrade
path exists for MLX must also exist for llama.cpp. Regression: an earlier
version upgraded mlx-lm but forgot llama.cpp entirely."""
import argparse

from yojit import cli


def _args():
    return argparse.Namespace()


def test_upgrade_covers_both_backends_when_both_are_installed(mocker):
    mocker.patch.object(cli.shutil, "which", return_value="/usr/bin/fake")
    mock_run = mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.manifest, "list_models", return_value={})

    cli.cmd_upgrade(_args())

    all_calls = [call.args[0] for call in mock_run.call_args_list]
    assert any("mlx-lm" in call for call in all_calls), "mlx-lm upgrade must run"
    assert any("llama.cpp" in call for call in all_calls), "llama.cpp upgrade must run -- this was missing before"


def test_upgrade_skips_a_backend_that_is_not_installed(mocker):
    def fake_which(name):
        return None if name == "llama-server" else "/usr/bin/fake"

    mocker.patch.object(cli.shutil, "which", side_effect=fake_which)
    mock_run = mocker.patch.object(cli.subprocess, "run")
    mocker.patch.object(cli.manifest, "list_models", return_value={})

    cli.cmd_upgrade(_args())

    all_calls = [call.args[0] for call in mock_run.call_args_list]
    assert not any("llama.cpp" in call for call in all_calls)
    assert any("mlx-lm" in call for call in all_calls)
