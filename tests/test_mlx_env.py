from pathlib import Path

from yojit import mlx_env


def test_venv_python_path_is_under_yojit_home():
    p = mlx_env.venv_python()
    assert p.parent.name == "bin"
    assert p.parent.parent.name == "venv"
    assert p.parent.parent.parent.name == ".yojit"


def test_module_command_uses_venv_python_explicitly_not_path(mocker):
    """module_command() must always build the venv's own interpreter path, never rely on PATH."""
    cmd = mlx_env.module_command("mlx_lm.server", "--model", "/some/path")
    assert cmd[0] == str(mlx_env.venv_python())
    assert cmd[1] == "-m"
    assert cmd[2] == "mlx_lm.server"
    assert cmd[3:] == ["--model", "/some/path"]


def test_module_command_does_not_touch_the_filesystem(mocker, tmp_path):
    """module_command() must be a pure path-builder, no ensure_venv() side effect."""
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mlx_env.module_command("mlx_vlm.server", "--port", "8080")
    mock_run.assert_not_called()


def test_ensure_venv_skips_creation_if_already_present(mocker):
    mocker.patch.object(Path, "exists", return_value=True)
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mlx_env.ensure_venv()
    mock_run.assert_not_called()


def test_ensure_venv_creates_and_upgrades_pip_if_missing(mocker):
    mocker.patch.object(Path, "exists", return_value=False)
    mocker.patch.object(Path, "mkdir")
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mlx_env.ensure_venv()
    assert mock_run.call_count == 2
    venv_call = mock_run.call_args_list[0][0][0]
    assert "venv" in venv_call
    pip_upgrade_call = mock_run.call_args_list[1][0][0]
    assert "pip" in pip_upgrade_call and "--upgrade" in pip_upgrade_call


def test_is_installed_checks_inside_the_venv(mocker):
    mocker.patch("yojit.mlx_env.ensure_venv")
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mock_run.return_value.returncode = 0
    assert mlx_env.is_installed("mlx_lm") is True
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == str(mlx_env.venv_python())
    assert "import mlx_lm" in cmd[2]


def test_is_installed_false_on_nonzero_exit(mocker):
    mocker.patch("yojit.mlx_env.ensure_venv")
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mock_run.return_value.returncode = 1
    assert mlx_env.is_installed("mlx_vlm") is False


def test_pip_install_targets_the_venv_pip_not_system(mocker):
    mocker.patch("yojit.mlx_env.ensure_venv")
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mlx_env.pip_install("mlx-vlm")
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == str(mlx_env.venv_python())
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert "mlx-vlm" in cmd
    assert "--upgrade" not in cmd


def test_pip_install_upgrade_flag(mocker):
    mocker.patch("yojit.mlx_env.ensure_venv")
    mock_run = mocker.patch("yojit.mlx_env.subprocess.run")
    mlx_env.pip_install("mlx-lm", "mlx-vlm", upgrade=True)
    cmd = mock_run.call_args[0][0]
    assert "--upgrade" in cmd
    assert "mlx-lm" in cmd and "mlx-vlm" in cmd
