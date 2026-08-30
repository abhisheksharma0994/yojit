"""_resolve_install's branch logic: GGUF vs MLX, root-level weights vs
bit-suffixed subfolders, explicit --bits vs auto-pick.
"""
import pytest

from yojit import cli


def test_resolve_install_routes_gguf_backend_to_install_gguf(mocker):
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=[{"path": "model.gguf", "size": 1024}])
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="llamacpp")
    mocker.patch.object(cli.hf_explore, "pick_best_gguf_file", return_value={"path": "model-Q4.gguf", "size": 1024})
    mock_install = mocker.patch.object(cli.installer, "install_gguf", return_value="Installed")

    cli._resolve_install("org/repo", None, None, 24.0)

    mock_install.assert_called_once_with("org/repo", "model-Q4.gguf", 24.0)


def test_resolve_install_uses_explicit_gguf_file_without_auto_pick(mocker):
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=[{"path": "model.gguf"}])
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="llamacpp")
    mock_pick = mocker.patch.object(cli.hf_explore, "pick_best_gguf_file")
    mock_install = mocker.patch.object(cli.installer, "install_gguf", return_value="Installed")

    cli._resolve_install("org/repo", None, "explicit.gguf", 24.0)

    mock_pick.assert_not_called()
    mock_install.assert_called_once_with("org/repo", "explicit.gguf", 24.0)


def test_resolve_install_exits_when_no_gguf_file_fits(mocker):
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=[{"path": "model.gguf"}])
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="llamacpp")
    mocker.patch.object(cli.hf_explore, "pick_best_gguf_file", return_value=None)

    with pytest.raises(SystemExit):
        cli._resolve_install("org/repo", None, None, 24.0)


def test_resolve_install_uses_root_weights_when_no_bits_requested(mocker):
    files = [{"path": "config.json"}, {"path": "model.safetensors"}]
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=files)
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="mlx")
    mock_install = mocker.patch.object(cli.installer, "install_mlx", return_value="Installed")

    cli._resolve_install("org/repo", None, None, 24.0)

    mock_install.assert_called_once_with("org/repo", None, 24.0)


def test_resolve_install_picks_explicit_bit_subfolder(mocker):
    files = [{"path": "4-bit/config.json"}, {"path": "4-bit/model.safetensors"},
             {"path": "8-bit/config.json"}, {"path": "8-bit/model.safetensors"}]
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=files)
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="mlx")
    mock_install = mocker.patch.object(cli.installer, "install_mlx", return_value="Installed")

    cli._resolve_install("org/repo", 8, None, 24.0)

    mock_install.assert_called_once_with("org/repo", "8-bit", 24.0)


def test_resolve_install_exits_when_requested_bits_not_available(mocker):
    files = [{"path": "4-bit/config.json"}, {"path": "4-bit/model.safetensors"}]
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=files)
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="mlx")

    with pytest.raises(SystemExit):
        cli._resolve_install("org/repo", 8, None, 24.0)


def test_resolve_install_auto_picks_largest_bit_subfolder_when_bits_unset(mocker):
    files = [{"path": "4-bit/config.json"}, {"path": "4-bit/model.safetensors"},
             {"path": "8-bit/config.json"}, {"path": "8-bit/model.safetensors"}]
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=files)
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value="mlx")
    mock_install = mocker.patch.object(cli.installer, "install_mlx", return_value="Installed")

    cli._resolve_install("org/repo", None, None, 24.0)

    mock_install.assert_called_once_with("org/repo", "8-bit", 24.0)


def test_resolve_install_exits_when_no_mlx_weights_found_anywhere(mocker):
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=[{"path": "README.md"}])
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value=None)

    with pytest.raises(SystemExit):
        cli._resolve_install("org/repo", None, None, 24.0)


def test_resolve_install_falls_back_to_root_weights_when_no_subfolders_but_root_has_them(mocker):
    files = [{"path": "config.json"}, {"path": "model.safetensors"}]
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=files)
    mocker.patch.object(cli.hf_explore, "detect_backend_from_files", return_value=None)
    mock_install = mocker.patch.object(cli.installer, "install_mlx", return_value="Installed")

    cli._resolve_install("org/repo", 4, None, 24.0)

    mock_install.assert_called_once_with("org/repo", None, 24.0)
