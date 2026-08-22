"""cmd_init's dynamic model-suggestion + implicit install+serve flow.

The suggestion pipeline is the same generic search/rank/fit logic `explore`
uses (hf_explore + classify) -- no hardcoded model list, so it generalizes
to any machine's RAM and stays current as new models are released. These
tests mock the HF-facing calls, never hitting the real network.
"""
import argparse

from yojit import cli, manifest
from yojit.specs import Specs


def _fake_specs(ram_gb=24.0):
    return Specs(platform="darwin", is_apple_silicon=True, chip="Apple M5 Pro",
                 total_ram_gb=ram_gb, free_disk_gb=500.0, cpu_cores=12)


def _args():
    return argparse.Namespace()


# --- _suggest_model_for_ram: pure logic, mocked HF calls -------------------

def test_suggest_model_picks_highest_ranked_candidate_that_fits(mocker):
    mocker.patch.object(cli.hf_explore, "search", return_value=[
        {"id": "org/too-big", "downloads": 9999, "pipeline_tag": "text-generation", "tags": []},
        {"id": "org/fits-well", "downloads": 500, "pipeline_tag": "text-generation", "tags": []},
    ])
    mocker.patch.object(cli.hf_explore, "repo_files", side_effect=lambda repo_id: {
        "org/too-big": [{"path": "config.json"}, {"path": "model.safetensors", "size": 20 * 1024**3}],
        "org/fits-well": [{"path": "config.json"}, {"path": "model.safetensors", "size": 5 * 1024**3}],
    }[repo_id])

    repo, reason = cli._suggest_model_for_ram(24.0)

    assert repo == "org/fits-well"  # too-big is 83% of RAM, skipped for being unsafe


def test_suggest_model_returns_none_when_nothing_fits(mocker):
    mocker.patch.object(cli.hf_explore, "search", return_value=[
        {"id": "org/too-big", "downloads": 100, "pipeline_tag": "text-generation", "tags": []},
    ])
    mocker.patch.object(cli.hf_explore, "repo_files", return_value=[
        {"path": "config.json"}, {"path": "model.safetensors", "size": 20 * 1024**3},
    ])

    repo, reason = cli._suggest_model_for_ram(24.0)

    assert repo is None


def test_suggest_model_returns_none_when_hf_is_unreachable(mocker):
    mocker.patch.object(cli.hf_explore, "search", side_effect=RuntimeError("network down"))
    repo, reason = cli._suggest_model_for_ram(24.0)
    assert repo is None


def test_suggest_model_skips_non_mlx_repos(mocker):
    mocker.patch.object(cli.hf_explore, "search", return_value=[
        {"id": "org/gguf-repo", "downloads": 9999, "pipeline_tag": "text-generation", "tags": []},
        {"id": "org/mlx-repo", "downloads": 100, "pipeline_tag": "text-generation", "tags": []},
    ])
    mocker.patch.object(cli.hf_explore, "repo_files", side_effect=lambda repo_id: {
        "org/gguf-repo": [{"path": "model.gguf"}],
        "org/mlx-repo": [{"path": "config.json"}, {"path": "model.safetensors", "size": 5 * 1024**3}],
    }[repo_id])

    repo, reason = cli._suggest_model_for_ram(24.0)

    assert repo == "org/mlx-repo"


# --- cmd_init integration: mocked suggestion + install + serve ------------

def test_init_skips_suggestion_flow_when_models_already_installed(models_root, opencode_config, mocker):
    manifest.add_model("org/existing", {"backend": "mlx", "store_path": "store/mlx/existing"})
    mocker.patch.object(cli.specs, "detect", return_value=_fake_specs())
    mocker.patch("shutil.which", return_value="/usr/bin/fake")
    mock_suggest = mocker.patch.object(cli, "_suggest_model_for_ram")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    mock_serve = mocker.patch.object(cli.server, "serve")

    cli.cmd_init(_args())

    mock_suggest.assert_not_called()
    mock_resolve.assert_not_called()
    mock_serve.assert_not_called()


def test_init_installs_and_serves_the_dynamic_suggestion_on_blank_or_y(models_root, opencode_config, monkeypatch, mocker):
    mocker.patch.object(cli.specs, "detect", return_value=_fake_specs())
    mocker.patch("shutil.which", return_value="/usr/bin/fake")
    mocker.patch.object(cli, "_suggest_model_for_ram", return_value=("org/dynamic-pick", "5.0GB, low tier, 100 downloads"))
    monkeypatch.setattr("builtins.input", lambda _: "")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    mock_serve = mocker.patch.object(cli.server, "serve")

    cli.cmd_init(_args())

    mock_resolve.assert_called_once()
    assert mock_resolve.call_args[0][0] == "org/dynamic-pick"
    mock_serve.assert_called_once_with(None, open_opencode=True)


def test_init_installs_a_custom_repo_id_overriding_the_suggestion(models_root, opencode_config, monkeypatch, mocker):
    mocker.patch.object(cli.specs, "detect", return_value=_fake_specs())
    mocker.patch("shutil.which", return_value="/usr/bin/fake")
    mocker.patch.object(cli, "_suggest_model_for_ram", return_value=("org/dynamic-pick", "reason"))
    monkeypatch.setattr("builtins.input", lambda _: "org/my-custom-model")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    mocker.patch.object(cli.server, "serve")

    cli.cmd_init(_args())

    assert mock_resolve.call_args[0][0] == "org/my-custom-model"


def test_init_skips_entirely_on_n(models_root, opencode_config, monkeypatch, mocker):
    mocker.patch.object(cli.specs, "detect", return_value=_fake_specs())
    mocker.patch("shutil.which", return_value="/usr/bin/fake")
    mocker.patch.object(cli, "_suggest_model_for_ram", return_value=("org/dynamic-pick", "reason"))
    monkeypatch.setattr("builtins.input", lambda _: "n")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    mock_serve = mocker.patch.object(cli.server, "serve")

    cli.cmd_init(_args())

    mock_resolve.assert_not_called()
    mock_serve.assert_not_called()


def test_init_with_no_suggestion_found_prompts_for_a_repo_id_and_blank_skips(models_root, opencode_config, monkeypatch, mocker):
    """When nothing could be auto-suggested (offline, nothing fits), blank
    input must mean 'skip', not accidentally install something -- there's no
    suggestion to fall back to."""
    mocker.patch.object(cli.specs, "detect", return_value=_fake_specs())
    mocker.patch("shutil.which", return_value="/usr/bin/fake")
    mocker.patch.object(cli, "_suggest_model_for_ram", return_value=(None, None))
    monkeypatch.setattr("builtins.input", lambda _: "")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    mock_serve = mocker.patch.object(cli.server, "serve")

    cli.cmd_init(_args())

    mock_resolve.assert_not_called()
    mock_serve.assert_not_called()


def test_init_with_no_suggestion_found_still_accepts_a_pasted_repo_id(models_root, opencode_config, monkeypatch, mocker):
    mocker.patch.object(cli.specs, "detect", return_value=_fake_specs())
    mocker.patch("shutil.which", return_value="/usr/bin/fake")
    mocker.patch.object(cli, "_suggest_model_for_ram", return_value=(None, None))
    monkeypatch.setattr("builtins.input", lambda _: "org/my-own-pick")
    mock_resolve = mocker.patch.object(cli, "_resolve_install")
    mocker.patch.object(cli.server, "serve")

    cli.cmd_init(_args())

    assert mock_resolve.call_args[0][0] == "org/my-own-pick"
