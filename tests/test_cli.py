import pytest

from yojit import cli, manifest

ALL_SUBCOMMANDS = [
    "init", "explore", "install", "list", "use", "config", "serve",
    "stop", "status", "remove", "sync", "doctor", "upgrade",
]


def test_every_documented_subcommand_is_registered():
    parser = cli.build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction)
    )
    assert set(subparsers_action.choices.keys()) == set(ALL_SUBCOMMANDS)


@pytest.mark.parametrize("subcommand,extra_args,func_name", [
    ("init", [], "cmd_init"),
    ("explore", [], "cmd_explore"),
    ("install", ["org/repo"], "cmd_install"),
    ("list", [], "cmd_list"),
    ("use", ["org/repo"], "cmd_use"),
    ("config", ["org/repo"], "cmd_config"),
    ("serve", [], "cmd_serve"),
    ("stop", [], "cmd_stop"),
    ("status", [], "cmd_status"),
    ("remove", ["org/repo"], "cmd_remove"),
    ("sync", [], "cmd_sync"),
    ("doctor", [], "cmd_doctor"),
    ("upgrade", [], "cmd_upgrade"),
])
def test_subcommand_dispatches_to_the_correct_function(subcommand, extra_args, func_name):
    parser = cli.build_parser()
    args = parser.parse_args([subcommand] + extra_args)
    assert args.func is getattr(cli, func_name)


def test_install_requires_a_repo_argument():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["install"])  # missing required positional


def test_serve_model_argument_is_optional():
    parser = cli.build_parser()
    args = parser.parse_args(["serve"])
    assert args.model is None
    args = parser.parse_args(["serve", "org/repo"])
    assert args.model == "org/repo"


def test_serve_no_open_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["serve"])
    assert args.no_open is False
    args = parser.parse_args(["serve", "--no-open"])
    assert args.no_open is True


def test_no_subcommand_exits_with_error():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_dispatches_with_explicit_argv(monkeypatch):
    """main() must accept an argv list (not just read sys.argv) so it's
    testable and embeddable."""
    called = {}
    monkeypatch.setattr(cli, "cmd_list", lambda args: called.setdefault("ran", True))
    cli.main(["list"])
    assert called.get("ran") is True


def test_config_parses_all_the_lm_studio_style_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "config", "org/repo",
        "--seed", "42",
        "--kv-cache-quant", "8",
        "--kv-group-size", "64",
        "--quantized-kv-start", "5000",
        "--max-concurrent-predictions", "2",
        "--context", "16384",
    ])
    assert args.seed == 42
    assert args.kv_cache_quant == "8"
    assert args.kv_group_size == 64
    assert args.quantized_kv_start == 5000
    assert args.max_concurrent_predictions == 2
    assert args.context == 16384


def test_config_defaults_are_all_none(models_root):
    parser = cli.build_parser()
    args = parser.parse_args(["config", "org/repo"])
    assert all(v is None for v in (args.seed, args.kv_cache_quant, args.kv_group_size,
                                    args.quantized_kv_start, args.max_concurrent_predictions, args.context))


def test_cmd_config_writes_overrides_and_context(models_root):
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "context": 4096})
    parser = cli.build_parser()
    args = parser.parse_args(["config", "org/model-a", "--seed", "42", "--context", "16384"])
    cli.cmd_config(args)

    entry = manifest.get_model("org/model-a")
    assert entry["overrides"] == {"seed": 42}
    assert entry["context"] == 16384


def test_cmd_config_with_no_flags_reports_current_state_without_erroring(models_root, capsys):
    manifest.add_model("org/model-a", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/a", "context": 4096})
    parser = cli.build_parser()
    args = parser.parse_args(["config", "org/model-a"])
    cli.cmd_config(args)
    out = capsys.readouterr().out
    assert "org/model-a" in out


def test_cmd_config_rejects_uninstalled_model(models_root):
    parser = cli.build_parser()
    args = parser.parse_args(["config", "org/not-installed", "--seed", "1"])
    with pytest.raises(SystemExit):
        cli.cmd_config(args)
