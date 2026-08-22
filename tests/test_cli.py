import pytest

from yojit import cli

ALL_SUBCOMMANDS = [
    "init", "explore", "install", "list", "use", "serve",
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
