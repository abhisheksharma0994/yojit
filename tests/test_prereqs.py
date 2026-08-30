"""ensure_opencode_installed(): closes a real gap where the README promised
"missing prerequisites are installed on demand" but opencode's absence was
only ever detected and printed, never actually installed."""
from yojit import prereqs


def test_returns_true_immediately_when_already_installed(mocker):
    mocker.patch.object(prereqs.shutil, "which",
                         side_effect=lambda name: "/usr/local/bin/opencode" if name == "opencode" else None)
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    assert prereqs.ensure_opencode_installed() is True
    mock_run.assert_not_called()  # no install attempted -- already present


def test_installs_via_homebrew_on_macos(mocker):
    """which() is queried by exact name -- opencode is missing, brew is present,
    and the post-install re-check for opencode succeeds."""
    calls = {"opencode": iter([None, "/opt/homebrew/bin/opencode"]), "brew": iter(["/opt/homebrew/bin/brew"])}
    mocker.patch.object(prereqs.shutil, "which", side_effect=lambda name: next(calls[name]))
    mocker.patch.object(prereqs.platform, "system", return_value="Darwin")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    result = prereqs.ensure_opencode_installed()

    assert result is True
    args, kwargs = mock_run.call_args
    assert args[0] == ["brew", "install", "anomalyco/tap/opencode"]
    assert kwargs["check"] is False  # a failed brew install must not raise -- caller checks the return value


def test_returns_false_with_manual_instructions_when_not_macos(mocker):
    mocker.patch.object(prereqs.shutil, "which", return_value=None)
    mocker.patch.object(prereqs.platform, "system", return_value="Linux")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    result = prereqs.ensure_opencode_installed()

    assert result is False
    mock_run.assert_not_called()  # never guesses a package manager on other platforms


def test_returns_false_when_macos_but_brew_is_missing(mocker):
    mocker.patch.object(prereqs.shutil, "which", return_value=None)  # opencode missing, brew missing
    mocker.patch.object(prereqs.platform, "system", return_value="Darwin")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    result = prereqs.ensure_opencode_installed()

    assert result is False
    mock_run.assert_not_called()


def test_returns_false_when_brew_install_did_not_actually_result_in_opencode_on_path(mocker):
    calls = {"opencode": iter([None, None]), "brew": iter(["/opt/homebrew/bin/brew"])}
    mocker.patch.object(prereqs.shutil, "which", side_effect=lambda name: next(calls[name]))
    mocker.patch.object(prereqs.platform, "system", return_value="Darwin")
    mocker.patch.object(prereqs.subprocess, "run")

    assert prereqs.ensure_opencode_installed() is False
