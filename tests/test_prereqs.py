"""ensure_opencode_installed(): closes a real gap where the README promised
"missing prerequisites are installed on demand" but opencode's absence was
only ever detected and printed, never actually installed."""
from yojit import prereqs


def test_returns_true_immediately_when_already_installed(mocker):
    mocker.patch.object(prereqs.shutil, "which", return_value="/usr/local/bin/opencode")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    assert prereqs.ensure_opencode_installed() is True
    mock_run.assert_not_called()  # no install attempted -- already present


def test_installs_via_homebrew_on_macos(mocker):
    which_results = iter([None, "/opt/homebrew/bin/brew", "/opt/homebrew/bin/opencode"])
    mocker.patch.object(prereqs.shutil, "which", side_effect=lambda name: next(which_results))
    mocker.patch.object(prereqs.platform, "system", return_value="Darwin")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    result = prereqs.ensure_opencode_installed()

    assert result is True
    args = mock_run.call_args[0][0]
    assert args == ["brew", "install", "anomalyco/tap/opencode"]


def test_returns_false_with_manual_instructions_when_not_macos(mocker):
    mocker.patch.object(prereqs.shutil, "which", return_value=None)
    mocker.patch.object(prereqs.platform, "system", return_value="Linux")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    result = prereqs.ensure_opencode_installed()

    assert result is False
    mock_run.assert_not_called()  # never guesses a package manager on other platforms


def test_returns_false_when_macos_but_brew_is_missing(mocker):
    which_results = iter([None, None])  # opencode missing, then brew missing
    mocker.patch.object(prereqs.shutil, "which", side_effect=lambda name: next(which_results))
    mocker.patch.object(prereqs.platform, "system", return_value="Darwin")
    mock_run = mocker.patch.object(prereqs.subprocess, "run")

    result = prereqs.ensure_opencode_installed()

    assert result is False
    mock_run.assert_not_called()


def test_returns_false_when_brew_install_did_not_actually_result_in_opencode_on_path(mocker):
    which_results = iter([None, "/opt/homebrew/bin/brew", None])  # install "succeeds" but still missing
    mocker.patch.object(prereqs.shutil, "which", side_effect=lambda name: next(which_results))
    mocker.patch.object(prereqs.platform, "system", return_value="Darwin")
    mocker.patch.object(prereqs.subprocess, "run")

    assert prereqs.ensure_opencode_installed() is False
