from yojit import specs


def test_total_ram_gb_returns_a_positive_number():
    assert specs.total_ram_gb() > 0


def test_total_ram_gb_falls_back_gracefully_when_sysctl_fails(mocker):
    """Must exercise the fallback on any OS: the Darwin path shells out via
    subprocess (mocked here), but Linux reads /proc/meminfo directly and
    never touches subprocess -- mocking only subprocess left this test
    silently no-op'd on Linux CI runners, since real RAM (not the documented
    16.0 fallback) got returned there instead. Mock unconditionally on the
    branch actually taken for platform.system()."""
    mocker.patch("yojit.specs.subprocess.check_output", side_effect=OSError("no sysctl here"))
    mocker.patch("builtins.open", side_effect=OSError("no /proc/meminfo here"))
    assert specs.total_ram_gb() == 16.0  # documented conservative fallback


def test_free_disk_gb_returns_a_non_negative_number():
    assert specs.free_disk_gb("/") >= 0


def test_free_disk_gb_falls_back_to_zero_on_error(mocker):
    mocker.patch("yojit.specs.shutil.disk_usage", side_effect=OSError("no such path"))
    assert specs.free_disk_gb("/nonexistent") == 0.0


def test_detect_returns_populated_specs():
    s = specs.detect()
    assert s.total_ram_gb > 0
    assert s.platform in ("darwin", "linux", "windows")
    assert isinstance(s.is_apple_silicon, bool)
