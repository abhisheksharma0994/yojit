import io

from yojit import specs


def test_cpu_cores_returns_os_cpu_count(mocker):
    mocker.patch("yojit.specs.os.cpu_count", return_value=12)
    assert specs.cpu_cores() == 12


def test_cpu_cores_falls_back_to_four_when_undetectable(mocker):
    mocker.patch("yojit.specs.os.cpu_count", return_value=None)
    assert specs.cpu_cores() == 4


def test_mac_chip_name_returns_real_value_on_success(mocker):
    mocker.patch("yojit.specs.subprocess.check_output", return_value=b"Apple M1\n")
    assert specs._mac_chip_name() == "Apple M1"


def test_mac_chip_name_falls_back_to_unknown_on_error(mocker):
    mocker.patch("yojit.specs.subprocess.check_output", side_effect=OSError("no sysctl"))
    assert specs._mac_chip_name() == "unknown"


def test_total_ram_gb_uses_sysctl_on_darwin(mocker):
    mocker.patch("yojit.specs.platform.system", return_value="Darwin")
    mocker.patch("yojit.specs.subprocess.check_output", return_value=b"17179869184\n")  # 16 GiB
    assert specs.total_ram_gb() == 16.0


def test_total_ram_gb_reads_proc_meminfo_on_linux(mocker):
    mocker.patch("yojit.specs.platform.system", return_value="Linux")
    fake_meminfo = io.StringIO("MemTotal:       16777216 kB\nMemFree:        1000 kB\n")
    mocker.patch("yojit.specs.builtins.open", return_value=fake_meminfo)
    assert specs.total_ram_gb() == 16.0


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


def test_detect_uses_platform_processor_as_chip_on_non_darwin(mocker):
    mocker.patch("yojit.specs.platform.system", return_value="Linux")
    mocker.patch("yojit.specs.platform.machine", return_value="x86_64")
    mocker.patch("yojit.specs.platform.processor", return_value="x86_64 Family")
    s = specs.detect()
    assert s.is_apple_silicon is False
    assert s.chip == "x86_64 Family"


def test_detect_falls_back_to_unknown_chip_when_processor_is_blank(mocker):
    mocker.patch("yojit.specs.platform.system", return_value="Linux")
    mocker.patch("yojit.specs.platform.machine", return_value="x86_64")
    mocker.patch("yojit.specs.platform.processor", return_value="")
    s = specs.detect()
    assert s.chip == "unknown"
