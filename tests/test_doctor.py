from yojit import doctor, manifest
from yojit.specs import Specs


def _specs(**overrides):
    base = dict(platform="darwin", is_apple_silicon=True, chip="Apple Silicon",
                total_ram_gb=24.0, free_disk_gb=100.0, cpu_cores=8)
    base.update(overrides)
    return Specs(**base)


def _base_mocks(mocker, port_pid=None, mlx_vlm_installed=True, llama_server=None, opencode=None):
    mocker.patch.object(doctor.specs, "detect", return_value=_specs())
    mocker.patch.object(doctor.server, "_port_pid", return_value=port_pid)
    mocker.patch.object(doctor.mlx_env, "is_installed", return_value=mlx_vlm_installed)
    mocker.patch.object(doctor.shutil, "which", side_effect=lambda name: {
        "llama-server": llama_server, "opencode": opencode,
    }.get(name))


def test_run_reports_platform_and_ram(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    findings = doctor.run()
    assert any("Platform: darwin" in f and "RAM: 24" in f for f in findings)


def test_run_warns_when_port_in_use(mocker, models_root, opencode_config):
    _base_mocks(mocker, port_pid=4242)
    findings = doctor.run()
    assert any("WARNING" in f and "4242" in f for f in findings)


def test_run_reports_port_free(mocker, models_root, opencode_config):
    _base_mocks(mocker, port_pid=None)
    findings = doctor.run()
    assert any("OK" in f and "free" in f for f in findings)


def test_run_warns_when_mlx_vlm_missing_on_apple_silicon(mocker, models_root, opencode_config):
    _base_mocks(mocker, mlx_vlm_installed=False)
    findings = doctor.run()
    assert any("mlx-vlm not installed" in f for f in findings)


def test_run_skips_mlx_vlm_check_on_non_apple_silicon(mocker, models_root, opencode_config):
    mocker.patch.object(doctor.specs, "detect", return_value=_specs(is_apple_silicon=False))
    mocker.patch.object(doctor.server, "_port_pid", return_value=None)
    mock_is_installed = mocker.patch.object(doctor.mlx_env, "is_installed", return_value=False)
    mocker.patch.object(doctor.shutil, "which", return_value="/usr/bin/fake")
    doctor.run()
    mock_is_installed.assert_not_called()


def test_run_reports_llama_server_and_opencode_missing(mocker, models_root, opencode_config):
    _base_mocks(mocker, llama_server=None, opencode=None)
    findings = doctor.run()
    assert any("llama-server not installed" in f for f in findings)
    assert any("WARNING: opencode not installed" in f for f in findings)


def test_run_reports_llama_server_and_opencode_present(mocker, models_root, opencode_config):
    _base_mocks(mocker, llama_server="/usr/bin/llama-server", opencode="/usr/bin/opencode")
    findings = doctor.run()
    assert not any("llama-server not installed" in f for f in findings)
    assert not any("opencode not installed" in f for f in findings)


def test_run_warns_when_opencode_json_missing(mocker, models_root, tmp_path, monkeypatch):
    _base_mocks(mocker)
    monkeypatch.setenv("YOJIT_OPENCODE_CONFIG", str(tmp_path / "does-not-exist.json"))
    findings = doctor.run()
    assert any("opencode.json not found" in f for f in findings)


def test_run_reports_valid_opencode_json(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    findings = doctor.run()
    assert any("opencode.json is valid JSON" in f for f in findings)


def test_run_reports_invalid_opencode_json(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    opencode_config.write_text("{not valid json")
    findings = doctor.run()
    assert any("ERROR" in f and "invalid JSON" in f for f in findings)


def test_run_reports_no_default_model_set(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    findings = doctor.run()
    assert any("no default model set yet" in f for f in findings)


def test_run_warns_when_default_model_is_risky(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    manifest.add_model("org/big", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/big",
                                    "tier": "high", "size_gb": 20.0})
    findings = doctor.run()
    assert any("WARNING" in f and "org/big" in f and "OOM crash risk" in f for f in findings)


def test_run_warns_when_default_model_has_no_manifest_entry(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    manifest.add_model("org/ghost", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/ghost"})
    manifest.remove_model("org/ghost")
    data = manifest.load()
    data["default_model"] = "org/ghost"  # simulate a manifest left pointing at a removed model
    manifest.save(data)
    findings = doctor.run()
    assert any("WARNING" in f and "org/ghost" in f and "no manifest entry" in f for f in findings)


def test_run_reports_default_model_fits_comfortably(mocker, models_root, opencode_config):
    _base_mocks(mocker)
    manifest.add_model("org/small", {"backend": "mlx_vlm", "store_path": "store/mlx_vlm/small",
                                      "tier": "low", "size_gb": 5.0})
    findings = doctor.run()
    assert any("OK" in f and "org/small" in f and "fits comfortably" in f for f in findings)
