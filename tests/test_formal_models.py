"""Runs the formal models as part of the test suite.

`formal/lean` proves the numeric core of `classify.py` for all inputs;
`formal/tla` model-checks the concurrency in the state writes and the server
lifecycle. Both exist to stop the code and its model drifting apart, and a
model nobody runs rots, so they are wired in here.

Two properties matter more than the assertions themselves:

  * The toolchains are OPTIONAL. A contributor without elan or Java gets skips,
    not failures -- that is the point of `_require`.
  * The constants are CHECKED AGAINST THE PYTHON SOURCE, not against a copy of
    themselves. `test_lean_constants_match_the_python_source` is the one test
    that makes every Lean theorem mean something about the shipped code.

Set YOJIT_REQUIRE_FORMAL=1 to turn skips into failures (CI does this, so the
guard cannot go quiet).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yojit import classify  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
LEAN_DIR = REPO_ROOT / "formal" / "lean"
TLA_DIR = REPO_ROOT / "formal" / "tla"
TLA_JAR = REPO_ROOT / "formal" / "tools" / "tla2tools.jar"

REQUIRE_FORMAL = os.environ.get("YOJIT_REQUIRE_FORMAL") == "1"


def _skip_or_fail(reason: str) -> None:
    if REQUIRE_FORMAL:
        pytest.fail(f"{reason} (YOJIT_REQUIRE_FORMAL=1, so this is not skippable)")
    pytest.skip(reason)


def _require_model_files() -> None:
    """The models have to be where this module thinks they are.

    Without this, a run from a directory that lacks `formal/` -- an installed
    sdist, a copied tree, a sandbox that only carries `src/` and `tests/` -- would
    assert on missing files rather than on the thing it claims to check. Skipping
    instead (and failing under YOJIT_REQUIRE_FORMAL) keeps the signal honest in
    both directions: CI still cannot pass without the models, and a checkout
    without them reports "skipped" rather than a misleading failure.
    """
    for path, what in ((LEAN_DIR / "Yojit", "the Lean model"), (TLA_DIR, "the TLA+ specs")):
        if not path.is_dir():
            _skip_or_fail(f"{what} is not present at {path} -- cannot run the models from here")


def _lake() -> str | None:
    """`lake` from PATH, or from the default elan install location."""
    found = shutil.which("lake")
    if found:
        return found
    elan_lake = Path.home() / ".elan" / "bin" / "lake"
    return str(elan_lake) if elan_lake.exists() else None


def _lean_env(lake: str) -> dict:
    env = dict(os.environ)
    env["PATH"] = str(Path(lake).parent) + os.pathsep + env.get("PATH", "")
    return env


# `lake build` compiles the whole development; a cold build is tens of seconds,
# a warm one under a second. Cached per session because every Lean assertion
# below reads the same output.
_lake_build_cache: dict = {}


def _lake_build() -> subprocess.CompletedProcess:
    _require_model_files()
    if "result" not in _lake_build_cache:
        lake = _lake()
        if not lake:
            _skip_or_fail(
                "no `lake` on PATH and no ~/.elan/bin/lake -- install elan to run the Lean model"
            )
        _lake_build_cache["result"] = subprocess.run(
            [lake, "build"], cwd=LEAN_DIR, env=_lean_env(lake),
            capture_output=True, text=True, timeout=900,
        )
    return _lake_build_cache["result"]


# A .cfg names a SPECIFICATION, not necessarily a file of the same name:
# ServeLifecycleConsistency.cfg checks ServeLifecycle.tla under a second invariant.
_CFG_MODULE = {"ServeLifecycleConsistency.cfg": "ServeLifecycle"}


def _tlc(cfg_filename: str, tmp_path: Path) -> tuple[int, str]:
    """Runs TLC on one .cfg, with `-metadir` under `tmp_path`.

    TLC derives its scratch directory name from the wall clock and refuses to
    start if it already exists, which makes two runs in the same second fail
    spuriously -- and without `-metadir` it would write into the repo's
    formal/tla/states/ rather than the temp dir."""
    _require_model_files()
    if not TLA_JAR.exists():
        _skip_or_fail(
            f"{TLA_JAR.relative_to(REPO_ROOT)} is missing -- run "
            "formal/tools/fetch-tla2tools.sh to download it"
        )
    if not shutil.which("java"):
        _skip_or_fail("no `java` on PATH -- TLC needs a JRE")
    cfg = TLA_DIR / cfg_filename
    module = TLA_DIR / f"{_CFG_MODULE.get(cfg_filename, cfg.stem)}.tla"
    assert module.exists(), f"{cfg_filename} names a module that does not exist: {module.name}"
    metadir = tmp_path / f"states-{cfg.stem}"
    proc = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-cp", str(TLA_JAR), "tlc2.TLC",
         "-config", str(cfg), "-metadir", str(metadir), "-deadlock", "-cleanup", str(module)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _violations(output: str) -> list[str]:
    """Every invariant/property TLC reported as violated."""
    return re.findall(r"Invariant (\S+) is violated|Property (\S+) is violated", output)


# --------------------------------------------------------------------------
# Lean
# --------------------------------------------------------------------------


def test_lean_model_builds_clean():
    result = _lake_build()
    assert result.returncode == 0, f"lake build failed:\n{result.stdout}\n{result.stderr}"


def test_lean_proofs_contain_no_sorry():
    """`sorry` compiles. A single one would turn every theorem below it into an
    unchecked claim, which is the failure mode this whole directory exists to
    avoid."""
    result = _lake_build()
    assert "sorry" not in result.stdout.lower(), (
        "the Lean sources mention `sorry` -- either a placeholder proof or a\n"
        "declaration that failed to build:\n" + result.stdout
    )


def test_lean_axiom_audit_is_complete_and_clean():
    """Every audited theorem must rest on at most propext/Classical.choice/
    Quot.sound. `sorryAx` would mean a placeholder proof reached the audit."""
    output = _lake_build().stdout
    audited = re.findall(r"'(\S+)' (?:does not depend on any axioms|depends on axioms: \[([^\]]*)\])", output)
    assert len(audited) >= 20, f"expected the axiom audit to cover the development, saw {len(audited)} lines"

    allowed = {"propext", "Classical.choice", "Quot.sound"}
    for name, axioms in audited:
        used = {a.strip() for a in axioms.split(",") if a.strip()}
        assert used <= allowed, f"{name} depends on non-standard axioms: {sorted(used - allowed)}"


# (Lean constant, classify.py attribute or None, how to get from one to the other).
# The transforms are the load-bearing part: Tier.lean and Limits.lean work in
# hundredths of a GiB, so `RESERVED_OS_GB = 8.0` must appear as 800, and a
# *fraction* like 0.25 is mirrored as the denominator 4 it is used as.
_LEAN_CONSTANT_MIRROR = [
    ("SCALE", None, 100),
    ("LOW_TIER_MAX_FRACTION_NUM", "LOW_TIER_MAX_FRACTION", lambda v: round(v * 100)),
    ("MEDIUM_TIER_MAX_FRACTION_NUM", "MEDIUM_TIER_MAX_FRACTION", lambda v: round(v * 100)),
    ("RESERVED_OS_UNITS", "RESERVED_OS_GB", lambda v: round(v * 100)),
    ("SAFETY_FACTOR_DENOM", "SAFETY_FACTOR", lambda v: round(1 / v)),
    ("HEADROOM_FLOOR_UNITS", "MIN_HEADROOM_GB", lambda v: round(v * 100)),
    ("MIN_CONTEXT", "MIN_CONTEXT", lambda v: v),
    ("MAX_CONTEXT_HARD_CAP", "MAX_CONTEXT_HARD_CAP", lambda v: v),
    ("MAX_OUTPUT_HARD_CAP", "MAX_OUTPUT_HARD_CAP", lambda v: v),
    ("MIN_OUTPUT", "MIN_OUTPUT", lambda v: v),
    ("CONTEXT_ROUND_TO", "CONTEXT_ROUND_TO", lambda v: v),
]


def _lean_abbrevs() -> dict:
    found = {}
    for lean_file in LEAN_DIR.rglob("Yojit/*.lean"):
        for name, value in re.findall(r"^abbrev (\w+) : Nat := (\d+)$", lean_file.read_text(), re.M):
            found[name] = int(value)
    return found


def test_lean_constants_match_the_python_source():
    """The Lean theorems are about the numbers in classify.py, but they are
    stated with Lean's own copies of them. This is the test that keeps those
    copies honest: edit a threshold in classify.py and this fails until the
    model is updated, which is the only reason the proofs still mean anything."""
    _require_model_files()
    lean = _lean_abbrevs()
    assert lean, "no `abbrev ... : Nat :=` constants found -- did the Lean sources move?"

    for name, py_attr, transform in _LEAN_CONSTANT_MIRROR:
        assert name in lean, f"Lean constant {name} disappeared from formal/lean"
        if py_attr is None:
            assert lean[name] == transform, f"{name} = {lean[name]}, expected {transform}"
            continue
        py_value = getattr(classify, py_attr)
        expected = transform(py_value)
        assert lean[name] == expected, (
            f"{name} = {lean[name]} in Lean but classify.{py_attr} = {py_value} "
            f"(expected {expected}; update the Lean model to match the code)"
        )


def test_lean_model_states_the_rules_the_python_actually_implements():
    """Named-theorem presence check. The Lean model is the spec of record for
    these rules; if a rule silently loses its theorem, the code can drift with
    the suite still green."""
    _require_model_files()
    sources = "\n".join(p.read_text() for p in (LEAN_DIR / "Yojit").glob("*.lean"))
    for theorem in (
        "outputOf_le_quarter",                    # the strict quarter rule
        "outputOf_le_self",                       # output never exceeds the window
        "output_ratio_holds_below_min_context",   # ...including below MIN_CONTEXT
        "legacy_output_exceeds_quarter_of_lt",    # and the old formula is kept as the counterexample
        "context_bounds",
        "kvPlanContext_fits_headroom",            # the KV plan now reports a context that fits
        "kvStartClamped_le_context",
        "tier_trichotomy",
    ):
        assert f"theorem {theorem}" in sources, f"theorem {theorem} is gone from the Lean model"


# --------------------------------------------------------------------------
# TLA+
# --------------------------------------------------------------------------


def test_tlc_finds_the_opencode_json_lost_update(tmp_path):
    """The unfixed read-modify-write must still be refuted, or this test is
    asserting that a bug is gone without noticing it was never modelled."""
    code, output = _tlc("OpencodeSync.cfg", tmp_path)
    assert code != 0, "TLC unexpectedly found no counterexample in OpencodeSync"
    assert _violations(output), f"expected a violated invariant, got:\n{output[-2000:]}"
    assert "SyncComplete" in output


def test_tlc_verifies_the_locked_opencode_sync(tmp_path):
    code, output = _tlc("OpencodeSyncLocked.cfg", tmp_path)
    assert code == 0, f"TLC reported a violation in the fixed spec:\n{output[-2000:]}"


@pytest.mark.parametrize("cfg,invariant", [
    ("ServeLifecycle.cfg", "ReportedServerIsAlive"),
    ("ServeLifecycleConsistency.cfg", "DefaultMatchesAdvertised"),
])
def test_tlc_refutes_the_pre_fix_serve_lifecycle(cfg, invariant, tmp_path):
    """Both pre-fix lifecycle claims are counterexamples -- the same invariant
    names that the fixed spec below must satisfy."""
    code, output = _tlc(cfg, tmp_path)
    assert code != 0, f"TLC unexpectedly found no counterexample for {invariant}"
    assert invariant in output, f"expected {invariant} to be violated:\n{output[-2000:]}"


def test_tlc_verifies_the_record_based_lifecycle(tmp_path):
    """The fixed lifecycle, with all four invariants checked: a foreign listener
    is never killed, the default and opencode.json agree, and status never
    reports a listener yojit did not start."""
    code, output = _tlc("ServeLifecycleRecord.cfg", tmp_path)
    assert code == 0, f"TLC found a counterexample in the fixed lifecycle:\n{output[-3000:]}"
    assert "distinct states found" in output, f"no state space was explored:\n{output[-2000:]}"


def test_tlc_configs_check_the_invariants_they_claim_to():
    """A cfg that silently drops its INVARIANT line would make the runs above
    pass while checking nothing."""
    _require_model_files()
    expected = {
        "OpencodeSync.cfg": ["SyncComplete"],
        "OpencodeSyncLocked.cfg": ["SyncComplete"],
        "ServeLifecycle.cfg": ["ReportedServerIsAlive"],
        "ServeLifecycleConsistency.cfg": ["DefaultMatchesAdvertised"],
        "ServeLifecycleRecord.cfg": [
            "ForeignListenerNeverKilled",
            "DefaultMatchesAdvertised",
            "StatusNeverReportsForeign",
            "StatusOnlyReportsLiveServers",
        ],
    }
    for cfg, invariants in expected.items():
        text = (TLA_DIR / cfg).read_text()
        declared = re.findall(r"^INVARIANT (\S+)$", text, re.M)
        assert declared == invariants, f"{cfg} declares {declared}, expected {invariants}"
