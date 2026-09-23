"""Tests for the advisory state lock.

The lock is what makes the read-modify-writes on manifest.json / opencode.json
safe, so the interesting cases are the failure ones: a second holder, a crashed
holder, and a timeout that must raise rather than proceed unlocked.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yojit import locking  # noqa: E402


@pytest.fixture
def lock_file(tmp_path, monkeypatch):
    """Points the lock at a throwaway path so no test can touch a real repo or
    ~/.yojit lock -- and so a hung lock can never block the developer's own
    `yojit serve` while the suite runs."""
    path = tmp_path / "yojit" / ".yojit.lock"
    monkeypatch.setenv(locking.LOCK_PATH_ENV_VAR, str(path))
    return path


# --- where the lock lives ---------------------------------------------------

def test_lock_path_honours_the_env_override(lock_file):
    assert locking.lock_path() == lock_file
    assert locking.state_dir() == lock_file.parent


def test_state_dir_falls_back_to_the_models_root(monkeypatch):
    monkeypatch.delenv(locking.LOCK_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv("YOJIT_HOME", "/tmp/yojit-lock-state-dir-test")
    from yojit import manifest

    assert locking.state_dir() == manifest.models_root()


def test_state_dir_falls_back_to_the_home_directory_when_the_manifest_cannot_resolve(
    monkeypatch, mocker
):
    monkeypatch.delenv(locking.LOCK_PATH_ENV_VAR, raising=False)
    from yojit import manifest

    mocker.patch.object(manifest, "models_root", side_effect=RuntimeError("not a dev checkout"))
    assert locking.state_dir() == Path.home() / ".yojit"


def test_state_dir_never_creates_anything(lock_file, tmp_path):
    locking.state_dir()
    assert not lock_file.parent.exists()


# --- mutual exclusion -------------------------------------------------------

def test_lock_is_taken_and_released_around_the_block(lock_file):
    assert not lock_file.exists()
    with locking.state_lock():
        assert lock_file.exists()
        assert lock_file.read_text().strip() == str(os.getpid())
    assert not lock_file.exists(), "the lock must be released, not left behind"


def test_lock_is_reentrant_within_a_process(lock_file):
    """Manifest/opencode helpers each take the lock; a caller holding it across
    several of them must not deadlock against itself."""
    with locking.state_lock():
        with locking.state_lock():
            with locking.state_lock():
                assert lock_file.read_text().strip() == str(os.getpid())
    assert not lock_file.exists()


def test_a_held_lock_times_out_instead_of_proceeding_unlocked(lock_file):
    """Proceeding unlocked is the one thing this must never do -- it would
    reintroduce the interleaving the lock exists to prevent."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(f"{os.getpid()}\n")  # a holder that is alive

    with pytest.raises(locking.StateLockTimeout) as exc:
        with locking.state_lock(timeout=0.05, poll=0.01):
            pytest.fail("the critical section must not run while another holder has the lock")

    assert str(os.getpid()) in str(exc.value), "the timeout should name the holder"


def test_timeout_message_omits_the_pid_when_the_lock_is_unreadable(lock_file):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("not-a-pid\n")

    with pytest.raises(locking.StateLockTimeout) as exc:
        with locking.state_lock(timeout=0.05, poll=0.01):
            pytest.fail("unreadable holder is still a holder")

    assert "PID" not in str(exc.value)


def test_a_lock_left_by_a_dead_process_is_broken(lock_file):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("999999999\n")  # no such process

    with locking.state_lock(timeout=1.0, poll=0.01):
        assert lock_file.read_text().strip() == str(os.getpid())


def test_a_lock_older_than_stale_after_is_broken_even_if_the_pid_is_alive(lock_file):
    """Covers a PID that was recycled onto an unrelated process and a machine
    resuming from sleep; the mtime is the backstop."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(f"{os.getpid()}\n")
    old = time.time() - 60
    os.utime(lock_file, (old, old))

    with locking.state_lock(timeout=1.0, poll=0.01, stale_after=1.0):
        assert lock_file.read_text().strip() == str(os.getpid())


def test_pid_liveness_never_reports_a_bad_pid_as_alive():
    assert locking._pid_alive(0) is False
    assert locking._pid_alive(-1) is False
    assert locking._pid_alive(os.getpid()) is True


def test_recorded_pid_is_none_when_the_lock_is_unreadable(lock_file):
    assert locking._recorded_pid(lock_file) is None
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("")
    assert locking._recorded_pid(lock_file) is None


def test_holder_is_gone_is_false_when_the_file_disappeared(lock_file):
    assert locking._holder_is_gone(lock_file, stale_after=0.0) is False


def test_pid_liveness_treats_a_permission_error_as_alive(lock_file, mocker):
    """We cannot signal it, so we must not assume it is gone and steal its lock."""
    mocker.patch.object(locking.os, "kill", side_effect=PermissionError)
    assert locking._pid_alive(1) is True


def test_pid_liveness_treats_an_unknown_oserror_as_alive(lock_file, mocker):
    mocker.patch.object(locking.os, "kill", side_effect=OSError("unsupported platform"))
    assert locking._pid_alive(1) is True


def test_a_stale_lock_that_cannot_be_removed_still_times_out(lock_file, mocker):
    """If the unlink fails we must not fall through and take a lock we do not
    own -- the old holder may still be inside its critical section."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("999999999\n")
    mocker.patch.object(Path, "unlink", side_effect=OSError("read-only"))

    with pytest.raises(locking.StateLockTimeout):
        with locking.state_lock(timeout=0.05, poll=0.01):
            pytest.fail("an unremovable stale lock must not be taken")


def test_an_unremovable_directory_does_not_stop_the_lock_being_taken(lock_file, mocker):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    mocker.patch.object(Path, "mkdir", side_effect=OSError("read-only parent"))
    with locking.state_lock():
        assert lock_file.exists()


def test_a_release_that_cannot_unlink_does_not_raise(lock_file, mocker):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    mocker.patch.object(Path, "unlink", side_effect=OSError("read-only"))
    with locking.state_lock():
        pass  # must not propagate out of the finally block


def test_the_lock_serializes_a_real_second_process(lock_file):
    """The property that actually matters: two processes, not two threads. The
    child must not be able to enter while the parent holds the lock."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    child_source = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(Path(__file__).parent.parent / 'src')!r})\n"
        "from yojit import locking\n"
        "try:\n"
        "    with locking.state_lock(timeout=0.2, poll=0.01):\n"
        "        print('ENTERED')\n"
        "except locking.StateLockTimeout:\n"
        "    print('BLOCKED')\n"
    )

    with locking.state_lock():
        out = subprocess.run([sys.executable, "-c", child_source],
                             capture_output=True, text=True, timeout=60).stdout

    assert "BLOCKED" in out, f"the child entered a held lock: {out!r}"

    out_after = subprocess.run([sys.executable, "-c", child_source],
                               capture_output=True, text=True, timeout=60).stdout
    assert "ENTERED" in out_after, f"the child could not take a released lock: {out_after!r}"


# --- atomic writes ----------------------------------------------------------

def test_atomic_write_text_replaces_the_contents(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old")
    locking.atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_atomic_write_text_creates_missing_parents(tmp_path):
    target = tmp_path / "nested" / "deeper" / "state.json"
    locking.atomic_write_text(target, "{}")
    assert target.read_text() == "{}"


def test_atomic_write_text_leaves_no_temp_file_behind(tmp_path):
    target = tmp_path / "state.json"
    locking.atomic_write_text(target, "x")
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_atomic_write_text_does_not_truncate_on_a_failed_write(tmp_path, mocker):
    """The reason for writing through a temp file: a failure mid-write must
    leave the previous document intact rather than a half-written one."""
    target = tmp_path / "state.json"
    target.write_text("complete")
    mocker.patch.object(Path, "write_text", side_effect=OSError("disk full"))

    with pytest.raises(OSError):
        locking.atomic_write_text(target, "partial")

    assert target.read_text() == "complete"
