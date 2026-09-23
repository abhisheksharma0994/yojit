"""One advisory lock over yojit's shared state.

Everything yojit persists is a read-modify-write on a JSON document
(manifest.json, opencode.json), or a decision about one shared resource (the
single server port). Two invocations interleaving inside any of those lose each
other's work: `formal/tla/OpencodeSync.tla` reproduces the lost update on the
config file and `formal/tla/ServeLifecycle.tla` reproduces the serve-side races.

The lock is an exclusively-created file: portable to every platform yojit claims
to support (fcntl is absent on Windows), needs no daemon, and survives a crashed
holder (a stale lock is broken when its recorded PID is gone).
"""
import errno
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE_NAME = ".yojit.lock"
LOCK_PATH_ENV_VAR = "YOJIT_LOCK"
DEFAULT_TIMEOUT = 10.0
DEFAULT_POLL = 0.05
DEFAULT_STALE_AFTER = 900.0

_held = threading.local()


class StateLockTimeout(RuntimeError):
    """Another yojit invocation held the lock past the timeout."""


def state_dir() -> Path:
    """Where the lock (and the server record) live.

    YOJIT_LOCK wins outright -- tests point it at a throwaway path. Otherwise the
    models root, which is already isolated per machine and per test."""
    explicit = os.environ.get(LOCK_PATH_ENV_VAR)
    if explicit:
        return Path(explicit).parent
    try:
        from . import manifest  # deferred: manifest imports this module
        return manifest.models_root()
    except Exception:
        return Path.home() / ".yojit"


def lock_path() -> Path:
    explicit = os.environ.get(LOCK_PATH_ENV_VAR)
    return Path(explicit) if explicit else state_dir() / LOCK_FILE_NAME


def _recorded_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check. Anything we cannot determine counts as alive,
    so an unknown platform never has its lock stolen from under it."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _holder_is_gone(path: Path, stale_after: float) -> bool:
    pid = _recorded_pid(path)
    if pid is not None and not _pid_alive(pid):
        return True
    try:
        return (time.time() - path.stat().st_mtime) > stale_after
    except OSError:
        return False


def _timeout_error(path: Path, timeout: float) -> StateLockTimeout:
    holder = _recorded_pid(path)
    return StateLockTimeout(
        f"another yojit invocation has held {path} for over {timeout:g}s"
        + (f" (PID {holder})" if holder else "")
        + "; retry once it finishes, or delete the lock if it crashed"
    )


def _take_lock_file(path: Path, timeout: float, poll: float, stale_after: float) -> int:
    """Exclusively creates the lock file and returns its open descriptor, or
    raises StateLockTimeout. The caller owns the descriptor."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as exc:
            # O_EXCL reports an existing lock as FileExistsError on CPython, but
            # check errno too: this path is the whole safety property, so it does
            # not get to depend on one platform's exception mapping.
            if not isinstance(exc, FileExistsError) and exc.errno != errno.EEXIST:
                raise
            # Break a lock whose holder is gone. The deadline is checked on THIS
            # path too, not just the "holder is alive" one: a stale lock we cannot
            # delete (read-only directory, or a filesystem that refuses the
            # unlink) would otherwise spin here without ever timing out.
            if _holder_is_gone(path, stale_after):
                try:
                    path.unlink()
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise _timeout_error(path, timeout) from None
            time.sleep(poll)


def _release(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass  # nothing we can do, and nothing that should fail the caller


@contextmanager
def state_lock(timeout: float = DEFAULT_TIMEOUT, poll: float = DEFAULT_POLL,
               stale_after: float = DEFAULT_STALE_AFTER):
    """Serializes one read-modify-write over yojit's shared state.

    Re-entrant within a process, so a caller can hold the lock across a sequence
    of mutation helpers instead of each helper taking it separately.

    Raises StateLockTimeout rather than proceeding unlocked: running the critical
    section anyway would reintroduce exactly the interleavings this prevents.
    """
    depth = getattr(_held, "depth", 0)
    if depth > 0:
        _held.depth = depth + 1
        try:
            yield
        finally:
            _held.depth = depth
        return

    path = lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    fd = _take_lock_file(path, timeout, poll, stale_after)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
    finally:
        os.close(fd)

    _held.depth = 1
    try:
        yield
    finally:
        _held.depth = 0
        _release(path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then os.replace.

    A concurrent reader can never observe a half-written document, and a crash
    mid-write cannot truncate what was already there -- which `write_text` on the
    real file can and does."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
