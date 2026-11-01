"""Cross-platform workflow lock and sleep prevention. SPDX-License-Identifier: CC0-1.0."""
from contextlib import contextmanager
import os


@contextmanager
def workflow_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def keep_awake():
    """Prevent Windows sleep while running; Linux users may use systemd-inhibit."""
    if os.name == "nt":
        import ctypes

        awake = ctypes.windll.kernel32.SetThreadExecutionState
        awake.argtypes, awake.restype = [ctypes.c_uint32], ctypes.c_uint32
        awake(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            awake(0x80000000)
