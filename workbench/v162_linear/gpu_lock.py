"""GPU serialization lock for the v162-independent branch (L side).

Historical contract from docs/superpowers/archive/plans/2026-09-06-v162-independent-linear-attention-plan.md §2:
atomic exclusive creation via open(path,'x'); the lock records side, PID,
run_id and start time; released in finally.  If the lock exists, wait; never
delete the other agent's lock.  The lock is an ignored run artifact and is
never committed to Git.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts" / "proxy_v3" / "v162-independent" / "gpu.lock"
)


class GPULock:
    def __init__(self, side: str, run_id: str, poll_seconds: float = 20.0):
        self.side = side
        self.run_id = run_id
        self.poll_seconds = poll_seconds
        self._handle = None

    def acquire(self, max_wait_seconds: float = 6 * 3600.0) -> None:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max_wait_seconds
        while True:
            try:
                self._handle = open(LOCK_PATH, "x")
                break
            except FileExistsError:
                holder = LOCK_PATH.read_text(encoding="utf-8", errors="replace")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"gpu.lock still held after {max_wait_seconds}s; holder: {holder}"
                    )
                print(
                    f"[gpu-lock] waiting: {holder.strip()!r}",
                    flush=True,
                )
                time.sleep(self.poll_seconds)
        payload = (
            f"side={self.side} pid={os.getpid()} run_id={self.run_id} "
            f"start={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        self._handle.write(payload)
        self._handle.flush()
        print(f"[gpu-lock] acquired: {payload.strip()}", flush=True)

    def release(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
                LOCK_PATH.unlink(missing_ok=True)
                print("[gpu-lock] released", flush=True)
            except OSError:
                pass
            self._handle = None

    def __enter__(self) -> "GPULock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
