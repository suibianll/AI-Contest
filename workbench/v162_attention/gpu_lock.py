"""GPU lock helper for the v162-independent workbench (plan §2 protocol).

Acquire with an atomic exclusive file create; release only the lock we own.
Usage:
    python gpu_lock.py acquire <side> <run_id>   # exits 0 when the lock is held
    python gpu_lock.py release <side> <run_id>   # exits 0 when our lock was freed
The lock file lives at artifacts/proxy_v3/v162-independent/gpu.lock and is an
ignored runtime artifact: never commit it, never delete a foreign lock.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "artifacts/proxy_v3/v162-independent/gpu.lock"
WAIT_SECONDS = 4 * 3600
POLL_SECONDS = 30


def _payload(side: str, run_id: str) -> str:
    return (
        f"side={side}\nrun_id={run_id}\npid={os.getpid()}\n"
        f"host={socket.gethostname()}\nstarted={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )


def acquire(side: str, run_id: str) -> int:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + WAIT_SECONDS
    while True:
        try:
            handle = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                print(f"gpu.lock still held after {WAIT_SECONDS}s; giving up", flush=True)
                return 2
            holder = LOCK_PATH.read_text(encoding="utf-8", errors="replace").strip().replace("\n", "; ")
            print(f"gpu.lock busy ({holder}); retrying in {POLL_SECONDS}s", flush=True)
            time.sleep(POLL_SECONDS)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_payload(side, run_id))
        print(f"gpu.lock acquired for {side}/{run_id}", flush=True)
        return 0


def release(side: str, run_id: str) -> int:
    if not LOCK_PATH.exists():
        print("gpu.lock absent; nothing to release", flush=True)
        return 0
    content = LOCK_PATH.read_text(encoding="utf-8", errors="replace")
    if f"side={side}" in content and f"run_id={run_id}" in content:
        LOCK_PATH.unlink()
        print(f"gpu.lock released for {side}/{run_id}", flush=True)
        return 0
    print("gpu.lock belongs to another run; refusing to delete", flush=True)
    return 3


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in {"acquire", "release"}:
        print(__doc__)
        return 1
    action, side, run_id = sys.argv[1], sys.argv[2], sys.argv[3]
    return acquire(side, run_id) if action == "acquire" else release(side, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
