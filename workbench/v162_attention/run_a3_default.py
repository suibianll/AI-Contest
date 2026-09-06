"""A3 fresh default run (compatibility backend, 168 Linear + 120 Attention).

Per plan §5: full six-API single file with the standard other side, fresh
default panel for this-side verification and complete API timing.  The API
seconds feed the official time model in the ledger.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import gpu_lock  # noqa: E402

CANDIDATE = ROOT / "workbench/v162_attention/candidate/solution.py"
OUT_JSON = ROOT / "artifacts/proxy_v3/v162-independent/attention/a3-default/default.json"
OUT_MD = ROOT / "artifacts/proxy_v3/v162-independent/attention/a3-default/default.md"
PY = ROOT / ".venv/Scripts/python.exe"


def main() -> int:
    code = gpu_lock.acquire("A", "a3-default-timing")
    if code != 0:
        return code
    try:
        command = [
            str(PY), str(ROOT / "evaluator/official_eval.py"),
            "--solution", str(CANDIDATE),
            "--name", "v162-attention-gate-default",
            "--cache-mode", "read",
            "--nvfp4-cache-mode", "auto",
            "--algorithm-device", "cuda",
            "--output", str(OUT_JSON),
            "--report", str(OUT_MD),
        ]
        print("=== a3 fresh default ===", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=str(ROOT))
        print(f"=== finished in {time.perf_counter() - started:.0f}s rc={completed.returncode} ===", flush=True)
        return completed.returncode
    finally:
        gpu_lock.release("A", "a3-default-timing")


if __name__ == "__main__":
    raise SystemExit(main())
