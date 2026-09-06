"""R1 (v189 attention stack + standard Linear) evaluation chain under one lock.

Steps: shards 0,2,3,5 screen -> six-shard ID -> OOD -> fresh default timing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import gpu_lock  # noqa: E402

ZERO = ROOT / "solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py"
CANDIDATE = ROOT / "workbench/v162_attention/candidate_v2/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt"
PY = ROOT / ".venv/Scripts/python.exe"


def eval_v3(name: str, shards: str, ood: bool) -> int:
    out_dir = ROOT / f"artifacts/proxy_v3/v162-independent/attention/{name}"
    command = [
        str(PY), str(ROOT / "evaluator/eval.py"),
        "--baseline-solution", str(ZERO),
        "--solution", str(CANDIDATE),
        "--name", name,
        "--attention-only",
        "--shards", shards,
        "--cache", str(CACHE),
        "--algorithm-device", "cuda",
        "--calibration-cache-mode", "auto",
        "--stop-after-nonpositive", "7",
        "--reuse-existing",
        "--output-dir", str(out_dir),
    ]
    if ood:
        command.append("--ood")
    print(f"=== {name} ===", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT))
    print(f"=== {name} done {time.perf_counter() - started:.0f}s rc={completed.returncode} ===", flush=True)
    return completed.returncode


def fresh_default() -> int:
    out_json = ROOT / "artifacts/proxy_v3/v162-independent/attention/r1-default/default.json"
    out_md = ROOT / "artifacts/proxy_v3/v162-independent/attention/r1-default/default.md"
    command = [
        str(PY), str(ROOT / "evaluator/official_eval.py"),
        "--solution", str(CANDIDATE),
        "--name", "r1-v189-attn-standard-linear-default",
        "--cache-mode", "read",
        "--nvfp4-cache-mode", "auto",
        "--algorithm-device", "cuda",
        "--output", str(out_json),
        "--report", str(out_md),
    ]
    print("=== r1 fresh default ===", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT))
    print(f"=== done {time.perf_counter() - started:.0f}s rc={completed.returncode} ===", flush=True)
    return completed.returncode


def main() -> int:
    code = gpu_lock.acquire("A", "r1-eval-chain")
    if code != 0:
        return code
    failures = []
    try:
        steps = [
            ("r1-screen", "0,2,3,5", False),
            ("r1-id", "0,1,2,3,4,5", False),
            ("r1-ood", "0,1,2,3,4,5", True),
        ]
        for name, shards, ood in steps:
            if eval_v3(name, shards, ood) != 0:
                failures.append(name)
        if not failures and fresh_default() != 0:
            failures.append("r1-default")
    finally:
        gpu_lock.release("A", "r1-eval-chain")
    if failures:
        print(f"R1 FAILURES: {failures}", flush=True)
        return 1
    print("R1 EVAL CHAIN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
