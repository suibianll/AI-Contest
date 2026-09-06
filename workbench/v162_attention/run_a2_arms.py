"""A2 four-arm + strong-control evaluator runs (shards 0,2,3,5, attention-only).

Holds the v162-independent GPU lock across all runs so the Linear agent is
never interleaved.  Runs:
  gate     = deployed candidate (_ROT_ARM="gate")
  h        = fixed-H arm (_ROT_ARM="h")
  learned  = ungated learned arm (_ROT_ARM="learned")
  v168     = strong control (historical attention parent)
  v189     = latest full parent, attention side read only
All runs use --baseline-solution = v162 zero point, per the plan contract.
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
V168 = ROOT / "solutions/20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA/solution.py"
V189 = ROOT / "solutions/20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt"
PY = ROOT / ".venv/Scripts/python.exe"

RUNS = [
    ("a2-gate", ROOT / "workbench/v162_attention/candidate/solution.py"),
    ("a2-h", ROOT / "workbench/v162_attention/research/arm_h/solution.py"),
    ("a2-learned", ROOT / "workbench/v162_attention/research/arm_learned/solution.py"),
    ("a2-strong-v168", V168),
    ("a2-strong-v189", V189),
]


def run_eval(name: str, solution: Path) -> int:
    out_dir = ROOT / f"artifacts/proxy_v3/v162-independent/attention/{name}"
    command = [
        str(PY), str(ROOT / "evaluator/eval.py"),
        "--baseline-solution", str(ZERO),
        "--solution", str(solution),
        "--name", name,
        "--attention-only",
        "--shards", "0,2,3,5",
        "--cache", str(CACHE),
        "--algorithm-device", "cuda",
        "--calibration-cache-mode", "auto",
        "--stop-after-nonpositive", "7",
        "--reuse-existing",
        "--output-dir", str(out_dir),
    ]
    print(f"=== {name} ===", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT))
    print(f"=== {name} finished in {time.perf_counter() - started:.0f}s rc={completed.returncode} ===", flush=True)
    return completed.returncode


def main() -> int:
    code = gpu_lock.acquire("A", "a2-four-arm")
    if code != 0:
        return code
    failures = []
    try:
        for name, solution in RUNS:
            rc = run_eval(name, solution)
            if rc != 0:
                failures.append((name, rc))
    finally:
        gpu_lock.release("A", "a2-four-arm")
    if failures:
        print(f"FAILURES: {failures}", flush=True)
        return 1
    print("ALL A2 RUNS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
