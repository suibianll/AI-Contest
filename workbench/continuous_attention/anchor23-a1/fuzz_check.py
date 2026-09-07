"""Run the official-contract fuzz surface against the A23 candidate."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import fuzz_official_contract as fuzz

failures = fuzz.run_candidate("a23", HERE / "solution.py")
if failures:
    print(f"--- a23: {len(failures)} failure(s)")
    for item in failures:
        print("   ", item)
    raise SystemExit(1)
print("--- a23: CLEAN")
