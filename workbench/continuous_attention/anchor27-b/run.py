"""A27-B run: registered stages under the shared GPU lock (4B panel, paired vs R3 and vs B)."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import gpu_lock

OUT = ROOT / "artifacts/proxy_v3/continuous/attention/anchor27-b"
PY = ROOT / ".venv/Scripts/python.exe"
CAND = HERE / "solution.py"
# Three-way: B = v189 root, P_old = R3, C = A27-B candidate
V189 = ROOT / "solution.py"
R3 = ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def panel(baseline, name, shards):
    directory = OUT / name
    cmd = [str(PY), "evaluator/eval.py", "--solution", str(CAND), "--baseline-solution", str(baseline),
           "--name", name, "--attention-only", "--shards", shards, "--cache", str(CACHE),
           "--calibration-cache-mode", "auto", "--algorithm-device", "cuda",
           "--stop-after-nonpositive", "7", "--output-dir", str(directory)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    stage = sys.argv[1]
    assert gpu_lock.acquire("A", "anchor27-b-" + stage) == 0
    try:
        if stage == "screen":
            panel(R3, "vs_r3", "0")
        elif stage == "full":
            panel(R3, "vs_r3", "0,1,2,3,4,5")
        elif stage == "full_vs_b":
            panel(V189, "vs_b", "0,1,2,3,4,5")
        else:
            raise ValueError(stage)
    finally:
        gpu_lock.release("A", "anchor27-b-" + stage)


if __name__ == "__main__":
    main()
