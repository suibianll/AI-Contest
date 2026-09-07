"""Run registered stages under the shared GPU lock (4B panel, paired vs A22-2)."""
from pathlib import Path
import hashlib
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import gpu_lock

OUT = ROOT / "artifacts/proxy_v3/continuous/attention/anchor23-a1"
PY = ROOT / ".venv/Scripts/python.exe"
CAND = HERE / "solution.py"
PARENT = ROOT / "solutions/continuous_attention_anchor22-a2/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def panel(shards):
    cmd = [str(PY), "evaluator/eval.py", "--solution", str(CAND), "--baseline-solution", str(PARENT),
           "--name", "id", "--attention-only", "--shards", shards, "--cache", str(CACHE),
           "--calibration-cache-mode", "auto", "--algorithm-device", "cuda",
           "--stop-after-nonpositive", "7", "--output-dir", str(OUT)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    stage = sys.argv[1]
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == "4686ad817128d60e1a1648e0e14793692097ee8171fac2bc5202466d5787c0b7"
    assert gpu_lock.acquire("A", "anchor23-a1-" + stage) == 0
    try:
        if stage == "screen":
            panel("0")
        elif stage == "full":
            panel("0,1,2,3,4,5")
        else:
            raise ValueError(stage)
    finally:
        gpu_lock.release("A", "anchor23-a1-" + stage)


if __name__ == "__main__":
    main()
