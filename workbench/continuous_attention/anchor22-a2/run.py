"""Run registered stages under the shared GPU lock, reusing immutable R3 data."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import gpu_lock
import eval_system

OUT = ROOT / "artifacts/proxy_v3/continuous/attention/anchor22-a2"
PY = ROOT / ".venv/Scripts/python.exe"
CAND = HERE / "solution.py"
# Retain the original recorded source path so reuse requires both path and SHA.
PARENT = ROOT / "workbench/v162_attention/candidate_v3d/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt"


def panel(ood, shards):
    name = "ood" if ood else "id"
    old = ROOT / f"artifacts/proxy_v3/v162-independent/attention/r3-{name}/r3-{name}"
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    copies = []
    for shard in range(6):
        suffix = f"{'ood-' if ood else ''}attention-shard{shard}.json"
        source = old / ("candidate-" + suffix)
        target = directory / ("baseline-" + suffix)
        result = eval_system._load_reusable_result(source, PARENT, "attention", shard, ood, CACHE)
        assert result is not None, f"R3 source/protocol/cache mismatch: {source}"
        if not target.exists():
            shutil.copyfile(source, target)
        assert source.read_bytes() == target.read_bytes()
        copies.append({"original": str(source), "copy": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (directory / "baseline-reuse.json").write_text(json.dumps(copies, indent=2) + "\n", encoding="utf-8")
    cmd = [str(PY), "evaluator/eval.py", "--solution", str(CAND), "--baseline-solution", str(PARENT),
           "--name", name, "--attention-only", "--shards", shards, "--cache", str(CACHE),
           "--calibration-cache-mode", "auto", "--algorithm-device", "cuda", "--reuse-existing",
           "--stop-after-nonpositive", "7", "--output-dir", str(OUT)]
    if ood:
        cmd.append("--ood")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    stage = sys.argv[1]
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == "a5c679d7a2b349a879b2019b4a613244f5fea7050e407b6475b9e29bd1c146dc"
    assert gpu_lock.acquire("A", "anchor22-a2-" + stage) == 0
    try:
        if stage == "screen":
            panel(False, "0,2")
        elif stage == "full":
            panel(False, "0,1,2,3,4,5")
        elif stage == "ood":
            panel(True, "0,1,2,3,4,5")
        elif stage == "timing":
            directory = OUT / "timing"
            directory.mkdir(parents=True, exist_ok=True)
            assert not (directory / "default.json").exists(), "Do not overwrite/repeat fresh timings"
            subprocess.run([str(PY), "evaluator/official_eval.py", "--solution", str(CAND), "--name", "anchor22-a2-default",
                            "--cache-mode", "read", "--nvfp4-cache-mode", "auto", "--algorithm-device", "cuda",
                            "--output", str(directory / "default.json"), "--report", str(directory / "default.md")], cwd=ROOT, check=True)
        elif stage == "cross":
            directory = OUT / "cross"
            directory.mkdir(parents=True, exist_ok=True)
            assert not (directory / "gpt2.json").exists()
            subprocess.run([str(PY), "evaluator/cross_model_eval.py", "--model", "gpt2", "--solution", str(CAND),
                            "--name", "anchor22-a2-gpt2", "--attention-only", "--compact-panel", "--cache-mode", "read",
                            "--algorithm-device", "cuda", "--output", str(directory / "gpt2.json"),
                            "--report", str(directory / "gpt2.md")], cwd=ROOT, check=True)
        else:
            raise ValueError(stage)
    finally:
        gpu_lock.release("A", "anchor22-a2-" + stage)


if __name__ == "__main__":
    main()
