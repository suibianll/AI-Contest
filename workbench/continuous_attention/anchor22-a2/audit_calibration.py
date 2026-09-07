"""Calibration-only audit: dump per-layer gate decisions from the real
cached calibration windows (no scoring, no baseline API calls)."""
from pathlib import Path
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev
import gpu_lock
import torch

spec_ev = __import__("importlib").util.spec_from_file_location("a22b_audit", HERE / "solution.py")
mod = __import__("importlib").util.module_from_spec(spec_ev)
spec_ev.loader.exec_module(mod)

OUT = HERE / "calibration-audit.json"
assert not OUT.exists(), "audit already recorded"


def main():
    pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt", map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor22-a2-calib-audit") == 0
    layers = []
    try:
        for layer in range(24):
            wins = [{role: tuple(t.cuda() for t in ev._pair(dense)) for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])} for f in range(5)]
            started = time.perf_counter()
            with torch.inference_mode():
                states = mod.hif4_calibration_attention(wins, 14, 2, 64)
            record = {"layer": layer, "wall_s": time.perf_counter() - started}
            for key, value in states["q_state"].items():
                if key.startswith(("a21_", "a22_", "a2_")):
                    record[key] = value
            learned_q = states["q_state"].get("learned_rotation")
            record["deployed_learned_rotation_q"] = learned_q is not None
            record["deployed_learned_center_k"] = "learned_center" in states["k_state"]
            layers.append(record)
            print(f"layer {layer:2d}: arm={record.get('a22_parent_arm')} accepted={record.get('a22_accepted')} "
                  f"parent={record.get('a22_gate_parent_mse'):.3e} cand={record.get('a22_gate_candidate_mse'):.3e} "
                  f"wall={record['wall_s']:.2f}s", flush=True)
    finally:
        gpu_lock.release("A", "anchor22-a2-calib-audit")
    OUT.write_text(json.dumps({"source_sha256": mod.__dict__.get("__file__") and __import__("hashlib").sha256((HERE / "solution.py").read_bytes()).hexdigest(), "layers": layers}, indent=2) + "\n", encoding="utf-8")
    accepted = [r["layer"] for r in layers if r.get("a22_accepted")]
    print("accepted layers:", accepted)


if __name__ == "__main__":
    main()
