"""A27-B smoke: end-to-end public entry on one FA layer + fallback behavior.

Checks (GPU lock):
  1. hif4_calibration_attention with 5 folds runs, returns valid states with
     the a27 info block, and the learned_rotation path keeps three API
     legality (dynamic q/k/v accept the states, finite outputs).
  2. Fallback: a single-fold call (len < 2) returns the base states unchanged.
  3. S=0 no-signal safety is covered by verify_math V3-force-zero (bit-equal).
"""
from pathlib import Path
import importlib.util
import json
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import gpu_lock  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
LAYER = 0


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load(HERE / "solution.py", "a27_smoke")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor27-b-smoke") == 0
    try:
        calib = [
            {role: tuple(t.cuda() for t in ev._pair(dense))
             for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][LAYER])}
            for f in range(5)
        ]
        result = {}
        with torch.no_grad():
            t0 = time.perf_counter()
            states = mod.hif4_calibration_attention(calib, 16, 4, 256)
            torch.cuda.synchronize()
            result["wall_s"] = time.perf_counter() - t0
            info = states["q_state"]
            result["info"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in info.items() if k.startswith("a27_") and isinstance(v, (int, float, str))}
            # legality: dynamic APIs accept the states, outputs finite
            q_out = mod.hif4_dynamic_quantize_q(*calib[-1]["q"], 16, 256, states["q_state"])
            k_out = mod.hif4_dynamic_quantize_k(*calib[-1]["k"], 4, 256, states["k_state"])
            v_out = mod.hif4_dynamic_quantize_v(*calib[-1]["v"], 4, 256, states["v_state"])
            finite = all(torch.isfinite(mod._dequantize_hif4(p)).all().item() for p in (q_out, k_out, v_out))
            result["finite_outputs"] = bool(finite)
            result["accepted"] = int(info.get("a27_accepted", 0))
            result["moved"] = int(info.get("a27_moved_coords", 0))
            result["fallback_single_fold_ok"] = None
        with torch.no_grad():
            single = mod.hif4_calibration_attention(calib[:1], 16, 4, 256)
            base = mod._V189_CALIBRATION_ATTENTION(calib[:1], 16, 4, 256)
            same = all(torch.equal(single["q_state"][key], base["q_state"][key])
                       for key in base["q_state"] if torch.is_tensor(base["q_state"][key]))
            result["fallback_single_fold_ok"] = bool(same)
        result["pass"] = bool(result["finite_outputs"] and result["fallback_single_fold_ok"]
                              and result["moved"] > 0)
        (HERE / "smoke.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, default=str))
    finally:
        gpu_lock.release("A", "anchor27-b-smoke")


if __name__ == "__main__":
    main()
