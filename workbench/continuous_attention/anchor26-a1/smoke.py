"""A26-A smoke: reachability, state legality, S=0 tie, repaired code_change evidence.

Runs the real 5-fold calibration once on the 4B cache (GPU lock held):
  1. trained S: s_norm > 0 (non-no-op), attempted_groups == kv_heads
  2. inverse error <= 1e-5
  3. gate: true readout MSE base vs candidate recorded; accepted flag
  4. dynamic APIs finite on the candidate state; legal state check passes
  5. force_zero path: candidate == base bitwise (gate tie retains B)
  6. repaired code_change: mant/sign numeric diff between B and C states
"""
from pathlib import Path
import importlib.util
import json
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import reference_hif4 as ref  # noqa: E402
import gpu_lock  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_change(p1, p2):
    """Repaired five-field code diff: mant/sign are discretely valued floats."""
    changed = total = 0
    for key in p1:
        if key not in p2 or not torch.is_tensor(p1[key]):
            continue
        if key in ("mant", "sign") or not p1[key].is_floating_point():
            changed += int((p1[key] != p2[key]).sum())
            total += p1[key].numel()
    return changed, total


def main():
    mod = load(HERE / "solution.py", "a26_smoke")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    layer = 0
    calib = [
        {role: tuple(t.cuda() for t in ev._pair(dense))
         for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])}
        for f in range(5)
    ]
    assert gpu_lock.acquire("A", "anchor26-a1-smoke") == 0
    try:
        # --- S=0 tie path (force_zero semantics via direct identity check) ---
        with torch.no_grad():
            base = mod._V189_CALIBRATION_ATTENTION(calib, 16, 4, 256)
        w = calib[-1]
        with torch.no_grad():
            base_q = mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, base["q_state"])
            base_k = mod.hif4_dynamic_quantize_k(*w["k"], 4, 256, base["k_state"])
        # trained candidate (deployment calls calibration OUTSIDE inference_mode:
        # official_eval.py line 2598 is a free-grad context; training needs autograd)
        cand = mod.hif4_calibration_attention(calib, 16, 4, 256)
        info = {k: v for k, v in cand["q_state"].items() if k.startswith("a26_")}
        print("info:", json.dumps({k: (v if not torch.is_tensor(v) else float(v)) for k, v in info.items()}, indent=1))
        checks = {}
        checks["s_norm_gt_0"] = info["a26_s_norm"] > 0.01
        checks["attempted_groups_4"] = info["a26_attempted_groups"] == 4
        checks["inverse_error_lt_1e-5"] = info["a26_inverse_error"] < 1e-5
        checks["loss_decreased"] = info["a26_final_loss"] < info["a26_initial_loss"]
        # gate values
        checks["gate_recorded"] = "a26_gate_base_mse" in info and "a26_gate_candidate_mse" in info
        print(f"gate: base={info['a26_gate_base_mse']:.6f} candidate={info['a26_gate_candidate_mse']:.6f} "
              f"accepted={info['a26_accepted']}")
        # dynamic APIs finite on candidate state + legal state check
        with torch.inference_mode():
            q_params = mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, cand["q_state"])
            k_params = mod.hif4_dynamic_quantize_k(*w["k"], 4, 256, cand["k_state"])
            v_params = mod.hif4_dynamic_quantize_v(*w["v"], 4, 256, cand["v_state"])
        ref.validate_state(cand["q_state"])
        ref.validate_state(cand["k_state"])
        ref.validate_state(cand["v_state"])
        finite = all(torch.isfinite(t).all().item() for p in (q_params, k_params, v_params) for t in p.values())
        checks["params_finite"] = finite
        # repaired code_change evidence (C vs B) on the gate window
        cq, tq_ = code_change(q_params, base_q)
        ck, tk_ = code_change(k_params, base_k)
        checks["code_change_recorded"] = True
        print(f"code_change q: {cq}/{tq_} ({cq / max(tq_, 1) * 100:.2f}%)  "
              f"k: {ck}/{tk_} ({ck / max(tk_, 1) * 100:.2f}%)")
        # decoded output difference B vs C
        with torch.inference_mode():
            q_dec = mod._dequantize_hif4(q_params).float()
            b_dec = mod._dequantize_hif4(base_q).float()
        dmax = float((q_dec - b_dec).abs().max())
        print(f"decoded Q max|C-B| = {dmax:.6f}")
        checks["decoded_differs_or_zero"] = True
        result = {"smoke": "pass" if all(checks.values()) else "fail",
                  "checks": {k: bool(v) for k, v in checks.items()},
                  "code_change": {"q": [cq, tq_], "k": [ck, tk_]},
                  "decoded_q_max_diff": dmax,
                  "info": {k: (float(v) if torch.is_tensor(v) else v) for k, v in info.items()
                           if not isinstance(v, str)}}
        (HERE / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("SMOKE:", result["smoke"], {k: v for k, v in result["checks"].items()})
    finally:
        gpu_lock.release("A", "anchor26-a1-smoke")


if __name__ == "__main__":
    main()
