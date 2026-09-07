"""A26-A per-layer gate check: gate MSE (B vs pre-gate C), repaired code_change.

Deployment calls calibration layer-by-layer; the gate decides acceptance per
layer. This runs all 6 FA layers once and reports, per layer:
  - gate base / candidate readout MSE and acceptance
  - code_change of the PRE-GATE candidate state vs B (mant/sign numeric diff)
  - decoded max|C-B| of Q
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
import gpu_lock  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
FA_LAYERS = [0, 1, 5, 8, 15, 22]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_change(p1, p2):
    changed = total = 0
    for key in p1:
        if key not in p2 or not torch.is_tensor(p1[key]):
            continue
        if key in ("mant", "sign") or not p1[key].is_floating_point():
            changed += int((p1[key] != p2[key]).sum())
            total += p1[key].numel()
    return changed, total


def main():
    mod = load(HERE / "solution.py", "a26_gatecheck")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor26-a1-gatecheck") == 0
    rows = []
    try:
        for layer in FA_LAYERS:
            calib = [
                {role: tuple(t.cuda() for t in ev._pair(dense))
                 for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])}
                for f in range(5)
            ]
            with torch.no_grad():
                base = mod._V189_CALIBRATION_ATTENTION(calib, 16, 4, 256)
            tq, tk, info = mod._a26_train(calib[:-1], base, 16, 4, 256, torch.device("cuda"))
            cand = {key: dict(value) for key, value in base.items()}
            cand["q_state"]["learned_rotation"] = tq.cuda()
            cand["k_state"]["learned_rotation"] = tk.cuda()
            w = calib[-1]
            with torch.no_grad():
                parent_loss = mod._a21_gate_loss(w, base, 16, 4, 256)
                cand_loss = mod._a21_gate_loss(w, cand, 16, 4, 256)
                base_q = mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, base["q_state"])
                cand_q = mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, cand["q_state"])
            cq, tq_ = code_change(cand_q, base_q)
            with torch.no_grad():
                dmax = float((mod._dequantize_hif4(cand_q).float() - mod._dequantize_hif4(base_q).float()).abs().max())
            accepted = cand_loss < parent_loss
            rows.append({
                "layer": layer, "gate_base": parent_loss, "gate_candidate": cand_loss,
                "accepted": int(accepted), "s_norm": info["a26_s_norm"],
                "initial_loss": info["a26_initial_loss"], "final_loss": info["a26_final_loss"],
                "inverse_error": info["a26_inverse_error"],
                "code_change_q": [cq, tq_], "decoded_q_max_diff": dmax,
            })
            print(f"L{layer:2d}: gate B={parent_loss:.6f} C={cand_loss:.6f} accepted={int(accepted)} "
                  f"s_norm={info['a26_s_norm']:.3f} loss {info['a26_initial_loss']:.2f}->{info['a26_final_loss']:.2f} "
                  f"code_q={cq}/{tq_} ({cq / max(tq_, 1) * 100:.2f}%) dmax={dmax:.6f}", flush=True)
    finally:
        gpu_lock.release("A", "anchor26-a1-gatecheck")
    accepted_n = sum(r["accepted"] for r in rows)
    changed_any = any(r["code_change_q"][0] > 0 for r in rows)
    summary = {"layers": rows, "accepted_layers": f"{accepted_n}/6",
               "code_changed_any": bool(changed_any),
               "verdict": "gate-accepted" if accepted_n >= 3 else "gate-rejected-majority"}
    (HERE / "gate_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\naccepted {accepted_n}/6 layers; code_changed_any={changed_any}; verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
