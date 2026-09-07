"""A27-B per-layer gate check (6 FA layers): gate acceptance, repaired
code_change (mant/sign numeric diff) of the PRE-GATE candidate vs B, decoded
max diff, training wall time.

Deployment calls calibration layer-by-layer; the gate decides acceptance per
layer. Pre-registered falsification: accepted < 3/6 -> close this card.
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
    mod = load(HERE / "solution.py", "a27_gatecheck")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor27-b-gatecheck") == 0
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
                t0 = time.perf_counter()
                tq, tk, info = mod._a27_train(calib[:-1], base, 16, 4, 256, torch.device("cuda"))
                torch.cuda.synchronize()
                train_s = time.perf_counter() - t0
                cand = {key: dict(value) for key, value in base.items()}
                cand["q_state"]["learned_rotation"] = tq.cuda()
                cand["k_state"]["learned_rotation"] = tk.cuda()
                w = calib[-1]
                parent_loss = mod._a21_gate_loss(w, base, 16, 4, 256)
                cand_loss = mod._a21_gate_loss(w, cand, 16, 4, 256)
                base_q = mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, base["q_state"])
                cand_q = mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, cand["q_state"])
                cq, tq_ = code_change(cand_q, base_q)
                dmax = float((mod._dequantize_hif4(cand_q).float()
                              - mod._dequantize_hif4(base_q).float()).abs().max())
            accepted = cand_loss < parent_loss
            rows.append({
                "layer": layer, "gate_base": parent_loss, "gate_candidate": cand_loss,
                "accepted": int(accepted), "train_s": train_s,
                "moved": info["a27_moved_coords"], "fd_nonzero": info["a27_fd_nonzero"],
                "initial_loss": info["a27_initial_loss"], "final_loss": info["a27_final_loss"],
                "inverse_error": info["a27_inverse_error"],
                "code_change_q": [cq, tq_], "decoded_q_max_diff": dmax,
            })
            print(f"L{layer:2d}: gate B={parent_loss:.6f} C={cand_loss:.6f} accepted={int(accepted)} "
                  f"moved={info['a27_moved_coords']}/32 loss {info['a27_initial_loss']:.6f}->"
                  f"{info['a27_final_loss']:.6f} train={train_s:.1f}s "
                  f"code_q={cq}/{tq_} ({cq / max(tq_, 1) * 100:.2f}%) dmax={dmax:.6f}", flush=True)
    finally:
        gpu_lock.release("A", "anchor27-b-gatecheck")
    accepted_n = sum(r["accepted"] for r in rows)
    changed_any = any(r["code_change_q"][0] > 0 for r in rows)
    summary = {"layers": rows, "accepted_layers": f"{accepted_n}/6",
               "code_changed_any": bool(changed_any),
               "verdict": "gate-accepted" if accepted_n >= 3 else "gate-rejected-majority"}
    (HERE / "gate_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\naccepted {accepted_n}/6 layers; code_changed_any={changed_any}; verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
