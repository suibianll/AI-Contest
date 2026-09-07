"""Decisive check: does the trained S worsen the T-coordinate logits MSE
(the quantity the A26 loss models) or only the post-softmax output MSE?

Per layer, on the gate window:
  qk_mse_B : ||Qb Kb^T - T(Q)T(K)^T||^2 / d   (B states, true residuals)
  qk_mse_C : same with the pre-gate C states
  out_mse_B / out_mse_C : attention-output MSE (softmax domain, gate value)
If qk_mse_C > qk_mse_B the first-order proxy itself is anti-correlated with
the real code error (model falsified at the logits level).
"""
from pathlib import Path
import importlib.util
import json
import math
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


def main():
    mod = load(HERE / "solution.py", "a26_logitcheck")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor26-a1-logitcheck") == 0
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
                q_ref = mod._dequantize_nvfp4_float32(*w["q"]).float()
                k_ref = mod._dequantize_nvfp4_float32(*w["k"]).float()
                v_ref = mod._dequantize_nvfp4_float32(*w["v"]).float()
                q_t = mod._a1_stack_transform(q_ref, 16, 256, base["q_state"], False)
                k_t = mod._a1_stack_transform(k_ref, 4, 256, base["k_state"], True)
                v_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_v(*w["v"], 4, 256, base["v_state"])).float()
                dim = 256
                out = {}
                for tag, st in (("B", base), ("C", cand)):
                    q_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, st["q_state"])).float()
                    k_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_k(*w["k"], 4, 256, st["k_state"])).float()
                    logits_hat = (q_hat.reshape(-1, 16, dim) @ k_hat.reshape(-1, 4, dim).transpose(-1, -2)) / math.sqrt(dim)
                    logits_t = (q_t.reshape(-1, 16, dim) @ k_t.reshape(-1, 4, dim).transpose(-1, -2)) / math.sqrt(dim)
                    out[f"qk_{tag}"] = float((logits_hat - logits_t).square().mean())
                    o_hat = mod._a2_attention_forward(q_hat[None], k_hat[None], v_hat[None], 16, 4, dim)[0]
                    o_t = mod._a2_attention_forward(q_t[None], k_t[None], v_hat[None], 16, 4, dim)[0]
                    out[f"out_{tag}"] = float((o_hat - o_t).square().mean())
            rows.append({"layer": layer, **out})
            print(f"L{layer:2d}: qk B={out['qk_B']:.4f} C={out['qk_C']:.4f} "
                  f"({'worse' if out['qk_C'] > out['qk_B'] else 'better'})  "
                  f"out B={out['out_B']:.6f} C={out['out_C']:.6f} "
                  f"({'worse' if out['out_C'] > out['out_B'] else 'better'})", flush=True)
    finally:
        gpu_lock.release("A", "anchor26-a1-logitcheck")
    (HERE / "logit_check.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    qk_worse = sum(r["qk_C"] > r["qk_B"] for r in rows)
    print(f"\nqk worse in {qk_worse}/6 layers (logits-level falsification if 6/6)")


if __name__ == "__main__":
    main()
