"""V-effect decomposition of the attention error ledger (R-2 localization).

Answers one question: of mse_player and mse_standard, how much comes from
V quantization (F5v) vs Q/K quantization (F1) vs softmax cross terms?

  F5v_player = ||attn(Q_hat, K_hat, v_hat) - attn(Q_hat, K_hat, v_ref)||^2
  F5v_std    = ||attn(Q_std, K_std, v_std) - attn(Q_std, K_std, v_ref)||^2
  F1_std     = ||attn(Q_std, K_std, v_hat) - attn(Q_ref, K_ref, v_hat)||^2

If F5v_player / mse_standard is large, the 0.9 target is structurally
unreachable with V frozen; if small, Q/K (F1/F2) remains the actionable grid.
"""
from pathlib import Path
import importlib.util
import json
import math
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import reference_hif4 as ref  # noqa: E402
import gpu_lock  # noqa: E402

PARENT_SOLUTION = ROOT / "workbench/continuous_attention/anchor23-a1/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
A23_DIR = ROOT / "artifacts/proxy_v3/continuous/attention/anchor23-a1/id"
FA_LAYERS = [0, 1, 5, 8, 15, 22]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attention_out(mod, q, k, v, qh, kh, dim):
    return mod._a2_attention_forward(q[None].float(), k[None].float(), v[None].float(), qh, kh, dim)[0]


def std_quant(role, packed, qh, kh, dim):
    """Standard HiF4 (reference, no player state) quantization + decode."""
    dense = _MOD_STDECODE_DENSE(role, packed)
    params = ref.encode_standard_hif4(dense)
    return ref.decode_standard_hif4(params).float()


_MOD_STDECODE_DENSE = None  # set in main(): mod._dequantize_nvfp4_float32


def main():
    global _MOD_STDECODE_DENSE
    mod = load(PARENT_SOLUTION, "a23_vdecomp")
    _MOD_STDECODE_DENSE = lambda role, packed: mod._dequantize_nvfp4_float32(*packed)
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    cases = []
    for shard in range(6):
        d = json.loads((A23_DIR / f"candidate-attention-shard{shard}.json").read_text(encoding="utf-8"))
        cases += d["results"][0]["case_scores"]["attention"]
    std_by_case = {(c["layer"], c["test_window"]): c["mse_standard"] for c in cases}
    player_by_case = {(c["layer"], c["test_window"]): c["mse_player"] for c in cases}

    assert gpu_lock.acquire("A", "attention-v-decompose") == 0
    rows = []
    try:
        for layer in FA_LAYERS:
            calib = [
                {role: tuple(t.cuda() for t in ev._pair(dense))
                 for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])}
                for f in range(5)
            ]
            with torch.inference_mode():
                states = mod.hif4_calibration_attention(calib, 16, 4, 256)
            for w in range(12):
                item = {role: tuple(t.cuda() for t in ev._pair(dense))
                        for role, dense in zip(("q", "k", "v"), pack["test_qkv"][w][layer])}
                with torch.inference_mode():
                    q_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_q(*item["q"], 16, 256, states["q_state"])).float()
                    k_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_k(*item["k"], 4, 256, states["k_state"])).float()
                    v_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_v(*item["v"], 4, 256, states["v_state"])).float()
                    v_ref = mod._dequantize_nvfp4_float32(*item["v"]).float()
                    q_ref = mod._dequantize_nvfp4_float32(*item["q"]).float()
                    k_ref = mod._dequantize_nvfp4_float32(*item["k"]).float()
                    q_std = std_quant("q", item["q"], 16, 4, 256)
                    k_std = std_quant("k", item["k"], 4, 4, 256)
                    v_std = std_quant("v", item["v"], 4, 4, 256)
                out_p = attention_out(mod, q_hat, k_hat, v_hat, 16, 4, 256)
                out_p_vref = attention_out(mod, q_hat, k_hat, v_ref, 16, 4, 256)
                out_s = attention_out(mod, q_std, k_std, v_std, 16, 4, 256)
                out_s_vref = attention_out(mod, q_std, k_std, v_ref, 16, 4, 256)
                out_s_vhat = attention_out(mod, q_std, k_std, v_hat, 16, 4, 256)
                f5v_player = float((out_p - out_p_vref).square().mean())
                f5v_std = float((out_s - out_s_vref).square().mean())
                f1_std = float((out_s_vhat - attention_out(mod, q_ref, k_ref, v_hat, 16, 4, 256)).square().mean())
                std = std_by_case[(layer, w)]
                rows.append({
                    "layer": layer, "test_window": w, "mse_standard": std,
                    "mse_player": player_by_case[(layer, w)],
                    "f5v_player": f5v_player, "f5v_player_share": f5v_player / std,
                    "f5v_std": f5v_std, "f5v_std_share": f5v_std / std,
                    "f1_std": f1_std, "f1_std_share": f1_std / std,
                })
                print(f"L{layer:2d} w{w:2d}: F5v_player/std={f5v_player/std:.4f} "
                      f"F5v_std/std={f5v_std/std:.4f} F1_std/std={f1_std/std:.4f}", flush=True)
    finally:
        gpu_lock.release("A", "attention-v-decompose")

    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    summary = {
        "definition": "V-effect decomposition of A23 parent error on 72 paired 4B cases",
        "f5v_player_share_mean": mean("f5v_player_share"),
        "f5v_std_share_mean": mean("f5v_std_share"),
        "f1_std_share_mean": mean("f1_std_share"),
        "per_layer": {str(L): {
            "f5v_player_share": sum(r["f5v_player_share"] for r in rows if r["layer"] == L) / 12,
            "f5v_std_share": sum(r["f5v_std_share"] for r in rows if r["layer"] == L) / 12,
            "f1_std_share": sum(r["f1_std_share"] for r in rows if r["layer"] == L) / 12,
        } for L in FA_LAYERS},
        "structural_reachability": {
            "floor_F4_if_QK_zero": mean("f5v_player_share"),
            "note": "F4 >= F5v_player/mse_standard even with perfect Q/K (cross terms ignored); if this floor exceeds 0.1, gain 0.9 is unreachable with V frozen at the parent quantized path",
        },
        "rows": rows,
    }
    out = ROOT / "artifacts/continuous/attention/error_ledger_v_decompose_2026-09-08.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nfloor F4 (QK=0): {summary['structural_reachability']['floor_F4_if_QK_zero']:.4f}")
    print(f"F5v_std share: {summary['f5v_std_share_mean']:.4f}  F1_std share: {summary['f1_std_share_mean']:.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
