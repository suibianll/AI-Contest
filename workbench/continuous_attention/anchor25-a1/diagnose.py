"""Four-dimension mechanism diagnostic for A25.

For each FA layer on the 4B panel, records under S=0 (before transform) and
S=trained (after transform):
  1. scale: per-role/block amax ratio (after/before)
  2. code change: count of HiF4 five-field code words that differ
  3. QK MSE: MSE of Q@K^T (quantized vs NVFP4 reference)
  4. Attention output MSE: MSE of softmax(QK/sqrt(d))@V (quantized vs reference)

Also runs the three-way comparison: B (base stack) / P_old (R3) / C (A25).
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev
import reference_hif4 as ref
import gpu_lock


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hif4_params_diff(p1, p2):
    """Count differing five-field code words between two HiF4 param dicts."""
    changed = 0
    total = 0
    for key in p1:
        if key not in p2:
            continue
        if torch.is_tensor(p1[key]) and p1[key].is_floating_point():
            continue
        if torch.is_tensor(p1[key]):
            diff = (p1[key] != p2[key]).sum().item()
            changed += diff
            total += p1[key].numel()
    return changed, total


def qk_mse(q_hat, k_hat, q_ref, k_ref, qh, kh, dim):
    """MSE of Q@K^T (quantized) vs Q_ref@K_ref^T (NVFP4 reference)."""
    group = qh // kh
    q3 = q_hat.reshape(q_hat.shape[0], qh, dim).transpose(0, 1)
    k3 = k_hat.reshape(k_hat.shape[0], kh, dim).transpose(0, 1).repeat_interleave(group, 0)
    qr3 = q_ref.reshape(q_ref.shape[0], qh, dim).transpose(0, 1)
    kr3 = k_ref.reshape(k_ref.shape[0], kh, dim).transpose(0, 1).repeat_interleave(group, 0)
    qk = (q3 @ k3.transpose(-1, -2)) / math.sqrt(dim)
    qk_ref = (qr3 @ kr3.transpose(-1, -2)) / math.sqrt(dim)
    return float((qk - qk_ref).square().mean())


def attn_mse(mod, q_hat, k_hat, v_hat, q_ref, k_ref, v_ref, qh, kh, dim):
    """Full attention output MSE (quantized vs NVFP4 reference)."""
    return float((mod._a2_attention_forward(q_hat[None], k_hat[None], v_hat[None], qh, kh, dim)[0]
                  - mod._a2_attention_forward(q_ref[None], k_ref[None], v_ref[None], qh, kh, dim)[0]).square().mean())


def main():
    mod = load(HERE / "solution.py", "a25_diag")
    r3 = load(ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py", "r3_diag")
    torch.manual_seed(21071)
    result = {"source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest()}
    pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt", map_location="cpu", weights_only=False)
    fa_layers = [0, 1, 5, 8, 15, 22]
    assert gpu_lock.acquire("A", "anchor25-a1-diag") == 0
    try:
        layers_diag = []
        for layer in fa_layers:
            wins = [{role: tuple(t.cuda() for t in ev._pair(dense)) for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])} for f in range(5)]
            # B: base stack
            with torch.inference_mode():
                b_states = mod._V189_CALIBRATION_ATTENTION(wins, 16, 4, 256)
            # P_old: R3 complete (old training)
            with torch.inference_mode():
                r3_states = r3.hif4_calibration_attention(wins, 16, 4, 256)
            # C: A25 single-trunk
            with torch.inference_mode():
                c_states = mod.hif4_calibration_attention(wins, 16, 4, 256)
            # Use first test window for diagnostic
            w = wins[0]
            diag = {"layer": layer}
            # --- dimension 1: scale ratio (C vs B) ---
            for role, heads in (("q", 16), ("k", 4)):
                dense = mod._dequantize_nvfp4_float32(*w[role])
                u_before = mod._a1_stack_transform(dense, heads, 256, b_states[role + "_state"], role == "k")
                amax_before = u_before.reshape(-1, 4, 64).abs().amax(-1).mean().item()
                learned = c_states[role + "_state"].get("learned_rotation")
                if learned is not None:
                    u_after = mod._a2_apply_group_rotation(u_before, heads, learned.to(u_before.device))
                    amax_after = u_after.reshape(-1, 4, 64).abs().amax(-1).mean().item()
                    diag[f"scale_ratio_{role}"] = amax_after / max(amax_before, 1e-12)
                else:
                    diag[f"scale_ratio_{role}"] = 1.0
            # --- dimension 2: code change count (C vs B) ---
            for role, heads in (("q", 16), ("k", 4)):
                p_before = mod.hif4_dynamic_quantize_q(*w[role], heads, 256, b_states[role + "_state"]) if role == "q" else mod.hif4_dynamic_quantize_k(*w[role], heads, 256, b_states[role + "_state"])
                p_after = mod.hif4_dynamic_quantize_q(*w[role], heads, 256, c_states[role + "_state"]) if role == "q" else mod.hif4_dynamic_quantize_k(*w[role], heads, 256, c_states[role + "_state"])
                changed, total = hif4_params_diff(p_before, p_after)
                diag[f"code_change_{role}"] = {"changed": changed, "total": total, "pct": changed / max(total, 1) * 100}
            # --- dimensions 3 & 4: QK and Attention MSE for B / P_old / C ---
            v_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_v(*w["v"], 4, 256, b_states["v_state"])).float()
            q_ref = mod._dequantize_nvfp4_float32(*w["q"]).float()
            k_ref = mod._dequantize_nvfp4_float32(*w["k"]).float()
            v_ref = mod._dequantize_nvfp4_float32(*w["v"]).float()
            for tag, st in (("B", b_states), ("P_old_R3", r3_states), ("C_A25", c_states)):
                q_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_q(*w["q"], 16, 256, st["q_state"])).float()
                k_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_k(*w["k"], 4, 256, st["k_state"])).float()
                diag[f"qk_mse_{tag}"] = qk_mse(q_hat, k_hat, q_ref, k_ref, 16, 4, 256)
                diag[f"attn_mse_{tag}"] = attn_mse(mod, q_hat, k_hat, v_hat, q_ref, k_ref, v_ref, 16, 4, 256)
            # --- A25 audit ---
            for k, v in c_states["q_state"].items():
                if k.startswith("a25_"):
                    diag[k] = v
            layers_diag.append(diag)
            print(f"layer {layer:2d}: scale_q={diag['scale_ratio_q']:.4f} scale_k={diag['scale_ratio_k']:.4f} "
                  f"code_q={diag['code_change_q']['pct']:.1f}% code_k={diag['code_change_k']['pct']:.1f}% "
                  f"qk B/P/C={diag['qk_mse_B']:.2e}/{diag['qk_mse_P_old_R3']:.2e}/{diag['qk_mse_C_A25']:.2e} "
                  f"attn B/P/C={diag['attn_mse_B']:.2e}/{diag['attn_mse_P_old_R3']:.2e}/{diag['attn_mse_C_A25']:.2e} "
                  f"accepted={diag.get('a25_accepted')} s_norm={diag.get('a25_s_norm', 0):.3f}", flush=True)
    finally:
        gpu_lock.release("A", "anchor25-a1-diag")
    result["layers"] = layers_diag
    out = HERE / "diagnostics.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nDiagnostics written to {out}")
    # Summary
    print("\n=== Four-dimension summary ===")
    for d in layers_diag:
        layer = d["layer"]
        scale_q = d["scale_ratio_q"]
        scale_k = d["scale_ratio_k"]
        code_q = d["code_change_q"]["pct"]
        code_k = d["code_change_k"]["pct"]
        qk_b, qk_c = d["qk_mse_B"], d["qk_mse_C_A25"]
        attn_b, attn_c = d["attn_mse_B"], d["attn_mse_C_A25"]
        qk_p = d["qk_mse_P_old_R3"]
        attn_p = d["attn_mse_P_old_R3"]
        state = "no_change" if d.get("a25_s_norm", 0) < 0.01 else \
                "scale_down_code_changed" if scale_q < 0.99 and code_q > 0 else \
                "output_improved" if attn_c < attn_b else "output_worse"
        print(f"L{layer:2d}: scale {scale_q:.3f}/{scale_k:.3f} code {code_q:.0f}%/{code_k:.0f}% "
              f"qk B→C {qk_b:.2e}→{qk_c:.2e} attn B→C {attn_b:.2e}→{attn_c:.2e} "
              f"| P_old qk {qk_p:.2e} attn {attn_p:.2e} | {state}")


if __name__ == "__main__":
    main()
