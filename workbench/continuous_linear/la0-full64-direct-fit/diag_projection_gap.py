"""L-A0 diagnostic: continuous vs legal-projected loss gap on real 4B data.
Answers: is the low S_fit caused by projection loss (clip/quantization) or by
insufficient continuous fit?  Reports per-role: L0 (parent), L_cont (best
continuous after full-64 solve), L_legal (after mant/sign projection), and
the clip ratio (fraction of entries beyond the fixed scale grid).
"""
import gc
import glob
import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluator"))
import official_eval as v2
import reference_hif4 as ref

DENSE_CACHE = os.path.join(ROOT, "artifacts", "official_eval", "cache", "qwen3.5-4b-proxy-v2.pt")
CALIB_DIR = os.path.join(ROOT, "artifacts", "official_eval", "cache", "proxy-v3-calibration")
CAND_SOL = os.path.join(ROOT, "workbench", "continuous_linear", "la0-full64-direct-fit", "candidate", "solution.py")
PAR_SHA = "acb16f764db80eda"


def load_solution(path):
    spec = importlib.util.spec_from_file_location("la0diag", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    torch.manual_seed(0)
    sol = load_solution(CAND_SOL)
    raw = torch.load(DENSE_CACHE, map_location="cpu", weights_only=False)
    weights = raw["weights"]
    cal_act = raw["calibration_activations"]
    roles = tuple(raw["roles"])
    del raw
    gc.collect()

    par_artifacts = []
    for p in sorted(glob.glob(os.path.join(CALIB_DIR, f"{PAR_SHA}-linear-*.pt"))):
        art = torch.load(p, map_location="cpu", weights_only=True)
        model = art.get("identity", {}).get("model_revision") or ""
        del art
        gc.collect()
        if "Qwen3.5-4B" in model:
            par_artifacts.append(p)
    par_map = {}
    for p in par_artifacts:
        art = torch.load(p, map_location="cpu", weights_only=True)
        for item in art.get("weight_states", []):
            key = (int(item["layer"]), str(item["role"]))
            par_map[key] = (item["state"], dict(item["params"]))
        del art
        gc.collect()

    shard0_layers = {5, 11, 17, 23}
    summary = []
    for (layer, role), (astate, params) in sorted(par_map.items()):
        if layer not in shard0_layers:
            continue
        w_dense = weights[layer][role].to(torch.float32)
        wp = v2._pair(w_dense)
        w_orig = sol._dequantize_nvfp4_float32(wp[0], wp[1])
        o, in_f = int(w_orig.shape[0]), int(w_orig.shape[1])
        acts_pairs = [(v2._pair(cal_act[role][f][layer].to(torch.float32))) for f in (0, 1)]
        # Xh, Y, omegas (reuse solver internals by calling the fit but capturing continuous)
        F = 2
        xh_folds, y_folds, mse_std_folds = [], [], []
        for pair in acts_pairs:
            x = sol._dequantize_nvfp4_float32(pair[0], pair[1]).to(torch.float32)
            actp = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], astate)
            xh = sol._dequantize_hif4(actp).to(torch.float32)
            if xh.shape != x.shape:
                xh = xh.reshape(x.shape)
            xh_folds.append(xh)
            y_folds.append(x @ w_orig.t())
        w_std = sol._dequantize_hif4(params).to(torch.float32)
        for f in range(F):
            x = sol._dequantize_nvfp4_float32(acts_pairs[f][0], acts_pairs[f][1]).to(torch.float32)
            xh_std = sol._dequantize_hif4(sol.hif4_dynamic_quantize_activation(acts_pairs[f][0], acts_pairs[f][1], astate)).to(torch.float32)
            if xh_std.shape != x.shape:
                xh_std = xh_std.reshape(x.shape)
            standard = xh_std @ w_std.t()
            reference = x @ w_orig.t()
            mse_std_folds.append(float(((standard - reference) ** 2).mean()) + 1e-12)
        omegas = [1.0 / mse_std_folds[f] for f in range(F)]
        # continuous full-64 solve, no projection
        n_blocks = in_f // sol._HIF4_BLOCK_SIZE
        W = sol._dequantize_hif4(params).to(torch.float32)
        for b in range(n_blocks):
            sl = slice(b * sol._HIF4_BLOCK_SIZE, (b + 1) * sol._HIF4_BLOCK_SIZE)
            H_B = torch.zeros(sol._HIF4_BLOCK_SIZE, sol._HIF4_BLOCK_SIZE, dtype=torch.float32)
            G_B = torch.zeros(sol._HIF4_BLOCK_SIZE, o, dtype=torch.float32)
            for f in range(F):
                Xb = xh_folds[f][:, sl]
                Rf = y_folds[f] - xh_folds[f] @ W.t()
                H_B += omegas[f] * (Xb.t() @ Xb)
                G_B += omegas[f] * (Xb.t() @ Rf)
            H_B = 0.5 * (H_B + H_B.t())
            diagm = float(H_B.diagonal().mean().clamp_min(1e-12))
            H_reg = H_B + 0.2 * diagm * torch.eye(sol._HIF4_BLOCK_SIZE, dtype=torch.float32)
            L_B = torch.linalg.cholesky(H_reg)
            dW = torch.linalg.solve_triangular(L_B.t(), torch.linalg.solve_triangular(L_B, G_B, upper=False), upper=True)
            W[:, sl] = (W[:, sl].t() + dW).t()
        # compute L_cont and L_legal
        sf = params["scale_factor"].to(torch.float32)
        lv2v = params["scale_lv2"].to(torch.float32)
        lv3v = params["scale_lv3"].to(torch.float32)
        scale_full = (sf * lv2v * lv3v).expand(-1, -1, 8, 2, 4).reshape(o, n_blocks, 64).reshape(o, in_f)
        L_cont = 0.0
        L_legal = 0.0
        clip_total = 0
        total = 0
        W_legal = W.clone()
        for f in range(F):
            L_cont += omegas[f] * float(((y_folds[f] - xh_folds[f] @ W.t()) ** 2).sum())
        for b in range(n_blocks):
            sl = slice(b * sol._HIF4_BLOCK_SIZE, (b + 1) * sol._HIF4_BLOCK_SIZE)
            scale_b = scale_full[:, sl]
            Wq = (W[:, sl] / scale_b.clamp_min(1e-12) / 0.25).round().clamp(-7.0, 7.0) * 0.25 * scale_b
            W_legal[:, sl] = Wq
            clip_total += int((W[:, sl].abs() > (scale_b * 1.75).clamp_min(1e-12)).sum())
            total += W[:, sl].numel()
        for f in range(F):
            L_legal += omegas[f] * float(((y_folds[f] - xh_folds[f] @ W_legal.t()) ** 2).sum())
        L0 = 0.0
        for f in range(F):
            L0 += omegas[f] * float(((y_folds[f] - xh_folds[f] @ w_std.t()) ** 2).sum())
        summary.append({
            "state": [layer, role], "L0": L0, "L_cont": L_cont, "L_legal": L_legal,
            "clip_ratio": clip_total / total, "L_cont_reduction": 1 - L_cont / L0,
            "L_legal_reduction": 1 - L_legal / L0,
        })
        print(f"L{layer}:{role} L0={L0:.3e} L_cont={L_cont:.3e} ({1-L_cont/L0:.2%}) "
              f"L_legal={L_legal:.3e} ({1-L_legal/L0:.2%}) clip={clip_total/total:.2%}")
    with open(os.path.join(os.path.dirname(CAND_SOL), "la0-diag-shard0.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()