"""L-A0 calibration S_fit verification (CPU, zero model forward).

For each weight state (layer, role), run L-A0's Full-64 Direct A@W solver on the
real calibration folds and compute S_fit = mean_i(1 - MSE_player,i/MSE_std,i),
the directive's primary acceptance metric (target >= 0.85, ideal ~0.88).

Reuses the L4 calibration artifact (ACB16F76) for activation_state + weight_params
(starting point), the dense cache for X/Y.  No 4B panel, no API re-run.
"""
import gc
import hashlib
import importlib.util
import glob
import json
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


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_solution(path):
    spec = importlib.util.spec_from_file_location("la0_verify", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    torch.manual_seed(0)
    print(f"candidate SHA: {sha_of(CAND_SOL)[:16]}")
    sol = load_solution(CAND_SOL)

    raw = torch.load(DENSE_CACHE, map_location="cpu", weights_only=False)
    weights = raw["weights"]
    cal_act = raw["calibration_activations"]
    cal_windows = raw["calibration_windows"]
    roles = tuple(raw["roles"])
    del raw
    gc.collect()

    # L4 calibration artifact for activation_state + starting weight_params
    par_artifacts = []
    for p in sorted(glob.glob(os.path.join(CALIB_DIR, f"{PAR_SHA}-linear-*.pt"))):
        art = torch.load(p, map_location="cpu", weights_only=True)
        model = art.get("identity", {}).get("model_revision") or ""
        del art
        gc.collect()
        if "Qwen3.5-4B" in model:
            par_artifacts.append(p)
    print(f"L4 artifacts: {len(par_artifacts)}")

    # build L4 state map (activation_state + weight_params) per (layer, role)
    par_map = {}
    for p in par_artifacts:
        art = torch.load(p, map_location="cpu", weights_only=True)
        for item in art.get("weight_states", []):
            key = (int(item["layer"]), str(item["role"]))
            par_map[key] = (item["state"], dict(item["params"]))
        del art
        gc.collect()
    print(f"L4 states: {len(par_map)}")

    # verify L-A0 solver on shard0 states first (28 states), then decide
    shard0_layers = {5, 11, 17, 23}
    rows = []
    for (layer, role), (astate, params) in sorted(par_map.items()):
        if layer not in shard0_layers:
            continue
        w_dense = weights[layer][role].to(torch.float32)
        weight_pair = v2._pair(w_dense)
        wq, ws = (weight_pair[0], weight_pair[1])
        w_orig = sol._dequantize_nvfp4_float32(wq, ws)
        acts_pairs = [(v2._pair(cal_act[role][f][layer].to(torch.float32))) for f in (0, 1)]
        cand_params = sol._la0_full64_direct_fit(
            dict(params), astate, acts_pairs, w_orig,
            int(w_orig.shape[0]), int(w_orig.shape[1]),
        )
        wp = cand_params
        W_std = ref.dequantize_hif4(dict(params), w_dense.shape).to(torch.float32)
        W_cand = ref.dequantize_hif4(dict(wp), w_dense.shape).to(torch.float32)
        W_ref = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
        for fold in (0, 1):
            x_dense = cal_act[role][fold][layer].to(torch.float32)
            ap = v2._pair(x_dense)
            X = v2.dequantize_nvfp4(*ap).to(torch.float32)
            X_std = ref.decode_standard_hif4(ref.encode_standard_hif4(X)).to(torch.float32)
            actp = sol.hif4_dynamic_quantize_activation(ap[0], ap[1], astate)
            Xh = ref.dequantize_hif4(v2._cpu_params(actp), X.shape).to(torch.float32)
            reference = X @ W_ref.t()
            standard = X_std @ W_std.t()
            player = Xh @ W_cand.t()
            mse_std = float((standard - reference).square().mean())
            mse_cand = float((player - reference).square().mean())
            gain = (mse_std - mse_cand) / mse_std if mse_std > 0 else float("nan")
            rows.append({"layer": layer, "role": role, "fold": fold,
                         "mse_std": mse_std, "mse_cand": mse_cand, "gain": gain})
    gains = [r["gain"] for r in rows if math.isfinite(r["gain"])]
    s_fit = sum(gains) / len(gains) if gains else float("nan")
    print(f"\n=== L-A0 shard0 S_fit (calibration folds) ===")
    print(f"S_fit = {s_fit:.4f} (n={len(gains)})")
    print(f"target >= 0.85, ideal ~0.88")
    per_role = {}
    for r in rows:
        if math.isfinite(r["gain"]):
            per_role.setdefault(r["role"], []).append(r["gain"])
    for role, g in sorted(per_role.items()):
        print(f"  {role:9s} S_fit={sum(g)/len(g):.4f} n={len(g)}")
    with open(os.path.join(os.path.dirname(CAND_SOL), "la0-sfit-shard0.json"), "w", encoding="utf-8") as f:
        json.dump({"s_fit": s_fit, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
