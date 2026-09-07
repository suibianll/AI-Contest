"""A27-B mathematical + deployment-fidelity verification (6 checks).

V1 (CPU)  _a27_project: zero-trace + box semantics (bisection formula).
V2 (CPU)  _a27_exp_pair: S=0 identity exact; exp(+S) exp(-S)^T = I; band expand.
V3 (GPU)  trainer reachability: real _a27_train on layer 0 has nonzero FD
          signals, moves coordinates, inverse error ~ 0; force_zero returns
          identity and bit-identical API readout (tie-keeps-B path).
V4 (GPU)  slice-readout fidelity: per-group c=0 readout == full dynamic API
          readout (proves the training objective IS the deployed chain).
V5 (GPU)  GQA decomposition: per-group attention == _a2_attention_forward.
V6 (CPU)  Hadamard basis orthonormality.
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
LAYER = 0
BOUND = math.log(2.0) / 2.0


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load(HERE / "solution.py", "a27_verify")
    results = {}

    # ---------- V1: projection semantics (CPU)
    torch.manual_seed(7)
    c = torch.randn(4, 8) * 0.6
    p = mod._a27_project(c.clone(), BOUND)
    v1_trace = float((p.sum(-1)).abs().max())
    v1_box = float(p.abs().max())
    # brute-force shift that zeroes the clamped sum, compared to the bisection
    def shift_zero(cc):
        lo, hi = -2.0, 2.0
        for _ in range(200):
            mid = (lo + hi) * 0.5
            if (cc - mid).clamp(-BOUND, BOUND).sum() > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5
    v1_shift_err = 0.0
    for g in range(4):
        m = shift_zero(c[g])
        ref = (c[g] - m).clamp(-BOUND, BOUND)
        v1_shift_err = max(v1_shift_err, float((ref - p[g]).abs().max()))
    results["V1_project"] = {
        "zero_trace_max": v1_trace, "box_max": v1_box, "shift_formula_err": v1_shift_err,
        "pass": bool(v1_trace < 1e-5 and v1_box <= BOUND + 1e-7 and v1_shift_err < 1e-5),
    }

    # ---------- V2: exp pair identity + band expand (CPU)
    u = mod._a27_hadamard(256, torch.device("cpu"))
    z = torch.zeros(4, 8)
    eq, ek = mod._a27_exp_pair(z, u, 8)
    v2_identity = float((eq - torch.eye(256)).abs().max())
    inv = (eq.to(torch.float64) @ ek.to(torch.float64).transpose(-1, -2) - torch.eye(256, dtype=torch.float64)).abs().max()
    cb = torch.randn(4, 8)
    d = mod._a27_band_expand(cb, 256, 8)
    band_ok = True
    for k in range(8):
        seg = d[:, k * 32:(k + 1) * 32]
        if not torch.allclose(seg, cb[:, k:k + 1].expand_as(seg), atol=0.0):
            band_ok = False
    results["V2_exp_pair"] = {
        "s0_identity_max": v2_identity, "inverse_max": float(inv), "band_expand_ok": band_ok,
        "pass": bool(v2_identity == 0.0 and float(inv) < 1e-9 and band_ok),
    }

    # ---------- V6: Hadamard orthonormality (CPU)
    gram = (u.to(torch.float64) @ u.to(torch.float64).t() - torch.eye(256, dtype=torch.float64)).abs().max()
    entries_ok = bool(((u * 16.0).round() == u * 16.0).all())
    results["V6_hadamard"] = {
        "orthonormal_max": float(gram), "entries_scaled_integers": entries_ok,
        "pass": bool(float(gram) < 1e-9 and entries_ok),
    }

    # ---------- GPU section
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor27-b-verify") == 0
    try:
        calib = [
            {role: tuple(t.cuda() for t in ev._pair(dense))
             for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][LAYER])}
            for f in range(5)
        ]
        dev = torch.device("cuda")
        with torch.no_grad():
            base = mod._V189_CALIBRATION_ATTENTION(calib, 16, 4, 256)
            qs, ks = base["q_state"], base["k_state"]

            # ---------- V4: slice readout fidelity at c=0 (fold 1, 128 tokens)
            item = calib[1]
            dense_q = mod._dequantize_nvfp4_float32(*item["q"]).to(dev, torch.float32)
            dense_k = mod._dequantize_nvfp4_float32(*item["k"]).to(dev, torch.float32)
            u_q = mod._a1_stack_transform(dense_q, 16, 256, qs, False)
            u_k = mod._a1_stack_transform(dense_k, 4, 256, ks, True)
            eye = torch.eye(256, device=dev).expand(1, 256, 256)
            imp_q_all = qs["importance"].detach().to(dev, torch.float32).reshape(-1)
            imp_k_all = ks["importance"].detach().to(dev, torch.float32).reshape(-1)
            slice_q = torch.empty_like(dense_q)
            slice_k = torch.empty_like(dense_k)
            for g in range(4):
                q_rot = mod._a2_apply_group_rotation(u_q.reshape(u_q.shape[0], 4, 1024)[:, g], 4, eye)
                k_rot = mod._a2_apply_group_rotation(u_k.reshape(u_k.shape[0], 4, 256)[:, g], 1, eye)
                pq = mod._dense_to_hif4(
                    q_rot, importance=imp_q_all[g * 1024:(g + 1) * 1024],
                    search_offsets=qs["offsets"], error_threshold=float(qs["error_threshold"]),
                    accept_margin=float(qs["accept_margin"]), max_refine_ratio=float(qs["max_refine_ratio"]),
                    max_refine_blocks=qs.get("max_refine_blocks"))
                pk = mod._dense_to_hif4(
                    k_rot, importance=imp_k_all[g * 256:(g + 1) * 256],
                    search_offsets=ks["offsets"], error_threshold=float(ks["error_threshold"]),
                    accept_margin=float(ks["accept_margin"]), max_refine_ratio=float(ks["max_refine_ratio"]),
                    max_refine_blocks=ks.get("max_refine_blocks"))
                slice_q.reshape(u_q.shape[0], 4, 1024)[:, g] = mod._dequantize_hif4(pq).to(torch.float32)
                slice_k.reshape(u_k.shape[0], 4, 256)[:, g] = mod._dequantize_hif4(pk).to(torch.float32)
            api_q = mod._dequantize_hif4(mod.hif4_dynamic_quantize_q(*item["q"], 16, 256, qs)).to(torch.float32)
            api_k = mod._dequantize_hif4(mod.hif4_dynamic_quantize_k(*item["k"], 4, 256, ks)).to(torch.float32)
            v4_q = float((slice_q - api_q).abs().max())
            v4_k = float((slice_k - api_k).abs().max())
            results["V4_slice_fidelity"] = {
                "q_max_diff": v4_q, "k_max_diff": v4_k,
                "pass": bool(v4_q == 0.0 and v4_k == 0.0),
            }

            # ---------- V5: GQA decomposition (fold 1)
            qh = api_q.reshape(api_q.shape[0], 4, 4, 256)
            kh_ = api_k.reshape(api_k.shape[0], 4, 1, 256)
            vh = mod._dequantize_hif4(mod.hif4_dynamic_quantize_v(*item["v"], 4, 256, base["v_state"])).to(torch.float32)
            vh = vh.reshape(vh.shape[0], 4, 256)
            per_group = [mod._a27_group_attn(qh[:, g], kh_[:, g], vh[:, g:g + 1], 256) for g in range(4)]
            joined = torch.stack(per_group, dim=1).reshape(api_q.shape[0], 16, 256)
            ref_out = mod._a2_attention_forward(
                api_q[None], api_k[None], mod._dequantize_hif4(mod.hif4_dynamic_quantize_v(*item["v"], 4, 256, base["v_state"])).to(torch.float32)[None],
                16, 4, 256)[0].reshape(api_q.shape[0], 4, 4, 256).reshape(api_q.shape[0], 16, 256)
            v5 = float((joined - ref_out).abs().max())
            results["V5_gqa_decomposition"] = {"max_diff": v5, "pass": bool(v5 < 1e-5)}

            # ---------- V3: trainer reachability (full run + force_zero)
            tq, tk, info = mod._a27_train(calib[:-1], base, 16, 4, 256, dev)
            results["V3_reachability"] = {
                "fd_nonzero": info["a27_fd_nonzero"], "moved": info["a27_moved_coords"],
                "attempted": info["a27_attempted_coords"],
                "initial_loss": info["a27_initial_loss"], "final_loss": info["a27_final_loss"],
                "c_norm": info["a27_c_norm"], "inverse_error": info["a27_inverse_error"],
                "pass": bool(info["a27_fd_nonzero"] > 0 and info["a27_moved_coords"] > 0
                             and info["a27_inverse_error"] < 1e-4),
            }
            tq0, tk0, info0 = mod._a27_train(calib[:-1], base, 16, 4, 256, dev, force_zero=True)
            ident = torch.eye(256).expand(4, 256, 256)
            v3_zero = bool(torch.equal(tq0, ident) and torch.equal(tk0, ident))
            cand = {k: dict(v) for k, v in base.items()}
            cand["q_state"]["learned_rotation"] = tq0.to(dev)
            cand["k_state"]["learned_rotation"] = tk0.to(dev)
            api_q_c = mod.hif4_dynamic_quantize_q(*item["q"], 16, 256, cand["q_state"])
            api_q_b = mod.hif4_dynamic_quantize_q(*item["q"], 16, 256, base["q_state"])
            bits_equal = all(torch.equal(api_q_c[key], api_q_b[key]) for key in api_q_b)
            results["V3_force_zero"] = {
                "identity_matrices": v3_zero, "api_readout_bit_equal": bits_equal,
                "pass": bool(v3_zero and bits_equal),
            }
    finally:
        gpu_lock.release("A", "anchor27-b-verify")

    all_pass = all(r["pass"] for r in results.values())
    results["ALL_PASS"] = all_pass
    (HERE / "verify_math.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, r in results.items():
        if name == "ALL_PASS":
            continue
        print(f"{name}: pass={r['pass']} " + ", ".join(f"{k}={v}" for k, v in r.items() if k != "pass"))
    print(f"ALL_PASS={all_pass}")


if __name__ == "__main__":
    main()
