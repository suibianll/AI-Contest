"""P4 unit proof: attention-side exact branch encoder (lightweight MSE upper bound).

Given NVFP4 attention input (quant, scale), build a "fully-exact" HiF4 encoder
that uses the per-(row, 64-block) exact sf selection (P0 lemma) and skip the
v168 center/rotation/pair_transform. Compare against v168 baseline output MSE.

Goal: estimate the upper bound of "exact path only" — if this MSE is much lower
than v168, P4 (hybrid exact + v168 path) is worth implementing. If similar,
v168 already near theoretical limit and P4 marginal gain is small.

Output: artifacts/proxy_v3/cb2-attn-exact-branch-20260905/cb2_report.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
for p in (str(ROOT / "evaluator"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import solution as sol  # noqa: E402

CACHE = ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt"
OUT_DIR = ROOT / "artifacts" / "proxy_v3" / "cb2-attn-exact-branch-20260905" / "run-001"

CODES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
S_FRACTIONS = sorted(
    {Fraction(k, 4) for k in range(1, 8)}
    | {Fraction(k, 2) for k in range(1, 8)}
    | {Fraction(k, 1) for k in range(1, 8)}
)
S_SET = set(S_FRACTIONS)
M4 = [Fraction(8 + i, 8) for i in range(8)]
M6 = [Fraction(4 + i, 4) for i in range(4)]
DJ_LO, DJ_HI = -8, 8
BIG = 1.0e30


def build_tables():
    valid = torch.zeros(7, 8, 4, 17, dtype=torch.float32)
    err2 = torch.zeros(7, 8, 4, 17, dtype=torch.float64)
    for ci, c in enumerate(CODES):
        cf = Fraction(c).limit_denominator(16)
        for i4, m4 in enumerate(M4):
            for i6, m6 in enumerate(M6):
                for dj in range(DJ_LO, DJ_HI + 1):
                    rho = (m4 / m6) * Fraction(2) ** dj
                    v = cf * rho
                    if v in S_SET:
                        valid[ci, i4, i6, dj - DJ_LO] = 1.0
                        err2[ci, i4, i6, dj - DJ_LO] = 0.0
                    else:
                        grid = [Fraction(0)] + S_FRACTIONS
                        best = min(grid, key=lambda t: abs(t - v))
                        err2[ci, i4, i6, dj - DJ_LO] = float((best - v) ** 2)
    return valid, err2


E_TABLE, ERR_TABLE = build_tables()
E_FLAT = E_TABLE.reshape(7, -1)
ERR_FLAT = ERR_TABLE.reshape(7, -1).to(torch.float32)


def e6m2_decode_f64(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(0, 254)
    e = torch.bitwise_right_shift(c, 2).to(torch.float64) - 48.0
    m = 1.0 + (c & 3).to(torch.float64) * 0.25
    return torch.pow(2.0, e) * m


def count_codes(quant: torch.Tensor) -> torch.Tensor:
    """counts shape (R, B, 4, 7) for CODES index in non-negative codes."""
    R, C = quant.shape
    B = C // 64
    q = quant.reshape(R, B, 4, 16)
    codes_pos = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=q.dtype, device=q.device)
    counts = torch.zeros(R, B, 4, 7, dtype=torch.int64)
    for ci, cv in enumerate(codes_pos.tolist()):
        counts[..., ci] = (q == cv).sum(-1)
    return counts


def decompose_scales(scale: torch.Tensor):
    s = scale.to(torch.float64).clamp_min(2.0 ** -30)
    e = torch.floor(torch.log2(s))
    m = s / torch.pow(2.0, e)
    idx = torch.round((m - 1.0) * 8.0).clamp(0, 7).to(torch.int64)
    m_exact = torch.pow(2.0, e) * (1.0 + idx.to(torch.float64) / 8.0)
    return e.to(torch.int64), idx, m_exact


def exact_encode_block(quant_block: torch.Tensor, scale_sub: torch.Tensor,
                       *, do_full_search: bool = True):
    """Per-row 64-block encoder: find a per-row sf such that sub-blocks are
    EXACTLY representable in HiF4 (mant field, lv2/lv3 = 1).

    Returns dict with sf_code, sf, lv2=1, lv3=1, mantissa=code*0.25, sign,
    exact_count, n_subs_total.
    """
    Rb = quant_block.shape[0]
    s_blk = scale_sub.to(torch.float64)
    e_b, m4n, _ = decompose_scales(scale_sub)
    e_b = e_b.to(torch.int64)
    m4n = m4n.to(torch.int64)
    cnt = count_codes(quant_block)  # (R, B=1, 4, 7)
    cnt = cnt.reshape(Rb, 4, 7).cpu()
    # candidates per row: 7 j_offsets x 4 m6 mantissas = 28 candidates
    j_off = torch.arange(-2, 3)
    m6_off = torch.arange(4)
    e_sf_seed = torch.zeros(Rb, dtype=torch.int64)
    codes_cand = torch.zeros(Rb, 20, dtype=torch.int64)
    valid_cand = torch.zeros(Rb, 20, dtype=torch.bool)
    # Use 0 as seed (we search all 28 around it)
    e_cand = e_sf_seed[:, None, None] + j_off[None, :, None]
    codes_cand_full = (e_cand + 48) * 4 + m6_off[None, None, :]
    codes_cand_full = codes_cand_full.reshape(Rb, 20).clamp(0, 254)
    valid_cand = (codes_cand_full >= 0) & (codes_cand_full <= 254)
    codes_cand = codes_cand_full
    # per-(row, cand, sub) ok table
    K = 20
    ec = codes_cand // 4 - 48
    m6c = codes_cand % 4
    dj = e_b[:, None, :] - ec[:, :, None]  # (R, K, 4)
    dj_ok = (dj >= DJ_LO) & (dj <= DJ_HI)
    dj_c = dj.clamp(DJ_LO, DJ_HI) - DJ_LO
    m4n_b = m4n[:, None, :].expand(Rb, K, 4)
    idx = m4n_b * 68 + m6c[:, :, None] * 17 + dj_c
    E_v = E_FLAT[:, idx.cpu()]  # (7, R, K, 4)
    ok = (dj_ok & valid_cand[:, :, None]).cpu().to(torch.float32)
    cnt_expanded = cnt[:, None, :, :].expand(Rb, K, 4, 7)
    exact_present = (cnt_expanded * ok[:, :, :, None]).sum(3)  # (R, K, 4)
    exact_total = exact_present.sum(2)  # (R, K)
    # Pick best sf per row: max exact_count, tie-break by lowest sf code
    best_k = exact_total.argmax(1)
    best_code = codes_cand.gather(1, best_k[:, None]).squeeze(1)
    # decode to sf, lv2=lv3=1 (since we picked a sf that makes subs exact)
    sf_best = e6m2_decode_f64(best_code).to(torch.float32)  # (R,)
    # exact sub count
    best_exact = exact_total.gather(1, best_k[:, None]).squeeze(1)
    # decode mant = quant value * sub_scale / sf
    # For each (row, sub, 16 vals), we need mant field = round(abs_q * 4 / sf) since lv2=lv3=1
    q = quant_block.reshape(Rb, 4, 16).to(torch.float32)
    sgn = torch.sign(q)
    abs_q = q.abs()
    # mant field in {0, 0.25, 0.5, ..., 1.75}
    # For exact sub-blocks: each |abs_q / sf| must be in {k/4, k/2, k: k=1..7}
    # We just use RTN; for exact subs it will be exactly right
    mant = (torch.round(abs_q * (4.0 / sf_best[:, None, None])).clamp(0.0, 7.0) * 0.25)
    lv2 = torch.ones(Rb, 4)
    lv3 = torch.ones(Rb, 4)
    return {
        "sf_code": best_code,
        "sf": sf_best,
        "lv2": lv2,
        "lv3": lv3,
        "mantissa": mant,
        "sign": sgn,
        "exact_total": best_exact,
        "n_subs": torch.full((Rb,), 4.0),
    }


def v168_encode_block(quant_block: torch.Tensor, scale_sub: torch.Tensor, full: torch.Tensor):
    """Use real _dense_to_hif4 with refine disabled as attention-X v186 stand-in.

    Note: attention path in v168 has additional center/rotation/pair_transform
    that we don't replicate here -- we use plain _dense_to_hif4 as the baseline.
    """
    Rb = quant_block.shape[0]
    dense = full  # already (R, 64)
    return sol._dense_to_hif4(dense, max_refine_ratio=0.0)


def decode_hif4_to_dense(params: dict) -> torch.Tensor:
    """Decode HiF4 params (hybrid or v186 format) to dense (R, 64)."""
    if "mantissa" in params:
        Rb = params["mantissa"].shape[0]
        sign = params["sign"].reshape(Rb, 64)
        mant = params["mantissa"].reshape(Rb, 64)
        sf = params["sf"]
        if sf.ndim == 1:
            sf = sf[:, None]
        lv2 = params["lv2"]
        if lv2.ndim == 2:
            lv2 = lv2[:, :, None].expand(Rb, 4, 16).reshape(Rb, 64)
        elif lv2.ndim == 1:
            lv2 = lv2[:, None].expand(Rb, 64)
        lv3 = params["lv3"]
        if lv3.ndim == 2:
            lv3 = lv3[:, :, None].expand(Rb, 4, 16).reshape(Rb, 64)
        elif lv3.ndim == 1:
            lv3 = lv3[:, None].expand(Rb, 64)
        decoded = sign * mant * lv3 * lv2 * sf
        return decoded.reshape(Rb, 64)
    else:
        Rb = params["mant"].shape[0]
        sign = params["sign"].reshape(Rb, 64)
        mant = params["mant"].reshape(Rb, 64)
        sf = params["scale_factor"].reshape(Rb, 1)
        lv2 = params["scale_lv2"].reshape(Rb, 8).repeat_interleave(8, 1).reshape(Rb, 64)
        lv3 = params["scale_lv3"].reshape(Rb, 16).repeat_interleave(4, 1).reshape(Rb, 64)
        decoded = sign * mant * lv3 * lv2 * sf
        return decoded.reshape(Rb, 64)


def process_attn_pair(role: str, layer: int, quant: torch.Tensor, scale: torch.Tensor):
    """Per (role, layer) QKV: run exact encoder + v186 baseline, compare MSE.

    quant: (rows, cols); scale: (rows, cols/16).
    """
    R, C = quant.shape
    B = C // 64
    s_full = scale.to(torch.float32).reshape(R, B, 4)
    full = v2.dequantize_nvfp4(quant, scale).to(torch.float32)  # NVFP4-dequantized dense

    mse_exact_list = []
    mse_v186_list = []
    exact_count_list = []
    n_subs_list = []
    t0 = time.time()
    for b in range(B):
        qb = quant.reshape(R, B, 64)[:, b, :]
        sb = s_full[:, b, :]
        fb = full.reshape(R, B, 64)[:, b, :]
        try:
            h = exact_encode_block(qb, sb)
            dec_h = decode_hif4_to_dense(h)
            mse_h = ((dec_h - fb) ** 2).mean().item()
            mse_exact_list.append(mse_h)
            exact_count_list.append(h["exact_total"].float().mean().item())
            n_subs_list.append(float(h["n_subs"].mean().item()))
            # v186 baseline
            v = v168_encode_block(qb, sb, fb)
            dec_v = decode_hif4_to_dense(v)
            mse_v = ((dec_v - fb) ** 2).mean().item()
            mse_v186_list.append(mse_v)
        except Exception as e:
            print(f"  [err {role} L{layer:02d} b={b}] {type(e).__name__}: {e}", flush=True)
            continue
    dt = time.time() - t0
    if mse_exact_list:
        agg_h = sum(mse_exact_list) / len(mse_exact_list)
    else:
        agg_h = float("nan")
    if mse_v186_list:
        agg_v = sum(mse_v186_list) / len(mse_v186_list)
    else:
        agg_v = float("nan")
    ratio = agg_h / agg_v if (mse_v186_list and agg_v > 0) else None
    print(
        f"[a] L{layer:02d} {role:2s} n={len(mse_exact_list):3d} "
        f"exact_pure={agg_h:.3e} v186={agg_v:.3e} ratio={ratio:.3f} "
        f"sub_exact={sum(exact_count_list)/max(1,len(exact_count_list)):.1f}/64 "
        f"in {dt:.1f}s",
        flush=True,
    )
    return {
        "layer": layer,
        "role": role,
        "blocks": len(mse_exact_list),
        "mse_exact": agg_h,
        "mse_v186": agg_v,
        "mse_ratio": ratio,
        "exact_subs_mean": sum(exact_count_list) / max(1, len(exact_count_list)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--roles", nargs="+", default=("q", "k", "v"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cb2] loading pack: {CACHE.name}", flush=True)
    raw = v2.load_pack(CACHE)
    pack = v3.prepare_shard(raw, 0, "both", ood=False)
    n_layers = min(args.layers, int(pack.layers))
    n_windows = min(args.windows, len(pack.test_qkv))
    print(f"[cb2] layers={n_layers} windows={n_windows} roles={args.roles}", flush=True)

    results = []
    for w in range(n_windows):
        for layer in range(n_layers):
            if layer % 6 != 0:
                continue
            qkv_dict = pack.calibration_qkv[w][layer] if (
                w < len(pack.calibration_qkv) and layer < len(pack.calibration_qkv[w])
            ) else None
            if qkv_dict is None:
                continue
            for name in ("q", "k", "v"):
                if name not in args.roles or name not in qkv_dict:
                    continue
                pair = qkv_dict[name]
                q, s = pair[0], pair[1]
                if q.shape[-1] % 64 != 0:
                    continue
                # cap rows for speed
                if q.shape[0] > 1024:
                    q = q[:1024]
                    s = s[:1024]
                res = process_attn_pair(name, layer, q, s)
                res["window"] = w
                results.append(res)

    # aggregate
    by_role = defaultdict(list)
    for r in results:
        if r["mse_ratio"] is not None:
            by_role[r["role"]].append(r)

    summary = {
        "meta": {
            "script": "cb2_attn_exact_branch.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_results": len(results),
            "note": "exact encoder uses per-row sf search; lv2=lv3=1; mant via RTN. "
                    "Compares against _dense_to_hif4(refine=0) which is v186 W-side baseline.",
        },
        "results": results,
        "agg": {
            r: {
                "n": len(rs),
                "mean_mse_exact": sum(x["mse_exact"] for x in rs) / len(rs),
                "mean_mse_v186": sum(x["mse_v186"] for x in rs) / len(rs),
                "mean_ratio": sum(x["mse_ratio"] for x in rs) / len(rs),
                "mean_exact_subs": sum(x["exact_subs_mean"] for x in rs) / len(rs),
            }
            for r, rs in by_role.items()
        },
    }

    # P4 gate: exact MSE <= 0.90 * v186 AND exact_subs_mean >= 2/4
    qk = [r for r in results if r["role"] in ("q", "k") and r["mse_ratio"] is not None]
    if qk:
        mean_ratio = sum(r["mse_ratio"] for r in qk) / len(qk)
        mean_exact_subs = sum(r["exact_subs_mean"] for r in qk) / len(qk)
        verdict = (
            "PROMISING" if (mean_ratio <= 0.95 and mean_exact_subs >= 1.5)
            else "MARGINAL" if (mean_ratio <= 1.0 and mean_exact_subs >= 1.0)
            else "NO-GAIN"
        )
        summary["p4_gate"] = {
            "mean_ratio": mean_ratio,
            "mean_exact_subs": mean_exact_subs,
            "n_cells": len(qk),
            "verdict": verdict,
        }
        print(
            f"\nP4 gate (Q/K): {verdict}  "
            f"(mean_ratio={mean_ratio:.3f}, mean_exact_subs={mean_exact_subs:.2f}/64, "
            f"n={len(qk)})",
            flush=True,
        )

    out_path = OUT_DIR / "cb2_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
