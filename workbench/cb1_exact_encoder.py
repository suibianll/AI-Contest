"""P1 unit proof: hybrid W pipeline with per-(row, 64-block) sf candidate grid.

Compares (product-level MSE) the cached v186 calibration state vs a hybrid arm
that replaces the sf-selection step with a 7x4 candidate grid search. Does NOT
modify solution.py. Read-only on cached proxy-v2 pack.

Mechanism (from P0 plan 2026-09-05-nvfp4-codebook-exact-conversion-plan §2, §3):
- Per (row, 64-block), enumerate 7 j-offsets x 4 m6 mantissas -> K=28 candidates.
  (P0 used K=112 including m6=0..7; here restricted to m6 in {0,1,2,3} of E6M2 =
  mantissa field 0..3, i.e. mantissa {1.0, 1.25, 1.5, 1.75}; same set as the
  E6M2 mantissas actually used in v186 refine path).
- For each candidate sf:
   * decode sf, lv2, lv3 from cands (lv2/lv3 greedy from max8/max4 with the new sf)
   * exactness check per sub-block: m4(s_b) matches m6(sf) AND j in J(c) for ALL
     present codes c in that sub-block
   * score = (exact_count, -MSE) lex order; pick argmax
- Hybrid arm keeps: sign (binary), lv2/lv3 (greedy with candidate sf),
  mantissa = RTN round to nearest valid mantissa for non-exact cells.
- v186 arm: same dense input, but uses _dense_to_hif4 with refine disabled
  (max_refine_ratio=0) to isolate the sf choice.

Pre-registered G1: hybrid product MSE <= 0.90 * v186 product MSE AND not worse on
>= 2/3 cells -> P2. (0.90, 1.0] -> plan revision once. > 1.0 -> close W side.

Run:
  python workbench/cb1_exact_encoder.py [--quick] [--layers N] [--roles ...]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from fractions import Fraction
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
OUT_DIR = ROOT / "artifacts" / "proxy_v3" / "cb1-exact-encoder-20260905" / "run-001"

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


def decode_nvfp4(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return dequantized NVFP4 weight in float32."""
    R, C = quant.shape
    assert C % 16 == 0
    B = C // 16
    q = quant.reshape(R, B, 16).to(torch.float32)
    s = scale.to(torch.float32).reshape(R, B, 1)
    return (q * s).reshape(R, C)


def decompose_scales(scale: torch.Tensor):
    """Per (row, sub-block): exponent e (int), mantissa idx m4 (0..7), float value."""
    s = scale.to(torch.float64).clamp_min(2.0 ** -30)
    e = torch.floor(torch.log2(s))
    m = s / torch.pow(2.0, e)
    idx = torch.round((m - 1.0) * 8.0).clamp(0, 7).to(torch.int64)
    m_exact = torch.pow(2.0, e) * (1.0 + idx.to(torch.float64) / 8.0)
    return e.to(torch.int64), idx, m_exact


def count_codes(quant: torch.Tensor):
    """counts shape (R, B, 4, 7) for code index in CODES.

    quant value mapping: -6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6
    We restrict to non-zero positive codes for table lookup; sign carried separately.
    """
    R, C = quant.shape
    B = C // 64
    q = quant.reshape(R, B, 4, 16)
    codes_pos = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=q.dtype, device=q.device)
    counts = torch.zeros(R, B, 4, 7, dtype=torch.int64)
    for ci, cv in enumerate(codes_pos.tolist()):
        counts[..., ci] = (q == cv).sum(-1)
    return counts


def hybrid_encode_block(quant_block: torch.Tensor, scale_sub: torch.Tensor,
                        sf_code: torch.Tensor, dense_block: torch.Tensor = None):
    """Given one 64-block (R=rows of this layer, 64 cols) and 4 sub-scales,
    pick a per-row sf_code (shape (R,)), and return mantissa+sign+lv2+lv3+sf.

    dense_block: (R, 64) the floating-point input to be quantized to HiF4.
        If None, fall back to quant*scale_sub (the NVFP4-dequantized value).
    """
    Rb, Cb = quant_block.shape  # Rb = rows of layer, Cb=64
    assert Cb == 64
    sf_val = e6m2_decode_f64(sf_code).to(torch.float32)  # (R,)
    # sub-scale mantissa idx (E4M3): shape (R, 4)
    e_b, m4n, s_b = decompose_scales(scale_sub)  # (R, 4) each
    e_b = e_b.to(torch.int64)
    m4n = m4n.to(torch.int64)
    # Use dense input (floating-point) for HiF4 quantization, not the NVFP4 code.
    if dense_block is None:
        dense_block = quant_block.to(torch.float32) * scale_sub[:, :, None].reshape(Rb, 64)
    q = dense_block.reshape(Rb, 4, 16).to(torch.float32)
    sgn = torch.sign(q)
    abs_q = q.abs()
    # energy-weighted relative error of each candidate (R, K)
    # K = 7 j_off x 4 m6 mantissas = 28 candidates; restrict to j in [-2, 2]
    j_off = torch.arange(-2, 3)  # 5
    m6_off = torch.arange(4)  # 4
    # sf exponent + 48*4 + mantissa
    e_sf = sf_code // 4 - 48  # (R,) int
    m6_sf = sf_code % 4        # (R,) int
    # candidate e for each (row, j, m6)
    e_cand = e_sf[:, None, None] + j_off[None, :, None]  # (R, 5, 4)
    codes_cand = (e_cand + 48) * 4 + m6_off[None, None, :]  # (R, 5, 4)
    codes_cand = codes_cand.reshape(Rb, 20).clamp(0, 254)
    # for each candidate, per sub: per code validity (R, K, 4) per code 7
    # exactness per (R, K, sub): all present codes exact
    cnt = torch.zeros(Rb, 4, 7, dtype=torch.int64)
    for ci, cv in enumerate(CODES):
        cnt[..., ci] = (abs_q.reshape(Rb, 4, 16) == cv).sum(-1)
    # build candidate e,m6 (R,K)
    K = 20
    ec = codes_cand // 4 - 48  # (R, K)
    m6c = codes_cand % 4       # (R, K)
    dj = e_b[:, None, :] - ec[:, :, None]  # (R, K, 4)
    dj_ok = (dj >= DJ_LO) & (dj <= DJ_HI)
    dj_c = dj.clamp(DJ_LO, DJ_HI) - DJ_LO
    m4n_b = m4n[:, None, :].expand(Rb, K, 4)  # (R, K, 4)
    idx = m4n_b * 68 + m6c[:, :, None] * 17 + dj_c  # (R, K, 4)
    E_v = E_FLAT[:, idx.cpu()]  # (7, R, K, 4)
    ER_v = ERR_FLAT[:, idx.cpu()]  # (7, R, K, 4)
    cnt_cpu = cnt.cpu()
    ok = (dj_ok & (codes_cand >= 0)[:, :, None] & (codes_cand <= 254)[:, :, None]).cpu().to(torch.float32)
    # exact count per (R, K, sub): sum_{code} cnt * ok
    # cnt (R, 4, 7) broadcast over K, ok (R, K, 4) per-sub: factor ok out
    # exact_present[r,k,sub] = sum_ci cnt[r,sub,ci] * ok[r,k,sub]
    # We need cnt with shape (R, K, 4, 7) for mul with ok[:, :, :, None] -> (R, K, 4, 7)
    cnt_expanded = cnt_cpu[:, None, :, :].expand(Rb, K, 4, 7)  # (R, K, 4, 7)
    exact_present = (cnt_expanded * ok[:, :, :, None]).sum(3)  # (R, K, 4)
    # per (R, K): exact_count = sum over subs
    exact_total = exact_present.sum(2)  # (R, K)
    # MSE per (R, K): sum over subs,codes of cnt * ER * sf2
    ER_perm = ER_v.permute(1, 2, 3, 0)  # (R, K, 4, 7)
    cnt_expanded2 = cnt_cpu[:, None, :, :].expand(Rb, K, 4, 7)
    err_rel = (cnt_expanded2 * ER_perm).sum((2, 3))  # (R, K)
    sf2 = (e6m2_decode_f64(codes_cand.cpu())) ** 2  # (R, K)
    err_act = err_rel.to(torch.float64) * sf2
    # total cells per (R, 4 subs, 16 vals) = 64
    total_cells = 64
    # tie-break by exact_total descending, err_act ascending
    # score = exact_total - 1e-9 * err_act (prefer more exact, lower MSE)
    score = exact_total.to(torch.float64) - 1e-12 * err_act
    best_k = score.argmax(dim=1)  # (R,)
    best_code = codes_cand.gather(1, best_k[:, None]).squeeze(1)  # (R,)
    # decode mantissa via RTN with the chosen sf, lv2/lv3 greedy
    sf_best = e6m2_decode_f64(best_code).to(torch.float32)  # (R,)
    max4 = abs_q.amax(dim=-1)  # (R, 4)
    max8 = max4.amax(dim=-1, keepdim=True)  # (R, 1)
    # lv2: per (row, sub) check if max4 >= 4*sf; all subs use the same lv2 (1 or 2)
    # per-row: max8 (R, 1) >= 4*sf_best[:, None] -> (R,)
    e2 = max8 >= (4.0 * sf_best[:, None])  # (R, 1)
    lv2_scalar = (1.0 + e2.to(torch.float32))  # (R, 1)
    lv2 = lv2_scalar.expand(Rb, 4)  # (R, 4) broadcast across subs
    # lv3: per (row, sub) check if max4 >= 2*sf*lv2_scalar
    e3 = max4 >= (2.0 * sf_best[:, None] * lv2_scalar)  # (R, 4)
    lv3 = (1.0 + e3.to(torch.float32))  # (R, 4)
    denom = sf_best[:, None, None] * lv2[:, :, None] * lv3[:, :, None]  # (R, 4, 16)
    mant = (torch.round(abs_q * (4.0 / denom)).clamp_(0.0, 7.0) * 0.25)
    return {
        "sf_code": best_code,
        "sf": sf_best,
        "lv2": lv2,
        "lv3": lv3,
        "mantissa": mant,
        "sign": sgn,
        "exact_total": exact_total.gather(1, best_k[:, None]).squeeze(1),
        "mse_rel": (err_act.gather(1, best_k[:, None]).squeeze(1) / max(1.0, total_cells)),
    }


def v186_encode_block(quant_block: torch.Tensor, scale_sub: torch.Tensor, full: torch.Tensor):
    """Use real _dense_to_hif4 with refine disabled to isolate sf choice."""
    # We need full block dense input -- here use quant*scale_sub as "dense"
    Rb = quant_block.shape[0]
    dense = (quant_block.reshape(Rb, 4, 16).to(torch.float32) * scale_sub[:, :, None]).reshape(Rb, 64)
    out = sol._dense_to_hif4(dense, max_refine_ratio=0.0)
    return out


def mse_hif4_decode(params: dict, dense_ref: torch.Tensor) -> float:
    """MSE between dense_ref (R, 64) and the dequantized HiF4 params.

    Accepts either:
      - hybrid format: {sign (R,4,16), mantissa (R,4,16), sf (R,), lv2 (R,4), lv3 (R,4)}
      - v186 _dense_to_hif4 format: {sign/mant (R,1,8,2,4), scale_factor (R,1,1,1,1),
        scale_lv2 (R,1,8,1,1), scale_lv3 (R,1,8,2,1)}
    """
    Rb = dense_ref.shape[0]
    if "mantissa" in params:
        sign = params["sign"]
        if sign.ndim == 2:
            sign = sign.reshape(Rb, 4, 16)
        mant = params["mantissa"]
        if mant.ndim == 2:
            mant = mant.reshape(Rb, 4, 16)
        sf = params["sf"]
        if sf.ndim == 1:
            sf = sf[:, None, None]
        elif sf.ndim == 2:
            sf = sf[:, :, None]
        lv2 = params["lv2"]
        if lv2.ndim == 1:
            lv2 = lv2[:, None]
        if lv2.ndim == 2:
            lv2 = lv2[:, :, None]
        lv3 = params["lv3"]
        if lv3.ndim == 1:
            lv3 = lv3[:, None]
        if lv3.ndim == 2:
            lv3 = lv3[:, :, None]
        decoded = sign * mant * lv3 * lv2 * sf
        decoded = decoded.reshape(Rb, 64)
    else:
        # v186 format: (R, 1, 8, 2, 4)
        sign = params["sign"].reshape(Rb, 64)
        mant = params["mant"].reshape(Rb, 64)
        sf = params["scale_factor"].reshape(Rb, 1)
        lv2 = params["scale_lv2"].reshape(Rb, 8)
        lv2 = lv2.repeat_interleave(8, dim=1).reshape(Rb, 64)  # (R, 64)
        lv3 = params["scale_lv3"].reshape(Rb, 16)
        lv3 = lv3.repeat_interleave(4, dim=1).reshape(Rb, 64)  # (R, 64)
        # need sf broadcast to (R, 64); lv2, lv3 broadcast to per-element granularity
        decoded = sign * mant * lv3 * lv2 * sf
    return ((decoded - dense_ref) ** 2).mean().item()


def process_layer(role: str, layer: int, quant: torch.Tensor, scale: torch.Tensor, full: torch.Tensor, *, do_v186: bool = True):
    """Returns per-block mse hybrid vs v186, aggregate stats."""
    R, C = quant.shape
    B = C // 64
    # NVFP4 scale shape is (R, C/16).  Each 64-block has 4 NVFP4 16-blocks, so
    # reshape to (R, B, 4 subs, 1) -- one scale per 16-block; the 4 sub-blocks of
    # a HiF4 64-block each carry their own NVFP4 scale.
    s_full = scale.to(torch.float32).reshape(R, B, 4)  # (R, B, 4) one per NVFP4 16-block
    # full dense block (R, B, 64)
    full_blk = full.reshape(R, B, 64)
    q_blk = quant.reshape(R, B, 64)
    s_blk = s_full  # (R, B, 4)
    # process per block, accumulate MSE
    mse_h_list = []
    mse_v_list = []
    ex_h_list = []
    n_blocks_done = 0
    t0 = time.time()
    for b in range(B):
        # q_blk[:, b, :] (R, 64); s_blk[:, b, :] (R, 4); full_blk[:, b, :] (R, 64)
        qb = q_blk[:, b, :]
        sb = s_blk[:, b, :]
        fb = full_blk[:, b, :]
        try:
            h = hybrid_encode_block(qb, sb, _default_candidate_sf(fb), dense_block=fb)
            mse_h = mse_hif4_decode(h, fb)
            ex_h_list.append(h["exact_total"].float().mean().item())
            mse_h_list.append(mse_h)
            if do_v186:
                v = v186_encode_block(qb, sb, fb)
                mse_v = mse_hif4_decode(v, fb)
                mse_v_list.append(mse_v)
        except Exception as e:
            import traceback
            print(f"  [err L{layer:02d} {role} b={b}] {type(e).__name__}: {e}", flush=True)
            print(f"  TRACEBACK:\n{traceback.format_exc()}", flush=True)
            continue
        n_blocks_done += 1
    dt = time.time() - t0
    if mse_h_list:
        agg_h = float(sum(mse_h_list) / len(mse_h_list))
    else:
        agg_h = float("nan")
    if mse_v_list:
        agg_v = float(sum(mse_v_list) / len(mse_v_list))
    else:
        agg_v = float("nan")
    ratio = agg_h / agg_v if (mse_v_list and agg_v > 0) else None
    return {
        "layer": layer,
        "role": role,
        "blocks": n_blocks_done,
        "mse_hybrid": agg_h,
        "mse_v186": agg_v,
        "mse_ratio_h_over_v": ratio,
        "exact_mean_per_block": float(sum(ex_h_list) / max(1, len(ex_h_list))),
        "duration_s": dt,
    }


def _default_candidate_sf(full_block: torch.Tensor) -> torch.Tensor:
    """Per-row candidate sf seed = standard E6M2 scale of |full_block| (R, 64)."""
    amax = full_block.abs().amax(dim=-1).to(torch.float32)  # (R,)
    code, _ = sol._standard_e6m2_scale(amax)
    return code.to(torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--layers", type=int, default=0,
                    help="0 = auto (3 if --quick, 8 otherwise)")
    ap.add_argument("--roles", nargs="+", default=None,
                    help="subset of {fc_gate, fc_up, proj, q, k, v, o}")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cb1] loading pack: {CACHE.name}", flush=True)
    raw = v2.load_pack(CACHE)
    roles_all = ("fc_gate", "fc_up", "proj", "q", "k", "v", "o")
    roles = tuple(args.roles) if args.roles else roles_all
    n_layers_target = args.layers if args.layers > 0 else (3 if args.quick else 8)
    print(f"[cb1] target layers={n_layers_target} roles={roles}", flush=True)

    results = []
    for shard_idx in (0,):  # single shard for P1 unit proof
        pack = v3.prepare_shard(raw, shard_idx, "both", ood=False)
        n_layers_total = int(pack.layers) if hasattr(pack, "layers") else len(pack.weights)
        for layer in range(min(n_layers_target, n_layers_total)):
            for role in roles:
                if role not in pack.weights[layer]:
                    continue
                q, s = pack.weights[layer][role]
                full = v2.dequantize_nvfp4(q, s).to(torch.float32)
                t0 = time.time()
                res = process_layer(role, layer, q, s, full, do_v186=True)
                dt = time.time() - t0
                ratio = res["mse_ratio_h_over_v"]
                ratio_s = f"{ratio:.3f}" if ratio is not None else "  n/a"
                ex_s = f"{res['exact_mean_per_block']:.3f}"
                print(
                    f"[w] L{layer:02d} {role:8s} n={res['blocks']:3d} "
                    f"hybrid={res['mse_hybrid']:.3e} v186={res['mse_v186']:.3e} "
                    f"ratio={ratio_s} ex={ex_s} "
                    f"in {dt:.1f}s",
                    flush=True,
                )
                results.append(res)

    # aggregate per role
    by_role: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r["mse_ratio_h_over_v"] is not None:
            by_role[r["role"]].append(r["mse_ratio_h_over_v"])

    summary = {
        "meta": {
            "script": "cb1_exact_encoder.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_results": len(results),
            "candidate_grid": "j in [-2,2] x m6 in {0..3}  (K=20 per row)",
        },
        "results": results,
        "agg": {
            role: {
                "n": len(vs),
                "mean_ratio": sum(vs) / len(vs),
                "min_ratio": min(vs),
                "max_ratio": max(vs),
            }
            for role, vs in by_role.items()
        },
    }

    # G1: mean ratio <= 0.90 AND not worse on >=2/3 cells (worst 1/3 cells allowed to be worse)
    fc_roles = ("fc_gate", "fc_up", "proj")
    fc_ratios = [r["mse_ratio_h_over_v"] for r in results
                 if r["role"] in fc_roles and r["mse_ratio_h_over_v"] is not None]
    if fc_ratios:
        mean_ratio = sum(fc_ratios) / len(fc_ratios)
        sorted_r = sorted(fc_ratios)
        n_better_or_equal = sum(1 for x in fc_ratios if x <= 1.0)
        verdict = (
            "PASS->P2" if mean_ratio <= 0.90 and n_better_or_equal >= max(1, 2 * len(fc_ratios) // 3)
            else "REVISE" if mean_ratio <= 1.0
            else "CLOSE_W->P4"
        )
        summary["g1"] = {
            "mean_ratio": mean_ratio,
            "n_cells": len(fc_ratios),
            "n_better_or_equal": n_better_or_equal,
            "verdict": verdict,
        }
        print(f"\nG1 verdict: {verdict}  (mean_ratio={mean_ratio:.3f}, "
              f"better_or_equal={n_better_or_equal}/{len(fc_ratios)})", flush=True)

    out_path = OUT_DIR / "cb1_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
