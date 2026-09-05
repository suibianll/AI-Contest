"""P0 data-fact proof for the NVFP4 codebook-exact conversion plan (2026-09-05).

Offline, read-only.  No version produced, no official timing recorded.

Measured on the cached qwen2.5-0.5b proxy-v2 pack:
  F0  structure sanity: quant values in E2M1; scales decompose exactly as
      (exponent, mantissa idx 0..7); scale shape == (rows, cols/16).
  F1  sub-scale alignment within 64-blocks: mantissa pattern stats, block
      mode-count histogram, exponent spread, cross-row/pair stability,
      even-eighth (E6M2-representable) fraction, subnormal fraction.
  F2  exact-value fraction ceiling: full E6M2 sf-lattice candidate search per
      64-block (per sub-block exponent window j in [-4, 2] x 4 mantissas,
      K=112 candidates), table-driven validity  c * (m4/m6) * 2^dj in S.
  F3  code-set statistics per sub-block.
  F4  encoder-level MSE comparison holding the snap machinery fixed (free-a
      per-value snap): baseline sf = _standard_e6m2_scale(amax/7) (real
      solution function, BF16 intermediate) vs best candidate sf; plus a real
      _dense_to_hif4 (refine off) anchor on a subset to calibrate the
      free-a optimism.

G0 gate (pre-registered in the plan) is evaluated on the W-side fc roles.

Usage: python workbench/cb0_codebook_proof.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys_path_added: list[str] = []
for p in (str(ROOT / "evaluator"), str(ROOT)):
    if p not in __import__("sys").path:
        __import__("sys").path.insert(0, p)
        sys_path_added.append(p)

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import solution as sol  # noqa: E402

CACHE = ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt"
OUT = ROOT / "artifacts" / "proxy_v3" / "cb0-codebook-proof-20260905" / "run-001"

CODES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
S_FRACTIONS = sorted(
    {Fraction(k, 4) for k in range(1, 8)}
    | {Fraction(k, 2) for k in range(1, 8)}
    | {Fraction(k, 1) for k in range(1, 8)}
)
S_FLOATS = torch.tensor([float(s) for s in S_FRACTIONS], dtype=torch.float64)
S_SET = set(S_FRACTIONS)
M4 = [Fraction(8 + i, 8) for i in range(8)]
M6 = [Fraction(4 + i, 4) for i in range(4)]
DJ_LO, DJ_HI = -8, 8
BIG = 1.0e30

# sampling plan (global layer indices)
FC_LAYER_MOD = 3          # fc roles: every 3rd layer
QKVO_LAYER_MOD = 6        # q/k/v/o: every 6th layer
X_WINDOWS = 2
X_LAYER_MOD = 8
X_ROW_CAP = 256
ANCHOR_LAYERS = 2         # real _dense_to_hif4 anchor layers for fc roles
CHUNK = 2048


def build_tables() -> tuple[torch.Tensor, torch.Tensor]:
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
E_FLAT = E_TABLE.reshape(7, -1)            # (7, 544) float32
ERR_FLAT = ERR_TABLE.reshape(7, -1).to(torch.float32)  # (7, 544) float32


def e6m2_decode_f64(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(0, 254)
    e = torch.bitwise_right_shift(c, 2).to(torch.float64) - 48.0
    m = 1.0 + (c & 3).to(torch.float64) * 0.25
    return torch.pow(2.0, e) * m


def decompose_scale(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    s = scale.to(torch.float64).clamp_min(2.0**-30)
    e = torch.floor(torch.log2(s))
    m = s / torch.pow(2.0, e)
    idx = torch.round((m - 1.0) * 8.0).clamp(0, 7).to(torch.int64)
    m_exact = torch.pow(2.0, e) * (1.0 + idx.to(torch.float64) / 8.0)
    resid = ((s - m_exact).abs() / s).max().item()
    return e.to(torch.int64), idx, float(resid)


def code_counts(quant: torch.Tensor) -> torch.Tensor:
    """counts[..., ci] = occurrences of CODES[ci] per (row, 64-block, sub)."""
    R, C = quant.shape
    B = C // 64
    a = quant.abs().to(torch.float32).reshape(-1)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    idx = torch.bucketize(a, boundaries).to(torch.int64)  # 0=zero, 1..7 codes
    rows = R * B * 4
    counts = torch.zeros(rows, 8, dtype=torch.int32)
    counts.scatter_add_(1, idx.reshape(rows, 16), torch.ones_like(idx.reshape(rows, 16), dtype=torch.int32))
    counts = counts.reshape(R, B, 4, 8)
    return counts[..., 1:].to(torch.int32)  # drop zero bin -> (R,B,4,7)


def analyze_pair(
    quant: torch.Tensor,
    scale: torch.Tensor,
    *,
    do_full: bool = True,
    real_anchor: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    R, C = int(quant.shape[0]), int(quant.shape[1])
    if C % 64 != 0:
        return {"skipped": f"last dim {C} not divisible by 64"}
    B = C // 64
    e_sub, m4_sub, resid = decompose_scale(scale)
    out["scale_decomp_max_resid"] = resid
    qv = quant.to(torch.float32)
    uniq_ok = bool(
        torch.isin(
            qv.abs(),
            torch.tensor([0.0] + list(CODES), dtype=torch.float32),
        ).all().item()
    )
    out["quant_in_e2m1"] = uniq_ok
    out["shape"] = [R, C]

    counts = code_counts(quant)  # (R,B,4,7) int32
    zero_cnt = 16 * 4 - counts.sum(-1)  # per (R,B,4) zeros among 16
    nonzero_blk = counts.sum((2, 3))  # (R,B) nonzero values per 64-block
    out["nonzero_frac"] = float(nonzero_blk.sum() / (R * B * 64))
    out["zero_sub_frac"] = float((zero_cnt == 16).float().mean())

    # ---- F1: alignment structure ----
    m4_blk = m4_sub.reshape(R, B, 4)
    e_blk = e_sub.reshape(R, B, 4)
    mode_cnt = torch.zeros(R * B, 8, dtype=torch.int32).scatter_add_(
        1, m4_blk.reshape(-1, 4).to(torch.int64), torch.ones(R * B, 4, dtype=torch.int32)
    ).amax(1)
    out["f1"] = {
        "p_all4_same_mantissa": float((mode_cnt == 4).float().mean()),
        "mode_count_hist": [float((mode_cnt == k).float().mean()) for k in (1, 2, 3, 4)],
        "even_m4_frac": float((m4_sub % 2 == 0).float().mean()),
        "subnormal_scale_frac": float((e_sub < -6).float().mean()),
        "exp_spread_p50": float(torch.quantile((e_blk.amax(-1) - e_blk.amin(-1)).to(torch.float32), 0.5)),
        "exp_spread_p90": float(torch.quantile((e_blk.amax(-1) - e_blk.amin(-1)).to(torch.float32), 0.9)),
    }
    # cross-row pair stability (sub-block columns with equal mantissa across rows)
    m4_cols = m4_sub  # (R, C/16)
    ncol = m4_cols.shape[1]
    if ncol >= 2 and R >= 8:
        a = m4_cols.to(torch.int16)
        pair_eq = []
        step = max(1, ncol // 32)
        for j1 in range(0, ncol, step):
            for j2 in range(j1 + 1, ncol, step):
                pair_eq.append((a[:, j1] == a[:, j2]).float().mean().item())
        pair_t = torch.tensor(pair_eq)
        out["f1"]["pair_m4_eq_mean"] = float(pair_t.mean())
        out["f1"]["pair_m4_eq_p90"] = float(torch.quantile(pair_t, 0.9))

    # ---- F3: code stats ----
    distinct = (counts > 0).sum(-1).to(torch.float32)  # (R,B,4)
    out["f3"] = {
        "distinct_codes_per_sub_mean": float(distinct.mean()),
        "distinct_codes_per_sub_p90": float(torch.quantile(distinct.reshape(-1), 0.9)),
        "code_freq": [
            float(counts[..., ci].sum()) / float(max(1, counts.sum()))
            for ci in range(7)
        ],
    }

    # ---- F2/F4 ----
    if not do_full:
        return out

    s_blk = scale.to(torch.float64).reshape(R, B, 4)
    cnt = counts.reshape(R * B, 4, 7).to(torch.float32)
    m4n = m4_blk.reshape(R * B, 4)
    en = e_blk.reshape(R * B, 4)
    s_n = s_blk.reshape(R * B, 4)
    N = R * B

    # energy per block (float64)
    code_t = torch.tensor(CODES, dtype=torch.float64)
    vals2 = (s_n[:, :, None] * code_t[None, None, :]) ** 2  # (N,4,7)
    energy = (vals2 * cnt.to(torch.float64)).sum((1, 2))  # (N,)
    nonzero_n = cnt.sum((1, 2))  # (N,)

    # baseline sf (real solution function, BF16 intermediate)
    mc = qv.abs().reshape(R, B, 4, 16).amax(-1)  # (R,B,4)
    amax = (mc.to(torch.float64) * s_blk).amax(-1).to(torch.float32)  # (R,B)
    base_code, _ = sol._standard_e6m2_scale(amax)
    base_code = base_code.reshape(-1).to(torch.int64)  # (N,)

    # candidates: per sub-block, j in [-4..2] x 4 mantissas
    j_off = torch.arange(-4, 3)  # 7
    m6_off = torch.arange(4)
    # shapes: en (N, 4), j_off (7), m6_off (4)  -> candidates (N, 4 subs, 7 j, 4 m6)
    e_cand = en[:, :, None, None] + j_off[None, None, :, None]  # (N, 4, 7, 1)
    codes_cand = (e_cand + 48) * 4 + m6_off[None, None, None, :]  # (N, 4, 7, 4)
    codes_cand = codes_cand.reshape(N, 112)
    cand_valid = (codes_cand >= 0) & (codes_cand <= 254)

    # baseline gather (vectorized, small)
    m6b = base_code % 4
    eb = base_code // 4 - 48
    djb = en - eb[:, None]  # (N,4)
    djb_ok = (djb >= DJ_LO) & (djb <= DJ_HI)
    djb_c = djb.clamp(DJ_LO, DJ_HI) - DJ_LO
    idx_b = m4n * 68 + m6b[:, None] * 17 + djb_c  # (N,4)
    E_b = E_FLAT[:, idx_b]  # (7,N,4)
    ERR_b = ERR_FLAT[:, idx_b]
    base_code_safe = ((base_code >= 0) & (base_code <= 254)).to(torch.float32)  # (N,)
    ok_b = djb_ok.to(torch.float32) * base_code_safe[:, None]  # (N,4)
    # E_b (7, N, 4), ok_b (N, 4): ok_b[None] -> (1, N, 4), broadcasts to (7, N, 4)
    Eb_w = E_b * ok_b[None]  # (7, N, 4)
    Er_w = ERR_b  # (7, N, 4)
    # permute -> (N, 4, 7), mul cnt (N, 4, 7), sum over (sub=1, code=2)
    base_exact = (cnt * Eb_w.permute(1, 2, 0)).sum((1, 2))
    base_err_rel = (cnt * Er_w.permute(1, 2, 0)).sum((1, 2)) + BIG * (1.0 - ok_b).sum(-1)
    sf2_b = e6m2_decode_f64(base_code.clamp(0, 254)) ** 2
    base_err_act = base_err_rel.to(torch.float64) * sf2_b
    base_err_act = torch.where(energy > 0, base_err_act, torch.zeros_like(base_err_act))

    best_exact = torch.zeros(N)
    exact_at_best_err = torch.zeros(N)
    err_best = torch.full((N,), float("inf"))
    best_code = torch.zeros(N, dtype=torch.int64)
    for n0 in range(0, N, CHUNK):
        sl = slice(n0, min(N, n0 + CHUNK))
        cc = codes_cand[sl]
        cv = cand_valid[sl]
        m6n_ = cc % 4
        ecn_ = cc // 4 - 48
        dj = en[sl][:, None, :] - ecn_[:, :, None]  # (n,K,4)
        dj_ok = (dj >= DJ_LO) & (dj <= DJ_HI)
        dj_c = dj.clamp(DJ_LO, DJ_HI) - DJ_LO
        idx = m4n[sl][:, None, :] * 68 + m6n_[:, :, None] * 17 + dj_c  # (n,K,4)
        E_v = E_FLAT[:, idx]  # (7,n,K,4) code,block,cand,sub
        ER_v = ERR_FLAT[:, idx]
        ok = (dj_ok & cv[:, :, None]).to(torch.float32)  # (n,K,4) block,cand,sub
        cnt_c = cnt[sl]  # (n, 4, 7) block,sub,code
        # permute E_v -> (n, K, sub, code); mul with ok[None] broadcast on code dim; then sum over (sub, code)
        Ev_p = E_v.permute(1, 2, 3, 0) * ok[:, :, :, None]  # (n,K,sub,code)
        ex = (cnt_c[:, None, :, :] * Ev_p).sum((2, 3))  # (n, K)
        er = (cnt_c[:, None, :, :] * ER_v.permute(1, 2, 3, 0)).sum((2, 3)) + BIG * (1.0 - ok).sum(-1)
        sf2 = e6m2_decode_f64(cc.clamp(0, 254)) ** 2  # (n,K)
        err_act = er.to(torch.float64) * sf2
        en_mask = (energy[sl] > 0).to(torch.float64)
        err_act = torch.where(en_mask[:, None] > 0, err_act, torch.zeros_like(err_act))
        bi = err_act.argmin(1)
        ar = torch.arange(err_act.shape[0])
        best_exact[sl] = ex.max(1).values
        err_best[sl] = err_act[ar, bi]
        exact_at_best_err[sl] = ex[ar, bi]
        best_code[sl] = cc[ar, bi]

    out["f2"] = {
        "exact_frac_best_nonzero": float(best_exact.sum() / max(1.0, nonzero_n.sum())),
        "exact_frac_best_perblk_p50": float(torch.quantile((best_exact / nonzero_n.clamp_min(1)).to(torch.float32), 0.5)),
        "exact_frac_best_perblk_p90": float(torch.quantile((best_exact / nonzero_n.clamp_min(1)).to(torch.float32), 0.9)),
        "exact_frac_at_mse_best": float(exact_at_best_err.sum() / max(1.0, nonzero_n.sum())),
        "baseline_exact_frac_nonzero": float(base_exact.sum() / max(1.0, nonzero_n.sum())),
    }
    out["f4"] = {
        "energy_total": float(energy.sum()),
        "table_mse_baseline": float(base_err_act.sum()),
        "table_mse_best_sf": float(err_best.sum()),
        "table_mse_ratio_best_over_baseline": float(
            err_best.sum() / max(1e-300, base_err_act.sum())
        ),
        "rel_mse_baseline": float(base_err_act.sum() / max(1e-300, energy.sum())),
        "rel_mse_best_sf": float(err_best.sum() / max(1e-300, energy.sum())),
    }

    if real_anchor:
        dense = v2.dequantize_nvfp4(quant, scale).to(torch.float32)
        params = sol._dense_to_hif4(dense)
        dense_hat = sol._dequantize_hif4(params).to(torch.float32)
        mse_real = float((dense_hat - dense).square().sum())
        energy_real = float(dense.square().sum())
        out["f4"]["real_dense_mse_refine_off"] = mse_real
        out["f4"]["real_dense_energy"] = energy_real
        out["f4"]["real_rel_mse_refine_off"] = mse_real / max(1e-300, energy_real)
        out["f4"]["anchor_real_over_table_baseline"] = float(
            mse_real / max(1e-300, base_err_act.sum())
        )
    return out


def iter_weight_pairs(pack: Any):
    for layer in range(pack.layers):
        roles = pack.weights[layer]
        if not roles:
            continue
        for role in sorted(roles):
            yield layer, role, roles[role]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    fc_mod = 6 if args.quick else FC_LAYER_MOD
    qkvo_mod = 12 if args.quick else QKVO_LAYER_MOD

    t0 = time.time()
    raw = v2.load_pack(CACHE)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"meta": {"cache": str(CACHE), "quick": args.quick}}

    w_results: list[dict[str, Any]] = []
    for shard in range(6):
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        for layer, role, pair in iter_weight_pairs(pack):
            is_fc = role in ("fc_gate", "fc_up", "proj")
            mod = fc_mod if is_fc else qkvo_mod
            if layer % mod != 0:
                continue
            do_anchor = is_fc and role == "fc_gate" and layer < 2 * fc_mod
            t1 = time.time()
            res = analyze_pair(pair[0], pair[1], do_full=True, real_anchor=do_anchor)
            res.update({"layer": layer, "role": role, "kind": "weight"})
            w_results.append(res)
            print(
                f"[w] L{layer:02d} {role:8s} blocks_done in {time.time()-t1:5.1f}s "
                f"exact={res.get('f2', {}).get('exact_frac_best_nonzero', float('nan')):.4f} "
                f"mse_ratio={res.get('f4', {}).get('table_mse_ratio_best_over_baseline', float('nan')):.4f}",
                flush=True,
            )
        del pack
    report["weights"] = w_results

    # ---- X side ----
    x_results: list[dict[str, Any]] = []
    for shard in range(2):
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        for role in ("fc_gate", "fc_up", "proj"):
            act = pack.test_activations.get(role)
            if not act:
                continue
            for w in range(min(X_WINDOWS, len(act))):
                for layer in range(pack.layers):
                    if layer % X_LAYER_MOD != 0 or w >= len(act[w]) or layer >= len(act[w]):
                        continue
                    pair = act[w][layer]
                    if pair is None or (isinstance(pair, tuple) and pair[0] is None):
                        continue
                    q, s = pair[0], pair[1]
                    q = q.reshape(q.shape[0], -1)[:X_ROW_CAP]
                    s = s.reshape(s.shape[0], -1)[:X_ROW_CAP]
                    res = analyze_pair(q, s, do_full=True)
                    res.update({"layer": layer, "role": role, "window": w, "kind": "activation"})
                    x_results.append(res)
        del pack
    report["activations"] = x_results

    # ---- QKV side ----
    qkv_results: list[dict[str, Any]] = []
    for shard in range(1):
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        for w in range(min(1, len(pack.test_qkv))):
            for layer in range(pack.layers):
                if layer % X_LAYER_MOD != 0 or layer >= len(pack.test_qkv[w]):
                    continue
                triple = pack.test_qkv[w][layer]
                if triple is None:
                    continue
                for name, pair in zip(("q", "k", "v"), triple):
                    q, s = pair[0], pair[1]
                    q = q.reshape(-1, q.shape[-1])[:X_ROW_CAP]
                    s = s.reshape(-1, s.shape[-1])[:X_ROW_CAP]
                    if q.shape[-1] % 64 != 0:
                        qkv_results.append({"kind": "qkv", "name": name, "layer": layer, "skipped": f"last dim {q.shape[-1]}"})
                        continue
                    res = analyze_pair(q, s, do_full=True)
                    res.update({"layer": layer, "role": name, "window": w, "kind": "qkv"})
                    qkv_results.append(res)
        del pack
    report["qkv"] = qkv_results

    # ---- G0 aggregation (W-side fc roles) ----
    fc = [r for r in w_results if r.get("role") in ("fc_gate", "fc_up", "proj")]
    if fc:
        agg_exact = float(
            sum(r["f2"]["exact_frac_best_nonzero"] for r in fc) / len(fc)
        )
        agg_ratio = float(
            sum(r["f4"]["table_mse_best_sf"] for r in fc)
            / max(1e-300, sum(r["f4"]["table_mse_baseline"] for r in fc))
        )
        per_role: dict[str, float] = {}
        for role in ("fc_gate", "fc_up", "proj"):
            sub = [r for r in fc if r["role"] == role]
            if sub:
                per_role[role] = {
                    "exact_frac": float(sum(r["f2"]["exact_frac_best_nonzero"] for r in sub) / len(sub)),
                    "mse_ratio": float(
                        sum(r["f4"]["table_mse_best_sf"] for r in sub)
                        / max(1e-300, sum(r["f4"]["table_mse_baseline"] for r in sub))
                    ),
                }
        report["g0"] = {
            "agg_exact_frac_best_nonzero": agg_exact,
            "agg_mse_ratio_best_over_baseline": agg_ratio,
            "per_role": per_role,
            "verdict": (
                "PASS->P1" if (agg_exact >= 0.20 and agg_ratio <= 0.85)
                else ("ATTENTION-PIVOT" if (agg_exact >= 0.10 or agg_ratio <= 0.95) else "CLOSE")
            ),
        }

    # x/qkv aggregates
    for key, res_list in (("x_agg", x_results), ("qkv_agg", qkv_results)):
        ok = [r for r in res_list if "f2" in r]
        if ok:
            report[key] = {
                "exact_frac_best_nonzero_mean": float(
                    sum(r["f2"]["exact_frac_best_nonzero"] for r in ok) / len(ok)
                ),
                "mse_ratio_mean": float(
                    sum(r["f4"]["table_mse_best_sf"] for r in ok)
                    / max(1e-300, sum(r["f4"]["table_mse_baseline"] for r in ok))
                ),
            }

    report["meta"]["elapsed_s"] = time.time() - t0
    out_path = OUT / "cb0_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nreport -> {out_path}  ({report['meta']['elapsed_s']:.0f}s)")
    if "g0" in report:
        g = report["g0"]
        print(f"G0: exact_frac={g['agg_exact_frac_best_nonzero']:.4f} "
              f"mse_ratio={g['agg_mse_ratio_best_over_baseline']:.4f} -> {g['verdict']}")


if __name__ == "__main__":
    main()
