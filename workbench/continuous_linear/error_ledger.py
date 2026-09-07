"""Linear error ledger (research-loop §2.2, R1).

Cells per weight state (layer, role), denominator = MSE_STD of the SAME
calibration folds (standard HiF4 full output vs dense X@W^T):

  E1 = ||Y - Xh @ W_cont^T||^2 / MSE_STD   (continuous solve, NO legal projection;
       the L23b block loop applied with every block's continuous delta)
  E2 = ||Xh @ (W_cont - W_legal)^T||^2 / MSE_STD  (legal projection cost;
       W_legal = final five-field decode from the candidate artifact)
  E3 = 1 - gain  from the 4B panel JSONs (real dynamic API, all 336 cases)
  E4 = E3 - E1 - E2   (calibration<->deployment mismatch, may be negative)

E1/E2 are computed on shard0 states only (the research-loop §2.5 minimal run
for missing cells); E3 covers the full panel.  E1/E2 reuse the candidate and
parent same-SHA calibration artifacts (no API re-run, no new model forward),
CPU-only, one artifact pair in memory at a time.  The continuous loop uses the
candidate's own _l23_block_solve and fold weights, so W_cont matches the
deployed candidate's internal solve.

Output:
  artifacts/continuous/linear/error_ledger_<date>.json
  artifacts/continuous/linear/error_ledger_<date>.md

Usage:
    .venv/Scripts/python.exe workbench/continuous_linear/error_ledger.py
"""

from __future__ import annotations

import gc
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "evaluator"))
import official_eval as v2  # noqa: E402
import reference_hif4 as ref  # noqa: E402

DENSE_CACHE = os.path.join(ROOT, "artifacts", "official_eval", "cache", "qwen3.5-4b-proxy-v2.pt")
CALIB_DIR = os.path.join(ROOT, "artifacts", "official_eval", "cache", "proxy-v3-calibration")
CAND_SHA = "13639fb22976b2c7"
PAR_SHA = "acb16f764db80eda"
CAND_SOL = os.path.join(HERE, "l23-residual-subspace", "candidate", "solution.py")
PAR_SOL = os.path.join(ROOT, "solutions", "v162_linear_l4-v189-linear-exact_officialNA_timeNA", "solution.py")
PANEL_DIR = os.path.join(ROOT, "artifacts", "proxy_v3", "continuous", "linear",
                         "l23-residual-subspace", "allrow-six-shard")
OUT_DIR = os.path.join(ROOT, "artifacts", "continuous", "linear")
CALIBRATION_FOLDS = (0, 1)
DATE = "2026-09-08"
SHARD0_ONLY = True


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_solution(path):
    spec = importlib.util.spec_from_file_location("ledger_sol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_artifacts(sha_prefix):
    out = []
    for path in sorted(glob.glob(os.path.join(CALIB_DIR, f"{sha_prefix}-linear-*.pt"))):
        art = torch.load(path, map_location="cpu", weights_only=True)
        model = art.get("identity", {}).get("model_revision") or ""
        keys = sorted((int(i["layer"]), str(i["role"])) for i in art.get("weight_states", []))
        del art
        gc.collect()
        if "Qwen3.5-4B" in model and keys:
            out.append((path, keys))
    return out


def load_params_only(path, keys):
    art = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for item in art.get("weight_states", []):
        key = (int(item["layer"]), str(item["role"]))
        if key in keys:
            out[key] = dict(item["params"])
    del art
    gc.collect()
    return out


def load_state_params(path, keys):
    art = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for item in art.get("weight_states", []):
        key = (int(item["layer"]), str(item["role"]))
        if key in keys:
            out[key] = (item["state"], dict(item["params"]))
    del art
    gc.collect()
    return out


def panel_e3():
    gains = {}
    for shard in range(6):
        for label in ("candidate", "baseline"):
            path = os.path.join(PANEL_DIR, label, f"{label}-linear-shard{shard}.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data.get("results", []):
                for case in item.get("case_scores", {}).get("linear", []):
                    key = (int(case["layer"]), str(case["role"]))
                    gains.setdefault(label, {}).setdefault(key, []).append(float(case["gain"]))
    return {
        label: {key: float(sum(v) / len(v)) for key, v in by_state.items() if v}
        for label, by_state in gains.items()
    }


def main():
    torch.manual_seed(0)
    cand_sha = sha_of(CAND_SOL)
    par_sha = sha_of(PAR_SOL)
    print(f"candidate SHA: {cand_sha[:16]} (expect {CAND_SHA})")
    print(f"parent    SHA: {par_sha[:16]} (expect {PAR_SHA})")

    raw = torch.load(DENSE_CACHE, map_location="cpu", weights_only=False)
    weights = raw["weights"]
    cal_act = raw["calibration_activations"]
    roles = tuple(raw["roles"])
    del raw
    gc.collect()

    cand_sol = load_solution(CAND_SOL)
    par_sol = load_solution(PAR_SOL)

    cand_artifacts = list_artifacts(CAND_SHA)
    par_artifacts = list_artifacts(PAR_SHA)
    print(f"candidate artifacts: {len(cand_artifacts)}; parent artifacts: {len(par_artifacts)}")

    target_keys = None
    if SHARD0_ONLY:
        target_keys = set(cand_artifacts[0][1])
        print(f"shard0-only E1/E2: {len(target_keys)} states")

    e3 = panel_e3()
    e3_cand = e3.get("candidate", {})
    e3_par = e3.get("baseline", {})

    rows = []
    for cand_path, cand_keys in cand_artifacts:
        keys = set(cand_keys)
        if target_keys is not None:
            keys &= target_keys
        if not keys:
            continue
        par_path = None
        for p, pkeys in par_artifacts:
            if set(pkeys) >= keys:
                par_path = p
                break
        if par_path is None:
            print(f"no matching parent artifact for {len(keys)} states; skip")
            continue
        print(f"pair: {os.path.basename(cand_path)[-24:]} <-> {os.path.basename(par_path)[-24:]}")

        par_params = load_params_only(par_path, keys)
        cand_states = load_state_params(cand_path, keys)

        for (layer, role) in sorted(keys):
            astate, cand_params = cand_states[(layer, role)]
            par_params_for_state = par_params[(layer, role)]

            w_dense = weights[layer][role].to(torch.float32)
            weight_pair = v2._pair(w_dense)
            W = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
            W_std = ref.decode_standard_hif4(ref.encode_standard_hif4(W)).to(torch.float32)
            W_legal = ref.dequantize_hif4(dict(cand_params), W.shape).to(torch.float32)
            W0 = ref.dequantize_hif4(dict(par_params_for_state), W.shape).to(torch.float32)
            o, in_f = int(W.shape[0]), int(W.shape[1])
            lam = (
                cand_sol._WEIGHT_GPTQ_REGULARIZATION_WIDE
                if (in_f >= cand_sol._WIDE_LAYER_MIN_DIM or o >= cand_sol._WIDE_LAYER_MIN_DIM)
                else cand_sol._WEIGHT_GPTQ_REGULARIZATION
            )
            bs = cand_sol._HIF4_BLOCK_SIZE
            n_blocks = in_f // bs

            xh_folds, y_folds, mse_std_folds = [], [], []
            for fold in CALIBRATION_FOLDS:
                x_dense = cal_act[role][fold][layer].to(torch.float32)
                act_pair = v2._pair(x_dense)
                X = v2.dequantize_nvfp4(*act_pair).to(torch.float32)
                X_std = ref.decode_standard_hif4(ref.encode_standard_hif4(X)).to(torch.float32)
                actp = cand_sol.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], astate)
                Xh = ref.dequantize_hif4(v2._cpu_params(actp), X.shape).to(torch.float32)
                xh_folds.append(Xh)
                y_folds.append(X @ W.t())
                mse_std_folds.append(float(((X_std @ W_std.t()) - (X @ W.t())).square().mean()))

            F = len(xh_folds)
            omegas = [1.0 / (F * max(float((y_folds[f] ** 2).sum()), 1e-12)) for f in range(F)]
            xs = [xh_folds[f] * math.sqrt(omegas[f]) for f in range(F)]
            ys = [y_folds[f] * math.sqrt(omegas[f]) for f in range(F)]
            xh_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)

            W_cont = W0.clone()
            R = y_all - xh_all @ W_cont.t()
            for b in range(n_blocks):
                sl = slice(b * bs, (b + 1) * bs)
                Xb = xh_all[:, sl]
                Wb0 = W_cont[:, sl].t().contiguous()
                dW = cand_sol._l23_block_solve(Xb, R, lam)
                Wc = Wb0 + dW
                W_cont[:, sl] = Wc.t()
                R = R - Xb @ dW

            e1_folds, e2_folds = [], []
            for f in range(F):
                num1 = float(((y_folds[f] - xh_folds[f] @ W_cont.t()) ** 2).mean())
                num2 = float(((xh_folds[f] @ (W_cont - W_legal).t()) ** 2).mean())
                denom = mse_std_folds[f]
                e1_folds.append(num1 / denom if denom > 0 else float("nan"))
                e2_folds.append(num2 / denom if denom > 0 else float("nan"))
            e1 = float(sum(e1_folds) / len(e1_folds))
            e2 = float(sum(e2_folds) / len(e2_folds))
            e3v = e3_cand.get((layer, role))
            e3p = e3_par.get((layer, role))
            e3_c = (1.0 - e3v) if e3v is not None else None
            e3_p = (1.0 - e3p) if e3p is not None else None
            e4 = (e3_c - e1 - e2) if e3_c is not None else None
            rows.append({
                "state": [layer, role], "e1_continuous_fit": e1, "e2_legal_projection": e2,
                "e3_deployment_candidate": e3_c, "e3_deployment_parent": e3_p,
                "e4_mismatch": e4,
                "mse_std_folds": [float(v) for v in mse_std_folds],
                "n_blocks": n_blocks, "fold_rows": [int(x.shape[0]) for x in xh_folds],
            })
        del par_params, cand_states
        gc.collect()

    covered = len(rows)
    agg = {
        "e1_mean": float(sum(r["e1_continuous_fit"] for r in rows) / covered) if covered else None,
        "e2_mean": float(sum(r["e2_legal_projection"] for r in rows) / covered) if covered else None,
        "e3_candidate_mean": float(sum(r["e3_deployment_candidate"] for r in rows if r["e3_deployment_candidate"] is not None)
                                   / sum(1 for r in rows if r["e3_deployment_candidate"] is not None)) if covered else None,
        "e4_mean": float(sum(r["e4_mismatch"] for r in rows if r["e4_mismatch"] is not None)
                         / sum(1 for r in rows if r["e4_mismatch"] is not None)) if covered else None,
        "cells": covered,
        "note": "E1/E2 on calibration folds [0,1] (shard0 states only, minimal run per loop §2.5); "
                "E3 = 1 - mean gain from the 6-shard panel for the same states; "
                "E4 = E3 - E1 - E2; no new forward, artifact reuse",
    }

    by_role = {}
    for r in rows:
        by_role.setdefault(r["state"][1], []).append(r)
    role_stats = {
        role: {
            "e1": float(sum(x["e1_continuous_fit"] for x in items) / len(items)),
            "e2": float(sum(x["e2_legal_projection"] for x in items) / len(items)),
            "e3": float(sum(x["e3_deployment_candidate"] for x in items if x["e3_deployment_candidate"] is not None)
                        / sum(1 for x in items if x["e3_deployment_candidate"] is not None)),
            "count": len(items),
        }
        for role, items in by_role.items()
    }

    payload = {
        "date": DATE, "side": "linear", "parent_label": "L4 (ACB16F76, 4607/247s)",
        "candidate_label": "L23b (13639FB2)", "parent_sha256": par_sha, "candidate_sha256": cand_sha,
        "aggregate": agg, "per_role": role_stats, "cells": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"error_ledger_linear_{DATE}.json")
    out_md = os.path.join(OUT_DIR, f"error_ledger_linear_{DATE}.md")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Linear error ledger {DATE}\n\n")
        fh.write(f"- E1 mean: {agg['e1_mean']:.4f} (continuous low-dim fit residual / MSE_STD)\n")
        fh.write(f"- E2 mean: {agg['e2_mean']:.4f} (legal projection loss / MSE_STD)\n")
        fh.write(f"- E3 candidate mean: {agg['e3_candidate_mean']:.4f} (1 - gain, 4B panel)\n")
        fh.write(f"- E4 mean: {agg['e4_mean']:.4f} (E3 - E1 - E2, deployment mismatch)\n")
        fh.write(f"- cells: {covered}\n\n")
        fh.write("| state | E1 | E2 | E3(cand) | E4 |\n|---|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda r: (r["state"][0], r["state"][1])):
            e3s = "--" if r["e3_deployment_candidate"] is None else f"{r['e3_deployment_candidate']:.4f}"
            e4s = "--" if r["e4_mismatch"] is None else f"{r['e4_mismatch']:.4f}"
            fh.write(f"| {r['state'][0]}:{r['state'][1]} | {r['e1_continuous_fit']:.4f} "
                     f"| {r['e2_legal_projection']:.4f} | {e3s} | {e4s} |\n")

    print("\n=== LEDGER ===")
    print(f"E1 continuous fit: {agg['e1_mean']:.4f}")
    print(f"E2 legal projection: {agg['e2_mean']:.4f}")
    print(f"E3 candidate (panel 1-gain): {agg['e3_candidate_mean']:.4f}")
    print(f"E4 mismatch: {agg['e4_mean']:.4f}")
    print(f"written: {out_json}, {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())