"""L23b fit_gain table (workpackage 2026-09-08 P0 / research-loop §2.2).

For every weight state (layer, role) and every calibration fold (window 0/1,
the SAME windows used by hif4_calibration_and_quantize_weight):
  - input identity (window key) and row count
  - NVFP4 teacher W (dense reference), X (NVFP4-decoded input)
  - MSE_STD   = mean((standard_hif4(X) @ standard_hif4(W)^T - X@W^T)^2)
  - MSE_PARENT= mean((Xh @ W_par^T - X@W^T)^2)
  - MSE_CAND  = mean((Xh @ W_cand^T - X@W^T)^2)
  - gain_parent = 1 - MSE_PARENT/MSE_STD ; gain_cand = 1 - MSE_CAND/MSE_STD
  - error reduction ratio = 1 - MSE_CAND/MSE_PARENT

fit_gain = equal-weight mean over folds within a state, then equal-weight mean
over all weight states.  The player output uses the REAL dynamic activation
(hif4_dynamic_quantize_activation with the cached activation_state) and the
FINAL five-field weight decode (cached weight_params) -- the evaluator scoring
path applied to the calibration folds.

Memory-safe two-pass design (RAM is ~12 GB free, artifacts are 5.7 GB each):
  pass 1 loads candidate artifacts shard by shard, keeps per-state Xh (small)
  and MSE_CAND/STD; pass 2 loads parent artifacts and merges MSE_PARENT.
  The activation_state is bit-identical between candidate and parent (L23b
  freezes the parent activation path), verified here as a control.

Reuses same-SHA calibration artifacts; no API re-run, no new model forward.
CPU-only.  Protocol matches evaluator/proxy_v3_eval._score and
official_eval._score_details.

Usage:
    .venv/Scripts/python.exe workbench/continuous_linear/l23-residual-subspace/fit_gain_table.py
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "evaluator"))
import official_eval as v2  # noqa: E402
import reference_hif4 as ref  # noqa: E402

DENSE_CACHE = os.path.join(
    ROOT, "artifacts", "official_eval", "cache", "qwen3.5-4b-proxy-v2.pt"
)
CALIB_DIR = os.path.join(ROOT, "artifacts", "official_eval", "cache", "proxy-v3-calibration")
CAND_SHA = "44d7e964f8264633"
PAR_SHA = "acb16f764db80eda"
CAND_SOL = os.path.join(HERE, "candidate", "solution.py")
PAR_SOL = os.path.join(
    ROOT, "solutions", "v162_linear_l4-v189-linear-exact_officialNA_timeNA", "solution.py"
)
OUT_JSON = os.path.join(HERE, "fit_gain_table.json")
OUT_MD = os.path.join(HERE, "fit_gain_table.md")

CALIBRATION_FOLDS = (0, 1)


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_solution(path):
    spec = importlib.util.spec_from_file_location("fit_sol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_artifacts(sha_prefix, model_marker="Qwen3.5-4B"):
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(CALIB_DIR, f"{sha_prefix}-linear-*.pt"))):
        art = torch.load(path, map_location="cpu", weights_only=True)
        model = (art.get("identity", {}).get("model_revision") or "")
        keys = sorted((int(i["layer"]), str(i["role"])) for i in art.get("weight_states", []))
        del art
        gc.collect()
        if model_marker in model and keys:
            out.append((path, keys))
    return out


def state_dicts_from_artifact(path):
    art = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for item in art.get("weight_states", []):
        out[(int(item["layer"]), str(item["role"]))] = (item["state"], dict(item["params"]))
    del art
    gc.collect()
    return out


def _state_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    torch.manual_seed(0)
    print(f"candidate SHA: {sha_of(CAND_SOL)[:16]} (expect {CAND_SHA})")
    print(f"parent    SHA: {sha_of(PAR_SOL)[:16]} (expect {PAR_SHA})")

    raw = torch.load(DENSE_CACHE, map_location="cpu", weights_only=False)
    weights = raw["weights"]
    cal_act = raw["calibration_activations"]
    cal_windows = raw["calibration_windows"]
    metadata = raw["metadata"]
    roles = tuple(raw["roles"])
    del raw
    gc.collect()

    cand_sol = load_solution(CAND_SOL)
    par_sol = load_solution(PAR_SOL)
    state_keys = [(layer, role) for layer in range(metadata["total_layers"]) for role in roles]

    cand_artifacts = list_artifacts(CAND_SHA)
    par_artifacts = list_artifacts(PAR_SHA)
    print(f"candidate artifacts: {len(cand_artifacts)}; parent artifacts: {len(par_artifacts)}")

    cand_rows = {}
    xh_store = {}
    act_state_store = {}
    for path, keys in cand_artifacts:
        states = state_dicts_from_artifact(path)
        print(f"pass1 artifact: {os.path.basename(path)[-24:]} ({len(keys)} states)")
        for (layer, role) in keys:
            astate, params = states[(layer, role)]
            act_state_store[(layer, role)] = astate
            w_dense = weights[layer][role].to(torch.float32)
            weight_pair = v2._pair(w_dense)
            W = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
            W_std = ref.decode_standard_hif4(ref.encode_standard_hif4(W)).to(torch.float32)
            W_cand = ref.dequantize_hif4(dict(params), W.shape).to(torch.float32)
            for fold in CALIBRATION_FOLDS:
                x_dense = cal_act[role][fold][layer].to(torch.float32)
                act_pair = v2._pair(x_dense)
                X = v2.dequantize_nvfp4(*act_pair).to(torch.float32)
                X_std = ref.decode_standard_hif4(ref.encode_standard_hif4(X)).to(torch.float32)
                actp = cand_sol.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], astate)
                ref.validate_hif4_params(actp, X.shape)
                Xh = ref.dequantize_hif4(v2._cpu_params(actp), X.shape).to(torch.float32)
                xh_store[(layer, role, fold)] = Xh
                reference = X @ W.t()
                standard = X_std @ W_std.t()
                player = Xh @ W_cand.t()
                mse_std = float((standard - reference).square().mean())
                mse_cand = float((player - reference).square().mean())
                winfo = cal_windows[fold]
                cand_rows[(layer, role, fold)] = {
                    "layer": layer, "role": role, "fold": fold,
                    "window_key": [winfo["split"], winfo["document_id"], winfo["token_start"], winfo["token_end"]],
                    "rows": int(X.shape[0]), "input_width": int(X.shape[-1]),
                    "output_width": int(W.shape[0]),
                    "mse_std": mse_std, "mse_candidate": mse_cand,
                    "gain_candidate": (mse_std - mse_cand) / mse_std if mse_std > 0 else float("nan"),
                }
        del states
        gc.collect()

    activation_identical = True
    for path, keys in par_artifacts:
        states = state_dicts_from_artifact(path)
        print(f"pass2 artifact: {os.path.basename(path)[-24:]} ({len(keys)} states)")
        for (layer, role) in keys:
            if (layer, role) not in act_state_store:
                continue
            astate, params = states[(layer, role)]
            ref.validate_state(astate)
            if not _state_equal(astate, act_state_store[(layer, role)]):
                activation_identical = False
            w_dense = weights[layer][role].to(torch.float32)
            weight_pair = v2._pair(w_dense)
            W = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
            W_par = ref.dequantize_hif4(dict(params), W.shape).to(torch.float32)
            for fold in CALIBRATION_FOLDS:
                Xh = xh_store.get((layer, role, fold))
                if Xh is None:
                    continue
                x_dense = cal_act[role][fold][layer].to(torch.float32)
                act_pair = v2._pair(x_dense)
                X = v2.dequantize_nvfp4(*act_pair).to(torch.float32)
                reference = X @ W.t()
                player = Xh @ W_par.t()
                mse_par = float((player - reference).square().mean())
                row = cand_rows[(layer, role, fold)]
                row["mse_parent"] = mse_par
                row["gain_parent"] = (row["mse_std"] - mse_par) / row["mse_std"] if row["mse_std"] > 0 else float("nan")
                row["error_reduction_ratio"] = (1.0 - row["mse_candidate"] / mse_par) if mse_par > 0 else float("nan")
                g_cand = row["gain_candidate"]; g_par = row["gain_parent"]
                row["cand_vs_parent"] = "win" if g_cand > g_par else ("loss" if g_cand < g_par else "tie")
                row["cand_vs_std"] = "win" if g_cand > 0 else ("loss" if g_cand < 0 else "tie")
        del states
        gc.collect()
    print(f"activation_state candidate==parent: {activation_identical}")

    rows = [r for r in cand_rows.values() if "mse_parent" in r]
    rows.sort(key=lambda r: (r["layer"], r["role"], r["fold"]))

    by_state = {}
    for r in rows:
        by_state.setdefault((r["layer"], r["role"]), []).append(r)
    state_gains = []
    for key, items in by_state.items():
        g = [r["gain_candidate"] for r in items if math.isfinite(r["gain_candidate"])]
        gp = [r["gain_parent"] for r in items if math.isfinite(r["gain_parent"])]
        if g:
            state_gains.append({
                "state": list(key),
                "fit_gain_candidate": float(sum(g) / len(g)),
                "fit_gain_parent": float(sum(gp) / len(gp)) if gp else None,
                "folds": len(items),
            })
    finite = [s["fit_gain_candidate"] for s in state_gains if math.isfinite(s["fit_gain_candidate"])]
    fit_gain = float(sum(finite) / len(finite)) if finite else float("nan")
    par_finite = [s["fit_gain_parent"] for s in state_gains
                  if s["fit_gain_parent"] is not None and math.isfinite(s["fit_gain_parent"])]
    fit_gain_parent = float(sum(par_finite) / len(par_finite)) if par_finite else float("nan")

    wins = sum(1 for r in rows if r["cand_vs_std"] == "win")
    losses = sum(1 for r in rows if r["cand_vs_std"] == "loss")
    ties = sum(1 for r in rows if r["cand_vs_std"] == "tie")
    vp_wins = sum(1 for r in rows if r["cand_vs_parent"] == "win")
    vp_losses = sum(1 for r in rows if r["cand_vs_parent"] == "loss")

    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r["gain_candidate"])
    role_stats = {
        role: {
            "mean_gain": float(sum(g) / len(g)) if g else None,
            "count": len(g),
            "wins_over_std": sum(1 for x in g if x > 0),
            "losses_over_std": sum(1 for x in g if x < 0),
        }
        for role, g in by_role.items()
    }

    summary = {
        "candidate_sha256": sha_of(CAND_SOL),
        "parent_sha256": sha_of(PAR_SOL),
        "candidate_label": "L23b residual-cross subspace (13639FB2)",
        "parent_label": "L4 (ACB16F76)",
        "protocol": "fit_gain on calibration folds [0,1]; gain = 1 - MSE_PLAYER/MSE_STD vs dense X@W^T; "
                    "real dynamic activation + final five-field decode; CPU; two-pass artifact reuse",
        "linear_fit_gain_candidate": fit_gain,
        "linear_fit_gain_parent": fit_gain_parent,
        "states_covered": len(state_gains),
        "total_states": len(state_keys),
        "fold_rows": len(rows),
        "folds_per_state": len(CALIBRATION_FOLDS),
        "activation_state_candidate_equals_parent": activation_identical,
        "cand_vs_std": {"win": wins, "loss": losses, "tie": ties},
        "cand_vs_parent": {"win": vp_wins, "loss": vp_losses},
        "per_role": role_stats,
        "note": "fit_gain is the calibration-data fit metric (research target >= 0.9); "
                "independent-window case gains remain record-only",
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("# L23b fit_gain table\n\n")
        fh.write(f"- Linear fit_gain (candidate): **{fit_gain:.6f}**\n")
        fh.write(f"- Linear fit_gain (parent): **{fit_gain_parent:.6f}**\n")
        fh.write(f"- activation_state candidate==parent: {activation_identical}\n")
        fh.write(f"- states covered: {len(state_gains)}/{len(state_keys)}; "
                 f"fold rows: {len(rows)} ({len(CALIBRATION_FOLDS)} folds/state)\n")
        fh.write(f"- candidate vs STD: {wins} win / {losses} loss / {ties} tie\n")
        fh.write(f"- candidate vs parent: {vp_wins} win / {vp_losses} loss\n\n")
        fh.write("| layer | role | fold | rows | MSE_STD | MSE_PARENT | MSE_CAND | gain_parent | gain_cand | cand_vs_std |\n")
        fh.write("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(
                f"| {r['layer']} | {r['role']} | {r['fold']} | {r['rows']} "
                f"| {r['mse_std']:.3e} | {r['mse_parent']:.3e} | {r['mse_candidate']:.3e} "
                f"| {r['gain_parent']:.4f} | {r['gain_candidate']:.4f} | {r['cand_vs_std']} |\n"
            )

    print(f"\n=== RESULT ===")
    print(f"Linear fit_gain candidate: {fit_gain:.6f}")
    print(f"Linear fit_gain parent:    {fit_gain_parent:.6f}")
    print(f"states {len(state_gains)}/{len(state_keys)}; fold rows {len(rows)}")
    print(f"cand vs STD: {wins}W/{losses}L/{ties}T; vs parent: {vp_wins}W/{vp_losses}L")
    print(f"written: {OUT_JSON}, {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())