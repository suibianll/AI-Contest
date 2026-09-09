"""L-H1 step-0 prechecks on a synthetic 64-column block (fixed seed, no tuning).

Precheck A (expressibility): per-column positive/negative elements may take
different magnitude codes while emitting only the legal five HiF4 fields (no
positive/negative dual scale, no zero-point, no custom decode).  The candidate
five fields must pass evaluator/reference_hif4.py and decode to different
legal values than the root encoding.

Precheck B (residual space): on the same synthetic block, freeze the final
deployed activation state, enumerate one column's legal sign/mant codes in
the structured per-column (m_pos, m_neg) family, and compute the complete
output error XW^T - Q(XR)Q(WR^{-T})^T.  At least one legal state that the
root Weight GPTQ does not select must be strictly better, otherwise the
direction is closed as NO_DISTINCT_RESIDUAL.
"""

from pathlib import Path
import importlib.util
import json
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def nvfp4_pair(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = int(dense.shape[-1]) // 16
    scale = torch.ones(*dense.shape[:-1], blocks, dtype=torch.float32)
    return dense.clone(), scale


def reconstruct_deployment_weight(root, weight: torch.Tensor, state: dict) -> torch.Tensor:
    channels = int(weight.shape[-1])
    device = weight.device
    smooth_inv = state.get("smooth_inv")
    if smooth_inv is None:
        d = torch.ones(channels, dtype=torch.float32, device=device)
    else:
        d = root._safe_positive_vector(smooth_inv, channels).reciprocal().to(device)
    permutation = state.get("permutation")
    if permutation is None:
        permutation = torch.arange(channels, device=device, dtype=torch.int64)
    else:
        permutation = permutation.to(device=device, dtype=torch.int64).reshape(-1)
    w1 = root._linear_pair_transform(
        weight,
        d,
        permutation,
        int(state.get("block_smooth_size", 0)),
        int(state.get("block_smooth_seed", 0)),
        weight_side=True,
    )
    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=device, dtype=torch.float32)
        v = residual_v.to(device=device, dtype=torch.float32)
        return w1 - (w1 @ v) @ u.transpose(0, 1)
    rank1_u = state["rank1_u"].to(device=device, dtype=torch.float32).reshape(-1)
    rank1_v = state["rank1_v"].to(device=device, dtype=torch.float32).reshape(-1)
    return w1 - (w1 @ rank1_v).unsqueeze(-1) * rank1_u


def main() -> None:
    root = load_module(ROOT / "solution.py", "lh1_root_solution")
    reference = load_module(ROOT / "evaluator" / "reference_hif4.py", "lh1_reference")

    device = torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(20260909)
    out_features, in_features, tokens = 64, 64, 256

    # Correlated activations with per-column scale variation; mixed-sign weight.
    mixing = torch.eye(in_features) + 0.3 * torch.randn(
        in_features, in_features, generator=gen
    ) / float(in_features) ** 0.5
    col_scale = torch.exp(0.5 * torch.randn(in_features, generator=gen))
    weight = torch.randn(out_features, in_features, generator=gen)

    windows = []
    for _ in range(2):
        z = torch.randn(tokens, in_features, generator=gen)
        windows.append((z @ mixing) * col_scale)
    calib_pairs = [nvfp4_pair(x) for x in windows]
    weight_pair = nvfp4_pair(weight)

    report: dict = {"device": "cpu", "seed": 20260909}

    # ------------------------------------------------------------------
    # Precheck A: expressibility of per-column sign-asymmetric codes.
    # ------------------------------------------------------------------
    root_block_params = root._dense_to_hif4(weight)
    candidate_params = {k: v.clone() for k, v in root_block_params.items()}
    column = 0
    # candidate: positives of the column -> mant 1.75, negatives -> mant 0.75.
    col_sign = torch.sign(weight[:, column])
    new_mant = torch.where(col_sign > 0, torch.full_like(col_sign, 1.75),
                           torch.full_like(col_sign, 0.75))
    new_mant = torch.where(col_sign == 0, torch.zeros_like(new_mant), new_mant)
    mant = candidate_params["mant"].reshape(out_features, 1, 8, 2, 4).clone()
    sign = candidate_params["sign"].reshape(out_features, 1, 8, 2, 4).clone()
    mant[:, 0, column // 8, (column % 8) // 4, column % 4] = new_mant
    sign[:, 0, column // 8, (column % 8) // 4, column % 4] = torch.where(
        new_mant == 0, torch.zeros_like(col_sign), col_sign
    )
    candidate_params["mant"] = mant.reshape(out_features, 1, 8, 2, 4)
    candidate_params["sign"] = sign.reshape(out_features, 1, 8, 2, 4)
    reference.validate_hif4_params(candidate_params, (out_features, in_features))
    reference_dec = reference.dequantize_hif4(
        candidate_params, (out_features, in_features)
    )
    root_dec = reference.dequantize_hif4(
        root_block_params, (out_features, in_features)
    )
    differs = not bool(torch.equal(reference_dec, root_dec))
    report["precheck_a"] = {
        "verdict": "EXPRESSIBLE" if differs else "NOT_EXPRESSIBLE",
        "legal_five_fields": True,
        "decoded_differs_from_root": differs,
        "changed_elements": int((reference_dec != root_dec).sum()),
        "definition": (
            "same input column: positive and negative weights may take "
            "different magnitude codes; output stays the legal five fields "
            "(no dual scale, no zero-point, no custom decode)"
        ),
    }

    # ------------------------------------------------------------------
    # Precheck B: residual space under the complete output objective.
    # ------------------------------------------------------------------
    result = root.hif4_calibration_and_quantize_weight(
        weight_pair[0], weight_pair[1], calib_pairs
    )
    state = result["activation_state"]
    weight_params = result["weight_params"]

    w_prime = reconstruct_deployment_weight(root, weight, state)
    w_hat0 = root._dequantize_hif4(weight_params).to(torch.float32)

    x_prime = torch.cat(
        [root._static_actorder_dense_from_state(p[0], p[1], state) for p in calib_pairs]
    ).to(torch.float32)
    q_hat = torch.cat(
        [
            root._dequantize_hif4(
                root.hif4_dynamic_quantize_activation(p[0], p[1], state)
            ).to(torch.float32)
            for p in calib_pairs
        ]
    )

    # Product preservation: X'W'^T must equal XW^T (continuous transforms).
    x_raw = torch.cat(windows).to(torch.float32)
    product_error = float(
        (x_prime @ w_prime.t() - x_raw @ weight.t()).abs().max()
    )
    product_scale = float((x_raw @ weight.t()).abs().max())

    reference_out = x_prime @ w_prime.t()
    residual = reference_out - q_hat @ w_hat0.t()
    base_loss = float(residual.square().sum())

    g_qr = q_hat.t() @ residual                       # [D, F]
    d_q = q_hat.square().sum(dim=0)                   # [D]
    denom = (
        weight_params["scale_factor"]
        * weight_params["scale_lv2"]
        * weight_params["scale_lv3"]
    ).expand(weight_params["sign"].shape).flatten(start_dim=-4).to(torch.float32)
    sign_w = torch.sign(w_prime)

    m_values = torch.arange(8, dtype=torch.float32) * 0.25
    best = None
    improving_states = 0
    improving_asymmetric = 0
    for c in range(in_features):
        wcol = w_prime[:, c]
        a = denom[:, c] * (wcol > 0).to(torch.float32)
        b = -denom[:, c] * (wcol < 0).to(torch.float32)
        w_old = w_hat0[:, c]
        g = g_qr[c]
        ga = float(g @ a)
        gb = float(g @ b)
        aa = float(a.square().sum())
        bb = float(b.square().sum())
        ab = float(a @ b)
        aw = float(a @ w_old)
        bw = float(b @ w_old)
        gw = float(g @ w_old)
        ww = float(w_old.square().sum())
        dq = float(d_q[c])
        # loss delta of replacing the column by m_pos*a + m_neg*b codes:
        # delta = -2*G.delta + ||q_c||^2*||delta||^2 with delta = w_new - w_old.
        delta = (
            -2.0 * (m_values[:, None] * ga + m_values[None, :] * gb - gw)
            + dq * (
                m_values[:, None] ** 2 * aa
                + m_values[None, :] ** 2 * bb
                + 2.0 * m_values[:, None] * m_values[None, :] * ab
                + ww
                - 2.0 * m_values[:, None] * aw
                - 2.0 * m_values[None, :] * bw
            )
        )
        improving = delta < -1e-12
        improving_states += int(improving.sum())
        asym = improving & (m_values[:, None] != m_values[None, :])
        improving_asymmetric += int(asym.sum())
        idx = int(delta.argmin())
        i, j = divmod(idx, 8)
        if best is None or float(delta[i, j]) < best[0]:
            best = (float(delta[i, j]), c, float(m_values[i]), float(m_values[j]))

    # Formula sanity: scalar-formula delta must match a direct recompute for
    # an arbitrary combo on an arbitrary column (guards against sign/indexing
    # bugs that could fake a negative result).
    check_col = 0
    check_mp, check_mn = 1.0, 1.25
    w_chk = torch.zeros_like(w_prime[:, check_col])
    pos_chk = w_prime[:, check_col] > 0
    neg_chk = w_prime[:, check_col] < 0
    w_chk[pos_chk] = denom[pos_chk, check_col] * check_mp
    w_chk[neg_chk] = -denom[neg_chk, check_col] * check_mn
    w_hat_chk = w_hat0.clone()
    w_hat_chk[:, check_col] = w_chk
    direct_chk = float(
        (reference_out - q_hat @ w_hat_chk.t()).square().sum()
    ) - base_loss
    a_chk = denom[:, check_col] * pos_chk.to(torch.float32)
    b_chk = -denom[:, check_col] * neg_chk.to(torch.float32)
    w_old_chk = w_hat0[:, check_col]
    d_chk = w_chk - w_old_chk
    scalar_chk = float(
        -2.0 * (g_qr[check_col] @ d_chk)
        + d_q[check_col] * d_chk.square().sum()
    )
    formula_ok = abs(direct_chk - scalar_chk) <= max(1e-4, 1e-4 * abs(direct_chk))

    # Diagnostic (not part of the verdict): per-element mant +-1 neighbor
    # enumeration under the same true objective, to distinguish "structured
    # family too coarse" from "objective gap ~0".
    element_neighbor_improving = 0
    element_neighbor_total = 0
    element_neighbor_best_delta = 0.0
    for c in range(in_features):
        for r in range(out_features):
            w_val = w_prime[r, c]
            if float(w_val) == 0.0:
                continue
            root_code = int(round(float(weight_params["mant"].flatten(start_dim=-4)[r, c]) * 4.0))
            for cand_code in (root_code - 1, root_code + 1):
                if cand_code < 0 or cand_code > 7:
                    continue
                element_neighbor_total += 1
                w_new = float(torch.sign(w_val)) * float(denom[r, c]) * cand_code * 0.25
                d_elem = w_new - float(w_hat0[r, c])
                delta_elem = float(
                    -2.0 * float(g_qr[c, r]) * float(d_elem)
                    + float(d_q[c]) * float(d_elem) ** 2
                )
                if delta_elem < -1e-9:
                    element_neighbor_improving += 1
                    element_neighbor_best_delta = min(
                        element_neighbor_best_delta, delta_elem
                    )

    # Direct verification of the scalar formula for the best combo.
    best_delta, best_col, best_mp, best_mn = best
    direct_ok = None
    direct_delta = None
    if best_delta < -1e-12:
        w_new_col = torch.zeros_like(w_prime[:, best_col])
        pos = w_prime[:, best_col] > 0
        neg = w_prime[:, best_col] < 0
        w_new_col[pos] = denom[pos, best_col] * best_mp
        w_new_col[neg] = -denom[neg, best_col] * best_mn
        w_hat_new = w_hat0.clone()
        w_hat_new[:, best_col] = w_new_col
        direct_delta = float(
            (reference_out - q_hat @ w_hat_new.t()).square().sum()
        ) - base_loss
        direct_ok = abs(direct_delta - best_delta) <= max(
            1e-6, 1e-6 * abs(direct_delta)
        )

    # Root's chosen codes for the best column (for the "root does not select"
    # check): the structured (m_pos, m_neg) family differs from the root's
    # per-element codes unless every positive and every negative element of
    # the column already uses exactly those codes.
    root_mant_col = weight_params["mant"].flatten(start_dim=-4)[:, best_col]
    root_pos_codes = root_mant_col[w_prime[:, best_col] > 0] * 4.0
    root_neg_codes = root_mant_col[w_prime[:, best_col] < 0] * 4.0
    root_matches_best = bool(
        root_pos_codes.numel() > 0
        and torch.all(root_pos_codes == best_mp * 4.0)
        and (
            root_neg_codes.numel() == 0
            or torch.all(root_neg_codes == best_mn * 4.0)
        )
    )

    verdict = (
        "DISTINCT_RESIDUAL"
        if (best_delta < -1e-12 and not root_matches_best and direct_ok)
        else "NO_DISTINCT_RESIDUAL"
    )
    report["precheck_b"] = {
        "verdict": verdict,
        "base_full_output_loss": base_loss,
        "best_delta": best_delta,
        "best_relative_improvement": best_delta / max(base_loss, 1e-12),
        "best_column": best_col,
        "best_m_pos": best_mp,
        "best_m_neg": best_mn,
        "best_is_asymmetric": best_mp != best_mn,
        "improving_structured_states": improving_states,
        "improving_asymmetric_states": improving_asymmetric,
        "total_structured_states": in_features * 64,
        "root_selects_best_state": root_matches_best,
        "scalar_formula_direct_check_ok": direct_ok,
        "formula_sanity_check_ok": formula_ok,
        "formula_sanity_direct_delta": direct_chk,
        "formula_sanity_scalar_delta": scalar_chk,
        "diag_element_neighbor_improving": element_neighbor_improving,
        "diag_element_neighbor_total": element_neighbor_total,
        "diag_element_neighbor_best_delta": element_neighbor_best_delta,
        "diag_element_neighbor_best_relative": (
            element_neighbor_best_delta / max(base_loss, 1e-12)
        ),
        "direct_delta": direct_delta,
        "product_preservation_abs_max": product_error,
        "product_preservation_rel": product_error / max(product_scale, 1e-12),
    }

    (HERE / "precheck.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
