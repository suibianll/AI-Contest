"""Headroom probe: exact-metric mantissa refinement of the deployed activation.

Read-only CPU probe on real 4B data.  For one (layer, role, test window) it
reconstructs the parent (R0 v202) deployment and measures:

  * the exact output-error decomposition
        R = X_hat W_hat^T - X W^T = A + B,
    with ``A`` the activation part ``(X_hat - dense) W_hat^T`` and ``B`` the
    weight part ``dense W_hat^T - X W^T``;
  * the achievable decrease of the true output squared error from an
    exact-metric coordinate descent over the 4-element mantissa groups,
    comparing two targets:
        target = dense   (C-free: what the parent already optimises)
        target = X M     (ideal: M = W^T W_hat G^{-1}, needs one extra in^2
                          matrix per layer at runtime)

Candidate sets per 4-element group (``--cand``):
  pm1  per element, code +-1 (8 candidates) -- the L-XR1 pre-registered set;
  fc   all 2^4 floor/ceil combinations of the *unquantised* deployed value.

Descent variants per target:
  frozen    one frozen gradient ``g = E G``, every group takes its best move
            independently (no coupling, no verification);
  seq       block-sequential with exact ``E``/``g`` updates after every
            accepted group move (needs G's rows; narrow-layer runtime);
  blockdiag exact within-block descent with a frozen cross-block gradient,
            then one exact per-row verification of the accumulated move
            (wide-layer runtime: two triangular solves, no per-group solve).

Everything is recomputed with the full deployed weight, so the reported
``dL`` is the true output squared-error change, not a surrogate.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402

PARENT = ROOT / "solutions" / "20260909_v202_linear-sample-energy-fusion_scoreNA_timeNA" / "solution.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_candidates(
    kind: str,
    dense: torch.Tensor,
    code_now: torch.Tensor,
    local_scale: torch.Tensor,
    sign_f: torch.Tensor,
):
    """Return (deltas, codes) as (K, rows, blocks, 16, 4) float64/int64.

    ``delta`` is decoded(new) - decoded(current), ``codes`` the candidate code
    value (0..7) or -1 when the candidate is not available for that element.
    """
    rows, blocks = code_now.shape[0], code_now.shape[1]
    if kind == "fc":
        x_abs = dense.reshape(rows, blocks, 16, 4).abs().to(torch.float64)
        raw = x_abs * 4.0 / local_scale
        floor_code = torch.floor(raw).clamp(0, 6).to(torch.int64)
        ceil_code = torch.ceil(raw).clamp(0, 7).to(torch.int64)
        bits = torch.arange(16)
        masks = ((bits.unsqueeze(1) >> torch.arange(4)) & 1).bool()
        codes = torch.where(
            masks.reshape(16, 1, 1, 1, 4), ceil_code.unsqueeze(0), floor_code.unsqueeze(0)
        )
    elif kind == "pm1":
        # candidate k = element j moves by step s, k = j*2 + s
        j_idx = torch.arange(4).reshape(4, 1).expand(4, 2).reshape(8)
        s_idx = torch.tensor([-1, 1]).reshape(1, 2).expand(4, 2).reshape(8)
        sel = torch.zeros(8, 4, dtype=torch.bool)
        sel[torch.arange(8), j_idx] = True
        moved = code_now.unsqueeze(0) + s_idx.reshape(8, 1, 1, 1, 1)
        codes = torch.where(
            sel.reshape(8, 1, 1, 1, 4), moved, code_now.unsqueeze(0)
        ).clamp(0, 7)
    else:
        raise ValueError(kind)
    deltas = sign_f.unsqueeze(0) * (codes.to(torch.float64) - code_now.unsqueeze(0)) * 0.25 * local_scale.unsqueeze(0)
    return deltas, codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--role", default="q")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cand", default="pm1")
    parser.add_argument("--variants", default="seq,blockdiag")
    parser.add_argument("--max-blocks", type=int, default=0)
    args = parser.parse_args()

    parent = load_module(PARENT, "r0_parent")

    payload = torch.load(args.cache, map_location="cpu", mmap=True, weights_only=False)
    entry = next(
        item
        for item in payload["weight_states"]
        if int(item["layer"]) == args.layer and str(item["role"]) == args.role
    )
    state = entry["state"]
    weight_params = entry["params"]

    pack = torch.load(args.pack, map_location="cpu", mmap=True, weights_only=False)
    raw = pack["test_activations"][args.role][args.window][args.layer]
    raw = raw.to(torch.float32)[: args.rows].contiguous()
    pair = v2._pair(raw)
    ref_activation = v2.dequantize_nvfp4(*pair).to(torch.float32)

    weight_pair = v2._pair(pack["weights"][args.layer][args.role].to(torch.float32))
    weight_shape = v2.dequantize_nvfp4(*weight_pair).shape
    ref_weight = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
    weight_hat = v2.dequantize_hif4(dict(weight_params), weight_shape).to(torch.float32)

    with torch.no_grad():
        act_params = parent.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
        dense = parent._static_actorder_dense_from_state(pair[0], pair[1], state)
    x_hat = v2.dequantize_hif4(dict(act_params), ref_activation.shape).to(torch.float32)

    rows, channels = x_hat.shape
    blocks = channels // 64
    gram = weight_hat.t().mm(weight_hat).to(torch.float64)
    cross = ref_weight.t().mm(weight_hat).to(torch.float64)
    ident = torch.eye(channels, dtype=torch.float64)
    m_matrix = torch.linalg.solve(gram + 1e-9 * gram.diagonal().mean() * ident, cross.t()).t()

    reference_out = ref_activation.to(torch.float64) @ ref_weight.to(torch.float64).t()
    ideal_target = ref_activation.to(torch.float64) @ m_matrix

    e_act = (x_hat - dense).to(torch.float64)
    a_part = e_act @ weight_hat.to(torch.float64).t()
    b_part = dense.to(torch.float64) @ weight_hat.to(torch.float64).t() - reference_out

    def true_loss(x_hat_cur: torch.Tensor) -> float:
        out = x_hat_cur.to(torch.float64) @ weight_hat.to(torch.float64).t()
        return float((out - reference_out).square().sum())

    loss0 = true_loss(x_hat)
    a_norm = float(a_part.square().sum())
    b_norm = float(b_part.square().sum())
    cross_term = float(2.0 * (a_part * b_part).sum())
    print(
        f"[decompose] layer{args.layer}/{args.role} w{args.window} rows={rows} ch={channels}\n"
        f"  L0={loss0:.6e}  |A|^2={a_norm:.6e} ({a_norm / loss0:6.3f})  "
        f"|B|^2={b_norm:.6e} ({b_norm / loss0:6.3f})  "
        f"2<A,B>={cross_term:.6e} ({cross_term / loss0:6.3f})  "
        f"residual={loss0 - a_norm - b_norm - cross_term:.3e}"
    )

    sign = act_params["sign"].to(torch.float32)
    mant = act_params["mant"].to(torch.float32)
    lv3 = act_params["scale_lv3"].to(torch.float32)
    lv2 = act_params["scale_lv2"].to(torch.float32)
    sf = act_params["scale_factor"].to(torch.float32)
    local_scale = (sf * lv2 * lv3).reshape(rows, blocks, 16, 1).to(torch.float64)
    sign_f = sign.reshape(rows, blocks, 16, 4).to(torch.float64)
    code_now = torch.round(mant.reshape(rows, blocks, 16, 4).to(torch.float64) * 4.0).to(torch.int64)
    del lv3, lv2, sf, mant, sign

    all_delta, all_codes = build_candidates(args.cand, dense, code_now, local_scale, sign_f)
    n_cand = all_delta.shape[0]
    print(f"[cand] {args.cand}  candidates/group={n_cand}  moved_elements={int((all_delta != 0).any(0).sum())}")

    variants = [name for name in args.variants.split(",") if name]
    block_limit = args.max_blocks if args.max_blocks > 0 else blocks

    for target_name in ("dense", "ideal"):
        target = dense.to(torch.float64) if target_name == "dense" else ideal_target
        e_cur = x_hat.to(torch.float64) - target
        g_cur = e_cur @ gram

        for variant in variants:
            x_cur = x_hat.to(torch.float64).clone()
            accepted = 0
            if variant == "frozen":
                g = g_cur.clone()
                for block in range(block_limit):
                    for group in range(16):
                        sl = slice(block * 64 + group * 4, block * 64 + group * 4 + 4)
                        delta = all_delta[:, :, block, group, :]
                        g4 = g[:, sl]
                        g44 = gram[sl, sl]
                        linear = torch.einsum("kri,ri->kr", delta, g4)
                        quad = torch.einsum("kri,ij,krj->kr", delta, g44, delta)
                        cost = 2.0 * linear + quad
                        best_k = cost.argmin(dim=0)
                        best_cost = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
                        take = best_cost < 0.0
                        chosen = delta.gather(
                            0, best_k.reshape(1, rows, 1).expand(1, rows, 4)
                        ).squeeze(0)
                        chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                        x_cur[:, sl] += chosen
                        accepted += int(take.sum())
                loss_new = true_loss(x_cur)
                print(
                    f"[{target_name}/{variant}/{args.cand}] accepted_groups={accepted} "
                    f"dL_true={loss_new - loss0:+.6e} rel={(loss_new - loss0) / loss0:+.4%}"
                )
            elif variant == "blockseq":
                # runtime-feasible vectorised block-sequential descent:
                # per block, all 16 groups propose simultaneously, the exact
                # pairwise coupling inside the block is applied, the accepted
                # subset is verified jointly, and the accepted block move is
                # propagated exactly to the remaining blocks.
                g = g_cur.clone()
                for block in range(block_limit):
                    bsl = slice(block * 64, (block + 1) * 64)
                    gb = g[:, bsl].reshape(rows, 16, 4)
                    gbb = gram[bsl, bsl]
                    gii = gbb.reshape(16, 4, 16, 4)[
                        torch.arange(16), :, torch.arange(16), :
                    ]
                    delta = all_delta[:, :, block, :, :]  # (K, rows, 16, 4)
                    lin = torch.einsum("kria,ria->kri", delta, gb)
                    quad = torch.einsum("kria,iab,krib->kri", delta, gii, delta)
                    c1 = 2.0 * lin + quad
                    best_k = c1.argmin(dim=0)
                    best_c = c1.gather(0, best_k.unsqueeze(0)).squeeze(0)
                    chosen = delta.gather(
                        0, best_k.reshape(1, rows, 16, 1).expand(1, rows, 16, 4)
                    ).squeeze(0)  # (rows, 16, 4)
                    # pairwise coupling between proposed group moves
                    g2 = gbb.reshape(16, 4, 16, 4).permute(0, 2, 1, 3)
                    m_pair = torch.einsum("ria,iIab,rIb->riI", chosen, g2, chosen)
                    off = m_pair.sum(dim=2) - m_pair.diagonal(dim1=1, dim2=2)
                    marginal = best_c + 2.0 * off
                    keep = (marginal < 0.0) & (best_c < 0.0)
                    chosen = torch.where(keep.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                    # exact joint cost of the kept subset (self + pairwise)
                    j_self = (c1.gather(0, best_k.unsqueeze(0)).squeeze(0) * keep).sum(dim=1)
                    kk = keep.to(torch.float64)
                    j_pair = torch.einsum("ri,riI,rI->r", kk, m_pair, kk)
                    joint = j_self + j_pair
                    ok = joint < 0.0
                    chosen = torch.where(ok.reshape(rows, 1, 1), chosen, torch.zeros_like(chosen))
                    block_delta = chosen.reshape(rows, 64)
                    x_cur[:, bsl] += block_delta
                    accepted += int(ok.sum())
                    if block < block_limit - 1:
                        g[:, bsl.start + 64:] += block_delta @ gram[bsl, bsl.start + 64:]
                loss_new = true_loss(x_cur)
                print(
                    f"[{target_name}/{variant}/{args.cand}] accepted_blocks={accepted}/{block_limit} "
                    f"dL_true={loss_new - loss0:+.6e} rel={(loss_new - loss0) / loss0:+.4%}"
                )
            elif variant == "blockdiag":
                x_try = x_hat.to(torch.float64).clone()
                row_delta = torch.zeros(rows, channels, dtype=torch.float64)
                for block in range(block_limit):
                    bsl = slice(block * 64, (block + 1) * 64)
                    gbb = gram[bsl, bsl]
                    g_cross = g_cur[:, bsl]
                    delta_accum = torch.zeros(rows, 64, dtype=torch.float64)
                    for group in range(16):
                        gs = slice(group * 4, group * 4 + 4)
                        delta = all_delta[:, :, block, group, :]
                        g_grp = g_cross[:, gs] + delta_accum @ gbb[:, gs]
                        g44 = gbb[gs, gs]
                        linear = torch.einsum("kri,ri->kr", delta, g_grp)
                        quad = torch.einsum("kri,ij,krj->kr", delta, g44, delta)
                        cost = 2.0 * linear + quad
                        best_k = cost.argmin(dim=0)
                        best_cost = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
                        take = best_cost < 0.0
                        chosen = delta.gather(
                            0, best_k.reshape(1, rows, 1).expand(1, rows, 4)
                        ).squeeze(0)
                        chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                        delta_accum[:, gs] += chosen
                    row_delta[:, bsl] += delta_accum
                    x_try[:, bsl] += delta_accum
                g_delta = row_delta @ gram
                cost_row = 2.0 * (row_delta * g_cur).sum(dim=1) + (row_delta * g_delta).sum(dim=1)
                keep = cost_row < 0.0
                x_cur = torch.where(keep.unsqueeze(-1), x_try, x_hat.to(torch.float64))
                accepted = int(keep.sum())
                loss_new = true_loss(x_cur)
                print(
                    f"[{target_name}/{variant}/{args.cand}] accepted_rows={accepted}/{rows} "
                    f"dL_true={loss_new - loss0:+.6e} rel={(loss_new - loss0) / loss0:+.4%}"
                )
            else:  # seq
                g = g_cur.clone()
                for block in range(block_limit):
                    for group in range(16):
                        sl = slice(block * 64 + group * 4, block * 64 + group * 4 + 4)
                        delta = all_delta[:, :, block, group, :]
                        g4 = g[:, sl]
                        g44 = gram[sl, sl]
                        linear = torch.einsum("kri,ri->kr", delta, g4)
                        quad = torch.einsum("kri,ij,krj->kr", delta, g44, delta)
                        cost = 2.0 * linear + quad
                        best_k = cost.argmin(dim=0)
                        best_cost = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
                        take = best_cost < 0.0
                        chosen = delta.gather(
                            0, best_k.reshape(1, rows, 1).expand(1, rows, 4)
                        ).squeeze(0)
                        chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                        x_cur[:, sl] += chosen
                        g += chosen @ gram[sl, :]
                        accepted += int(take.sum())
                loss_new = true_loss(x_cur)
                print(
                    f"[{target_name}/{variant}/{args.cand}] accepted_groups={accepted} "
                    f"dL_true={loss_new - loss0:+.6e} rel={(loss_new - loss0) / loss0:+.4%}"
                )


if __name__ == "__main__":
    main()
