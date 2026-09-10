"""Accuracy frontier of the L-EM1 descent under iteration-count budgets.

``probe_headroom.py`` established the reference numbers for one pass of each
descent variant.  The measured runtime (``time_paired.py``: ~0.93 ms/group,
i.e. +0.57 s per (layer, role) call and ~+96 s over the official 168 Linear
calls) puts the *sequential* loop -- 640/1024 Python iterations per call -- over
the 300 s gate on its own, so the only way to keep the mechanism is to cut the
number of sequential iterations.

This probe measures the accuracy side of that trade with the true output
squared error on real 4B data:

  seq       one Python iteration per 4-element group (640/1024 per call);
            exact gradient refresh after every accepted move.
  blockseq  one iteration per 64-channel block (40 per call): all 16 groups of
            the block propose simultaneously, the exact within-block pairwise
            coupling is applied, and the accepted subset is verified jointly.
  jacobi    one iteration per *layer*: every group proposes simultaneously and
            the row's whole move is verified exactly against the full metric.

All three share the mechanism's fixed choices: ideal target ``J``, pm1
candidate set (one element of a natural 4-element group, code +-1), natural
ascending order, exact quadratic accept rule.  With ``--passes K`` every
variant repeats the sweep K times, recomputing the exact gradient between
sweeps, so the comparison is at equal sweep count.

Read-only CPU probe on real data; prints only.
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


def pm1_candidates(code_now, sign_f, local_scale):
    """(deltas, codes) for the 8 pm1 candidates of every group.

    ``code_now`` is (rows, blocks, 16, 4) int64; ``deltas`` is
    (8, rows, blocks, 16, 4) float64 and ``codes`` the candidate code value.
    """
    j_idx = torch.arange(4).reshape(4, 1).expand(4, 2).reshape(8)
    s_idx = torch.tensor([-1, 1]).reshape(1, 2).expand(4, 2).reshape(8)
    sel = torch.zeros(8, 4, dtype=torch.bool)
    sel[torch.arange(8), j_idx] = True
    moved = (code_now.unsqueeze(0) + s_idx.reshape(8, 1, 1, 1, 1)).clamp(0, 7)
    codes = torch.where(sel.reshape(8, 1, 1, 1, 4), moved, code_now.unsqueeze(0))
    deltas = (
        sign_f.unsqueeze(0)
        * (codes.to(torch.float64) - code_now.unsqueeze(0).to(torch.float64))
        * 0.25
        * local_scale.unsqueeze(0)
    )
    return deltas, codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--role", default="q")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--variants", default="seq,blockseq,jacobi")
    parser.add_argument("--passes", default="1")
    args = parser.parse_args()

    pass_list = [int(item) for item in args.passes.split(",") if item]
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
    groups = blocks * 16
    gram = weight_hat.t().mm(weight_hat).to(torch.float64)
    cross = ref_weight.t().mm(weight_hat).to(torch.float64)
    ident = torch.eye(channels, dtype=torch.float64)
    m_matrix = torch.linalg.solve(gram + 1e-9 * gram.diagonal().mean() * ident, cross.t()).t()
    reference_out = ref_activation.to(torch.float64) @ ref_weight.to(torch.float64).t()
    ideal_target = ref_activation.to(torch.float64) @ m_matrix

    def true_loss(x_hat_cur: torch.Tensor) -> float:
        out = x_hat_cur.to(torch.float64) @ weight_hat.to(torch.float64).t()
        return float((out - reference_out).square().sum())

    loss0 = true_loss(x_hat)

    sign = act_params["sign"].to(torch.float32)
    mant = act_params["mant"].to(torch.float32)
    lv3 = act_params["scale_lv3"].to(torch.float32)
    lv2 = act_params["scale_lv2"].to(torch.float32)
    sf = act_params["scale_factor"].to(torch.float32)
    local_scale = (sf * lv2 * lv3).reshape(rows, blocks, 16, 1).to(torch.float64)
    sign_f = sign.reshape(rows, blocks, 16, 4).to(torch.float64)
    code_now = torch.round(mant.reshape(rows, blocks, 16, 4).to(torch.float64) * 4.0).to(torch.int64)
    del lv3, lv2, sf, mant, sign

    # (groups, 4, 4) diagonal 4x4 metric blocks, used by the per-group quadratic
    gram_blocks = gram.reshape(blocks, 16, 4, blocks, 16, 4)
    bi = torch.arange(blocks)
    gi = torch.arange(16)
    gii = gram_blocks[bi[:, None, None, None], gi[None, :, None, None], :,
                      bi[:, None, None, None], gi[None, :, None, None], :]
    gii = gii.reshape(groups, 4, 4).contiguous()

    target = ideal_target
    print(
        f"[case] layer{args.layer}/{args.role} w{args.window} rows={rows} ch={channels} "
        f"blocks={blocks} groups={groups} L0={loss0:.6e}"
    )

    def run_seq(passes: int):
        x_cur = x_hat.to(torch.float64).clone()
        code_cur = code_now.clone()
        g = (x_cur - target) @ gram
        accepted = 0
        for _ in range(passes):
            for block in range(blocks):
                base = block * 64
                for group in range(16):
                    lo = base + group * 4
                    sl = slice(lo, lo + 4)
                    ci = code_cur[:, block, group, :]
                    deltas, codes = pm1_candidates(
                        ci.reshape(rows, 1, 1, 4), sign_f[:, block, group, :].reshape(rows, 1, 1, 4),
                        local_scale[:, block, group, :].reshape(rows, 1, 1, 1),
                    )
                    deltas = deltas.reshape(8, rows, 4)
                    codes = codes.reshape(8, rows, 4)
                    g4 = g[:, sl]
                    lin = torch.einsum("kri,ri->kr", deltas, g4)
                    quad = torch.einsum("kri,ij,krj->kr", deltas, gii[block * 16 + group], deltas)
                    cost = 2.0 * lin + quad
                    best_k = cost.argmin(dim=0)
                    best_c = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
                    take = best_c < 0.0
                    chosen = deltas.gather(0, best_k.reshape(1, rows, 1).expand(1, rows, 4)).squeeze(0)
                    chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                    new_code = codes.gather(
                        0, best_k.reshape(1, rows, 1).expand(1, rows, 4)
                    ).squeeze(0)
                    code_cur[:, block, group, :] = torch.where(
                        take.unsqueeze(-1), new_code, ci
                    )
                    x_cur[:, sl] += chosen
                    g += chosen @ gram[sl, :]
                    accepted += int(take.sum())
        return x_cur, accepted, "groups"

    def run_blockseq(passes: int):
        x_cur = x_hat.to(torch.float64).clone()
        code_cur = code_now.clone()
        accepted = 0
        for _ in range(passes):
            g = (x_cur - target) @ gram
            for block in range(blocks):
                base = block * 64
                bsl = slice(base, base + 64)
                gb = g[:, bsl].reshape(rows, 16, 4)
                gbb = gram[bsl, bsl]
                diag4 = gbb.reshape(16, 4, 16, 4)[
                    torch.arange(16), :, torch.arange(16), :
                ].reshape(16, 4, 4)
                deltas, codes = pm1_candidates(
                    code_cur[:, block : block + 1, :, :],
                    sign_f[:, block : block + 1, :, :],
                    local_scale[:, block : block + 1, :, :],
                )
                deltas = deltas.reshape(8, rows, 16, 4)
                codes = codes.reshape(8, rows, 16, 4)
                lin = torch.einsum("kria,ria->kri", deltas, gb)
                quad = torch.einsum("kria,iab,krib->kri", deltas, diag4, deltas)
                c1 = 2.0 * lin + quad
                best_k = c1.argmin(dim=0)
                best_c = c1.gather(0, best_k.unsqueeze(0)).squeeze(0)
                pick = best_k.reshape(1, rows, 16, 1).expand(1, rows, 16, 4)
                chosen = deltas.gather(0, pick).squeeze(0)
                chosen_codes = codes.gather(0, pick).squeeze(0)
                g2 = gbb.reshape(16, 4, 16, 4).permute(0, 2, 1, 3)
                m_pair = torch.einsum("ria,iIab,rIb->riI", chosen, g2, chosen)
                off = m_pair.sum(dim=2) - m_pair.diagonal(dim1=1, dim2=2)
                keep = ((best_c + 2.0 * off) < 0.0) & (best_c < 0.0)
                kk = keep.to(torch.float64)
                joint = (best_c * kk).sum(dim=1) + torch.einsum("ri,riI,rI->r", kk, m_pair, kk)
                ok = joint < 0.0
                keep = keep & ok.reshape(rows, 1)
                chosen = torch.where(keep.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                block_delta = chosen.reshape(rows, 64)
                x_cur[:, bsl] += block_delta
                code_cur[:, block, :, :] = torch.where(
                    keep.unsqueeze(-1), chosen_codes, code_cur[:, block, :, :]
                )
                g[:, bsl] = gb.reshape(rows, 64) + block_delta @ gbb
                if block + 1 < blocks:
                    g[:, bsl.stop:] += block_delta @ gram[bsl, bsl.stop:]
                accepted += int(ok.sum())
        return x_cur, accepted, "blocks"

    def run_jacobi(passes: int):
        x_cur = x_hat.to(torch.float64).clone()
        code_cur = code_now.clone()
        accepted = 0
        for _ in range(passes):
            g = (x_cur - target) @ gram
            deltas, codes = pm1_candidates(code_cur, sign_f, local_scale)
            deltas = deltas.reshape(8, rows, groups, 4)
            codes = codes.reshape(8, rows, groups, 4)
            g4 = g.reshape(rows, groups, 4)
            lin = torch.einsum("kria,ria->kri", deltas, g4)
            quad = torch.einsum("kria,iab,krib->kri", deltas, gii, deltas)
            cost = 2.0 * lin + quad
            best_k = cost.argmin(dim=0)
            best_c = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
            take = best_c < 0.0
            pick = best_k.reshape(1, rows, groups, 1).expand(1, rows, groups, 4)
            chosen = deltas.gather(0, pick).squeeze(0)
            chosen_codes = codes.gather(0, pick).squeeze(0)
            chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
            row_delta = chosen.reshape(rows, channels)
            g_delta = row_delta @ gram
            joint = 2.0 * (row_delta * g).sum(dim=1) + (row_delta * g_delta).sum(dim=1)
            ok = joint < 0.0
            row_delta = row_delta * ok.reshape(rows, 1)
            x_cur += row_delta
            keep = take & ok.reshape(rows, 1)
            code_cur = torch.where(
                keep.reshape(rows, blocks, 16).unsqueeze(-1),
                chosen_codes.reshape(rows, blocks, 16, 4),
                code_cur,
            )
            accepted += int(keep.sum())
        return x_cur, accepted, "rows"

    def run_groupstep(passes: int):
        """16 iterations per pass: group ``k`` of every block moves together."""
        x_cur = x_hat.to(torch.float64).clone()
        code_cur = code_now.clone()
        gii_step = gii.reshape(blocks, 16, 4, 4)
        accepted = 0
        for _ in range(passes):
            g = (x_cur - target) @ gram
            for step in range(16):
                cg = code_cur[:, :, step, :].reshape(rows, blocks, 1, 4)
                deltas, codes = pm1_candidates(
                    cg,
                    sign_f[:, :, step, :].reshape(rows, blocks, 1, 4),
                    local_scale[:, :, step, :].reshape(rows, blocks, 1, 1),
                )
                deltas = deltas.reshape(8, rows, blocks, 4)
                codes = codes.reshape(8, rows, blocks, 4)
                gblk = g.reshape(rows, blocks, 16, 4)[:, :, step, :]
                lin = torch.einsum("krba,rba->krb", deltas, gblk)
                quad = torch.einsum("krba,bij,krbj->krb", deltas, gii_step[:, step], deltas)
                cost = 2.0 * lin + quad
                best_k = cost.argmin(dim=0)
                best_c = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
                take = best_c < 0.0
                pick = best_k.reshape(1, rows, blocks, 1).expand(1, rows, blocks, 4)
                chosen = deltas.gather(0, pick).squeeze(0)
                chosen_codes = codes.gather(0, pick).squeeze(0)
                chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                placed = torch.zeros(rows, blocks, 16, 4, dtype=torch.float64)
                placed[:, :, step, :] = chosen
                row_delta = placed.reshape(rows, channels)
                g_delta = row_delta @ gram
                joint = 2.0 * (row_delta * g).sum(dim=1) + (row_delta * g_delta).sum(dim=1)
                ok = joint < 0.0
                row_delta = row_delta * ok.reshape(rows, 1)
                x_cur += row_delta
                g += row_delta @ gram
                keep = take & ok.reshape(rows, 1)
                code_cur[:, :, step, :] = torch.where(
                    keep.unsqueeze(-1), chosen_codes, code_cur[:, :, step, :]
                )
                accepted += int(keep.sum())
        return x_cur, accepted, "groups"

    runners = {
        "seq": run_seq,
        "blockseq": run_blockseq,
        "jacobi": run_jacobi,
        "groupstep": run_groupstep,
    }
    for variant in [name for name in args.variants.split(",") if name]:
        for passes in pass_list:
            x_cur, accepted, unit = runners[variant](passes)
            loss_new = true_loss(x_cur)
            print(
                f"[ideal/{variant}/pm1/p{passes}] accepted_{unit}={accepted} "
                f"dL_true={loss_new - loss0:+.6e} rel={(loss_new - loss0) / loss0:+.4%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
