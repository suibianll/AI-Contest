"""GPU dispatch-cost benchmark for the L-EM1 sequential descent.

``bench_seq_loop.py`` is a CPU proxy (the plan's "~50-100 ms/call" estimate came
from it); ``time_paired.py`` measured the real thing at ~0.93 ms/group, i.e.
~600 ms per (layer, role) call and ~96 s of the 300 s official gate.  This
script isolates where that time goes and what an analytic rewrite would cost:

  impl       mirror of ``implementation.py``: 8-candidate tensor, einsum
             quadratic, two host syncs per group
  impl_nosync  same op sequence with the two syncs removed
  analytic   closed form (``cost = 0.0625 G_jj +- 0.5 g_j``), block-local
             gradient, no syncs, no per-group host round trip
  jacobi     one fully vectorised whole-layer pass (all groups at once)

Shapes match the real panel: q is in=2560 rows=128, o is in=4096 rows=512.
Read-only: prints only.
"""

from __future__ import annotations

import time

import torch

BLOCK = 64
GROUPS = 16
STEP = 0.25


def _setup(channels: int, rows: int, seed: int = 0) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gram = torch.randn(channels, channels, device="cuda", generator=gen)
    gram = gram.t() @ gram / channels
    gram += 2.0 * torch.eye(channels, device="cuda")
    return {
        "gram": gram,
        "x": torch.randn(rows, channels, device="cuda", generator=gen) * 0.1,
        "g": torch.randn(rows, channels, device="cuda", generator=gen),
        "code": torch.randint(0, 8, (rows, channels), device="cuda", generator=gen).float(),
        "sign": torch.where(
            torch.rand(rows, channels, device="cuda", generator=gen) < 0.5, -1.0, 1.0
        ),
        "scale": torch.rand(
            rows, channels // BLOCK, GROUPS, 1, device="cuda", generator=gen
        )
        * 0.05
        + 0.01,
    }


def _time(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / reps


def _candidate_tensor(state: dict, rows: int, channels: int) -> torch.Tensor:
    code = state["code"].reshape(rows, channels // BLOCK, GROUPS, 4)
    sign = state["sign"].reshape(rows, channels // BLOCK, GROUPS, 4)
    scale = state["scale"].reshape(rows, channels // BLOCK, GROUPS, 1)
    j_idx = torch.arange(4, device="cuda").reshape(4, 1).expand(4, 2).reshape(8)
    s_idx = torch.tensor([-1.0, 1.0], device="cuda").reshape(1, 2).expand(4, 2).reshape(8)
    mask = torch.zeros(8, 4, dtype=torch.bool, device="cuda")
    mask[torch.arange(8, device="cuda"), j_idx] = True
    moved = (code.unsqueeze(0) + s_idx.reshape(8, 1, 1, 1, 1)).clamp_(0.0, 7.0)
    candidates = torch.where(mask.reshape(8, 1, 1, 1, 4), moved, code.unsqueeze(0))
    return sign.unsqueeze(0) * (candidates - code.unsqueeze(0)) * STEP * scale.unsqueeze(0)


def bench_impl(state: dict, rows: int, channels: int, syncs: bool) -> float:
    gram = state["gram"]
    x = state["x"].clone()
    g = state["g"].clone()
    code = state["code"].clone()
    blocks = channels // BLOCK
    delta = _candidate_tensor(state, rows, channels)
    element_index = torch.arange(4, device="cuda").reshape(4, 1).expand(4, 2).reshape(8)
    step_values = torch.tensor([-1.0, 1.0], device="cuda").reshape(1, 2).expand(4, 2).reshape(8)
    element_mask = torch.zeros(8, 4, dtype=torch.bool, device="cuda")
    element_mask[torch.arange(8, device="cuda"), element_index] = True
    step_column = step_values.reshape(8, 1, 1)
    mask_row = element_mask.reshape(8, 1, 4)
    acc = 0.0

    def run() -> None:
        nonlocal acc
        acc = 0.0
        for block in range(blocks):
            base = block * BLOCK
            for group in range(GROUPS):
                lo = base + group * 4
                sl = slice(lo, lo + 4)
                code_g = code[:, sl]
                moved = (code_g.unsqueeze(0) + step_column).clamp_(0.0, 7.0)
                cand = torch.where(mask_row, moved, code_g.unsqueeze(0))
                d = delta[:, :, block, group, :]
                local = gram[sl, sl]
                linear = (d * g[:, sl].unsqueeze(0)).sum(dim=-1)
                quad = torch.einsum("kri,ij,krj->kr", d, local, d)
                cost = 2.0 * linear + quad
                best = cost.argmin(dim=0)
                best_cost = cost.gather(0, best.unsqueeze(0)).squeeze(0)
                take = best_cost < 0.0
                if syncs and not bool(take.any()):
                    continue
                chosen = d.gather(0, best.reshape(1, rows, 1).expand(1, rows, 4)).squeeze(0)
                chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                x[:, sl] += chosen
                g.add_(chosen.mm(gram[sl, :]))
                if syncs:
                    acc += float(torch.where(take, best_cost, torch.zeros_like(best_cost)).sum())
                code_step = torch.where(
                    take.unsqueeze(-1),
                    element_mask[best].to(torch.float32) * step_values[best].unsqueeze(-1),
                    torch.zeros(rows, 4, device="cuda", dtype=torch.float32),
                )
                code[:, sl] = (code_g + code_step).clamp_(0.0, 7.0)
        state["_acc"] = acc

    return _time(run, 1)


def bench_analytic(state: dict, rows: int, channels: int) -> float:
    gram = state["gram"]
    diag = (STEP * STEP * gram.diagonal()).reshape(channels // BLOCK, GROUPS, 4)
    x = state["x"].clone()
    g = state["g"].clone()
    code = state["code"].clone()
    blocks = channels // BLOCK

    def run() -> None:
        for block in range(blocks):
            base = block * BLOCK
            bsl = slice(base, base + BLOCK)
            gb = g[:, bsl]
            cb = code[:, bsl]
            db = x[:, bsl]
            origin = db.clone()
            gbb = gram[bsl, bsl]
            d4 = diag[block]
            for group in range(GROUPS):
                lo = group * 4
                sl = slice(lo, lo + 4)
                g4 = gb[:, sl]
                c4 = cb[:, sl]
                plus = d4[group].unsqueeze(0) + 0.5 * g4
                minus = d4[group].unsqueeze(0) - 0.5 * g4
                plus = torch.where(c4 < 7.0, plus, torch.zeros_like(plus))
                minus = torch.where(c4 > 0.0, minus, torch.zeros_like(minus))
                value = torch.minimum(plus, minus)
                pick = value.argmin(dim=1, keepdim=True)
                best = value.gather(1, pick)
                take = (best < 0.0).to(torch.float32)
                sign = torch.where(
                    plus.gather(1, pick) < minus.gather(1, pick),
                    torch.ones_like(best),
                    -torch.ones_like(best),
                )
                move = sign * STEP * take
                index = pick + lo
                db.scatter_add_(1, index, move)
                cb.scatter_add_(1, index, sign * take)
                rows_index = index.expand(-1, BLOCK)
                gb.add_(move * gbb.gather(0, rows_index))
            g[:, bsl] = gb
            code[:, bsl] = cb
            if block + 1 < blocks:
                g[:, bsl.stop:] += (db - origin) @ gram[bsl, bsl.stop:]

    return _time(run, 1)


def bench_jacobi(state: dict, rows: int, channels: int, passes: int) -> float:
    gram = state["gram"]
    diag = (STEP * STEP * gram.diagonal()).reshape(channels // BLOCK, GROUPS, 4)
    x = state["x"].clone()
    g = state["g"].clone()
    code = state["code"].clone()
    blocks = channels // BLOCK
    d4_all = diag.reshape(channels // 4, 4)

    def run() -> None:
        for _ in range(passes):
            g4 = g.reshape(rows, channels // 4, 4)
            c4 = code.reshape(rows, channels // 4, 4)
            plus = d4_all + 0.5 * g4
            minus = d4_all - 0.5 * g4
            plus = torch.where(c4 < 7.0, plus, torch.zeros_like(plus))
            minus = torch.where(c4 > 0.0, minus, torch.zeros_like(minus))
            value = torch.minimum(plus, minus)
            pick = value.argmin(dim=2, keepdim=True)
            best = value.gather(2, pick)
            take = (best < 0.0).to(torch.float32)
            sign = torch.where(
                plus.gather(2, pick) < minus.gather(2, pick),
                torch.ones_like(best),
                -torch.ones_like(best),
            )
            move = sign * STEP * take
            index = (pick + 4 * torch.arange(channels // 4, device="cuda").reshape(1, -1, 1))
            delta = torch.zeros(rows, channels, device="cuda")
            delta.scatter_(1, index.reshape(rows, -1), move.reshape(rows, -1))
            gdelta = delta @ gram
            joint = 2.0 * (delta * g).sum(dim=1) + (delta * gdelta).sum(dim=1)
            keep = (joint < 0.0).to(torch.float32).unsqueeze(1)
            delta = delta * keep
            x.add_(delta)
            g.add_(delta @ gram)
            code.scatter_add_(1, index.reshape(rows, -1), (sign * take).reshape(rows, -1) * keep)

    return _time(run, 1)


def bench_groupstep(state: dict, rows: int, channels: int, passes: int) -> float:
    """``groupstep``: 16 sequential steps per pass, group ``k`` of every block.

    This is the re-plan candidate.  ``probe_frontier.py`` measured its accuracy
    (layer0/q: -19.10% / -26.06% at p1/p2 against seq p1 = -21.80%), so the
    question here is only what the 16-step schedule costs.  Mirrors the probe
    exactly, except that the redundant second ``row_delta @ gram`` (the probe
    computed it once for the joint test and again for the refresh) is reused.
    """
    gram = state["gram"]
    diag = (STEP * STEP * gram.diagonal()).reshape(channels // BLOCK, GROUPS, 4)
    x = state["x"].clone()
    g = state["g"].clone()
    code = state["code"].clone()
    sign = state["sign"]
    scale = state["scale"]
    blocks = channels // BLOCK
    d4_step = diag.permute(1, 0, 2).contiguous()
    gram6 = gram.reshape(blocks, GROUPS, 4, blocks, GROUPS, 4)
    bi = torch.arange(blocks, device="cuda").reshape(blocks, 1, 1, 1)
    gi = torch.arange(GROUPS, device="cuda").reshape(1, GROUPS, 1, 1)
    gii_step = gram6[bi, gi, :, bi, gi, :].reshape(blocks, GROUPS, 4, 4)
    target = state["x"] - 0.05 * state["g"]

    j_idx = torch.arange(4).reshape(4, 1).expand(4, 2).reshape(8)
    s_idx = torch.tensor([-1.0, 1.0], device="cuda").reshape(1, 2).expand(4, 2).reshape(8)
    sel = torch.zeros(8, 4, dtype=torch.bool, device="cuda")
    sel[torch.arange(8), j_idx] = True
    ar = torch.arange(blocks, device="cuda")

    def run() -> None:
        for _ in range(passes):
            g = (x - target) @ gram
            for step in range(GROUPS):
                cg = code.reshape(rows, blocks, GROUPS, 4)[:, :, step, :]
                moved = (cg.unsqueeze(0) + s_idx.reshape(8, 1, 1, 1, 1)).clamp(0.0, 7.0)
                codes = torch.where(sel.reshape(8, 1, 1, 1, 4), moved, cg.unsqueeze(0))
                sf = sign.reshape(rows, blocks, GROUPS, 4)[:, :, step, :]
                sc = scale.reshape(rows, blocks, GROUPS, 1)[:, :, step, :]
                deltas = (
                    sf.unsqueeze(0)
                    * (codes - cg.unsqueeze(0).to(torch.float32))
                    * STEP
                    * sc.unsqueeze(0)
                ).reshape(8, rows, blocks, 4)
                codes = codes.reshape(8, rows, blocks, 4)
                gblk = g.reshape(rows, blocks, GROUPS, 4)[:, :, step, :]
                d_step = d4_step[step].reshape(1, 1, blocks, 4)
                diag_quad = (deltas * deltas * d_step).sum(dim=3)
                off = torch.einsum(
                    "krba,bij,krbj->krb", deltas, gii_step[:, step], deltas
                ) - diag_quad
                lin = torch.einsum("krba,rba->krb", deltas, gblk)
                cost = 2.0 * lin + 2.0 * off + diag_quad
                best_k = cost.argmin(dim=0)
                best_c = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
                take = best_c < 0.0
                pick = best_k.reshape(1, rows, blocks, 1).expand(1, rows, blocks, 4)
                chosen = deltas.gather(0, pick).squeeze(0)
                chosen_codes = codes.gather(0, pick).squeeze(0)
                chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
                placed = torch.zeros(rows, blocks, GROUPS, 4, device="cuda")
                placed[:, :, step, :] = chosen
                row_delta = placed.reshape(rows, channels)
                g_delta = row_delta @ gram
                joint = 2.0 * (row_delta * g).sum(dim=1) + (row_delta * g_delta).sum(dim=1)
                ok = joint < 0.0
                row_delta = row_delta * ok.reshape(rows, 1)
                x.add_(row_delta)
                g.add_(g_delta * ok.reshape(rows, 1))
                keep = take & ok.reshape(rows, 1)
                codeslot = code.reshape(rows, blocks, GROUPS, 4)[:, :, step, :]
                codeslot.copy_(
                    torch.where(keep.unsqueeze(-1), chosen_codes, codeslot)
                )

    return _time(run, 1)


def main() -> None:
    print(f"device={torch.cuda.get_device_name(0)}")
    for channels, rows, label in ((2560, 128, "q/in2560"), (4096, 512, "o/in4096")):
        groups = (channels // BLOCK) * GROUPS
        state = _setup(channels, rows)
        impl = bench_impl(state, rows, channels, syncs=True)
        nosync = bench_impl(state, rows, channels, syncs=False)
        analytic = bench_analytic(state, rows, channels)
        jacobi = bench_jacobi(state, rows, channels, passes=1)
        gs1 = bench_groupstep(state, rows, channels, passes=1)
        gs2 = bench_groupstep(state, rows, channels, passes=2)
        print(
            f"{label}: groups={groups}\n"
            f"  impl          {impl * 1e3:8.1f} ms/call  ({impl / groups * 1e6:6.1f} us/group)\n"
            f"  impl_nosync   {nosync * 1e3:8.1f} ms/call  ({nosync / groups * 1e6:6.1f} us/group)\n"
            f"  analytic      {analytic * 1e3:8.1f} ms/call  ({analytic / groups * 1e6:6.1f} us/group)\n"
            f"  jacobi_1pass  {jacobi * 1e3:8.1f} ms/call  ({jacobi / groups * 1e6:6.1f} us/group)\n"
            f"  groupstep_p1  {gs1 * 1e3:8.1f} ms/call\n"
            f"  groupstep_p2  {gs2 * 1e3:8.1f} ms/call",
            flush=True,
        )


if __name__ == "__main__":
    main()
