"""Where does a groupstep step spend its time, and what does the sparse form buy?

The K=1 step body in ``implementation.py`` builds a **dense** ``row_delta`` of
shape ``(rows, channels)`` whose only non-zero columns are this step's group --
``blocks * 4`` of ``channels`` columns, i.e. 1/16 of them -- and then contracts
the whole thing against ``metric`` (``channels x channels``).  For in=4096,
rows=512 that is ``512 x 4096 x 4096`` ~ 8.6 GFLOP per step, times 16 steps.

The same quantity is available from the `blocks * 4` non-zero columns alone.  In
exact arithmetic the two agree, because the dropped columns are exactly zero;
in fp32 they agree up to the reduction order inside the matmul, which is the
only thing this candidate changes.

Variants:
  dense     mirror of ``implementation.py``: dense zeros buffer, dense mm
  slice     the step's own columns only, index_copy back into a dense buffer
  slicebuf  same, with the dense buffer allocated once outside the loop

Shapes match the real panel: q/o are in=2560/4096; rows 128 and 512 both occur.
Read-only: prints only.
"""

from __future__ import annotations

import statistics
import time

import torch

BLOCK = 64
GROUPS = 16
STEP = 0.25
CODE_MAX = 7.0
REPEATS = 10


def _setup(channels: int, rows: int, seed: int = 0) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(channels, channels, device="cuda", generator=gen)
    metric = weight.t() @ weight / channels
    metric += 2.0 * torch.eye(channels, device="cuda")
    blocks = channels // BLOCK
    return {
        "metric": metric,
        "blocks": blocks,
        "gradient": torch.randn(rows, channels, device="cuda", generator=gen) * 0.01,
        "sign": torch.where(
            torch.rand(rows, blocks, GROUPS, 4, device="cuda", generator=gen) < 0.5,
            -1.0,
            1.0,
        ),
        "scale": (
            torch.rand(rows, blocks, GROUPS, 1, device="cuda", generator=gen) * 0.05
            + 0.01
        ),
        "code": torch.randint(
            0, 8, (rows, blocks, GROUPS, 4), device="cuda", generator=gen
        ).float(),
        "rows": rows,
        "channels": channels,
    }


def _constants(blocks: int, channels: int, device) -> dict:
    element_index = torch.arange(4, device=device).reshape(4, 1).expand(4, 2).reshape(8)
    step_values = torch.tensor([-1.0, 1.0], device=device).reshape(1, 2).expand(4, 2).reshape(8)
    element_mask = torch.zeros(8, 4, dtype=torch.bool, device=device)
    element_mask[torch.arange(8, device=device), element_index] = True
    metric6 = metric_view = None
    return {
        "element_index": element_index,
        "step_values": step_values,
        "element_mask": element_mask,
        "step_column": step_values.reshape(8, 1, 1, 1),
        "mask_row": element_mask.reshape(8, 1, 1, 4),
        "metric6": metric_view,
        "metric6_placeholder": metric6,
    }


def _group_gram(metric, blocks: int, channels: int) -> torch.Tensor:
    metric6 = metric.reshape(blocks, GROUPS, 4, blocks, GROUPS, 4)
    block_index = torch.arange(blocks, device="cuda").reshape(blocks, 1, 1, 1)
    group_index = torch.arange(GROUPS, device="cuda").reshape(1, GROUPS, 1, 1)
    return metric6[block_index, group_index, :, block_index, group_index, :].reshape(
        blocks, GROUPS, 4, 4
    )


def _step_common(ctx: dict, data: dict, step: int, con: dict):
    rows = data["rows"]
    blocks = data["blocks"]
    code_g = data["code"][:, :, step, :]
    moved = (code_g.unsqueeze(0) + con["step_column"]).clamp_(0.0, CODE_MAX)
    candidates = torch.where(con["mask_row"], moved, code_g.unsqueeze(0))
    delta = (
        data["sign"][:, :, step, :].unsqueeze(0)
        * (candidates - code_g.unsqueeze(0))
        * STEP
        * data["scale"][:, :, step, :].unsqueeze(0)
    )
    local_gram = data["group_gram"][:, step]
    gradient4 = data["gradient"].reshape(rows, blocks, GROUPS, 4)
    linear = (delta * gradient4[:, :, step, :].unsqueeze(0)).sum(dim=-1)
    quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)
    cost = 2.0 * linear + quadratic
    best_index = cost.argmin(dim=0)
    best_cost = cost.gather(0, best_index.unsqueeze(0)).squeeze(0)
    take = best_cost < 0.0
    chosen = delta.gather(
        0, best_index.reshape(1, rows, blocks, 1).expand(1, rows, blocks, 4)
    ).squeeze(0)
    chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
    return chosen, take


def _dense(data: dict, con: dict, steps: int) -> torch.Tensor:
    rows = data["rows"]
    channels = data["channels"]
    blocks = data["blocks"]
    metric = data["metric"]
    total = torch.zeros((), device="cuda")
    for step in range(steps):
        chosen, take = _step_common(data, data, step, con)
        row_delta = torch.zeros(rows, channels, device="cuda")
        row_delta.reshape(rows, blocks, GROUPS, 4)[:, :, step, :].copy_(chosen)
        g_delta = row_delta.mm(metric)
        joint = 2.0 * (row_delta * data["gradient"]).sum(dim=1) + (row_delta * g_delta).sum(dim=1)
        total = total + joint.sum()
    return total


def _column_slice(blocks: int, step: int):
    base = torch.arange(blocks, device="cuda").reshape(blocks, 1) * BLOCK + step * 4
    return (base + torch.arange(4, device="cuda").reshape(1, 4)).reshape(-1)


def _metric_by_step(metric: torch.Tensor, blocks: int, channels: int) -> torch.Tensor:
    """Rows of ``metric`` grouped by the step that would move them.

    ``metric[cols_of_step_s]`` for every s, as one view: the row index of
    ``metric`` is the channel being moved, and channel ``b*64 + s*4 + j`` belongs
    to step ``s``.  Reordering once removes the per-step gather.
    """
    return (
        metric.reshape(blocks, GROUPS, 4, channels)
        .permute(1, 0, 2, 3)
        .reshape(GROUPS, blocks * 4, channels)
        .contiguous()
    )


def _slice(data: dict, con: dict, steps: int, preallocated: bool) -> torch.Tensor:
    rows = data["rows"]
    channels = data["channels"]
    blocks = data["blocks"]
    total = torch.zeros((), device="cuda")
    for step in range(steps):
        chosen, take = _step_common(data, data, step, con)
        cols = _column_slice(blocks, step)
        chosen_flat = chosen.reshape(rows, blocks * 4)
        # row_delta @ metric == chosen_flat @ metric[cols] -- row_delta's only
        # non-zero columns are cols, so the dropped columns contribute exactly 0.
        g_delta = chosen_flat.mm(data["metric_by_step"][step])
        gradient_slice = data["gradient"].index_select(1, cols)
        joint = 2.0 * (chosen_flat * gradient_slice).sum(dim=1) + (
            chosen_flat * g_delta.index_select(1, cols)
        ).sum(dim=1)
        total = total + joint.sum()
    return total


def main() -> int:
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    for channels, rows in ((2560, 128), (2560, 512), (4096, 512)):
        data = _setup(channels, rows)
        data["group_gram"] = _group_gram(data["metric"], data["blocks"], channels)
        data["metric_by_step"] = _metric_by_step(data["metric"], data["blocks"], channels)
        con = _constants(data["blocks"], channels, "cuda")
        variants = {
            "dense": lambda: _dense(data, con, GROUPS),
            "slice": lambda: _slice(data, con, GROUPS, False),
            "slicebuf": lambda: _slice(data, con, GROUPS, True),
        }
        print(f"\nin={channels} rows={rows} blocks={data['blocks']}", flush=True)
        values: dict[str, float] = {}
        for name, fn in variants.items():
            fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            values[name] = statistics.median(samples)
            print(
                f"  {name:<9} median {values[name] * 1e3:8.2f} ms   "
                f"min {min(samples) * 1e3:8.2f} ms   spread "
                f"{(max(samples) - min(samples)) * 1e3:6.2f} ms",
                flush=True,
            )
        print(
            f"  slice/dense = {values['slice'] / values['dense']:.3f}   "
            f"slicebuf/dense = {values['slicebuf'] / values['dense']:.3f}",
            flush=True,
        )
        del data, con
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
