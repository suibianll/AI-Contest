"""Diagnostic: how much of the deployed weight Gram lives inside a 64-block?

The parent stores only the 4x4 per-group diagonal of ``W_hat^T W_hat``
(``state["gram"]``) for the within-block mantissa solver, while the
block-level GPTQ compensation uses the exact full Hessian inverse.  This
script measures, on the real deployed weights from a calibration cache,
where the Gram energy actually sits:

  * 4-element group diagonal (what the solver sees),
  * the rest of the 64-element block diagonal (unmodelled inside a block),
  * cross-64-block coupling (handled only by the sequential GPTQ update).

Read-only; no GPU, no solution import side effects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluator"))


def load_states(path: Path):
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return payload


def dequantize(params: dict) -> torch.Tensor:
    """Decoded HiF4 weight, flattened to (rows, channels)."""
    scale = params["scale_factor"].to(torch.float32)
    lv2 = params["scale_lv2"].to(torch.float32)
    lv3 = params["scale_lv3"].to(torch.float32)
    sign = params["sign"].to(torch.float32)
    mant = params["mant"].to(torch.float32)
    value = sign * mant * lv3 * lv2 * scale
    return value.reshape(value.shape[0], -1)


def masks(channels: int, device):
    idx = torch.arange(channels, device=device)
    group = idx // 4
    block = idx // 64
    same_group = group[:, None] == group[None, :]
    same_block = block[:, None] == block[None, :]
    return same_group, same_block


def report(gram: torch.Tensor, label: str) -> None:
    channels = int(gram.shape[0])
    same_group, same_block = masks(channels, gram.device)
    total = float(gram.square().sum())
    in_group = float(gram.square().masked_select(same_group).sum())
    in_block = float(gram.square().masked_select(same_block).sum())
    cross = total - in_block
    within_block_off_group = in_block - in_group
    print(
        f"{label:>28s}  ch={channels:5d}  ||G||_F^2={total:.6e}  "
        f"4-group={in_group / total:6.3f}  "
        f"64-block-off-group={within_block_off_group / total:6.3f}  "
        f"cross-64-block={cross / total:6.3f}"
    )
    # Per-block conditioning of the exact 64x64 metric the solver would use.
    blocks = channels // 64
    g64 = gram.reshape(blocks, 64, blocks, 64).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
    eig = torch.linalg.eigvalsh(g64.to(torch.float64))
    cond = (eig[:, -1] / eig[:, 0].clamp_min(1e-30)).clamp_max(1e30)
    off_rel = (
        g64.square().sum(dim=(-1, -2))
        - g64.reshape(blocks, 16, 4, 16, 4)
        .diagonal(dim1=1, dim2=3)
        .permute(0, 3, 1, 2)
        .square()
        .sum(dim=(-1, -2))
    ) / g64.square().sum(dim=(-1, -2))
    print(
        f"{'':>28s}  64x64 cond: median={float(cond.median()):.3e} "
        f"max={float(cond.max()):.3e}; off-4-group share per block: "
        f"median={float(off_rel.median()):.3f} max={float(off_rel.max()):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--roles", default="q,o,proj,fc_up")
    args = parser.parse_args()

    payload = load_states(args.cache)
    wanted_layers = {int(v) for v in args.layers.split(",")}
    wanted_roles = {v for v in args.roles.split(",") if v}
    for entry in payload["weight_states"]:
        layer = int(entry["layer"])
        role = str(entry["role"])
        if layer not in wanted_layers or role not in wanted_roles:
            continue
        weight = dequantize(entry["params"])
        gram = weight.t().mm(weight).to(torch.float64)
        report(gram, f"layer{layer}/{role}")
        del weight, gram
    print("done")


if __name__ == "__main__":
    main()
