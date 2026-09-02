"""Fast screening wrapper for the local-only L3 fc oracle.

The canonical D0 oracle is intentionally conservative and uses sequential
coordinate updates.  This wrapper is for iteration only: it limits the scope
to the known worst layer (layer 3), keeps the joint edit class, and replaces
the 64-coordinate Gauss--Seidel pass with one batched Jacobi pass.  Its output
is a screening artifact, never a candidate score or a replacement for the
canonical D0 record.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workbench" / "l3_fc_legal_oracle.py"
spec = importlib.util.spec_from_file_location("l3_fc_legal_oracle", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load oracle source: {SOURCE}")
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def _coordinate_pass_jacobi(
    q: torch.Tensor,
    denominator: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    target: torch.Tensor,
    coordinates: Sequence[int],
    sweeps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One fixed batched pass; no Python loop over individual coordinates."""
    levels = oracle._levels(q.device)
    diagonal = torch.diagonal(hessian).clamp_min(oracle.EPS)
    accepted_moves = 0
    accepted_values = 0
    accepted_gain = 0.0
    q_work = q.clone()
    for _ in range(max(1, int(sweeps))):
        gradient = torch.einsum("ij,rj->ri", hessian, q_work) - torch.einsum(
            "ij,rj->ri", cross, target
        )
        coordinate_ids = torch.as_tensor(tuple(coordinates), device=q.device, dtype=torch.long)
        current = q_work.index_select(1, coordinate_ids)
        denom = denominator.index_select(1, coordinate_ids)
        options = denom[..., None] * levels[None, None, :]
        step = options - current[..., None]
        grad = gradient.index_select(1, coordinate_ids)[..., None]
        diag = diagonal.index_select(0, coordinate_ids)[None, :, None]
        change = 2.0 * step * grad + diag * step.square()
        best_change, best_index = change.min(dim=-1)
        improve = torch.isfinite(best_change) & (best_change < -oracle.EPS)
        accepted = step.gather(-1, best_index[..., None]).squeeze(-1)
        accepted = torch.where(improve, accepted, torch.zeros_like(accepted))
        q_work[:, coordinate_ids] += accepted
        accepted_moves += int(improve.sum().item())
        accepted_values += int((accepted != 0.0).sum().item())
        accepted_gain += float((-best_change[improve]).sum().item())
    return q_work, {
        "accepted_moves": float(accepted_moves),
        "accepted_values": float(accepted_values),
        "accepted_quadratic_gain": oracle._finite(accepted_gain),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "official_eval" / "l3-fc-fast-layer3.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "logs" / "execution" / "2026-09-02-l3-fc-fast-layer3.md",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    oracle.DEFAULT_LAYERS = (3,)
    oracle.EDIT_CLASSES = ("joint",)
    oracle._coordinate_pass = _coordinate_pass_jacobi
    namespace = SimpleNamespace(
        parent=ROOT / "workbench" / "pre-a3-v147-parent.py",
        cache=ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt",
        output=args.output,
        report=args.report,
        device=args.device,
        layers="3",
        roles=",".join(oracle.DEFAULT_ROLES),
    )
    result = oracle.run(namespace)
    # Make the approximation explicit in both JSON and Markdown without
    # changing the canonical artifact written by the parent script.
    result["diagnostic"] = "l3-fc-fast-layer3-jacobi-screen-v1"
    result["scope"] = "research-oracle"
    result["screening_scope"] = "layer3 / fc_gate+fc_up / joint only"
    result["teacher_note"] = "batched Jacobi screening; not canonical D0 teacher"
    args.output.write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with args.report.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\nScreening scope: layer 3 only, joint class only, one batched Jacobi "
            "pass per component. This is not a deployable candidate and does not "
            "replace canonical D0.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
