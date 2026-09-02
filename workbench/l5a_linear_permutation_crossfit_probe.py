"""Local-only cross-fold L5a permutation control.

This probe keeps the fixed quartile-interleave wiring from the mode-3 family,
but derives a proposal from one calibration fold and tests it on the other
fold before storing it.  A proposal is stored only when both fold-specific
proposals improve the opposite fold and the two proposals have identical
indices.  The consensus requirement is deliberately strict: it is a test of
whether the pressure ordering is a stable structural signal rather than a
same-fold output oracle.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workbench" / "l5a_linear_permutation_mode3_probe.py"
spec = importlib.util.spec_from_file_location("l5a_crossfit_base", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
impl = base.base


def _proposal_from_fold(
    weight: torch.Tensor,
    sample: torch.Tensor,
    balance: torch.Tensor,
) -> torch.Tensor:
    pressure = impl._pressure(weight, [sample], balance)
    return impl._low_high_interleave(pressure)


def _choose_crossfit(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, int, int]:
    impl._ACTIVE_PERM = None
    balance, seed, block_size = impl._ORIGINAL_CHOOSE(weight, calibration)
    rows, channels = map(int, weight.shape)
    diagnostic: dict[str, Any] = {
        "shape": [rows, channels],
        "expansive": bool(rows > channels),
        "accepted": False,
        "crossfit_consensus": False,
    }
    if rows <= channels or channels % impl.BLOCK != 0 or len(calibration) < 2:
        impl._LAST_DIAGNOSTIC = diagnostic
        return balance, seed, block_size
    try:
        proposal0 = _proposal_from_fold(weight, calibration[0], balance)
        proposal1 = _proposal_from_fold(weight, calibration[1], balance)
        same = bool(torch.equal(proposal0, proposal1))
        base0, base_loss0 = impl._product_score(
            weight, [calibration[0]], balance, seed, block_size, None
        )
        base1, base_loss1 = impl._product_score(
            weight, [calibration[1]], balance, seed, block_size, None
        )
        cross01, loss01 = impl._product_score(
            weight, [calibration[1]], balance, seed, block_size, proposal0
        )
        cross10, loss10 = impl._product_score(
            weight, [calibration[0]], balance, seed, block_size, proposal1
        )
        # The candidate that will be deployed is proposal0.  It is admitted
        # only after a symmetric cross-fold check, plus exact proposal
        # consensus, so no fold-specific ordering can leak into state.
        accepted = (
            same
            and all(math.isfinite(v) for v in (base0, base1, cross01, cross10))
            and cross01 < base1
            and cross10 < base0
            and len(loss01) == len(loss10) == 1
            and all(v <= b for v, b in zip(loss01, base_loss1))
            and all(v <= b for v, b in zip(loss10, base_loss0))
        )
        if accepted:
            impl._ACTIVE_PERM = proposal0
        diagnostic.update({
            "base_fold_scores": [float(base0), float(base1)],
            "cross_fold_scores": [float(cross01), float(cross10)],
            "proposal_consensus": same,
            "accepted": bool(accepted),
            "changed_channels": int((proposal0 != impl._identity(channels, weight.device)).sum().item()),
        })
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        diagnostic["error"] = f"{type(exc).__name__}: {exc}"
    impl._LAST_DIAGNOSTIC = diagnostic
    return balance, seed, block_size


impl.parent._choose_boat = _choose_crossfit

hif4_calibration_and_quantize_weight = base.hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation = base.hif4_dynamic_quantize_activation
hif4_calibration_attention = base.hif4_calibration_attention
hif4_dynamic_quantize_q = base.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = base.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = base.hif4_dynamic_quantize_v

__all__ = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
