"""Local-only L5a proposal with a fold-uncertainty stability gate.

The proposal is still built from both calibration folds and uses the fixed
quartile-interleave wiring.  Unlike the original mode-3 admission, the
candidate must reduce each fold's product loss by at least the observed
cross-fold disagreement.  This is a parameter-free lower-confidence check:
small or unstable same-fold gains are left at the parent state.
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
spec = importlib.util.spec_from_file_location("l5a_stability_base", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
impl = base.base


def _choose_stable(
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
        "stability_gate": "min_fold_reduction_ge_fold_disagreement",
    }
    if rows <= channels or channels % impl.BLOCK != 0 or len(calibration) < 2:
        impl._LAST_DIAGNOSTIC = diagnostic
        return balance, seed, block_size
    try:
        pressure = impl._pressure(weight, calibration, balance)
        candidate = impl._low_high_interleave(pressure)
        base_score, base_losses = impl._product_score(
            weight, calibration, balance, seed, block_size, None
        )
        candidate_score, candidate_losses = impl._product_score(
            weight, calibration, balance, seed, block_size, candidate
        )
        reductions = [
            (float(b) - float(c)) / max(abs(float(b)), impl.EPS)
            for b, c in zip(base_losses, candidate_losses)
        ]
        disagreement = abs(reductions[0] - reductions[1]) if len(reductions) == 2 else math.inf
        accepted = (
            math.isfinite(candidate_score)
            and math.isfinite(base_score)
            and candidate_score < base_score
            and len(base_losses) == len(candidate_losses) == 2
            and all(c <= b for c, b in zip(candidate_losses, base_losses))
            and min(reductions) >= disagreement
        )
        if accepted:
            impl._ACTIVE_PERM = candidate
        diagnostic.update({
            "base_score": float(base_score),
            "candidate_score": float(candidate_score),
            "base_fold_losses": [float(value) for value in base_losses],
            "candidate_fold_losses": [float(value) for value in candidate_losses],
            "fold_reductions": reductions,
            "fold_disagreement": float(disagreement),
            "accepted": bool(accepted),
            "changed_channels": int((candidate != impl._identity(channels, weight.device)).sum().item()),
        })
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        diagnostic["error"] = f"{type(exc).__name__}: {exc}"
    impl._LAST_DIAGNOSTIC = diagnostic
    return balance, seed, block_size


impl.parent._choose_boat = _choose_stable

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
