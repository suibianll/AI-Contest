"""Local-only L5a mode-3 (quartile-interleave) permutation control.

This wrapper uses the same parent, score gate, and state path as
``l5a_linear_permutation_probe.py``; only the fixed within-block wiring changes
from low/high interleave to four pressure-quartile interleave.  It exists as a
single workbench family comparison, not as a numbered candidate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workbench" / "l5a_linear_permutation_probe.py"
spec = importlib.util.spec_from_file_location("l5a_mode3_base", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def _quartile_interleave(pressure: torch.Tensor) -> torch.Tensor:
    channels = int(pressure.numel())
    order = base._identity(channels, pressure.device)
    for start in range(0, channels, base.BLOCK):
        local = torch.argsort(pressure[start : start + base.BLOCK], stable=True)
        quarter = base.BLOCK // 4
        chosen = torch.stack(
            (
                local[:quarter],
                local[quarter : 2 * quarter],
                local[2 * quarter : 3 * quarter],
                local[3 * quarter :],
            ),
            dim=1,
        ).reshape(-1)
        order[start : start + base.BLOCK] = chosen + start
    return order


base._low_high_interleave = _quartile_interleave

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
