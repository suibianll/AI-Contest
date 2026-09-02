"""Local-only L4 Weight-decoupled stored-scale probe.

The parent already performs its single output-supervised weight pass.  This probe
does not add another oracle pass: after that pass it keeps every HiF4 code,
lv2/lv3 and transform fixed, solves one closed-form stored-scale per
row/block under the transformed calibration Gram, projects the scalar to the
nearest legal E6M2 value, and admits the whole state only when both folds' real
deployed output loss do not worsen.  It is restricted to expansive FFN shapes
so q/k/v/o/proj remain controls for the first effect panel.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workbench" / "pre-a3-v147-parent.py"
spec = importlib.util.spec_from_file_location("l4_weight_decoupled_parent", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

_PARENT_CALIBRATE = base.hif4_calibration_and_quantize_weight


def _block_gram(calibration: Sequence[torch.Tensor], blocks: int) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for sample in calibration[:2]:
        value = sample.to(torch.float32)
        if value.ndim != 2 or int(value.shape[-1]) != blocks * base._BLOCK:
            raise ValueError("transformed calibration shape mismatch")
        view = value.reshape(-1, blocks, base._BLOCK)
        pieces.append(
            torch.bmm(
                view.permute(1, 2, 0).contiguous(),
                view.permute(1, 0, 2).contiguous(),
            )
        )
    if not pieces:
        raise ValueError("empty transformed calibration")
    return torch.stack(pieces, dim=0).sum(dim=0)


def _fit_stored_scale(
    weight_t: torch.Tensor,
    parent: dict[str, torch.Tensor],
    calibration_t: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    rows, channels = map(int, weight_t.shape)
    if rows <= channels or channels % base._BLOCK != 0 or len(calibration_t) < 2:
        return parent
    blocks = channels // base._BLOCK
    gram = _block_gram(calibration_t, blocks).to(weight_t.device, dtype=torch.float32)
    coeff = (
        parent["sign"].to(torch.float32)
        * parent["mant"].to(torch.float32)
        * parent["scale_lv2"].to(torch.float32)
        * parent["scale_lv3"].to(torch.float32)
    ).reshape(rows, blocks, base._BLOCK)
    target = weight_t.to(torch.float32).reshape(rows, blocks, base._BLOCK)
    numerator = torch.einsum("rbi,bij,rbj->rb", coeff, gram, target)
    denominator = torch.einsum("rbi,bij,rbj->rb", coeff, gram, coeff)
    fitted = numerator / denominator.clamp_min(base._EPS)
    old_scale = parent["scale_factor"].to(torch.float32).reshape(rows, blocks)
    fitted = torch.where(
        torch.isfinite(fitted) & (fitted > base._E6M2_MIN), fitted, old_scale
    )
    fitted = fitted.clamp(base._E6M2_MIN, base._E6M2_MAX)
    codes = base._e6m2_encode_nearest(fitted)
    scale = base._e6m2_decode(codes).reshape_as(parent["scale_factor"])
    candidate = base._clone_params(parent)
    candidate["scale_factor"] = scale.to(
        device=parent["scale_factor"].device,
        dtype=parent["scale_factor"].dtype,
    )
    return candidate


def _robust(losses: Sequence[float]) -> float:
    if not losses or not all(math.isfinite(float(value)) for value in losses):
        return math.inf
    return sum(float(value) for value in losses) / len(losses) + base._WEIGHT_HSDQ_ROBUST_MIX * max(losses)


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    result = _PARENT_CALIBRATE(weight_quant, weight_scale, calib_activation_list)
    try:
        state = result.get("activation_state")
        if not isinstance(state, dict):
            return result
        weight_t, calibration_t = base._active_transformed_calibration(
            weight_quant, weight_scale, calib_activation_list, state
        )
        parent_params = result["weight_params"]
        candidate_params = _fit_stored_scale(weight_t, parent_params, calibration_t)
        if candidate_params is parent_params:
            return result
        parent_deployed = base._active_deploy_from_state(
            parent_params, calibration_t, state
        )
        candidate_state, candidate_deployed = base._active_rebuild_state(
            weight_t, candidate_params, state, calibration_t
        )
        parent_losses = base._active_joint_loss(
            weight_t, parent_params, calibration_t, parent_deployed
        )
        candidate_losses = base._active_joint_loss(
            weight_t, candidate_params, calibration_t, candidate_deployed
        )
        accepted = (
            _robust(candidate_losses) < _robust(parent_losses)
            and len(parent_losses) == len(candidate_losses) == 2
            and all(
                float(candidate) <= float(parent)
                for candidate, parent in zip(candidate_losses, parent_losses)
            )
        )
        if accepted:
            result["weight_params"] = candidate_params
            candidate_state = dict(candidate_state)
            candidate_state["l4_weight_decoupled"] = True
            candidate_state["l4_parent_losses"] = [float(value) for value in parent_losses]
            candidate_state["l4_candidate_losses"] = [float(value) for value in candidate_losses]
            result["activation_state"] = candidate_state
    except (RuntimeError, ValueError, FloatingPointError, KeyError, TypeError):
        return result
    return result


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
