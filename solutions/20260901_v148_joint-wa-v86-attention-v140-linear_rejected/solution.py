"""A3 structural control: one joint Weight--Activation residual round.

The Attention APIs are the immutable v86 implementation and the initial
Linear APIs are v140.  After v140 has produced legal parameters, this module
performs one fixed alternating round in calibration space:

1. reconstruct the transformed calibration activations actually deployed by
   v140;
2. apply v140's legal output-supervised block refiner to the deployed weight;
3. rebuild the deployed-weight Gram and activation parameters;
4. apply the same block refiner once more and accept the result only when the
   two calibration folds improve the true product residual.

No alpha/offset/seed/block-budget sweep is introduced.  This is a local
attribution implementation; it loads the two immutable parent archives so
the experiment changes Linear only and keeps v86 Attention fixed.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import torch


_ROOT = Path(__file__).resolve().parents[2]
_V86_PATH = (
    _ROOT
    / "solutions"
    / "20260830_v086_c86-attn-block-final_scoreNA_timeNA"
    / "solution.py"
)
_V140_PATH = (
    _ROOT
    / "solutions"
    / "20260901_v140_linear-roab-pair_rejected"
    / "solution.py"
)
_EPS = 1.0e-12


def _load_archive(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"archive source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load archive source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v86 = _load_archive("_hif4_v86_attention", _V86_PATH)
_v140 = _load_archive("_hif4_v140_linear", _V140_PATH)


def _transformed_calibration(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: Sequence,
    state: dict[str, Any],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Recreate the exact v140 Linear-side coordinate system."""

    weight = _v140._dequantize_nvfp4_float32(weight_quant, weight_scale)
    weight = weight.to(torch.float32)
    samples = [
        _v140._dequantize_nvfp4_float32(item[0], item[1])
        .to(device=weight.device, dtype=torch.float32)
        for item in calib_activation_list
    ]
    pair = state.get("roab_pairs")
    if torch.is_tensor(pair):
        pair = pair.to(device=weight.device, dtype=torch.float32)
        inverse = torch.linalg.inv(pair)
        weight_t = _v140._pair_transform(weight, inverse.transpose(-1, -2))
        activation_t = [_v140._pair_transform(sample, pair) for sample in samples]
        return weight_t.to(torch.float32), [item.to(torch.float32) for item in activation_t]

    smooth_inv = state.get("smooth_inv")
    if not torch.is_tensor(smooth_inv):
        raise ValueError("v140 state has no Linear transform")
    smooth_inv = smooth_inv.to(device=weight.device, dtype=torch.float32).reshape(-1)
    balance = smooth_inv.reciprocal()
    seed = int(state.get("block_smooth_seed", -1))
    block_size = int(state.get("block_smooth_size", 0))
    weight_t = _v140._apply_boat_rotation(
        weight * balance.reshape(1, -1), seed, block_size
    )
    activation_t = [
        _v140._apply_boat_rotation(
            sample * smooth_inv.reshape(1, -1), seed, block_size
        )
        for sample in samples
    ]
    return weight_t.to(torch.float32), [item.to(torch.float32) for item in activation_t]


def _deploy_from_state(
    weight_params: dict[str, torch.Tensor],
    activation_t: Sequence[torch.Tensor],
    state: dict[str, Any],
) -> list[torch.Tensor]:
    """Run the v140 activation quantizer in transformed calibration space."""

    gram = state.get("gram64")
    cross = state.get("output_cross64")
    gain = state.get("output_gain")
    if not torch.is_tensor(gram) or not torch.is_tensor(gain):
        raise ValueError("v140 state has no deployed-weight curvature")
    device = activation_t[0].device
    gram = gram.to(device=device, dtype=torch.float32)
    cross = cross.to(device=device, dtype=torch.float32) if torch.is_tensor(cross) else None
    gain = gain.to(device=device, dtype=torch.float32).reshape(1, -1)
    deployed: list[torch.Tensor] = []
    for sample in activation_t:
        dense = sample.to(torch.float32) * gain
        params = _v140._dense_to_hif4(
            dense,
            offsets=_v140._BASE_OFFSETS,
            gram64=gram,
        )
        params = _v140._refine_activation(
            dense,
            params,
            gram,
            output_cross64=cross,
            output_target=sample,
        )
        deployed.append(_v140._dequantize_hif4(params).to(torch.float32))
    return deployed


def _rebuild_state(
    weight_t: torch.Tensor,
    weight_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    activation_t: Sequence[torch.Tensor],
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Compile Q(W)'s new output metric and its matching Q(A) state."""

    deployed_weight = _v140._dequantize_hif4(weight_params).to(torch.float32)
    gain = (
        (weight_t * deployed_weight).sum(dim=0)
        / deployed_weight.square().sum(dim=0).clamp_min(_EPS)
    )
    gain = torch.nan_to_num(gain, nan=1.0, posinf=2.0, neginf=0.5).clamp(0.5, 2.0)
    gram = _v140._gram64(deployed_weight)
    cross = _v140._block_cross64(deployed_weight, weight_t)
    next_state = dict(state)
    next_state["output_gain"] = _v140._cpu_tensor(gain)
    next_state["gram64"] = _v140._cpu_tensor(gram)
    next_state["output_cross64"] = _v140._cpu_tensor(cross)
    next_state["version"] = 4
    deployed = _deploy_from_state(weight_params, activation_t, next_state)
    return next_state, deployed


def _joint_loss(
    weight_t: torch.Tensor,
    weight_params: dict[str, torch.Tensor],
    activation_t: Sequence[torch.Tensor],
    deployed_activation: Sequence[torch.Tensor],
) -> list[float]:
    deployed_weight = _v140._dequantize_hif4(weight_params).to(torch.float32)
    losses: list[float] = []
    for raw, deployed in zip(activation_t[:2], deployed_activation[:2]):
        target = raw.to(torch.float32).mm(weight_t.to(torch.float32).t())
        prediction = deployed.to(torch.float32).mm(deployed_weight.t())
        losses.append(
            float(
                (target - prediction).square().mean()
                / (target.square().mean() + _EPS)
            )
        )
    return losses


def _robust_score(losses: Sequence[float]) -> float:
    if not losses:
        return math.inf
    return sum(losses) / len(losses) + float(_v140._WEIGHT_HSDQ_ROBUST_MIX) * max(losses)


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    base = _v140.hif4_calibration_and_quantize_weight(
        weight_quant,
        weight_scale,
        calib_activation_list,
    )
    try:
        state = base["activation_state"]
        if not isinstance(state, dict):
            return base
        weight_t, activation_t = _transformed_calibration(
            weight_quant,
            weight_scale,
            calib_activation_list,
            state,
        )
        parent_params = base["weight_params"]

        # First A3 half-step: reproduce v140's deployed Q(A), then optimize
        # legal Q(W) using its real product residual.
        parent_deployed = _deploy_from_state(parent_params, activation_t, state)
        first_params = _v140._crossfold_weight_output(
            weight_t,
            parent_params,
            activation_t,
            parent_deployed,
        )

        # Second half-step: the new Q(W) changes H_W, so compile a new Q(A)
        # state before applying the same legal block oracle once more.
        first_state, first_deployed = _rebuild_state(
            weight_t,
            first_params,
            state,
            activation_t,
        )
        final_params = _v140._crossfold_weight_output(
            weight_t,
            first_params,
            activation_t,
            first_deployed,
        )
        final_state, final_deployed = _rebuild_state(
            weight_t,
            final_params,
            first_state,
            activation_t,
        )

        base_losses = _joint_loss(
            weight_t,
            parent_params,
            activation_t,
            parent_deployed,
        )
        final_losses = _joint_loss(
            weight_t,
            final_params,
            activation_t,
            final_deployed,
        )
        # Use the same cross-fold acceptance principle as v140, but on the
        # complete two-sided product residual after both alternating steps.
        if _robust_score(final_losses) < _robust_score(base_losses):
            return {"weight_params": final_params, "activation_state": final_state}
    except (RuntimeError, ValueError, FloatingPointError, torch.linalg.LinAlgError):
        # A structural control must never make a layer invalid; v140 remains
        # the exact fallback whenever a rank/shape corner case appears.
        pass
    return base


# v140's dynamic Linear path is retained; the returned state is rebuilt above.
hif4_dynamic_quantize_activation = _v140.hif4_dynamic_quantize_activation

# v86 C86 Attention is the fixed comparison arm.
hif4_calibration_attention = _v86.hif4_calibration_attention
hif4_dynamic_quantize_q = _v86.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _v86.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _v86.hif4_dynamic_quantize_v


__all__ = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
