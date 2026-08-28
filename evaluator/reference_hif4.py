"""Evaluator-owned implementations of the official HiF4 scoring primitives.

This module is the single source of truth for the standard HiF4 baseline,
NVFP4 dequantization, HiF4 validation/dequantization, and calibration-state
validation used by local scoring.  It never imports an evaluated solution.

- standard amax/7 base scale with a BF16 intermediate,
- unsigned E6M2 scale codes,
- MSE-optimal selection over the eight legal lv2/lv3 configurations,
- 3-bit mantissa rounding, clamped to [0, 7] * 0.25,
- canonical zero sign.

It deliberately contains no offset search, no refinement, and no
candidate-side logic.  It must never import or call into the evaluated
solution: candidate changes to ``_dense_to_hif4`` cannot alter the
standard denominator by construction.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


_HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_HIF4_MAX_INNER = 7.0
_BF16_ONE_SEVENTH = 0.142578125

__all__ = [
    "dequantize_nvfp4",
    "dequantize_hif4",
    "encode_standard_hif4",
    "decode_standard_hif4",
    "e6m2_encode_nearest",
    "e6m2_decode",
    "standard_e6m2_scale",
    "validate_hif4_params",
    "validate_state",
]

_ALLOWED_STATE_DTYPES = {
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    """Official NVFP4 dequantization, including the BF16 rounding point."""

    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"Last dim {channels} not divisible by NVFP4 block size {blk_size}"
        )
    expected_scale_shape = tuple(quant_float.shape[:-1]) + (channels // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 scale shape {tuple(scale_float.shape)} != {expected_scale_shape}"
        )
    grouped = quant_float.unflatten(-1, (-1, blk_size))
    result = grouped * scale_float.unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)


def e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    """Encode non-negative FP32 values into finite unsigned E6M2 codes."""

    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_E6M2_MIN,
        posinf=_E6M2_MAX,
        neginf=_E6M2_MIN,
    ).clamp(min=_E6M2_MIN, max=_E6M2_MAX)

    exponent = torch.floor(torch.log2(x))
    base = torch.pow(2.0, exponent)
    mantissa_field = torch.round((x / base - 1.0) * 4.0).to(torch.int64)

    carry = mantissa_field >= 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa_field = torch.where(
        carry, torch.zeros_like(mantissa_field), mantissa_field
    ).clamp(min=0, max=3)

    exponent_field = (exponent.to(torch.int64) + 48).clamp(min=0, max=63)
    code = exponent_field * 4 + mantissa_field
    return code.clamp(min=0, max=254).to(torch.int16)


def e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(min=0, max=254)
    exponent_field = torch.bitwise_right_shift(c, 2)
    mantissa_field = torch.bitwise_and(c, 3)
    exponent = exponent_field.to(torch.float32) - 48.0
    return torch.pow(2.0, exponent) * (
        1.0 + mantissa_field.to(torch.float32) * 0.25
    )


def standard_e6m2_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the official amax/7 base scale with a BF16 intermediate."""

    high_precision_scale = (
        amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = e6m2_encode_nearest(high_precision_scale)
    return code, e6m2_decode(code)


def _solve_standard_hierarchy(
    absolute: torch.Tensor,
    scale_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the minimum-MSE legal hierarchy for each eight-value group."""

    losses: list[torch.Tensor] = []
    for total_exponent in (0, 1, 2):
        local_scale = scale_factor[..., None, None, None] * float(
            1 << total_exponent
        )
        mantissa = (
            torch.round(absolute * (4.0 / local_scale)).clamp_(0.0, 7.0) * 0.25
        )
        losses.append((absolute - mantissa * local_scale).square().sum(dim=-1))

    loss_0, loss_1, loss_2 = losses
    choose_01 = loss_1 < loss_0
    choose_12 = loss_2 < loss_1
    cost_lv2_1 = torch.minimum(loss_0, loss_1).sum(dim=-1)
    cost_lv2_2 = torch.minimum(loss_1, loss_2).sum(dim=-1)
    use_lv2_2 = cost_lv2_2 < cost_lv2_1
    use_lv3_2 = torch.where(use_lv2_2[..., None], choose_12, choose_01)

    scale_lv2 = 1.0 + use_lv2_2.to(torch.float32)
    scale_lv3 = 1.0 + use_lv3_2.to(torch.float32)
    denominator = (
        scale_factor[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )
    return scale_lv2, scale_lv3, mantissa


def encode_standard_hif4(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor with the standard HiF4 codec (frozen)."""

    if dense.ndim < 1:
        raise ValueError("dense must have at least one dimension")
    prefix = tuple(int(v) for v in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by HiF4 block size 64"
        )
    blocks = channels // _HIF4_BLOCK_SIZE

    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    sign = torch.sign(x_grouped)

    amax = x_abs.amax(dim=(-1, -2, -3))
    _, standard_scale = standard_e6m2_scale(amax)
    scale_lv2, scale_lv3, mantissa = _solve_standard_hierarchy(
        x_abs, standard_scale
    )

    sign_out = sign.reshape(*prefix, blocks, 8, 2, 4)
    mantissa_out = mantissa.reshape(*prefix, blocks, 8, 2, 4)
    sign_out = torch.where(
        mantissa_out == 0.0, torch.zeros_like(sign_out), sign_out
    )
    return {
        "scale_factor": standard_scale.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign_out,
        "mant": mantissa_out,
    }


def decode_standard_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Dequantize standard HiF4 parameters back to a dense tensor."""

    dense = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return dense.flatten(start_dim=-4, end_dim=-1)


def validate_hif4_params(
    params: Mapping[str, torch.Tensor],
    logical_shape: Sequence[int],
) -> None:
    """Validate all official HiF4 fields and their logical tensor shape."""

    expected_keys = {"scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"}
    if not isinstance(params, Mapping) or set(params) != expected_keys:
        raise ValueError(f"HiF4 keys must be {sorted(expected_keys)}")
    shape = tuple(int(value) for value in logical_shape)
    if not shape or shape[-1] % _HIF4_BLOCK_SIZE != 0:
        raise ValueError("logical shape is not HiF4 block aligned")
    blocks = shape[-1] // _HIF4_BLOCK_SIZE
    prefix = shape[:-1]
    expected_shapes = {
        "scale_factor": prefix + (blocks, 1, 1, 1),
        "scale_lv2": prefix + (blocks, 8, 1, 1),
        "scale_lv3": prefix + (blocks, 8, 2, 1),
        "sign": prefix + (blocks, 8, 2, 4),
        "mant": prefix + (blocks, 8, 2, 4),
    }
    for name, tensor in params.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"HiF4 {name} must be a tensor")
        if tensor.layout != torch.strided or tensor.requires_grad or tensor.is_complex():
            raise ValueError(f"HiF4 {name} must be a real dense tensor without gradients")
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"HiF4 {name} shape {tuple(tensor.shape)} != {expected_shapes[name]}"
            )
        if not bool(torch.isfinite(tensor.to(torch.float32)).all()):
            raise ValueError(f"HiF4 {name} must be finite")

    scale_factor = params["scale_factor"].to(torch.float32)
    nearest_scale = e6m2_decode(e6m2_encode_nearest(scale_factor))
    if not bool(torch.all(scale_factor == nearest_scale)):
        raise ValueError("HiF4 scale_factor contains a non-E6M2 value")
    if not bool(torch.all((params["scale_lv2"] == 1) | (params["scale_lv2"] == 2))):
        raise ValueError("HiF4 scale_lv2 must be 1 or 2")
    if not bool(torch.all((params["scale_lv3"] == 1) | (params["scale_lv3"] == 2))):
        raise ValueError("HiF4 scale_lv3 must be 1 or 2")
    if not bool(torch.all((params["sign"] == -1) | (params["sign"] == 0) | (params["sign"] == 1))):
        raise ValueError("HiF4 sign must be -1, 0, or 1")
    mantissa = params["mant"].to(torch.float32)
    if not bool(torch.all((mantissa >= 0.0) & (mantissa <= 1.75))):
        raise ValueError("HiF4 mantissa is outside [0, 1.75]")
    if not bool(torch.all(mantissa * 4.0 == torch.round(mantissa * 4.0))):
        raise ValueError("HiF4 mantissa must be a multiple of 0.25")


def dequantize_hif4(
    params: Mapping[str, torch.Tensor],
    logical_shape: Sequence[int],
) -> torch.Tensor:
    """Validate and independently dequantize candidate HiF4 parameters."""

    validate_hif4_params(params, logical_shape)
    dense = (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    )
    return dense.flatten(start_dim=-4, end_dim=-1)


def validate_state(value: Any, *, max_depth: int = 8, max_nodes: int = 4096) -> None:
    """Validate an activation/Q/K/V state exactly at the official API boundary."""

    nodes = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError(f"state node count exceeds {max_nodes}")
        if depth > max_depth:
            raise ValueError(f"state depth exceeds {max_depth}")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("state float must be finite")
            return
        if isinstance(item, str):
            return
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                raise ValueError("state tensors must be CPU tensors")
            if item.layout != torch.strided or item.requires_grad or item.is_complex():
                raise ValueError("state tensor must be real, dense, and gradient-free")
            if item.dtype not in _ALLOWED_STATE_DTYPES:
                raise ValueError(f"unsupported state tensor dtype: {item.dtype}")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise ValueError("state tensor must be finite")
            return
        if isinstance(item, dict):
            identity = id(item)
            if identity in active:
                raise ValueError("state contains a cycle")
            active.add(identity)
            try:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise ValueError("state dict keys must be strings")
                    visit(child, depth + 1)
            finally:
                active.remove(identity)
            return
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("state contains a cycle")
            active.add(identity)
            try:
                for child in item:
                    visit(child, depth + 1)
            finally:
                active.remove(identity)
            return
        raise ValueError(f"state contains unsupported type: {type(item).__name__}")

    visit(value, 0)
