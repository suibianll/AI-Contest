"""Standard HiF4 baseline for the official side-weight calibration experiment.

Both the Linear and Attention sides mirror the reference codec exactly:
every API decodes its NVFP4 input through the official BF16 intermediate
and returns the standard HiF4 encoding (amax/7 E6M2 scale, MSE-optimal
lv2/lv3 hierarchy, 3-bit mantissa, canonical zero sign).  There is no
calibration, no search, and no state content, so the local proxy gain is
zero by construction and the official score of this file anchors the
"standard behaviour" point of the official scoring curve.

Companion candidates of the same experiment:
- v163 keeps the v160 Linear and swaps Attention to this standard codec;
- v164 keeps the v160 Attention and swaps Linear to this standard codec.
"""

from __future__ import annotations

from typing import Any

import torch


_HIF4_BLOCK_SIZE = 64
_NVFP4_BLOCK_SIZE = 16
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_HIF4_MAX_INNER = 7.0
_BF16_ONE_SEVENTH = 0.142578125


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Official NVFP4 dequantization with the BF16 rounding point."""

    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by NVFP4 block size {blk_size}"
        )
    expected_scale_shape = tuple(quant_float.shape[:-1]) + (channels // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 scale shape {tuple(scale_float.shape)} != {expected_scale_shape}"
        )
    grouped = quant_float.detach().to(torch.float32).unflatten(-1, (-1, blk_size))
    result = grouped * scale_float.detach().to(torch.float32).unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
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


def _e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(min=0, max=254)
    exponent_field = torch.bitwise_right_shift(c, 2)
    mantissa_field = torch.bitwise_and(c, 3)
    exponent = exponent_field.to(torch.float32) - 48.0
    return torch.pow(2.0, exponent) * (
        1.0 + mantissa_field.to(torch.float32) * 0.25
    )


def _standard_e6m2_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official amax/7 base scale with a BF16 intermediate."""

    high_precision_scale = (
        amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = _e6m2_encode_nearest(high_precision_scale)
    return code, _e6m2_decode(code)


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


def _encode_standard_hif4(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor with the standard HiF4 codec."""

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
    _, standard_scale = _standard_e6m2_scale(amax)
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


def _standard_params(quant_float: torch.Tensor, scale_float: torch.Tensor):
    dense = dequantize_nvfp4(quant_float, scale_float).to(torch.float32)
    return _encode_standard_hif4(dense)


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Standard baseline: encode the weight, ignore calibration samples."""

    return {
        "weight_params": _standard_params(weight_quant, weight_scale),
        "activation_state": {},
    }


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    return _standard_params(activation_quant, activation_scale)


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Standard baseline: no attention state is needed."""

    return {"q_state": {}, "k_state": {}, "v_state": {}}


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _standard_params(q_quant, q_scale)


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _standard_params(k_quant, k_scale)


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _standard_params(v_quant, v_scale)
