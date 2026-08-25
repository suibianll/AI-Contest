from __future__ import annotations

from typing import Mapping, Sequence

import torch


NVFP4_BLOCK_SIZE = 16
HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0 ** -48
_E6M2_MAX = 49152.0
_BF16_ONE_SEVENTH = 0.142578125

# FP4 E2M1 finite carrier values used by the synthetic competition suite.
NVFP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    if quant_float.ndim == 0 or scale_float.ndim == 0:
        raise ValueError("NVFP4 tensors must have at least one dimension")
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(f"last dim {channels} is not divisible by {blk_size}")
    expected = quant_float.shape[:-1] + (channels // blk_size,)
    if tuple(scale_float.shape) != tuple(expected):
        raise ValueError(
            f"scale shape {tuple(scale_float.shape)} does not match {tuple(expected)}"
        )
    grouped = quant_float.reshape(*quant_float.shape[:-1], -1, blk_size)
    decoded = grouped.to(torch.float32) * scale_float.to(torch.float32).unsqueeze(-1)
    return decoded.flatten(-2, -1).to(torch.bfloat16)


def quantize_to_nvfp4(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense tensor to the synthetic suite's NVFP4 carrier."""
    if dense.ndim == 0:
        raise ValueError("dense must have at least one dimension")
    channels = int(dense.shape[-1])
    if channels % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"last dim {channels} is not divisible by 16")
    x = torch.nan_to_num(dense.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    grouped = x.reshape(*x.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    amax = grouped.abs().amax(dim=-1)
    scale = (amax / 6.0).clamp_min(_E6M2_MIN)
    levels = NVFP4_LEVELS.to(device=x.device)
    normalized = grouped / scale.unsqueeze(-1)
    distance = (normalized.unsqueeze(-1) - levels).abs()
    carrier = levels[distance.argmin(dim=-1)].reshape_as(x).to(torch.bfloat16)
    return carrier, scale


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_E6M2_MIN,
        posinf=_E6M2_MAX,
        neginf=_E6M2_MIN,
    ).clamp(min=_E6M2_MIN, max=_E6M2_MAX)
    exponent = torch.floor(torch.log2(x))
    base = torch.pow(2.0, exponent)
    mantissa = torch.round((x / base - 1.0) * 4.0).to(torch.int64)
    carry = mantissa >= 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa).clamp(0, 3)
    exponent_field = (exponent.to(torch.int64) + 48).clamp(0, 63)
    return (exponent_field * 4 + mantissa).clamp(0, 254).to(torch.int16)


def _e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    value = code.to(torch.int64).clamp(0, 254)
    exponent_field = torch.bitwise_right_shift(value, 2)
    mantissa_field = torch.bitwise_and(value, 3)
    exponent = exponent_field.to(torch.float32) - 48.0
    return torch.pow(2.0, exponent) * (1.0 + mantissa_field.to(torch.float32) * 0.25)


def _standard_e6m2_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    high_precision = (amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH).to(torch.float32)
    code = _e6m2_encode_nearest(high_precision)
    return code, _e6m2_decode(code)


def _pack_hif4(
    prefix: tuple[int, ...],
    blocks: int,
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
    sign: torch.Tensor,
    mantissa: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sign_out = sign.reshape(*prefix, blocks, 8, 2, 4)
    mantissa_out = mantissa.reshape(*prefix, blocks, 8, 2, 4)
    sign_out = torch.where(mantissa_out == 0.0, torch.zeros_like(sign_out), sign_out)
    return {
        "scale_factor": scale_factor.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign_out,
        "mant": mantissa_out,
    }


def standard_hif4_quantize(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    """Independent standard HiF4 quantization used as the reference baseline."""
    if dense.ndim == 0:
        raise ValueError("dense must have at least one dimension")
    prefix = tuple(int(value) for value in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % HIF4_BLOCK_SIZE != 0:
        raise ValueError(f"last dim {channels} is not divisible by 64")
    blocks = channels // HIF4_BLOCK_SIZE
    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * 7.0,
        neginf=-_E6M2_MAX * 7.0,
    )
    grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    absolute = grouped.abs()
    sign = torch.sign(grouped)
    max4 = absolute.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    _, scale_factor = _standard_e6m2_scale(amax)
    scale_lv2 = 1.0 + (max8 >= 4.0 * scale_factor[..., None]).to(torch.float32)
    max4_scale = scale_factor[..., None, None] * scale_lv2[..., None]
    scale_lv3 = 1.0 + (max4 >= 2.0 * max4_scale).to(torch.float32)
    denominator = (
        scale_factor[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = torch.round(absolute * (4.0 / denominator)).clamp(0.0, 7.0) * 0.25
    return _pack_hif4(prefix, blocks, scale_factor, scale_lv2, scale_lv3, sign, mantissa)


def dequantize_hif4(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    validate_hif4_params(params)
    dense = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return dense.flatten(start_dim=-4, end_dim=-1)


def validate_hif4_params(
    params: Mapping[str, torch.Tensor], logical_shape: Sequence[int] | None = None
) -> None:
    expected_keys = {"scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"}
    if set(params) != expected_keys:
        raise ValueError(f"HiF4 keys must be {sorted(expected_keys)}")
    for name, tensor in params.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"HiF4 {name} must be a tensor")
        if tensor.is_complex() or not torch.isfinite(tensor.to(torch.float32)).all():
            raise ValueError(f"HiF4 {name} must be finite and real")
    sign = params["sign"]
    mant = params["mant"]
    if not bool(torch.all((sign == -1) | (sign == 0) | (sign == 1))):
        raise ValueError("HiF4 sign contains an invalid value")
    if not bool(torch.all((mant >= 0) & (mant <= 1.75))):
        raise ValueError("HiF4 mantissa is outside [0, 1.75]")
    if not bool(torch.all((params["scale_lv2"] == 1) | (params["scale_lv2"] == 2))):
        raise ValueError("HiF4 scale_lv2 must be 1 or 2")
    if not bool(torch.all((params["scale_lv3"] == 1) | (params["scale_lv3"] == 2))):
        raise ValueError("HiF4 scale_lv3 must be 1 or 2")
    if logical_shape is not None:
        shape = tuple(int(value) for value in logical_shape)
        if len(shape) == 0 or shape[-1] % HIF4_BLOCK_SIZE != 0:
            raise ValueError("logical shape is not HiF4 block aligned")
        blocks = shape[-1] // HIF4_BLOCK_SIZE
        prefix = shape[:-1]
        expected = {
            "scale_factor": prefix + (blocks, 1, 1, 1),
            "scale_lv2": prefix + (blocks, 8, 1, 1),
            "scale_lv3": prefix + (blocks, 8, 2, 1),
            "sign": prefix + (blocks, 8, 2, 4),
            "mant": prefix + (blocks, 8, 2, 4),
        }
        for name, expected_shape in expected.items():
            if tuple(params[name].shape) != expected_shape:
                raise ValueError(f"HiF4 {name} shape {tuple(params[name].shape)} != {expected_shape}")
