"""HiF4 solution for the 2026 Huawei algorithm competition.

The implementation keeps the official HiF4 conversion as an explicit fallback,
selects calibration-gated equivalent scaling/reordering transforms, and applies
bounded scale/hierarchy refinement to difficult blocks. All calibration states
are plain CPU data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Union

import torch


_NVFP4_BLOCK_SIZE = 16
_HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_HIF4_MAX_INNER = 7.0
_BF16_ONE_SEVENTH = 0.142578125
_EPS = 1.0e-12

_LINEAR_STATS_TOKENS = 4096
_LINEAR_EVAL_TOKENS = 128
_LINEAR_WEIGHT_EVAL_ROWS = 256
_ATTN_STATS_TOKENS = 4096
_ATTN_EVAL_TOKENS = 128
_ATTN_TRUE_TOKENS = 64
_ATTN_FLAT_PRESSURE_SPAN = 0.75

# E6M2 code offsets.  Offset +2 is roughly the E6M2 analogue of the
# alternative 1.5x scale mode seen in microscaling scale search.  The
# extended set is attempted first; when its calibrated true-metric gate
# fails, the conservative set is the fallback stored per case.
_DYNAMIC_OFFSETS = (-2, -1, 1, 2, 3)
_DYNAMIC_OFFSET_SETS = ((-2, -1, 1, 2, 3), (-1, 2))
# The batched candidate solve materializes [candidates, blocks, 8, 2, 4]
# temporaries; beyond this element budget the sequential loop is used.
_BATCH_SOLVE_MAX_CANDIDATES = 32_768
_WEIGHT_OFFSETS = (-2, -1, 1, 2, 3)
_DYNAMIC_TOP_RANKS = (2,)
_LINEAR_DYNAMIC_OFFSETS = (-2, -1, 1, 2, 3)
_LINEAR_FAST_OFFSETS = (-2, -1, 2)
_LINEAR_DYNAMIC_TOP_RANKS: tuple[int, ...] = ()
_WEIGHT_TOP_RANKS: tuple[int, ...] = ()
_LINEAR_FULL_SEARCH_ELEMENTS = 262_144
_ATTN_FULL_SEARCH_ELEMENTS = 262_144


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize an NVFP4 carrier/scale pair to BF16."""

    if not torch.is_tensor(quant_float) or not torch.is_tensor(scale_float):
        raise TypeError("quant_float and scale_float must be torch.Tensor")
    if quant_float.ndim < 1:
        raise ValueError("quant_float must have at least one dimension")
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(
            f"Last dim {c} is not divisible by NVFP4 block size {blk_size}"
        )
    expected_scale_shape = tuple(quant_float.shape[:-1]) + (c // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"scale_float shape {tuple(scale_float.shape)} does not match "
            f"expected {expected_scale_shape}"
        )

    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)


def _dequantize_nvfp4_float32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
) -> torch.Tensor:
    """Match the supplied BF16 dequantizer, then use FP32 for optimization."""

    return dequantize_nvfp4(quant_float, scale_float).to(torch.float32)


def _sample_rows(x: torch.Tensor, limit: int) -> torch.Tensor:
    """Deterministically sample at most ``limit`` rows without random state."""

    rows = int(x.shape[0])
    if rows <= limit:
        return x
    step = max(1, (rows + limit - 1) // limit)
    return x[::step][:limit]


def _safe_positive_vector(x: torch.Tensor, length: int) -> torch.Tensor:
    """Return a finite, positive FP32 vector of the requested length."""

    y = x.detach().to(dtype=torch.float32).reshape(-1)
    if int(y.numel()) != length:
        raise ValueError(f"Expected vector of length {length}, got {y.numel()}")
    return torch.nan_to_num(
        y, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(_EPS)


def _normalize_importance(
    importance: Optional[torch.Tensor],
    length: int,
) -> Optional[torch.Tensor]:
    if importance is None:
        return None
    w = importance.detach().to(dtype=torch.float32).reshape(-1)
    if int(w.numel()) != length:
        raise ValueError(
            f"Expected importance of length {length}, got {w.numel()}"
        )
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = w.mean()
    if float(mean) <= _EPS:
        return torch.ones_like(w)
    return w / mean


def _identity_permutation(length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(length, dtype=torch.int64, device=device)


def _hierarchy_aware_permutation(
    first_range: torch.Tensor,
    second_range: torch.Tensor,
    combine_mode: int = 0,
    preserve_blocks: bool = False,
) -> torch.Tensor:
    """Cluster similarly scaled channels for the 64/8/4 HiF4 hierarchy.

    The two ranges describe the paired operands of an exactly equivalent
    transform (X/W or Q/K). Log-domain median normalization makes the
    ordering insensitive to the operands' unrelated global units.
    """

    if tuple(first_range.shape) != tuple(second_range.shape):
        raise ValueError("Paired channel ranges must have identical shapes")
    log_first = torch.log2(first_range.to(torch.float32).clamp_min(_EPS))
    log_second = torch.log2(second_range.to(torch.float32).clamp_min(_EPS))
    log_first = log_first - torch.median(log_first)
    log_second = log_second - torch.median(log_second)
    if int(combine_mode) == 0:
        pressure = torch.maximum(log_first, log_second).reshape(-1)
    elif int(combine_mode) == 1:
        pressure = (0.5 * (log_first + log_second)).reshape(-1)
    else:
        raise ValueError("Unsupported hierarchy pressure mode")
    if int(pressure.numel()) == 0:
        return torch.empty(0, dtype=torch.int64, device=pressure.device)
    if bool(preserve_blocks):
        if int(pressure.numel()) % _HIF4_BLOCK_SIZE != 0:
            raise ValueError("Block-preserving ordering requires a width divisible by 64")
        grouped = pressure.reshape(-1, _HIF4_BLOCK_SIZE)
        local_permutation = torch.argsort(grouped, dim=-1, descending=True)
        spread = grouped.amax(dim=-1) - grouped.amin(dim=-1)
        local_identity = torch.arange(
            _HIF4_BLOCK_SIZE,
            dtype=torch.int64,
            device=pressure.device,
        )[None, :].expand_as(local_permutation)
        local_permutation = torch.where(
            spread[:, None] >= 0.25,
            local_permutation,
            local_identity,
        )
        block_base = (
            torch.arange(
                int(grouped.shape[0]),
                dtype=torch.int64,
                device=pressure.device,
            )[:, None]
            * _HIF4_BLOCK_SIZE
        )
        return (local_permutation + block_base).reshape(-1)
    if float(pressure.max() - pressure.min()) < 0.25:
        return _identity_permutation(int(pressure.numel()), pressure.device)
    return torch.argsort(pressure, descending=True)


def _headwise_hierarchy_permutation(
    q_range: torch.Tensor,
    k_range: torch.Tensor,
    combine_mode: int = 0,
) -> torch.Tensor:
    """Return a local feature permutation for each paired Q/KV head."""

    if q_range.ndim != 2 or tuple(q_range.shape) != tuple(k_range.shape):
        raise ValueError("Headwise Q/K ranges must have shape [heads, head_dim]")
    q_log = torch.log2(q_range.to(torch.float32).clamp_min(_EPS))
    k_log = torch.log2(k_range.to(torch.float32).clamp_min(_EPS))
    q_log = q_log - q_log.median(dim=-1, keepdim=True).values
    k_log = k_log - k_log.median(dim=-1, keepdim=True).values
    if int(combine_mode) == 0:
        pressure = torch.maximum(q_log, k_log)
    elif int(combine_mode) == 1:
        pressure = 0.5 * (q_log + k_log)
    else:
        raise ValueError("Unsupported headwise hierarchy pressure mode")
    permutation = torch.argsort(pressure, dim=-1, descending=True)

    spread = pressure.amax(dim=-1) - pressure.amin(dim=-1)
    identity = torch.arange(
        int(pressure.shape[-1]), dtype=torch.int64, device=pressure.device
    ).expand_as(permutation)
    return torch.where(spread[:, None] >= 0.25, permutation, identity)


def _flatten_head_permutation(local_permutation: torch.Tensor) -> torch.Tensor:
    heads, head_dim = map(int, local_permutation.shape)
    base = torch.arange(
        heads, dtype=torch.int64, device=local_permutation.device
    )[:, None] * head_dim
    return (local_permutation.to(torch.int64) + base).reshape(-1)


def _candidate_is_safe(
    candidate: tuple[float, tuple[float, ...]],
    baseline: tuple[float, tuple[float, ...]],
    *,
    min_mean_improvement: float,
    worst_tolerance: float,
) -> bool:
    candidate_mean, candidate_cases = candidate
    baseline_mean, baseline_cases = baseline
    if not math.isfinite(candidate_mean):
        return False
    if candidate_mean > baseline_mean * (1.0 - min_mean_improvement):
        return False
    if len(candidate_cases) != len(baseline_cases):
        return False
    for current, reference in zip(candidate_cases, baseline_cases):
        if current > reference * (1.0 + worst_tolerance) + 1.0e-8:
            return False
    return True


def _attention_mask_consensus(
    candidate: tuple[float, tuple[float, ...]],
    baseline: tuple[float, tuple[float, ...]],
    min_improvement: float = 0.002,
) -> bool:
    """Require both non-causal and causal calibration domains to improve."""

    if len(candidate[1]) != len(baseline[1]):
        return False
    for mask_index in (0, 1):
        current = candidate[1][mask_index::2]
        reference = baseline[1][mask_index::2]
        if not current or len(current) != len(reference):
            return False
        current_mean = sum(current) / float(len(current))
        reference_mean = sum(reference) / float(len(reference))
        if current_mean > reference_mean * (1.0 - min_improvement):
            return False
    return True


def _center_attention_k(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    center_mode: int,
) -> torch.Tensor:
    """Apply a token-invariant K shift; softmax(QK^T) is unchanged."""

    mode = int(center_mode)
    if mode == 0:
        return dense
    if dense.ndim != 2 or int(dense.shape[0]) <= 0:
        raise ValueError("Attention centering expects a non-empty 2D tensor")
    if int(dense.shape[1]) != int(num_heads) * int(head_dim):
        raise ValueError("Invalid dimensions for attention centering")
    grouped = dense.reshape(-1, int(num_heads), int(head_dim))
    if mode == 1:
        center = grouped.mean(dim=0, keepdim=True)
    elif mode == 2:
        center = 0.5 * (
            grouped.amax(dim=0, keepdim=True)
            + grouped.amin(dim=0, keepdim=True)
        )
    else:
        raise ValueError("Unsupported attention center mode")
    return (grouped - center).reshape_as(dense)


def _attention_heavy_tail_guard(
    calib_qkv_list: Sequence[dict[str, Any]],
) -> bool:
    """Detect a broad K tail before enabling midrange centering.

    Midrange centering can amplify block-scale error for long, heavy-tailed
    heads. The guard uses only calibration K magnitudes and no scenario name;
    it relies only on the observed distribution, so the same guard generalizes
    across head sizes instead of treating shape as a scenario label.
    """

    samples: list[torch.Tensor] = []
    for sample in calib_qkv_list:
        k = _dequantize_nvfp4_float32(*sample["k"])
        rows = _sample_rows(k, 2048).abs().reshape(-1)
        if int(rows.numel()) > 0:
            samples.append(rows)
    if not samples:
        return False
    values = torch.cat(samples)
    median = torch.quantile(
        values, torch.tensor(0.50, dtype=values.dtype, device=values.device)
    )
    upper = torch.quantile(
        values, torch.tensor(0.99, dtype=values.dtype, device=values.device)
    )
    tail_ratio = upper / median.clamp_min(_EPS)
    return bool(float(tail_ratio) > 6.0)


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    """Encode non-negative FP32 values into finite unsigned E6M2 codes.

    Codes 0..254 are finite and monotonic.  Code 255 is NaN and is never
    produced.  Round-to-nearest-even is inherited from ``torch.round``.
    """

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
    """Compute the official amax/7 base scale with a BF16 intermediate."""

    high_precision_scale = (
        amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = _e6m2_encode_nearest(high_precision_scale)
    return code, _e6m2_decode(code)


def _offsets_as_tuple(offsets: Optional[Iterable[int]]) -> tuple[int, ...]:
    ordered = [0]
    if offsets is None:
        return (0,)
    if torch.is_tensor(offsets):
        values = offsets.detach().to("cpu").reshape(-1).tolist()
    else:
        values = list(offsets)
    for raw in values:
        value = int(raw)
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _solve_exact_hierarchy(
    x_abs: torch.Tensor,
    scale: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly solve lv2/lv3 for fixed scales using three loss tables.

    Args:
        x_abs: ``[num_blocks, 8, 2, 4]`` absolute values.
        scale: ``[num_blocks]`` finite E6M2 values.
        importance: optional tensor with the same shape as ``x_abs``.
    """

    losses: list[torch.Tensor] = []
    mantissas: list[torch.Tensor] = []

    for total_exponent in (0, 1, 2):
        local_scale = scale[:, None, None, None] * float(1 << total_exponent)
        mant_code = torch.round(x_abs * (4.0 / local_scale)).clamp_(0.0, 7.0)
        mantissa = mant_code * 0.25
        error = (x_abs - mantissa * local_scale).square()
        if importance is not None:
            error = error * importance
        losses.append(error.sum(dim=-1))
        mantissas.append(mantissa)

    loss_0, loss_1, loss_2 = losses
    choose_01 = loss_1 < loss_0
    choose_12 = loss_2 < loss_1

    cost_e2_0 = torch.minimum(loss_0, loss_1).sum(dim=-1)
    cost_e2_1 = torch.minimum(loss_1, loss_2).sum(dim=-1)
    e2 = cost_e2_1 < cost_e2_0
    e3 = torch.where(e2[..., None], choose_12, choose_01)

    block_loss = torch.where(e2, cost_e2_1, cost_e2_0).sum(dim=-1)
    total_exponent = e2.to(torch.int64)[..., None] + e3.to(torch.int64)

    # [N, 8, 2, 3, 4], gather the mantissa matching k=e2+e3.
    mantissa_stack = torch.stack(mantissas, dim=3)
    gather_index = total_exponent[..., None, None].expand(-1, -1, -1, 1, 4)
    mantissa = torch.gather(mantissa_stack, 3, gather_index).squeeze(3)

    scale_lv2 = 1.0 + e2.to(torch.float32)
    scale_lv3 = 1.0 + e3.to(torch.float32)
    return block_loss, scale_lv2, scale_lv3, mantissa


def _solve_hierarchy_candidates_batched(
    x_abs: torch.Tensor,
    candidate_codes: list[torch.Tensor],
    importance: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly solve lv2/lv3 for all candidate scales in one batched pass.

    Numerically identical to evaluating ``_solve_exact_hierarchy`` once per
    candidate and keeping the first strictly-better candidate (ties keep the
    earliest candidate, matching the sequential update order).

    Args:
        x_abs: ``[num_blocks, 8, 2, 4]`` absolute values.
        candidate_codes: list of ``[num_blocks]`` E6M2 codes.
        importance: optional tensor with the same shape as ``x_abs``.

    Returns:
        (loss [num_blocks], scale [num_blocks], scale_lv2 [num_blocks, 8],
         scale_lv3 [num_blocks, 8, 2], mantissa [num_blocks, 8, 2, 4]).
    """

    code_all = torch.stack(candidate_codes, dim=0)
    scale_all = _e6m2_decode(code_all)
    num_candidates = int(code_all.shape[0])
    num_blocks = int(x_abs.shape[0])

    losses: list[torch.Tensor] = []
    for total_exponent in (0, 1, 2):
        local_scale = (
            scale_all[:, :, None, None, None] * float(1 << total_exponent)
        )
        mant_code = torch.round(
            x_abs[None] * (4.0 / local_scale)
        ).clamp_(0.0, 7.0)
        mantissa = mant_code * 0.25
        error = (x_abs[None] - mantissa * local_scale).square()
        if importance is not None:
            error = error * importance[None]
        losses.append(error.sum(dim=-1))

    loss_0, loss_1, loss_2 = losses
    choose_01 = loss_1 < loss_0
    choose_12 = loss_2 < loss_1
    cost_e2_0 = torch.minimum(loss_0, loss_1).sum(dim=-1)
    cost_e2_1 = torch.minimum(loss_1, loss_2).sum(dim=-1)
    e2 = cost_e2_1 < cost_e2_0
    e3 = torch.where(e2[..., None], choose_12, choose_01)
    block_loss = torch.where(e2, cost_e2_1, cost_e2_0).sum(dim=-1)

    candidate_min = block_loss.min(dim=0).values
    is_min = block_loss == candidate_min[None, :]
    order = torch.arange(
        num_candidates, dtype=torch.int64, device=code_all.device
    )
    pick = torch.where(is_min, order[:, None], num_candidates)
    candidate_index = pick.min(dim=0).values

    flat_index = candidate_index[None]
    scale_sel = scale_all.gather(0, flat_index.expand_as(scale_all))[0]
    e2_sel = e2.gather(
        0,
        flat_index[..., None].expand(-1, -1, x_abs.shape[1]),
    )[0]
    e3_sel = e3.gather(
        0,
        flat_index[..., None, None].expand(
            -1, -1, x_abs.shape[1], x_abs.shape[2]
        ),
    )[0]

    scale_lv2 = 1.0 + e2_sel.to(torch.float32)
    scale_lv3 = 1.0 + e3_sel.to(torch.float32)
    denominator = (
        scale_sel[:, None, None, None]
        * scale_lv2[:, :, None, None]
        * scale_lv3[:, :, :, None]
    )
    mantissa = (
        torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )
    return candidate_min, scale_sel, scale_lv2, scale_lv3, mantissa


def _pack_hif4_params(
    prefix: tuple[int, ...],
    blocks: int,
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
    sign: torch.Tensor,
    mantissa: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Canonical zero: it is numerically irrelevant, but avoids relying on a
    # checker accepting sign=+/-1 when the final mantissa is zero.
    sign_out = sign.reshape(*prefix, blocks, 8, 2, 4)
    mantissa_out = mantissa.reshape(*prefix, blocks, 8, 2, 4)
    sign_out = torch.where(
        mantissa_out == 0.0, torch.zeros_like(sign_out), sign_out
    )
    return {
        "scale_factor": scale_factor.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign_out,
        "mant": mantissa_out,
    }


def _dense_to_hif4(
    dense: torch.Tensor,
    *,
    importance: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    search_top_ranks: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
    selection_mode: str = "relative",
) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor into valid HiF4 parameters."""

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

    max4 = x_abs.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    standard_code, standard_scale = _standard_e6m2_scale(amax)

    e2 = max8 >= (4.0 * standard_scale[..., None])
    scale_lv2 = 1.0 + e2.to(torch.float32)
    e3 = max4 >= (
        2.0 * standard_scale[..., None, None] * scale_lv2[..., None]
    )
    scale_lv3 = 1.0 + e3.to(torch.float32)

    denominator = (
        standard_scale[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )

    offsets = _offsets_as_tuple(search_offsets)
    if search_top_ranks is None:
        top_ranks: tuple[int, ...] = ()
    elif torch.is_tensor(search_top_ranks):
        top_ranks = tuple(
            int(v) for v in search_top_ranks.detach().to("cpu").reshape(-1).tolist()
            if 1 <= int(v) <= _HIF4_BLOCK_SIZE
        )
    else:
        top_ranks = tuple(
            int(v) for v in search_top_ranks
            if 1 <= int(v) <= _HIF4_BLOCK_SIZE
        )
    top_ranks = tuple(dict.fromkeys(top_ranks))
    refine_ratio = max(0.0, min(float(max_refine_ratio), 1.0))
    if refine_ratio <= 0.0 or (len(offsets) == 0 and len(top_ranks) == 0):
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is not None:
        channel_importance = channel_importance.to(x.device)
    if channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
        weighted_energy = x_abs.square()
        importance_view = None
    else:
        importance_view = channel_importance.reshape(
            *([1] * len(prefix)), blocks, 8, 2, 4
        )
        weighted_error = (x_abs - mantissa * denominator).square() * importance_view
        weighted_energy = x_abs.square() * importance_view

    standard_loss = weighted_error.sum(dim=(-1, -2, -3))
    energy = weighted_energy.sum(dim=(-1, -2, -3))
    if selection_mode == "hybrid":
        # For static Weight, pure relative error over-prioritizes tiny-energy
        # blocks while pure absolute error can be dominated by a few rows.
        # L/sqrt(E) is a stable compromise under a fixed refinement budget.
        normalized_error = standard_loss / torch.sqrt(energy + _EPS)
    elif selection_mode == "absolute":
        normalized_error = standard_loss
    elif selection_mode == "relative":
        normalized_error = standard_loss / (energy + _EPS)
    else:
        raise ValueError("selection_mode must be relative, hybrid, or absolute")

    flat_error = normalized_error.reshape(-1)
    hard_mask = flat_error > float(error_threshold)
    hard_indices = torch.nonzero(hard_mask, as_tuple=False).reshape(-1)
    if int(hard_indices.numel()) == 0:
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    total_blocks = int(flat_error.numel())
    refine_cap = max(1, int(math.ceil(total_blocks * refine_ratio)))
    if max_refine_blocks is not None:
        refine_cap = min(refine_cap, max(1, int(max_refine_blocks)))
    if int(hard_indices.numel()) > refine_cap:
        hard_indices = torch.topk(flat_error, k=refine_cap, largest=True).indices

    x_flat = x_abs.reshape(-1, 8, 2, 4)
    x_hard = x_flat.index_select(0, hard_indices)
    standard_loss_hard = standard_loss.reshape(-1).index_select(0, hard_indices)
    standard_code_hard = standard_code.reshape(-1).index_select(0, hard_indices)

    best_loss = standard_loss_hard.clone()
    best_scale = standard_scale.reshape(-1).index_select(0, hard_indices).clone()
    best_lv2 = scale_lv2.reshape(-1, 8).index_select(0, hard_indices).clone()
    best_lv3 = scale_lv3.reshape(-1, 8, 2).index_select(0, hard_indices).clone()
    best_mantissa = mantissa.reshape(-1, 8, 2, 4).index_select(
        0, hard_indices
    ).clone()

    if channel_importance is None:
        importance_hard = None
    else:
        block_importance = channel_importance.reshape(blocks, 8, 2, 4)
        channel_block_ids = torch.remainder(hard_indices, blocks)
        importance_hard = block_importance.index_select(0, channel_block_ids)

    candidate_codes: list[torch.Tensor] = []
    for offset in offsets:
        candidate_codes.append(
            (standard_code_hard.to(torch.int64) + int(offset)).clamp(
                min=0, max=254
            )
        )
    if top_ranks:
        largest_rank = max(top_ranks)
        ranked_values = torch.topk(
            x_hard.reshape(-1, _HIF4_BLOCK_SIZE),
            k=largest_rank,
            dim=-1,
            largest=True,
            sorted=True,
        ).values
        for rank in top_ranks:
            rank_scale = (
                ranked_values[:, int(rank) - 1].to(torch.bfloat16)
                * _BF16_ONE_SEVENTH
            ).to(torch.float32)
            candidate_codes.append(
                _e6m2_encode_nearest(rank_scale).to(torch.int64)
            )

    if candidate_codes and (
        len(candidate_codes) * int(candidate_codes[0].numel())
        <= _BATCH_SOLVE_MAX_CANDIDATES
    ):
        # Batched solve: one vectorized pass over all candidate scales with
        # semantics identical to the sequential loop below (ties keep the
        # earliest candidate, and the standard path always wins ties).
        cand_loss, cand_scale, cand_lv2, cand_lv3, cand_mantissa = (
            _solve_hierarchy_candidates_batched(
                x_hard, candidate_codes, importance_hard
            )
        )
        improve = cand_loss < best_loss
        best_loss = torch.where(improve, cand_loss, best_loss)
        best_scale = torch.where(improve, cand_scale, best_scale)
        best_lv2 = torch.where(improve[:, None], cand_lv2, best_lv2)
        best_lv3 = torch.where(improve[:, None, None], cand_lv3, best_lv3)
        best_mantissa = torch.where(
            improve[:, None, None, None], cand_mantissa, best_mantissa
        )
    else:
        for candidate_code in candidate_codes:
            candidate_scale = _e6m2_decode(candidate_code)
            candidate_loss, candidate_lv2, candidate_lv3, candidate_mantissa = (
                _solve_exact_hierarchy(x_hard, candidate_scale, importance_hard)
            )

            improve = candidate_loss < best_loss
            best_loss = torch.where(improve, candidate_loss, best_loss)
            best_scale = torch.where(improve, candidate_scale, best_scale)
            best_lv2 = torch.where(improve[:, None], candidate_lv2, best_lv2)
            best_lv3 = torch.where(
                improve[:, None, None], candidate_lv3, best_lv3
            )
            best_mantissa = torch.where(
                improve[:, None, None, None], candidate_mantissa, best_mantissa
            )

    margin = max(0.0, min(float(accept_margin), 0.99))
    accept = best_loss <= ((1.0 - margin) * standard_loss_hard)
    if not bool(torch.any(accept)):
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    selected_indices = hard_indices[accept]
    out_scale = standard_scale.reshape(-1).clone()
    out_lv2 = scale_lv2.reshape(-1, 8).clone()
    out_lv3 = scale_lv3.reshape(-1, 8, 2).clone()
    out_mantissa = mantissa.reshape(-1, 8, 2, 4).clone()

    out_scale.index_copy_(0, selected_indices, best_scale[accept])
    out_lv2.index_copy_(0, selected_indices, best_lv2[accept])
    out_lv3.index_copy_(0, selected_indices, best_lv3[accept])
    out_mantissa.index_copy_(0, selected_indices, best_mantissa[accept])

    return _pack_hif4_params(
        prefix,
        blocks,
        out_scale,
        out_lv2,
        out_lv3,
        sign,
        out_mantissa,
    )


def _nvfp4_to_hif4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    *,
    multiplier: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    center_mode: int = 0,
    center_num_heads: Optional[int] = None,
    center_head_dim: Optional[int] = None,
    importance: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    search_top_ranks: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(quant_float, scale_float)
    channels = int(dense.shape[-1])
    if int(center_mode) != 0:
        if center_num_heads is None or center_head_dim is None:
            raise ValueError("Attention centering requires head metadata")
        dense = _center_attention_k(
            dense,
            int(center_num_heads),
            int(center_head_dim),
            int(center_mode),
        )
    if multiplier is not None:
        scale = _safe_positive_vector(multiplier, channels).to(dense.device)
        dense.mul_(scale.reshape(*([1] * (dense.ndim - 1)), channels))
    if permutation is not None:
        order = permutation.detach().to(
            device=dense.device, dtype=torch.int64
        ).reshape(-1)
        if int(order.numel()) != channels:
            raise ValueError("Permutation width does not match tensor width")
        dense = dense.index_select(-1, order)
    return _dense_to_hif4(
        dense,
        importance=importance,
        search_offsets=search_offsets,
        search_top_ranks=search_top_ranks,
        error_threshold=error_threshold,
        accept_margin=accept_margin,
        max_refine_ratio=max_refine_ratio,
        max_refine_blocks=max_refine_blocks,
    )


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    dense = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return dense.flatten(start_dim=-4, end_dim=-1)


def _smooth_scale(
    activation_amax: torch.Tensor,
    weight_amax: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (activation_amax + _EPS).pow(alpha) / (
        weight_amax + _EPS
    ).pow(1.0 - alpha)
    d = torch.nan_to_num(d, nan=1.0, posinf=8.0, neginf=1.0 / 8.0)
    d = d.clamp(min=1.0 / 8.0, max=8.0)
    # A global normalization prevents an arbitrary overall scale drift while
    # retaining the relative channel smoothing.
    geometric_mean = torch.exp(torch.log(d).mean())
    return (d / geometric_mean).clamp(min=1.0 / 8.0, max=8.0)


def _linear_candidate_metrics(
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    refinement_aware: bool = False,
) -> tuple[float, tuple[float, ...]]:
    """Score an equivalent Linear transform from operand-side statistics."""

    channels = int(weight.shape[1])
    order = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
    if int(order.numel()) != channels:
        raise ValueError("Linear candidate permutation has an invalid width")

    weight_smooth = (weight * d.unsqueeze(0)).index_select(-1, order)
    h_x = (activation_second_moment / d.square()).index_select(0, order)
    if refinement_aware:
        weight_params = _dense_to_hif4(
            weight_smooth,
            importance=h_x,
            search_offsets=_WEIGHT_OFFSETS,
            search_top_ranks=_WEIGHT_TOP_RANKS,
            error_threshold=1.0e-7,
            accept_margin=0.0,
            max_refine_ratio=0.30,
            max_refine_blocks=16_384,
        )
    else:
        weight_params = _dense_to_hif4(weight_smooth)
    weight_hat = _dequantize_hif4(weight_params)

    weight_error = (
        (weight_smooth - weight_hat).square() * h_x.unsqueeze(0)
    ).sum()
    weight_energy = (weight_smooth.square() * h_x.unsqueeze(0)).sum()
    weight_score = weight_error / (weight_energy + _EPS)

    h_w = _normalize_importance(weight_hat.square().sum(dim=0), channels)
    if h_w is None:
        h_w = torch.ones(channels, dtype=torch.float32, device=weight.device)

    case_scores: list[float] = []
    for sample in activation_samples:
        smooth = (sample / d.unsqueeze(0)).index_select(-1, order)
        if refinement_aware:
            params = _dense_to_hif4(
                smooth,
                importance=h_w,
                search_offsets=_LINEAR_DYNAMIC_OFFSETS,
                search_top_ranks=_LINEAR_DYNAMIC_TOP_RANKS,
                error_threshold=1.0e-7,
                accept_margin=0.02,
                max_refine_ratio=0.20,
                max_refine_blocks=8_192,
            )
        else:
            params = _dense_to_hif4(smooth)
        reconstructed = _dequantize_hif4(params)
        error = ((smooth - reconstructed).square() * h_w.unsqueeze(0)).sum()
        energy = (smooth.square() * h_w.unsqueeze(0)).sum()
        score = torch.nan_to_num(
            weight_score + error / (energy + _EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

    if not case_scores:
        case_scores.append(float(torch.nan_to_num(weight_score, nan=1.0e30)))
    mean_score = sum(case_scores) / float(len(case_scores))
    return mean_score, tuple(case_scores)


def _cpu_state_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        x.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Calibrate SmoothQuant state and quantize a static Linear weight."""

    if not isinstance(calib_activation_list, list) or not calib_activation_list:
        raise ValueError("calib_activation_list must be a non-empty list")

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale)
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor [out_features, in_features]")
    out_features, in_features = map(int, weight.shape)
    if in_features % _HIF4_BLOCK_SIZE != 0:
        raise ValueError("in_features must be divisible by 64")

    sum_square = torch.zeros(in_features, dtype=torch.float32, device=weight.device)
    activation_amax = torch.zeros_like(sum_square)
    token_count = 0
    activation_samples: list[torch.Tensor] = []

    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each calibration activation must be a (quant, scale) pair")
        activation = _dequantize_nvfp4_float32(pair[0], pair[1])
        if activation.ndim != 2 or int(activation.shape[1]) != in_features:
            raise ValueError("Calibration activation shape is incompatible with weight")
        stats_sample = _sample_rows(activation, _LINEAR_STATS_TOKENS)
        sum_square += stats_sample.square().sum(dim=0)
        activation_amax = torch.maximum(
            activation_amax, stats_sample.abs().amax(dim=0)
        )
        token_count += int(stats_sample.shape[0])
        activation_samples.append(
            _sample_rows(activation, _LINEAR_EVAL_TOKENS).clone()
        )

    activation_second_moment = sum_square / float(max(token_count, 1))
    weight_amax = weight.abs().amax(dim=0)

    identity_d = torch.ones(
        in_features, dtype=torch.float32, device=weight.device
    )
    identity_perm = _identity_permutation(in_features, weight.device)
    smooth_candidates = [
        identity_d,
        _smooth_scale(activation_amax, weight_amax, 0.25),
        _smooth_scale(activation_amax, weight_amax, 0.50),
    ]

    # Candidate search touches only sampled output rows.  The selected
    # transform is then applied to the full Weight exactly once.
    weight_sample = _sample_rows(weight, _LINEAR_WEIGHT_EVAL_ROWS)
    # On middle/small matrices, select equivalent transforms with the same
    # difficult-block policy used by the final Weight/Activation paths.  The
    # former standard-only proxy can rank transforms incorrectly after
    # refinement.  Large matrices retain the cheap proxy to protect the
    # five-minute global runtime limit; this changes calibration cost only.
    refinement_aware_selection = bool(int(weight.numel()) <= 4_194_304)
    baseline_metrics = _linear_candidate_metrics(
        weight_sample,
        activation_second_moment,
        activation_samples,
        identity_d,
        identity_perm,
        refinement_aware=refinement_aware_selection,
    )
    best_metrics = baseline_metrics
    best_d = identity_d
    best_perm = identity_perm

    for candidate_index, candidate_d in enumerate(smooth_candidates):
        candidate_permutations = [identity_perm]
        sorted_perm = _hierarchy_aware_permutation(
            activation_amax / candidate_d,
            weight_amax * candidate_d,
        )
        if not torch.equal(sorted_perm, identity_perm):
            candidate_permutations.append(sorted_perm)

        for candidate_perm in candidate_permutations:
            if candidate_index == 0 and torch.equal(candidate_perm, identity_perm):
                continue
            metrics = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                candidate_d,
                candidate_perm,
                refinement_aware=refinement_aware_selection,
            )
            uses_reordering = not torch.equal(candidate_perm, identity_perm)
            if (
                metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02 if uses_reordering else 0.01,
                    worst_tolerance=0.005 if uses_reordering else 0.02,
                )
            ):
                best_metrics = metrics
                best_d = candidate_d
                best_perm = candidate_perm

    # The max-pressure ordering above is a strong global default.  Evaluate
    # two complementary layouts only for the selected smoothing scale: a
    # joint-pressure global sort and a 64-block-preserving local sort.  This
    # adds hierarchy diversity without multiplying the full D/P grid.
    selected_d = best_d
    selected_first_range = activation_amax / selected_d
    selected_second_range = weight_amax * selected_d
    for combine_mode, preserve_blocks in ((1, False), (0, True)):
        candidate_perm = _hierarchy_aware_permutation(
            selected_first_range,
            selected_second_range,
            combine_mode=combine_mode,
            preserve_blocks=preserve_blocks,
        )
        if torch.equal(candidate_perm, identity_perm) or torch.equal(
            candidate_perm, best_perm
        ):
            continue
        metrics = _linear_candidate_metrics(
            weight_sample,
            activation_second_moment,
            activation_samples,
            selected_d,
            candidate_perm,
            refinement_aware=refinement_aware_selection,
        )
        if (
            metrics[0] < best_metrics[0]
            and _candidate_is_safe(
                metrics,
                baseline_metrics,
                min_mean_improvement=0.02,
                worst_tolerance=0.005,
            )
        ):
            best_metrics = metrics
            best_d = selected_d
            best_perm = candidate_perm

    weight_smooth = (weight * best_d.unsqueeze(0)).index_select(
        -1, best_perm
    )
    h_x_smooth = (activation_second_moment / best_d.square()).index_select(
        0, best_perm
    )
    weight_params = _dense_to_hif4(
        weight_smooth,
        importance=h_x_smooth,
        search_offsets=_WEIGHT_OFFSETS,
        search_top_ranks=_WEIGHT_TOP_RANKS,
        error_threshold=1.0e-7,
        accept_margin=0.0,
        max_refine_ratio=0.30 if int(weight.numel()) <= 4_194_304 else 0.15,
        max_refine_blocks=65_536,
        selection_mode="hybrid",
    )

    weight_hat = _dequantize_hif4(weight_params)
    activation_importance = _normalize_importance(
        weight_hat.square().sum(dim=0), in_features
    )
    if activation_importance is None:
        activation_importance = torch.ones_like(best_d)

    permutation_state = None
    if not torch.equal(best_perm, identity_perm):
        permutation_state = best_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
    smooth_inv_state = None
    if not torch.equal(best_d, identity_d):
        smooth_inv_state = _cpu_state_tensor(best_d.reciprocal())

    activation_state = {
        "smooth_inv": smooth_inv_state,
        "permutation": permutation_state,
        "importance": _cpu_state_tensor(activation_importance),
        "offsets": torch.tensor(
            _LINEAR_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"
        ),
        "fast_offsets": torch.tensor(
            _LINEAR_FAST_OFFSETS, dtype=torch.int8, device="cpu"
        ),
        "top_ranks": torch.tensor(
            _LINEAR_DYNAMIC_TOP_RANKS, dtype=torch.int8, device="cpu"
        ),
        "error_threshold": 1.0e-7,
        "accept_margin": 0.005,
        "max_refine_ratio": 0.40,
        "max_refine_blocks": 32_768,
        "in_features": int(in_features),
        "version": 8,
    }
    return {
        "weight_params": weight_params,
        "activation_state": activation_state,
    }


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(activation_state, dict):
        raise TypeError("activation_state must be a dict")
    channels = int(activation_quant.shape[-1])
    if channels != int(activation_state.get("in_features", -1)):
        raise ValueError("Activation hidden size does not match calibration state")
    search_offsets = activation_state["offsets"]
    if int(activation_quant.numel()) > _LINEAR_FULL_SEARCH_ELEMENTS:
        search_offsets = activation_state["fast_offsets"]
    return _nvfp4_to_hif4(
        activation_quant,
        activation_scale,
        multiplier=activation_state["smooth_inv"],
        permutation=activation_state["permutation"],
        importance=activation_state["importance"],
        search_offsets=search_offsets,
        search_top_ranks=activation_state["top_ranks"],
        error_threshold=float(activation_state["error_threshold"]),
        accept_margin=float(activation_state["accept_margin"]),
        max_refine_ratio=float(activation_state["max_refine_ratio"]),
        max_refine_blocks=int(activation_state["max_refine_blocks"]),
    )


def _smooth_qk_scale(
    q_peak: torch.Tensor,
    k_peak: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (k_peak + _EPS).pow(alpha) / (q_peak + _EPS).pow(1.0 - alpha)
    return torch.nan_to_num(
        d, nan=1.0, posinf=16.0, neginf=1.0 / 16.0
    ).clamp(min=1.0 / 16.0, max=16.0)


def _attention_candidate_metrics(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    d_kv: torch.Tensor,
    q_second_moment: torch.Tensor,
    k_effective_second_moment: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
) -> tuple[float, tuple[float, ...]]:
    """Q/K quantization proxy with GQA-aligned equivalent transforms."""

    group_size = q_num_heads // kv_num_heads
    d_q = d_kv.repeat_interleave(group_size, dim=0)
    d_k = d_kv.reciprocal()
    q_order = q_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)
    k_order = k_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)

    q_second_kv = q_second_moment.reshape(
        kv_num_heads, group_size, head_dim
    ).mean(dim=1)
    h_k = k_effective_second_moment * d_k.square()
    h_q = q_second_kv * d_kv.square()
    h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
    h_q_for_k = h_q.reshape(-1)
    h_k_for_q = h_k_for_q.index_select(0, q_order)
    h_q_for_k = h_q_for_k.index_select(0, k_order)
    h_k_for_q = _normalize_importance(h_k_for_q, q_num_heads * head_dim)
    h_q_for_k = _normalize_importance(h_q_for_k, kv_num_heads * head_dim)
    if h_k_for_q is None or h_q_for_k is None:
        raise RuntimeError("Attention importance construction failed")

    case_scores: list[float] = []
    for q_sample, k_sample in zip(q_samples, k_samples):
        q_smooth = (q_sample * d_q.reshape(1, -1)).index_select(
            -1, q_order
        )
        k_centered = _center_attention_k(
            k_sample, kv_num_heads, head_dim, center_mode
        )
        k_smooth = (k_centered * d_k.reshape(1, -1)).index_select(
            -1, k_order
        )
        q_hat = _dequantize_hif4(_dense_to_hif4(q_smooth))
        k_hat = _dequantize_hif4(_dense_to_hif4(k_smooth))

        q_error = (
            (q_smooth - q_hat).square() * h_k_for_q.reshape(1, -1)
        ).sum()
        q_energy = (q_smooth.square() * h_k_for_q.reshape(1, -1)).sum()
        k_error = (
            (k_smooth - k_hat).square() * h_q_for_k.reshape(1, -1)
        ).sum()
        k_energy = (k_smooth.square() * h_q_for_k.reshape(1, -1)).sum()
        score = torch.nan_to_num(
            q_error / (q_energy + _EPS) + k_error / (k_energy + _EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)


def _attention_dense_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    """Small-window FP32 GQA used only during calibration."""

    seq_len = int(q.shape[0])
    group_size = q_num_heads // kv_num_heads
    qh = q.to(torch.float32).reshape(
        seq_len, q_num_heads, head_dim
    ).transpose(0, 1)
    kh = k.to(torch.float32).reshape(
        seq_len, kv_num_heads, head_dim
    ).repeat_interleave(group_size, dim=1).transpose(0, 1)
    vh = v.to(torch.float32).reshape(
        seq_len, kv_num_heads, head_dim
    ).repeat_interleave(group_size, dim=1).transpose(0, 1)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * (
        1.0 / math.sqrt(float(head_dim))
    )
    if causal and seq_len > 1:
        mask = torch.triu(
            torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=logits.device
            ),
            diagonal=1,
        )
        logits = logits.masked_fill(mask[None, :, :], -float("inf"))
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, vh)
    return output.transpose(0, 1).reshape(seq_len, q_num_heads * head_dim)


def _transform_attention_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    d_kv: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = q_num_heads // kv_num_heads
    d_q = d_kv.repeat_interleave(group_size, dim=0).reshape(-1)
    d_k = d_kv.reciprocal().reshape(-1)
    q_order = q_permutation.to(device=q.device, dtype=torch.int64).reshape(-1)
    k_order = k_permutation.to(device=k.device, dtype=torch.int64).reshape(-1)
    q_transformed = (q * d_q.reshape(1, -1)).index_select(-1, q_order)
    k_centered = _center_attention_k(
        k, kv_num_heads, head_dim, center_mode
    )
    k_transformed = (k_centered * d_k.reshape(1, -1)).index_select(
        -1, k_order
    )
    return q_transformed, k_transformed


def _attention_true_metrics(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    v_samples: Sequence[torch.Tensor],
    d_kv: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
    q_importance: Optional[torch.Tensor] = None,
    k_importance: Optional[torch.Tensor] = None,
    refine_mode: int = 0,
    refine_ratios: Optional[tuple[float, float, float]] = None,
    search_offsets: Optional[tuple[int, ...]] = None,
) -> tuple[float, tuple[float, ...]]:
    """Evaluate real Attention MSE under causal and non-causal masks.

    refine_mode: 0 = standard transformed quantization, 1 = refine Q/K only,
    2 = refine Q/K/V. Every sample/mask pair is a separate risk case.
    """

    if search_offsets is None:
        search_offsets = _DYNAMIC_OFFSETS

    case_scores: list[float] = []
    if refine_ratios is None:
        q_ratio = 0.08 if int(refine_mode) >= 1 else 0.0
        k_ratio = 0.12 if int(refine_mode) >= 1 else 0.0
        v_ratio = 0.10 if int(refine_mode) >= 2 else 0.0
    else:
        q_ratio, k_ratio, v_ratio = (
            max(0.0, min(float(value), 1.0)) for value in refine_ratios
        )
    for q, k, v in zip(q_samples, k_samples, v_samples):
        q_transformed, k_transformed = _transform_attention_pair(
            q,
            k,
            d_kv,
            q_num_heads,
            kv_num_heads,
            head_dim,
            q_permutation,
            k_permutation,
            center_mode,
        )
        if q_ratio > 0.0:
            q_params = _dense_to_hif4(
                q_transformed,
                importance=q_importance,
                search_offsets=search_offsets,
                search_top_ranks=_DYNAMIC_TOP_RANKS,
                error_threshold=1.0e-7,
                accept_margin=0.03,
                max_refine_ratio=q_ratio,
                max_refine_blocks=4_096,
            )
        else:
            q_params = _dense_to_hif4(q_transformed)
        if k_ratio > 0.0:
            k_params = _dense_to_hif4(
                k_transformed,
                importance=k_importance,
                search_offsets=search_offsets,
                search_top_ranks=_DYNAMIC_TOP_RANKS,
                error_threshold=1.0e-7,
                accept_margin=0.03,
                max_refine_ratio=k_ratio,
                max_refine_blocks=6_144,
            )
        else:
            k_params = _dense_to_hif4(k_transformed)
        if v_ratio > 0.0:
            v_params = _dense_to_hif4(
                v,
                search_offsets=search_offsets,
                search_top_ranks=_DYNAMIC_TOP_RANKS,
                error_threshold=1.0e-7,
                accept_margin=0.01,
                max_refine_ratio=v_ratio,
                max_refine_blocks=6_144,
            )
        else:
            v_params = _dense_to_hif4(v)
        q_hat = _dequantize_hif4(q_params)
        k_hat = _dequantize_hif4(k_params)
        v_hat = _dequantize_hif4(v_params)
        for causal in (False, True):
            reference = _attention_dense_output(
                q,
                k,
                v,
                q_num_heads,
                kv_num_heads,
                head_dim,
                causal,
            )
            candidate = _attention_dense_output(
                q_hat,
                k_hat,
                v_hat,
                q_num_heads,
                kv_num_heads,
                head_dim,
                causal,
            )
            loss = (reference - candidate).square().mean()
            energy = reference.square().mean()
            normalized = torch.nan_to_num(
                loss / (energy + _EPS),
                nan=1.0e30,
                posinf=1.0e30,
                neginf=1.0e30,
            )
            case_scores.append(float(normalized))
    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)


def _attention_jacobian_importance(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    v_samples: Sequence[torch.Tensor],
    d_kv: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal softmax-output sensitivity in selected Q/K coordinates."""

    group_size = q_num_heads // kv_num_heads
    q_importance = torch.zeros(
        q_num_heads, head_dim, dtype=torch.float32, device=d_kv.device
    )
    k_importance_q_heads = torch.zeros_like(q_importance)
    contribution_count = 0
    for q, k, v in zip(q_samples, k_samples, v_samples):
        q_transformed, k_transformed = _transform_attention_pair(
            q,
            k,
            d_kv,
            q_num_heads,
            kv_num_heads,
            head_dim,
            q_permutation,
            k_permutation,
            center_mode,
        )
        seq_len = int(q.shape[0])
        qh = q_transformed.reshape(
            seq_len, q_num_heads, head_dim
        ).transpose(0, 1)
        kh = k_transformed.reshape(
            seq_len, kv_num_heads, head_dim
        ).repeat_interleave(group_size, dim=1).transpose(0, 1)
        vh = v.reshape(
            seq_len, kv_num_heads, head_dim
        ).repeat_interleave(group_size, dim=1).transpose(0, 1)
        base_logits = torch.matmul(qh, kh.transpose(-1, -2)) * (
            1.0 / math.sqrt(float(head_dim))
        )
        for causal in (False, True):
            logits = base_logits
            if causal and seq_len > 1:
                mask = torch.triu(
                    torch.ones(
                        seq_len,
                        seq_len,
                        dtype=torch.bool,
                        device=logits.device,
                    ),
                    diagonal=1,
                )
                logits = logits.masked_fill(mask[None, :, :], -float("inf"))
            probabilities = torch.softmax(logits, dim=-1)
            output = torch.matmul(probabilities, vh)
            v_norm = vh.square().sum(dim=-1)
            o_norm = output.square().sum(dim=-1)
            cross = torch.matmul(output, vh.transpose(-1, -2))
            distance = (
                o_norm[..., None] + v_norm[:, None, :] - 2.0 * cross
            ).clamp_min(0.0)
            sensitivity = probabilities.square() * distance
            q_importance += torch.einsum(
                "hij,hjd->hd", sensitivity, kh.square()
            ) * (1.0 / float(head_dim))
            k_importance_q_heads += torch.einsum(
                "hij,hid->hd", sensitivity, qh.square()
            ) * (1.0 / float(head_dim))
            contribution_count += 1
    if contribution_count > 0:
        q_importance /= float(contribution_count)
        k_importance_q_heads /= float(contribution_count)
    k_importance = k_importance_q_heads.reshape(
        kv_num_heads, group_size, head_dim
    ).mean(dim=1)
    q_flat = _normalize_importance(
        q_importance.reshape(-1), q_num_heads * head_dim
    )
    k_flat = _normalize_importance(
        k_importance.reshape(-1), kv_num_heads * head_dim
    )
    if q_flat is None:
        q_flat = torch.ones(
            q_num_heads * head_dim, dtype=torch.float32, device=d_kv.device
        )
    if k_flat is None:
        k_flat = torch.ones(
            kv_num_heads * head_dim, dtype=torch.float32, device=d_kv.device
        )
    return q_flat, k_flat


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Calibrate static Smooth-QK and output-sensitive Q/K weights."""

    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if q_num_heads <= 0 or kv_num_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    q_channels = q_num_heads * head_dim
    kv_channels = kv_num_heads * head_dim
    if q_channels % 64 != 0 or kv_channels % 64 != 0:
        raise ValueError("Flattened Q/K/V dimensions must be divisible by 64")

    stats_device = calib_qkv_list[0]["q"][0].device
    q_sum_square = torch.zeros(
        q_num_heads, head_dim, dtype=torch.float32, device=stats_device
    )
    k_sum_square = torch.zeros(
        kv_num_heads, head_dim, dtype=torch.float32, device=stats_device
    )
    k_mean_sum_square = torch.zeros_like(k_sum_square)
    k_mid_sum_square = torch.zeros_like(k_sum_square)
    q_peak_square = torch.zeros_like(q_sum_square)
    k_peak_square = torch.zeros_like(k_sum_square)
    k_mean_peak_square = torch.zeros_like(k_sum_square)
    k_mid_peak_square = torch.zeros_like(k_sum_square)
    q_token_count = 0
    k_token_count = 0
    sample_count = 0
    q_samples: list[torch.Tensor] = []
    k_samples: list[torch.Tensor] = []
    q_true_samples: list[torch.Tensor] = []
    k_true_samples: list[torch.Tensor] = []
    v_true_samples: list[torch.Tensor] = []

    for sample in calib_qkv_list:
        if not isinstance(sample, dict) or set(sample.keys()) != {"q", "k", "v"}:
            raise ValueError("Each attention calibration sample must contain q/k/v")
        q = _dequantize_nvfp4_float32(*sample["q"])
        k = _dequantize_nvfp4_float32(*sample["k"])
        if not isinstance(sample["v"], (tuple, list)) or len(sample["v"]) != 2:
            raise ValueError("V calibration data must be an NVFP4 pair")
        v_quant, v_scale = sample["v"]
        if not torch.is_tensor(v_quant) or not torch.is_tensor(v_scale):
            raise TypeError("V calibration pair must contain tensors")
        if q.ndim != 2 or k.ndim != 2 or v_quant.ndim != 2:
            raise ValueError("Q/K/V calibration tensors must be 2D")
        if int(q.shape[1]) != q_channels:
            raise ValueError("Q calibration width does not match head metadata")
        if int(k.shape[1]) != kv_channels or int(v_quant.shape[1]) != kv_channels:
            raise ValueError("K/V calibration width does not match head metadata")
        expected_v_scale_shape = (int(v_quant.shape[0]), kv_channels // 16)
        if tuple(v_scale.shape) != expected_v_scale_shape:
            raise ValueError("V calibration scale shape is invalid")
        if int(q.shape[0]) != int(k.shape[0]) or int(k.shape[0]) != int(v_quant.shape[0]):
            raise ValueError("Q/K/V in a calibration sample must share seq_len")
        v = _dequantize_nvfp4_float32(v_quant, v_scale)

        q_stats = _sample_rows(q, _ATTN_STATS_TOKENS).reshape(
            -1, q_num_heads, head_dim
        )
        k_stats = _sample_rows(k, _ATTN_STATS_TOKENS).reshape(
            -1, kv_num_heads, head_dim
        )
        k_mean_stats = _center_attention_k(
            k_stats.reshape(-1, kv_channels),
            kv_num_heads,
            head_dim,
            1,
        ).reshape(-1, kv_num_heads, head_dim)
        k_mid_stats = _center_attention_k(
            k_stats.reshape(-1, kv_channels),
            kv_num_heads,
            head_dim,
            2,
        ).reshape(-1, kv_num_heads, head_dim)
        q_sum_square += q_stats.square().sum(dim=0)
        k_sum_square += k_stats.square().sum(dim=0)
        k_mean_sum_square += k_mean_stats.square().sum(dim=0)
        k_mid_sum_square += k_mid_stats.square().sum(dim=0)
        q_peak_square += q_stats.abs().amax(dim=0).square()
        k_peak_square += k_stats.abs().amax(dim=0).square()
        k_mean_peak_square += k_mean_stats.abs().amax(dim=0).square()
        k_mid_peak_square += k_mid_stats.abs().amax(dim=0).square()
        q_token_count += int(q_stats.shape[0])
        k_token_count += int(k_stats.shape[0])
        sample_count += 1
        q_samples.append(_sample_rows(q, _ATTN_EVAL_TOKENS).clone())
        k_samples.append(_sample_rows(k, _ATTN_EVAL_TOKENS).clone())
        if len(q_true_samples) < 2:
            q_true_samples.append(_sample_rows(q, _ATTN_TRUE_TOKENS).clone())
            k_true_samples.append(_sample_rows(k, _ATTN_TRUE_TOKENS).clone())
            v_true_samples.append(_sample_rows(v, _ATTN_TRUE_TOKENS).clone())

    q_second_moment = q_sum_square / float(max(q_token_count, 1))
    k_second_moment = k_sum_square / float(max(k_token_count, 1))
    k_mean_second_moment = k_mean_sum_square / float(max(k_token_count, 1))
    k_mid_second_moment = k_mid_sum_square / float(max(k_token_count, 1))
    q_peak = torch.sqrt(q_peak_square / float(max(sample_count, 1)))
    k_peak = torch.sqrt(k_peak_square / float(max(sample_count, 1)))
    k_mean_peak = torch.sqrt(k_mean_peak_square / float(max(sample_count, 1)))
    k_mid_peak = torch.sqrt(k_mid_peak_square / float(max(sample_count, 1)))

    group_size = q_num_heads // kv_num_heads
    q_peak_kv = q_peak.reshape(kv_num_heads, group_size, head_dim).amax(dim=1)
    # Finite-token peak noise can make an intrinsically balanced Q/K profile
    # look transformable on calibration data, while the selected scale/order
    # hurts unseen samples.  A robust 5--95% log-pressure span distinguishes
    # these flat cases from useful K-shift/heavy-tail/QK-imbalance cases.
    q_peak_log = torch.log2(q_peak_kv.clamp_min(_EPS))
    k_peak_log = torch.log2(k_peak.clamp_min(_EPS))
    q_peak_log = q_peak_log - q_peak_log.median(
        dim=-1, keepdim=True
    ).values
    k_peak_log = k_peak_log - k_peak_log.median(
        dim=-1, keepdim=True
    ).values
    attention_pressure = torch.maximum(q_peak_log, k_peak_log).reshape(-1)
    attention_pressure_span = (
        torch.quantile(attention_pressure, 0.95)
        - torch.quantile(attention_pressure, 0.05)
    )
    flat_attention_profile = bool(
        float(attention_pressure_span) < _ATTN_FLAT_PRESSURE_SPAN
    )
    heavy_tail_guard = _attention_heavy_tail_guard(calib_qkv_list)
    identity_d = torch.ones(
        kv_num_heads,
        head_dim,
        dtype=torch.float32,
        device=q_second_moment.device,
    )
    local_identity = torch.arange(
        head_dim, dtype=torch.int64, device=q_second_moment.device
    )[None, :].expand(kv_num_heads, -1)
    k_identity_perm = _flatten_head_permutation(local_identity)
    q_identity_perm = _flatten_head_permutation(
        local_identity.repeat_interleave(group_size, dim=0)
    )

    baseline_metrics = _attention_candidate_metrics(
        q_samples,
        k_samples,
        identity_d,
        q_second_moment,
        k_second_moment,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_identity_perm,
        k_identity_perm,
        0,
    )
    best_metrics = baseline_metrics
    best_d = identity_d
    best_center_mode = 0
    best_q_perm = q_identity_perm
    best_k_perm = k_identity_perm

    # Mean and midrange K-centering are exact softmax invariances.  First
    # select the centering/smoothing pair with identity ordering, then test a
    # small hierarchy-aware ordering set for the selected pair.
    for center_mode, effective_second, effective_peak in (
        (0, k_second_moment, k_peak),
        (1, k_mean_second_moment, k_mean_peak),
        (2, k_mid_second_moment, k_mid_peak),
    ):
        smooth_candidates = (
            identity_d,
            _smooth_qk_scale(q_peak_kv, effective_peak, 0.25),
            _smooth_qk_scale(q_peak_kv, effective_peak, 0.50),
        )
        for candidate_index, candidate_d in enumerate(smooth_candidates):
            if center_mode == 0 and candidate_index == 0:
                continue
            metrics = _attention_candidate_metrics(
                q_samples,
                k_samples,
                candidate_d,
                q_second_moment,
                effective_second,
                q_num_heads,
                kv_num_heads,
                head_dim,
                q_identity_perm,
                k_identity_perm,
                center_mode,
            )
            if (
                metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    metrics,
                    baseline_metrics,
                    min_mean_improvement=0.01,
                    worst_tolerance=0.02,
                )
            ):
                best_metrics = metrics
                best_d = candidate_d
                best_center_mode = center_mode

    if best_center_mode == 1:
        selected_k_peak = k_mean_peak
        selected_k_second = k_mean_second_moment
    elif best_center_mode == 2:
        selected_k_peak = k_mid_peak
        selected_k_second = k_mid_second_moment
    else:
        selected_k_peak = k_peak
        selected_k_second = k_second_moment

    # Max pressure protects the harder Q/K side; joint pressure clusters
    # channels that are simultaneously large.  Both are exact paired feature
    # permutations, and calibration gates them against every sampled case.
    for combine_mode in (0, 1):
        local_permutation = _headwise_hierarchy_permutation(
            q_peak_kv * best_d,
            selected_k_peak * best_d.reciprocal(),
            combine_mode=combine_mode,
        )
        candidate_k_perm = _flatten_head_permutation(local_permutation)
        candidate_q_perm = _flatten_head_permutation(
            local_permutation.repeat_interleave(group_size, dim=0)
        )
        if torch.equal(candidate_k_perm, k_identity_perm) or torch.equal(
            candidate_k_perm, best_k_perm
        ):
            continue
        permutation_metrics = _attention_candidate_metrics(
            q_samples,
            k_samples,
            best_d,
            q_second_moment,
            selected_k_second,
            q_num_heads,
            kv_num_heads,
            head_dim,
            candidate_q_perm,
            candidate_k_perm,
            best_center_mode,
        )
        if (
            permutation_metrics[0] < best_metrics[0]
            and _candidate_is_safe(
                permutation_metrics,
                baseline_metrics,
                min_mean_improvement=0.02,
                worst_tolerance=0.005,
            )
        ):
            best_metrics = permutation_metrics
            best_q_perm = candidate_q_perm
            best_k_perm = candidate_k_perm

    # The proxy above is cheap but does not model softmax or V.  Recheck the
    # selected transform against the exact short-window Attention output under
    # both causal and non-causal masks.  Any unstable transform is discarded.
    baseline_true_metrics = _attention_true_metrics(
        q_true_samples,
        k_true_samples,
        v_true_samples,
        identity_d,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_identity_perm,
        k_identity_perm,
        0,
        refine_mode=0,
    )
    chosen_d = identity_d
    chosen_center_mode = 0
    chosen_q_perm = q_identity_perm
    chosen_k_perm = k_identity_perm
    best_true_metrics = baseline_true_metrics
    # Evaluate the proxy-selected centering/smoothing both before and after
    # hierarchy-aware ordering.  This is a bounded Top-2 second stage rather
    # than a full Cartesian search.
    true_candidates = (() if flat_attention_profile else (
        (best_d, best_center_mode, q_identity_perm, k_identity_perm),
        (best_d, best_center_mode, best_q_perm, best_k_perm),
    ))
    seen_true_candidates: set[tuple[int, int, int]] = set()
    for candidate_d, candidate_center, candidate_q_perm, candidate_k_perm in true_candidates:
        candidate_key = (
            int(candidate_center),
            int(torch.sum(torch.abs(candidate_d - identity_d)) > 0),
            int(torch.sum(candidate_k_perm != k_identity_perm)),
        )
        if candidate_key in seen_true_candidates:
            continue
        seen_true_candidates.add(candidate_key)
        candidate_true_metrics = _attention_true_metrics(
            q_true_samples,
            k_true_samples,
            v_true_samples,
            candidate_d,
            q_num_heads,
            kv_num_heads,
            head_dim,
            candidate_q_perm,
            candidate_k_perm,
            candidate_center,
            refine_mode=0,
        )
        if (
            candidate_true_metrics[0] < best_true_metrics[0]
            and _candidate_is_safe(
                candidate_true_metrics,
                baseline_true_metrics,
                min_mean_improvement=0.005,
                worst_tolerance=0.005,
            )
        ):
            best_true_metrics = candidate_true_metrics
            chosen_d = candidate_d
            chosen_center_mode = candidate_center
            chosen_q_perm = candidate_q_perm
            chosen_k_perm = candidate_k_perm

    best_d = chosen_d
    best_center_mode = chosen_center_mode
    best_q_perm = chosen_q_perm
    best_k_perm = chosen_k_perm
    if int(best_center_mode) == 0:
        selected_k_peak = k_peak
        selected_k_second = k_second_moment
    elif int(best_center_mode) == 1:
        selected_k_peak = k_mean_peak
        selected_k_second = k_mean_second_moment
    else:
        selected_k_peak = k_mid_peak
        selected_k_second = k_mid_second_moment

    d_q = best_d.repeat_interleave(group_size, dim=0)
    d_k = best_d.reciprocal()
    q_second_kv = q_second_moment.reshape(
        kv_num_heads, group_size, head_dim
    ).mean(dim=1)
    h_k = selected_k_second * d_k.square()
    h_q = q_second_kv * best_d.square()
    h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
    h_q_for_k = h_q.reshape(-1)
    h_k_for_q = _normalize_importance(
        h_k_for_q.index_select(0, best_q_perm), q_channels
    )
    h_q_for_k = _normalize_importance(
        h_q_for_k.index_select(0, best_k_perm), kv_channels
    )
    if h_k_for_q is None:
        h_k_for_q = torch.ones(q_channels, dtype=torch.float32)
    if h_q_for_k is None:
        h_q_for_k = torch.ones(kv_channels, dtype=torch.float32)

    # Blend stable second-moment sensitivity with a softmax-Jacobian estimate.
    # The latter measures how Q/K perturbations propagate into the actual
    # Attention output and includes V geometry.
    jacobian_q, jacobian_k = _attention_jacobian_importance(
        q_true_samples,
        k_true_samples,
        v_true_samples,
        best_d,
        q_num_heads,
        kv_num_heads,
        head_dim,
        best_q_perm,
        best_k_perm,
        best_center_mode,
    )
    h_k_for_q = _normalize_importance(
        0.25 * h_k_for_q + 0.75 * jacobian_q, q_channels
    )
    h_q_for_k = _normalize_importance(
        0.25 * h_q_for_k + 0.75 * jacobian_k, kv_channels
    )
    if h_k_for_q is None:
        h_k_for_q = torch.ones(q_channels, dtype=torch.float32)
    if h_q_for_k is None:
        h_q_for_k = torch.ones(kv_channels, dtype=torch.float32)

    # Decide whether online difficult-block refinement is worth its risk.  V is
    # evaluated jointly here even though it has no legal paired transform.
    refine_baseline = _attention_true_metrics(
        q_true_samples,
        k_true_samples,
        v_true_samples,
        best_d,
        q_num_heads,
        kv_num_heads,
        head_dim,
        best_q_perm,
        best_k_perm,
        best_center_mode,
        q_importance=h_k_for_q,
        k_importance=h_q_for_k,
        refine_mode=0,
    )
    best_refine_metrics = refine_baseline
    best_refine_mode = 0
    # Unvalidated offset sets must never leak into the runtime state: start
    # from the conservative set and only keep an extended set that was
    # actually accepted by the refinement gate below.
    best_refine_offsets = _DYNAMIC_OFFSET_SETS[-1]
    for candidate_offsets in _DYNAMIC_OFFSET_SETS:
        if best_refine_mode > 0:
            break
        for candidate_refine_mode in (1, 2):
            candidate_refine_metrics = _attention_true_metrics(
                q_true_samples,
                k_true_samples,
                v_true_samples,
                best_d,
                q_num_heads,
                kv_num_heads,
                head_dim,
                best_q_perm,
                best_k_perm,
                best_center_mode,
                q_importance=h_k_for_q,
                k_importance=h_q_for_k,
                refine_mode=candidate_refine_mode,
                search_offsets=candidate_offsets,
            )
            if (
                candidate_refine_metrics[0] < best_refine_metrics[0]
                and _candidate_is_safe(
                    candidate_refine_metrics,
                    refine_baseline,
                    min_mean_improvement=0.005,
                    worst_tolerance=0.005,
                )
            ):
                best_refine_metrics = candidate_refine_metrics
                best_refine_mode = candidate_refine_mode
                best_refine_offsets = candidate_offsets

    best_refine_ratios = (
        0.08 if best_refine_mode >= 1 else 0.0,
        0.12 if best_refine_mode >= 1 else 0.0,
        0.10 if best_refine_mode >= 2 else 0.0,
    )
    fallback_refine_ratios = best_refine_ratios

    # The base 0/QK/QKV gate remains the mandatory fallback.  One calibrated
    # high-budget QKV candidate captures nearly all of the score gain of a
    # larger grid with one third of its extra calibration work.  It may replace
    # the fallback only when every case is non-worse, both mask domains improve,
    # and the gain survives a small online-cost penalty.  Each ratio candidate
    # is evaluated under every offset set: the extended set can degrade a
    # candidate's per-case profile enough to fail the worst-case gate even
    # when the conservative set would have passed, so the best gate-passing
    # (ratios, offsets) combination wins on objective value.
    escalation_baseline = best_refine_metrics
    best_budget_objective = best_refine_metrics[0] + 0.001 * sum(
        best_refine_ratios
    )
    for candidate_ratios in ((0.12, 0.16, 0.10), (0.25, 0.30, 0.20)):
        for candidate_offsets in _DYNAMIC_OFFSET_SETS:
            candidate_metrics = _attention_true_metrics(
                q_true_samples,
                k_true_samples,
                v_true_samples,
                best_d,
                q_num_heads,
                kv_num_heads,
                head_dim,
                best_q_perm,
                best_k_perm,
                best_center_mode,
                q_importance=h_k_for_q,
                k_importance=h_q_for_k,
                refine_ratios=candidate_ratios,
                search_offsets=candidate_offsets,
            )
            candidate_objective = candidate_metrics[0] + 0.001 * sum(
                candidate_ratios
            )
            if (
                candidate_objective < best_budget_objective
                and _candidate_is_safe(
                    candidate_metrics,
                    escalation_baseline,
                    min_mean_improvement=0.01,
                    worst_tolerance=0.0,
                )
                and _attention_mask_consensus(
                    candidate_metrics,
                    escalation_baseline,
                    min_improvement=0.002,
                )
            ):
                best_refine_metrics = candidate_metrics
                best_refine_ratios = candidate_ratios
                best_refine_mode = 3
                best_refine_offsets = candidate_offsets
                best_budget_objective = candidate_objective

    q_refine_ratio, k_refine_ratio, v_refine_ratio = best_refine_ratios
    # Keep calibration-derived importance and smoothing, but avoid applying a
    # midrange K shift online when a large head has a broad calibration tail.
    # The shift is an exact attention invariance before quantization, yet its
    # finite-precision interaction is harmful for this distribution.
    online_center_mode = (
        0 if heavy_tail_guard and best_center_mode == 2 else best_center_mode
    )

    offsets = torch.tensor(
        best_refine_offsets, dtype=torch.int8, device="cpu"
    )
    top_ranks = torch.tensor(
        _DYNAMIC_TOP_RANKS, dtype=torch.int8, device="cpu"
    )
    q_permutation_state = None
    k_permutation_state = None
    if not torch.equal(best_k_perm, k_identity_perm):
        q_permutation_state = best_q_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        k_permutation_state = best_k_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
    q_multiplier_state = None
    k_multiplier_state = None
    if not torch.equal(best_d, identity_d):
        q_multiplier_state = _cpu_state_tensor(d_q.reshape(-1))
        k_multiplier_state = _cpu_state_tensor(d_k.reshape(-1))

    q_state = {
        "multiplier": q_multiplier_state,
        "permutation": q_permutation_state,
        "importance": _cpu_state_tensor(h_k_for_q),
        "offsets": offsets.clone(),
        "top_ranks": top_ranks.clone(),
        "error_threshold": 1.0e-7,
        "accept_margin": 0.03,
        "max_refine_ratio": float(q_refine_ratio),
        "fast_refine_ratio": float(fallback_refine_ratios[0]),
        "max_refine_blocks": 16_384,
        "num_heads": int(q_num_heads),
        "head_dim": int(head_dim),
        "refine_mode": int(best_refine_mode),
        "flat_profile": bool(flat_attention_profile),
        "version": 11,
    }
    k_state = {
        "multiplier": k_multiplier_state,
        "permutation": k_permutation_state,
        "center_mode": int(online_center_mode),
        "importance": _cpu_state_tensor(h_q_for_k),
        "offsets": offsets.clone(),
        "top_ranks": top_ranks.clone(),
        "error_threshold": 1.0e-7,
        "accept_margin": 0.03,
        "max_refine_ratio": float(k_refine_ratio),
        "fast_refine_ratio": float(fallback_refine_ratios[1]),
        "max_refine_blocks": 24_576,
        "num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "refine_mode": int(best_refine_mode),
        "flat_profile": bool(flat_attention_profile),
        "heavy_tail_guard": bool(heavy_tail_guard),
        "version": 11,
    }
    v_state = {
        "offsets": offsets.clone(),
        "top_ranks": top_ranks.clone(),
        "error_threshold": 1.0e-7,
        "accept_margin": 0.01,
        "max_refine_ratio": float(v_refine_ratio),
        "fast_refine_ratio": float(fallback_refine_ratios[2]),
        "max_refine_blocks": 24_576,
        "num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "refine_mode": int(best_refine_mode),
        "flat_profile": bool(flat_attention_profile),
        "heavy_tail_guard": bool(heavy_tail_guard),
        "version": 11,
    }
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


def _check_attention_state(
    state: Any,
    num_heads: int,
    head_dim: int,
    name: str,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError(f"{name}_state must be a dict")
    if int(state.get("num_heads", -1)) != int(num_heads):
        raise ValueError(f"{name} head count does not match calibration state")
    if int(state.get("head_dim", -1)) != int(head_dim):
        raise ValueError(f"{name} head_dim does not match calibration state")
    return state


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(q_state, q_num_heads, head_dim, "q")
    if int(q_quant.shape[-1]) != q_num_heads * head_dim:
        raise ValueError("Q width does not match q_num_heads * head_dim")
    refine_ratio = float(state["max_refine_ratio"])
    if int(q_quant.numel()) > _ATTN_FULL_SEARCH_ELEMENTS:
        refine_ratio = float(state.get("fast_refine_ratio", refine_ratio))
    return _nvfp4_to_hif4(
        q_quant,
        q_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        importance=state["importance"],
        search_offsets=state["offsets"],
        search_top_ranks=state["top_ranks"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=refine_ratio,
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(k_state, kv_num_heads, head_dim, "k")
    if int(k_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("K width does not match kv_num_heads * head_dim")
    refine_ratio = float(state["max_refine_ratio"])
    if int(k_quant.numel()) > _ATTN_FULL_SEARCH_ELEMENTS:
        refine_ratio = float(state.get("fast_refine_ratio", refine_ratio))
    return _nvfp4_to_hif4(
        k_quant,
        k_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        importance=state["importance"],
        search_offsets=state["offsets"],
        search_top_ranks=state["top_ranks"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=refine_ratio,
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(v_state, kv_num_heads, head_dim, "v")
    if int(v_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("V width does not match kv_num_heads * head_dim")
    refine_ratio = float(state["max_refine_ratio"])
    if int(v_quant.numel()) > _ATTN_FULL_SEARCH_ELEMENTS:
        refine_ratio = float(state.get("fast_refine_ratio", refine_ratio))
    return _nvfp4_to_hif4(
        v_quant,
        v_scale,
        search_offsets=state["offsets"],
        search_top_ranks=state["top_ranks"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=refine_ratio,
        max_refine_blocks=int(state["max_refine_blocks"]),
    )
