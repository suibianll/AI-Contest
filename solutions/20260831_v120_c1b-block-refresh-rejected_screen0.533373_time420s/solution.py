"""Clean HiF4 entry: BOAT + cross-fold HSDQ/LRH.

The file intentionally contains only the six competition APIs and the small
set of primitives needed by the active algorithm.  Historical C1--C88
experiments remain recoverable from Git and ``solutions/``; rejected flags and
dormant research branches do not live in this submission file.

Linear calibration has three stages:

1. BOAT/L5a selects an invertible diagonal + block permutation +
   signed-Hadamard input transform from operand-local quantization errors.  It
   never constructs a Linear output.
2. The transformed weight is quantized to the legal HiF4 hierarchy.
3. Cross-fold HSDQ uses exact low-rank Hessians ``A.T @ A`` to polish Q(W).
   Products are calibration-local and can only change ``weight_params``.

Online Q(A) uses the frozen BOAT/L5a transform and a bounded block-Hessian HSDQ
whose state contains only static transformed-weight Gram blocks.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch


# ---------------------------------------------------------------------------
# Format and budgets
# ---------------------------------------------------------------------------

_BLOCK = 64
_EPS = 1.0e-12
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_BF16_ONE_SEVENTH = 0.142578125
_SIGNED_LEVELS = tuple(value * 0.25 for value in range(-7, 8))

_BASE_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
_PROXY_OFFSETS = (-1, 0, 1)
_WEIGHT_ROW_CHUNK = 256

_BOAT_ALPHAS = (0.0, 0.5, 0.75)
_BOAT_ROTATION_SIZES = (4, 8, 16, 64)
_BOAT_ROTATION_SEEDS = (0, 1)
_BOAT_PROXY_WEIGHT_ROWS = 192
_BOAT_SCALE_MIN = 1.0 / 16.0
_BOAT_SCALE_MAX = 16.0

# L2: a single low-degree CAT balance for expansive FFN shapes.  The route is
# structural (rows > channels), not a role/model identifier; no permutation or
# Householder state is introduced in this first L2 candidate.
_EXPANSIVE_CAT_ALPHA = 0.25

_WEIGHT_HSDQ_BLOCKS = 2
_WEIGHT_HSDQ_SWEEPS = 1
_WEIGHT_HSDQ_MIN_CHANNELS = 256
_WEIGHT_HSDQ_MAX_ROWS = 256
_WEIGHT_HSDQ_ROBUST_MIX = 0.5
_WEIGHT_HSDQ_MIN_GAIN = 1.0e-5

_ACT_HSDQ_BLOCKS = 128
_ACT_HSDQ_SWEEPS = 2
_ACT_GRAM_MAX_CHANNELS = 8192
_ACT_GLOBAL_LRH_MAX_CHANNELS = 1024
# L6a candidate: double the narrow-input off-block proposal rank.  The
# candidate is screened before it can become part of the precision parent.
_ACT_GLOBAL_LRH_RANK = 16
# L6b candidate: extend the same compressed off-block proposal to wide input
# matrices (currently Qwen's 4864-channel projection) with a deliberately
# small rank.  The dense deployed Gram is already part of the established
# wide-shape path; this adds only one compressed CPU state tensor.
_ACT_GLOBAL_LRH_WIDE_MAX_CHANNELS = 8192
_ACT_GLOBAL_LRH_WIDE_RANK = 4
_ACT_GLOBAL_LRH_OVERSAMPLE = 4
_ACT_GLOBAL_LRH_POWER_STEPS = 2
_ACT_GLOBAL_LRH_MIX = 0.1
_ACT_GLOBAL_LRH_BLOCKS = 4
# L6c candidate: one bounded hierarchy-coordinate sweep on the same 64-channel
# Gram blocks already stored for activation HSDQ.  Scale factors stay fixed;
# only legal lv2/lv3 choices and their mantissas are regenerated.
_ACT_G64_HIERARCHY_ENABLED = True
_ACT_G64_HIERARCHY_BLOCKS = 4
_ACT_G64_HIERARCHY_SWEEPS = 1
# L6d candidate: a block-circulant low-rank kernel for wide inputs.  The
# stored representation contains at most four 64x64 kernels and per-distance
# coefficients; it never stores all block pairs as activation state.
_ACT_STRUCTURED_LRH_COMPONENTS = 4
_ACT_STRUCTURED_LRH_MAX_CHANNELS = 8192
_ACT_STRUCTURED_LRH_BLOCKS = 4
# C1a keeps a reference implementation for bit/tolerance-level regression.
# The vectorized path only batches independent row/block proposals; the
# coordinate order and exact deployed-Gram gate remain unchanged.
_ACT_STRUCTURED_LRH_VECTORIZED = True
# C1b candidate knob.  ``none`` is the accepted v119 equivalent path;
# ``block`` refreshes the structured proposal after every selected block;
# ``sweep2`` repeats that refreshed block sweep twice.
_ACT_STRUCTURED_LRH_REFRESH_MODE = "block"
_L4_FINAL_GRAM_MAX_CHANNELS = 1024
# Keep the L4a final-weight-Gram ablation isolated from the later GALS
# candidate.  This switch is flipped only in the L4b candidate snapshot.
_L4_GALS_FINAL_ENABLED = True

# L5a: one fixed, block-local channel permutation is selected from
# operand-local outlier pressure.  The candidate list is deliberately tiny:
# identity, monotone pressure grouping, and low/high interleaving.  A
# permutation is written to state only when it improves both calibration folds;
# otherwise the v110 parent frame is retained exactly.
_L5A_PERMUTATION_ENABLED = True
_L5A_PERMUTATION_MIN_GAIN = 1.0e-5
_L5A_PERMUTATION_BLOCK = _BLOCK

_ATTN_OFFSETS = (-2, -1, 0, 1, 2)
_ATTN_ROTATION_SIZES = (0, 16, 32, 64)
_ATTN_SMOOTH_ALPHAS = (0.0, 0.25, 0.5, 0.75)
_ATTN_ROTATION_SEEDS = (0, 1)
_ATTN_GQRB_WIDTHS = (2, 4)
_ATTN_GQRB_ANGLES = (math.pi / 8.0, -math.pi / 8.0, math.pi / 4.0, -math.pi / 4.0)
_ATTN_GQRB_MIN_GAIN = 1.0e-3
_ATTN_PAWV_RANK = 8
_ATTN_PAWV_SWEEPS = 1

# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@torch.no_grad()
def _dequantize_nvfp4_float32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    channels = int(quant_float.shape[-1])
    if channels % block_size != 0:
        raise ValueError("NVFP4 last dimension must be divisible by 16")
    expected = tuple(quant_float.shape[:-1]) + (channels // block_size,)
    if tuple(scale_float.shape) != expected:
        raise ValueError("NVFP4 scale shape mismatch")
    grouped = quant_float.detach().to(torch.float32).reshape(
        *quant_float.shape[:-1], channels // block_size, block_size
    )
    return (grouped * scale_float.detach().to(torch.float32).unsqueeze(-1)).reshape(
        *quant_float.shape[:-1], channels
    ).to(torch.bfloat16).to(torch.float32)


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_E6M2_MIN,
        posinf=_E6M2_MAX,
        neginf=_E6M2_MIN,
    ).clamp(_E6M2_MIN, _E6M2_MAX)
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
    exponent = torch.bitwise_right_shift(value, 2).to(torch.float32) - 48.0
    mantissa = torch.bitwise_and(value, 3).to(torch.float32)
    return torch.pow(2.0, exponent) * (1.0 + 0.25 * mantissa)


def _standard_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = (amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH).to(torch.float32)
    code = _e6m2_encode_nearest(raw)
    return code, _e6m2_decode(code)


def _solve_hierarchy(
    absolute: torch.Tensor,
    scale: torch.Tensor,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact lv2/lv3 solution for fixed E6M2 scale.

    ``absolute`` is ``[..., 8, 2, 4]`` and ``scale`` is ``[...]``.
    """

    weight = 1.0 if importance is None else importance
    losses: list[torch.Tensor] = []
    for exponent in (0, 1, 2):
        local = scale[..., None, None, None] * float(1 << exponent)
        mantissa = torch.round(absolute * (4.0 / local.clamp_min(_EPS))).clamp(
            0.0, 7.0
        ) * 0.25
        losses.append(((absolute - mantissa * local).square() * weight).sum(dim=-1))
    loss0, loss1, loss2 = losses
    choose01 = loss1 < loss0
    choose12 = loss2 < loss1
    cost1 = torch.minimum(loss0, loss1).sum(dim=-1)
    cost2 = torch.minimum(loss1, loss2).sum(dim=-1)
    lv2_two = cost2 < cost1
    lv3_two = torch.where(lv2_two[..., None], choose12, choose01)
    lv2 = 1.0 + lv2_two.to(torch.float32)
    lv3 = 1.0 + lv3_two.to(torch.float32)
    denominator = (
        scale[..., None, None, None]
        * lv2[..., None, None]
        * lv3[..., None]
    )
    mantissa = torch.round(
        absolute * (4.0 / denominator.clamp_min(_EPS))
    ).clamp(0.0, 7.0) * 0.25
    loss = ((absolute - mantissa * denominator).square() * weight).sum(
        dim=(-1, -2, -3)
    )
    return loss, lv2, lv3, mantissa


def _encode_rows(
    dense: torch.Tensor,
    offsets: Sequence[int],
    importance: torch.Tensor | None = None,
    gram64: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    rows, channels = map(int, dense.shape)
    blocks = channels // _BLOCK
    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * 7.0,
        neginf=-_E6M2_MAX * 7.0,
    ).reshape(rows, blocks, 8, 2, 4)
    absolute = x.abs()
    sign = torch.sign(x)
    hierarchy_weight = None
    if importance is not None:
        hierarchy_weight = importance.to(device=dense.device, dtype=torch.float32).reshape(
            1, blocks, 8, 2, 4
        ).clamp_min(0.0)
    standard_code, _ = _standard_scale(absolute.amax(dim=(-1, -2, -3)))

    best_loss = torch.full(
        (rows, blocks), torch.inf, dtype=torch.float32, device=dense.device
    )
    best_scale = torch.ones_like(best_loss)
    best_lv2 = torch.ones(rows, blocks, 8, device=dense.device)
    best_lv3 = torch.ones(rows, blocks, 8, 2, device=dense.device)
    best_mantissa = torch.zeros_like(absolute)
    for offset in tuple(dict.fromkeys(int(item) for item in offsets)):
        code = (standard_code.to(torch.int64) + offset).clamp(0, 254)
        scale = _e6m2_decode(code)
        loss, lv2, lv3, mantissa = _solve_hierarchy(
            absolute, scale, hierarchy_weight
        )
        if gram64 is not None:
            denominator = (
                scale[..., None, None, None]
                * lv2[..., None, None]
                * lv3[..., None]
            )
            error = (sign * mantissa * denominator - x).reshape(
                rows, blocks, _BLOCK
            )
            loss = torch.einsum(
                "rbi,bij,rbj->rb", error, gram64, error
            )
        better = torch.isfinite(loss) & (loss < best_loss)
        best_loss = torch.where(better, loss, best_loss)
        best_scale = torch.where(better, scale, best_scale)
        best_lv2 = torch.where(better[..., None], lv2, best_lv2)
        best_lv3 = torch.where(better[..., None, None], lv3, best_lv3)
        best_mantissa = torch.where(
            better[..., None, None, None], mantissa, best_mantissa
        )
    sign = torch.where(best_mantissa == 0.0, torch.zeros_like(sign), sign)
    return {
        "scale_factor": best_scale.reshape(rows, blocks, 1, 1, 1),
        "scale_lv2": best_lv2.reshape(rows, blocks, 8, 1, 1),
        "scale_lv3": best_lv3.reshape(rows, blocks, 8, 2, 1),
        "sign": sign,
        "mant": best_mantissa,
    }


@torch.no_grad()
def _dense_to_hif4(
    dense: torch.Tensor,
    *,
    offsets: Sequence[int] = _BASE_OFFSETS,
    search_offsets: Sequence[int] | None = None,
    importance: torch.Tensor | None = None,
    gram64: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if dense.ndim < 1 or int(dense.shape[-1]) % _BLOCK != 0:
        raise ValueError("HiF4 last dimension must be divisible by 64")
    if search_offsets is not None:
        offsets = tuple(search_offsets) if tuple(search_offsets) else (0,)
    logical_shape = tuple(int(value) for value in dense.shape)
    channels = logical_shape[-1]
    flat = dense.detach().to(torch.float32).reshape(-1, channels)
    chunks: list[dict[str, torch.Tensor]] = []
    for start in range(0, int(flat.shape[0]), _WEIGHT_ROW_CHUNK):
        chunks.append(
            _encode_rows(
                flat[start : start + _WEIGHT_ROW_CHUNK],
                offsets,
                importance,
                gram64,
            )
        )
    merged = {
        key: torch.cat([part[key] for part in chunks], dim=0)
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")
    }
    prefix = logical_shape[:-1]
    blocks = channels // _BLOCK
    shapes = {
        "scale_factor": prefix + (blocks, 1, 1, 1),
        "scale_lv2": prefix + (blocks, 8, 1, 1),
        "scale_lv3": prefix + (blocks, 8, 2, 1),
        "sign": prefix + (blocks, 8, 2, 4),
        "mant": prefix + (blocks, 8, 2, 4),
    }
    return {key: value.reshape(shapes[key]) for key, value in merged.items()}


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    ).flatten(start_dim=-4, end_dim=-1)


def _clone_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in params.items()}


# ---------------------------------------------------------------------------
# BOAT transform
# ---------------------------------------------------------------------------


def _rotation_signs(
    channels: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    index = torch.arange(channels, device=device, dtype=torch.int64)
    value = index * 1103515245 + (int(seed) + 17) * 12345
    return torch.where(
        torch.bitwise_and(value, 1) == 0,
        torch.ones(channels, device=device, dtype=dtype),
        -torch.ones(channels, device=device, dtype=dtype),
    )


def _fwht_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        return x
    channels = int(x.shape[-1])
    if channels % block_size != 0 or block_size & (block_size - 1):
        return x
    y = x.reshape(-1, channels // block_size, block_size).clone()
    width = 1
    while width < block_size:
        view = y.reshape(*y.shape[:-1], -1, 2, width)
        left = view[..., 0, :].clone()
        right = view[..., 1, :].clone()
        view[..., 0, :] = left + right
        view[..., 1, :] = left - right
        y = view.reshape_as(y)
        width *= 2
    return (y / math.sqrt(float(block_size))).reshape_as(x)


def _validate_permutation(
    permutation: torch.Tensor | None,
    channels: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a device-local permutation or ``None`` for the identity frame."""

    if permutation is None:
        return None
    order = permutation.detach().to(device=device, dtype=torch.int64).reshape(-1)
    if int(order.numel()) != int(channels):
        raise ValueError("Linear permutation width does not match feature width")
    if int(order.numel()) == 0:
        return None
    if int(order.min().item()) < 0 or int(order.max().item()) >= int(channels):
        raise ValueError("Linear permutation contains an out-of-range index")
    if int(torch.unique(order).numel()) != int(channels):
        raise ValueError("Linear permutation must contain every feature once")
    return order


def _apply_boat_rotation(
    x: torch.Tensor,
    seed: int,
    block_size: int = _BLOCK,
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(seed) < 0 or int(block_size) <= 0:
        if permutation is None:
            return x
        order = _validate_permutation(permutation, int(x.shape[-1]), x.device)
        return x.index_select(-1, order) if order is not None else x
    order = _validate_permutation(permutation, int(x.shape[-1]), x.device)
    if order is not None:
        x = x.index_select(-1, order)
    signs = _rotation_signs(int(x.shape[-1]), seed, x.device, x.dtype)
    return _fwht_blocks(x * signs, int(block_size))


def _linear_pair_transform(
    tensor: torch.Tensor,
    balance: torch.Tensor,
    permutation: torch.Tensor | None = None,
    block_smooth_size: int = 0,
    block_smooth_seed: int = -1,
    *,
    weight_side: bool,
    **_: Any,
) -> torch.Tensor:
    """Compatibility helper used by local fixed-frame diagnostics."""

    d = balance.to(device=tensor.device, dtype=torch.float32).reshape(1, -1)
    transformed = tensor.to(torch.float32) * (d if weight_side else d.reciprocal())
    order = _validate_permutation(permutation, int(tensor.shape[-1]), tensor.device)
    if order is not None:
        transformed = transformed.index_select(-1, order)
    if int(block_smooth_size) > 0:
        transformed = _apply_boat_rotation(
            transformed, int(block_smooth_seed), int(block_smooth_size)
        )
    return transformed


def _sample_rows(x: torch.Tensor, count: int) -> torch.Tensor:
    rows = int(x.shape[0])
    if rows <= count:
        return x
    indices = torch.linspace(0, rows - 1, count, device=x.device).round().to(torch.int64)
    return x.index_select(0, indices)


def _relative_quant_error(
    x: torch.Tensor,
    offsets: Sequence[int],
) -> float:
    q = _dequantize_hif4(_dense_to_hif4(x, offsets=offsets)).to(torch.float32)
    return float((q - x).square().sum() / (x.square().sum() + _EPS))


def _l5a_channel_pressure(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    balance: torch.Tensor,
) -> torch.Tensor:
    """Estimate per-channel hierarchy pressure without forming a product.

    A channel with a large ``amax / rms`` ratio is more likely to force a
    shared lv2/lv3 scale away from its neighbours.  The pressure combines the
    independently observed weight-side and activation-side ratios after the
    already selected diagonal balance.  It is intentionally a first-order
    statistic: no evaluator output, residual, or cross-operand contraction is
    used to build the state permutation.
    """

    channels = int(weight.shape[1])
    if (
        not calibration
        or channels % _L5A_PERMUTATION_BLOCK != 0
        or int(balance.numel()) != channels
    ):
        return torch.zeros(channels, device=weight.device, dtype=torch.float32)
    w = _sample_rows(weight, _BOAT_PROXY_WEIGHT_ROWS).to(torch.float32)
    d = balance.to(device=weight.device, dtype=torch.float32).reshape(1, -1)
    w = w * d
    joined = torch.cat(
        [_sample_rows(item, 128).to(torch.float32) for item in calibration],
        dim=0,
    )
    a = joined / d
    w_rms = w.square().mean(dim=0).add(_EPS).sqrt()
    a_rms = a.square().mean(dim=0).add(_EPS).sqrt()
    w_tail = w.abs().amax(dim=0) / w_rms
    a_tail = a.abs().amax(dim=0) / a_rms
    pressure = 0.5 * (
        torch.log1p(w_tail.clamp_min(0.0))
        + torch.log1p(a_tail.clamp_min(0.0))
    )
    return torch.nan_to_num(pressure, nan=0.0, posinf=32.0, neginf=0.0)


def _l5a_block_permutation(
    pressure: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """Build one deterministic 64-channel permutation family.

    ``mode=0`` is identity, ``mode=1`` groups channels with similar pressure,
    and ``mode=2`` alternates low/high pressure.  All modes preserve 64-channel
    boundaries, so the legal HiF4 hierarchy remains unchanged; only which
    channels share each hierarchy scale is changed.
    """

    channels = int(pressure.numel())
    block = int(_L5A_PERMUTATION_BLOCK)
    identity = torch.arange(channels, device=pressure.device, dtype=torch.int64)
    if mode == 0 or channels % block != 0:
        return identity
    result = identity.clone()
    for start in range(0, channels, block):
        local = torch.argsort(
            pressure[start : start + block], stable=True
        ).to(torch.int64)
        if mode == 1:
            chosen = local
        elif mode == 2:
            # Interleave the low and high halves.  Reversing the high half
            # keeps adjacent pairs at similar pressure distance while making
            # every lv2/lv3 group see both ends of the local distribution.
            low = local[: block // 2]
            high = local[block // 2 :].flip(0)
            chosen = torch.empty_like(local)
            chosen[0::2] = low
            chosen[1::2] = high
        else:
            # A second low-degree grouping: four pressure quartiles are
            # interleaved, without introducing a free per-channel search.
            quarter = block // 4
            chosen = torch.stack(
                (
                    local[:quarter],
                    local[quarter : 2 * quarter],
                    local[2 * quarter : 3 * quarter],
                    local[3 * quarter :],
                ),
                dim=1,
            ).reshape(-1)
        result[start : start + block] = (
            chosen + start
        ).to(torch.int64)
    return result


def _choose_l5a_permutation(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    balance: torch.Tensor,
    seed: int,
    block_size: int,
) -> torch.Tensor | None:
    """Select at most one fixed block permutation with a two-fold gate.

    The score is the same operand-local BOAT proxy used for the existing
    diagonal/Hadamard choice.  A proposal must improve the aggregate score and
    not worsen either calibration fold; otherwise ``None`` preserves v110
    exactly.  This makes the new state field a strict precision candidate
    rather than an unconditional change to every layer.
    """

    if (
        not _L5A_PERMUTATION_ENABLED
        or not calibration
        or weight.ndim != 2
        or int(weight.shape[1]) % _L5A_PERMUTATION_BLOCK != 0
        or int(balance.numel()) != int(weight.shape[1])
    ):
        return None
    pressure = _l5a_channel_pressure(weight, calibration, balance)
    weight_proxy = _sample_rows(weight, _BOAT_PROXY_WEIGHT_ROWS)

    def score(order: torch.Tensor) -> tuple[float, list[float]]:
        w_t = _apply_boat_rotation(
            weight_proxy * balance.reshape(1, -1),
            int(seed),
            int(block_size),
            permutation=order,
        )
        weight_error = _relative_quant_error(w_t, _PROXY_OFFSETS)
        fold_errors: list[float] = []
        for sample in calibration:
            a_t = _apply_boat_rotation(
                _sample_rows(sample, 128) / balance.reshape(1, -1),
                int(seed),
                int(block_size),
                permutation=order,
            )
            fold_errors.append(_relative_quant_error(a_t, _PROXY_OFFSETS))
        act_mean = sum(fold_errors) / len(fold_errors)
        act_worst = max(fold_errors)
        aggregate = (
            math.sqrt(max(weight_error, 0.0))
            + math.sqrt(max(act_mean, 0.0))
            + math.sqrt(max(weight_error * act_mean, 0.0))
            + 0.25 * math.sqrt(max(act_worst, 0.0))
        )
        return aggregate, fold_errors

    identity = torch.arange(
        int(weight.shape[1]), device=weight.device, dtype=torch.int64
    )
    try:
        base_score, base_folds = score(identity)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    if not math.isfinite(base_score) or not all(
        math.isfinite(value) for value in base_folds
    ):
        return None
    best_score = base_score
    best_order: torch.Tensor | None = None
    # Three low-degree families are enough to test the hypothesis.  The state
    # still carries only one selected order, never the candidate list.
    for mode in (1, 2, 3):
        order = _l5a_block_permutation(pressure, mode)
        try:
            candidate_score, candidate_folds = score(order)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if not math.isfinite(candidate_score) or not all(
            math.isfinite(value) for value in candidate_folds
        ):
            continue
        robust = all(
            candidate <= base + 1.0e-7
            for candidate, base in zip(candidate_folds, base_folds)
        )
        if (
            robust
            and candidate_score < best_score - _L5A_PERMUTATION_MIN_GAIN
        ):
            best_score = candidate_score
            best_order = order
    if best_order is None or torch.equal(best_order, identity):
        return None
    return best_order


def _choose_boat(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, int, int]:
    channels = int(weight.shape[1])
    if not calibration or channels % _BLOCK != 0:
        return torch.ones(channels, device=weight.device), -1, 0
    joined = torch.cat([_sample_rows(item, 128) for item in calibration], dim=0)
    a_rms = joined.square().mean(dim=0).add(_EPS).sqrt()
    w_rms = weight.square().mean(dim=0).add(_EPS).sqrt()
    weight_proxy = _sample_rows(weight, _BOAT_PROXY_WEIGHT_ROWS)

    best_score = math.inf
    best_balance = torch.ones(channels, device=weight.device)
    best_seed = -1
    best_block = 0

    def score_candidate(balance: torch.Tensor, seed: int, block_size: int) -> float:
        w_t = _apply_boat_rotation(
            weight_proxy * balance, int(seed), int(block_size)
        )
        weight_error = _relative_quant_error(w_t, _PROXY_OFFSETS)
        fold_errors = []
        for sample in calibration:
            a_t = _apply_boat_rotation(
                _sample_rows(sample, 128) / balance, int(seed), int(block_size)
            )
            fold_errors.append(_relative_quant_error(a_t, _PROXY_OFFSETS))
        act_mean = sum(fold_errors) / len(fold_errors)
        act_worst = max(fold_errors)
        return (
            math.sqrt(max(weight_error, 0.0))
            + math.sqrt(max(act_mean, 0.0))
            + math.sqrt(max(weight_error * act_mean, 0.0))
            + 0.25 * math.sqrt(max(act_worst, 0.0))
        )

    # First solve the diagonal balance, then search the orthogonal member.
    for alpha in _BOAT_ALPHAS:
        if float(alpha) == 0.0:
            balance = torch.ones_like(a_rms)
        else:
            balance = (a_rms / w_rms).pow(float(alpha)).clamp(
                _BOAT_SCALE_MIN, _BOAT_SCALE_MAX
            )
            balance = balance / torch.exp(torch.log(balance).mean())
        score = score_candidate(balance, -1, 0)
        if score < best_score:
            best_score = score
            best_balance = balance.clone()
    for block_size in _BOAT_ROTATION_SIZES:
        for seed in _BOAT_ROTATION_SEEDS:
            score = score_candidate(best_balance, seed, block_size)
            if score < best_score:
                best_score = score
                best_seed = int(seed)
                best_block = int(block_size)
    return best_balance, best_seed, best_block


def _choose_expansive_cat_balance(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    base_balance: torch.Tensor,
    seed: int,
    block_size: int,
) -> torch.Tensor:
    """Try one fixed CAT balance only on structurally expansive matrices.

    The candidate is an invertible diagonal re-balance.  Its score uses only
    operand-local quantization errors, with the already selected BOAT rotation
    held fixed.  Returning the parent balance on every failure keeps this
    route a no-op for non-expansive shapes and for proxy regressions.
    """

    rows, channels = map(int, weight.shape)
    if (
        rows <= channels
        or channels % _BLOCK != 0
        or not calibration
        or base_balance.numel() != channels
    ):
        return base_balance
    joined = torch.cat([_sample_rows(item, 128) for item in calibration], dim=0)
    activation_rms = joined.to(torch.float32).square().mean(dim=0).add(_EPS).sqrt()
    weight_rms = weight.to(torch.float32).square().mean(dim=0).add(_EPS).sqrt()
    ratio = (activation_rms / weight_rms).pow(float(_EXPANSIVE_CAT_ALPHA)).clamp(
        _BOAT_SCALE_MIN, _BOAT_SCALE_MAX
    )
    ratio = ratio / torch.exp(torch.log(ratio).mean()).clamp_min(_EPS)
    candidate = base_balance.to(torch.float32) * ratio
    candidate = candidate / torch.exp(torch.log(candidate).mean()).clamp_min(_EPS)

    weight_proxy = _sample_rows(weight, _BOAT_PROXY_WEIGHT_ROWS)

    def score(balance: torch.Tensor) -> float:
        w_t = _apply_boat_rotation(weight_proxy * balance, int(seed), int(block_size))
        weight_error = _relative_quant_error(w_t, _PROXY_OFFSETS)
        fold_errors = []
        for sample in calibration:
            a_t = _apply_boat_rotation(
                _sample_rows(sample, 128) / balance, int(seed), int(block_size)
            )
            fold_errors.append(_relative_quant_error(a_t, _PROXY_OFFSETS))
        act_mean = sum(fold_errors) / len(fold_errors)
        act_worst = max(fold_errors)
        return (
            math.sqrt(max(weight_error, 0.0))
            + math.sqrt(max(act_mean, 0.0))
            + math.sqrt(max(weight_error * act_mean, 0.0))
            + 0.25 * math.sqrt(max(act_worst, 0.0))
        )

    try:
        base_score = score(base_balance)
        candidate_score = score(candidate)
    except (RuntimeError, ValueError, FloatingPointError):
        return base_balance
    if not math.isfinite(candidate_score) or candidate_score >= base_score:
        return base_balance
    return candidate


# ---------------------------------------------------------------------------
# HSDQ / exact low-rank Hessian refiners
# ---------------------------------------------------------------------------


def _denominator(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
    ).repeat_interleave(4, dim=-1).flatten(start_dim=-4, end_dim=-1)


def _write_codes(
    params: dict[str, torch.Tensor],
    codes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = _clone_params(params)
    sign = torch.where(codes == 0.0, torch.zeros_like(codes), torch.sign(codes))
    mantissa = codes.abs() * 0.25
    result["sign"] = sign.reshape_as(result["sign"])
    result["mant"] = mantissa.reshape_as(result["mant"])
    return result


def _polish_weight(
    weight: torch.Tensor,
    parent: dict[str, torch.Tensor],
    activation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows, channels = map(int, weight.shape)
    # Very expansive FFN matrices expose too many independently polished rows
    # for two calibration folds; their apparent fit does not transfer.
    if (
        channels < _WEIGHT_HSDQ_MIN_CHANNELS
        or channels % _BLOCK != 0
        or rows > 2 * channels
    ):
        return parent
    z = _sample_rows(activation, _WEIGHT_HSDQ_MAX_ROWS).to(torch.float32)
    if z.ndim != 2 or int(z.shape[1]) != channels:
        return parent
    blocks = channels // _BLOCK
    q = _dequantize_hif4(parent).to(torch.float32).clone()
    den = _denominator(parent).reshape(rows, blocks, _BLOCK)
    codes = torch.round(
        q.reshape(rows, blocks, _BLOCK) * 4.0 / den.clamp_min(_EPS)
    ).clamp(-7.0, 7.0)
    residual = z.mm((weight - q).t())
    leverage = []
    for block in range(blocks):
        lo = block * _BLOCK
        hi = lo + _BLOCK
        leverage.append(z[:, lo:hi].t().mm(residual).square().sum())
    count = min(blocks, max(1, int(_WEIGHT_HSDQ_BLOCKS)))
    selected = torch.topk(torch.stack(leverage), k=count).indices.tolist()
    levels = torch.as_tensor(_SIGNED_LEVELS, device=weight.device)
    for block in selected:
        lo = int(block) * _BLOCK
        hi = lo + _BLOCK
        local_z = z[:, lo:hi]
        gram = local_z.t().mm(local_z)
        diagonal = gram.diagonal().clamp_min(_EPS)
        local_q = q[:, lo:hi]
        local_den = den[:, int(block)]
        local_codes = codes[:, int(block)]
        for _ in range(max(1, int(_WEIGHT_HSDQ_SWEEPS))):
            correlation = local_z.t().mm(residual).t()
            for coordinate in range(_BLOCK):
                current = local_q[:, coordinate]
                options = local_den[:, coordinate, None] * levels[None, :]
                step = options - current[:, None]
                change = (
                    -2.0 * step * correlation[:, coordinate, None]
                    + diagonal[coordinate] * step.square()
                )
                best_change, best_index = change.min(dim=-1)
                improve = torch.isfinite(best_change) & (best_change < -_EPS)
                accepted = step.gather(-1, best_index[:, None]).squeeze(-1)
                accepted = torch.where(improve, accepted, torch.zeros_like(accepted))
                local_q[:, coordinate] += accepted
                local_codes[:, coordinate] = torch.round(
                    local_q[:, coordinate] * 4.0 / local_den[:, coordinate].clamp_min(_EPS)
                ).clamp(-7.0, 7.0)
                residual.add_(-local_z[:, coordinate, None] * accepted[None, :])
                correlation.add_(-accepted[:, None] * gram[coordinate][None, :])
        q[:, lo:hi] = local_q
        codes[:, int(block)] = local_codes
    return _write_codes(parent, codes.reshape(rows, channels))


def _product_loss(
    activation: torch.Tensor,
    weight: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> float:
    q = _dequantize_hif4(params).to(torch.float32)
    z = _sample_rows(activation, _WEIGHT_HSDQ_MAX_ROWS).to(torch.float32)
    residual = z.mm((weight - q).t())
    reference = z.mm(weight.t())
    return float(residual.square().mean() / (reference.square().mean() + _EPS))


def _crossfold_weight_hsdq(
    weight: torch.Tensor,
    parent: dict[str, torch.Tensor],
    calibration: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    if len(calibration) < 2:
        return _polish_weight(weight, parent, calibration[0]) if calibration else parent
    folds = [item.to(torch.float32) for item in calibration[:2]]
    candidates = [parent]
    cand0 = _polish_weight(weight, parent, folds[0])
    cand1 = _polish_weight(weight, parent, folds[1])

    parent_losses = [_product_loss(fold, weight, parent) for fold in folds]
    # Cross-fit admission: the candidate generated on one fold must improve
    # the other fold before it can enter the final robust selector.
    if _product_loss(folds[1], weight, cand0) < parent_losses[1]:
        candidates.append(cand0)
    if _product_loss(folds[0], weight, cand1) < parent_losses[0]:
        candidates.append(cand1)

    best = parent
    best_score = sum(parent_losses) / 2.0 + _WEIGHT_HSDQ_ROBUST_MIX * max(parent_losses)
    for candidate in candidates[1:]:
        losses = [_product_loss(fold, weight, candidate) for fold in folds]
        score = sum(losses) / 2.0 + _WEIGHT_HSDQ_ROBUST_MIX * max(losses)
        if score < best_score * (1.0 - _WEIGHT_HSDQ_MIN_GAIN):
            best = candidate
            best_score = score
    return best


def _gram64(weight: torch.Tensor) -> torch.Tensor:
    channels = int(weight.shape[1])
    blocks = channels // _BLOCK
    gram = weight.t().mm(weight)
    index = torch.arange(channels, device=weight.device).reshape(blocks, _BLOCK)
    return gram[index[:, :, None], index[:, None, :]]


def _global_activation_lrh(
    weight: torch.Tensor,
    gram64: torch.Tensor | None,
    *,
    rank: int = _ACT_GLOBAL_LRH_RANK,
    max_channels: int = _ACT_GLOBAL_LRH_MAX_CHANNELS,
) -> torch.Tensor | None:
    """Return a small PSD approximation to the off-block weight Gram.

    ``gram64`` is the legal block-diagonal state already used by the local
    activation refiner.  The randomized range iteration applies the residual
    ``W.T @ W - blockdiag(W.T @ W)`` without materializing that dense matrix
    during calibration.  The state is only a static weight statistic; the
    online candidate is still accepted by the exact deployed Gram gate below.
    """

    rows, channels = map(int, weight.shape)
    if (
        gram64 is None
        or channels > int(max_channels)
        or channels % _BLOCK != 0
        or int(rank) <= 0
    ):
        return None
    blocks = channels // _BLOCK
    if tuple(gram64.shape) != (blocks, _BLOCK, _BLOCK):
        return None
    width = min(channels, int(rank) + _ACT_GLOBAL_LRH_OVERSAMPLE)
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(0x6A1 + channels * 17 + rows)
    probe = torch.randn(
        channels,
        width,
        device=weight.device,
        dtype=torch.float32,
        generator=generator,
    )

    def offblock_matmul(value: torch.Tensor) -> torch.Tensor:
        full = weight.t().mm(weight.mm(value))
        grouped = value.reshape(blocks, _BLOCK, -1)
        block_part = torch.einsum(
            "bij,bjr->bir", gram64.to(value), grouped
        )
        return full - block_part.reshape_as(full)

    try:
        basis, _ = torch.linalg.qr(probe, mode="reduced")
        for _ in range(max(1, int(_ACT_GLOBAL_LRH_POWER_STEPS))):
            basis, _ = torch.linalg.qr(offblock_matmul(basis), mode="reduced")
        reduced = basis.t().mm(offblock_matmul(basis))
        eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    except RuntimeError:
        return None
    positive = eigenvalues > eigenvalues.abs().max().clamp_min(_EPS) * 1.0e-6
    indices = torch.nonzero(positive, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    indices = indices[-min(int(rank), int(indices.numel())) :]
    values = eigenvalues.index_select(0, indices).clamp_min(0.0)
    vectors = basis.mm(eigenvectors.index_select(1, indices))
    result = vectors * values.sqrt()[None, :]
    if not torch.isfinite(result).all():
        return None
    return result


def _refine_activation(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    *,
    max_blocks: int = _ACT_HSDQ_BLOCKS,
    sweeps: int = _ACT_HSDQ_SWEEPS,
) -> dict[str, torch.Tensor]:
    if gram64 is None or dense.ndim != 2:
        return parent
    rows, channels = map(int, dense.shape)
    if channels % _BLOCK != 0:
        return parent
    blocks = channels // _BLOCK
    h = gram64.to(device=dense.device, dtype=torch.float32)
    if tuple(h.shape) != (blocks, _BLOCK, _BLOCK):
        return parent
    q = _dequantize_hif4(parent).to(torch.float32).reshape(rows, blocks, _BLOCK)
    x = dense.to(torch.float32).reshape_as(q)
    den = _denominator(parent).reshape_as(q)
    codes = torch.round(q * 4.0 / den.clamp_min(_EPS)).clamp(-7.0, 7.0)
    error = q - x
    losses = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    count = min(blocks, max(1, int(max_blocks)))
    chosen = torch.topk(losses, k=count, dim=1).indices
    row_ids = torch.arange(rows, device=dense.device)[:, None].expand(rows, count).reshape(-1)
    block_ids = chosen.reshape(-1)
    q_work = q[row_ids, block_ids].clone()
    x_work = x[row_ids, block_ids]
    den_work = den[row_ids, block_ids]
    h_work = h.index_select(0, block_ids)
    levels = torch.as_tensor(_SIGNED_LEVELS, device=dense.device)
    diagonal = h_work.diagonal(dim1=-2, dim2=-1).clamp_min(_EPS)
    for _ in range(max(1, int(sweeps))):
        gradient = torch.einsum("nij,nj->ni", h_work, q_work - x_work)
        for coordinate in range(_BLOCK):
            current = q_work[:, coordinate]
            options = den_work[:, coordinate, None] * levels[None, :]
            step = options - current[:, None]
            change = 2.0 * step * gradient[:, coordinate, None] + diagonal[:, coordinate, None] * step.square()
            best_change, best_index = change.min(dim=-1)
            improve = torch.isfinite(best_change) & (best_change < -_EPS)
            accepted = step.gather(-1, best_index[:, None]).squeeze(-1)
            accepted = torch.where(improve, accepted, torch.zeros_like(accepted))
            q_work[:, coordinate] += accepted
            gradient.add_(accepted[:, None] * h_work[:, :, coordinate])
    q[row_ids, block_ids] = q_work
    codes[row_ids, block_ids] = torch.round(
        q_work * 4.0 / den_work.clamp_min(_EPS)
    ).clamp(-7.0, 7.0)
    return _write_codes(parent, codes.reshape(rows, channels))


@torch.no_grad()
def _refine_activation_hierarchy_g64(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    *,
    max_blocks: int = _ACT_G64_HIERARCHY_BLOCKS,
    sweeps: int = _ACT_G64_HIERARCHY_SWEEPS,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Run a bounded exact ``G_64`` hierarchy-coordinate proposal.

    The E6M2 block scale is held fixed.  For at most ``max_blocks`` blocks per
    row, one sweep visits each legal ``lv2`` and ``lv3`` coordinate and tries
    the two values ``{1, 2}``.  Every trial re-encodes the affected 64-channel
    block atomically and is scored by the exact quadratic increment

    ``Delta J = 2 e.T G Delta_q + Delta_q.T G Delta_q``.

    This is an activation-side proposal only: ``gram64`` is static calibration
    state, while the caller's existing deployed-Gram gate remains the final
    acceptance rule.  No output/residual statistic is stored in state.
    """

    diagnostics = {
        "rows": 0,
        "selected_blocks": 0,
        "accepted_coordinates": 0,
        "accepted_blocks": 0,
    }

    def finish(
        value: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, int]]:
        if return_diagnostics:
            return value, diagnostics
        return value

    if (
        not _ACT_G64_HIERARCHY_ENABLED
        or gram64 is None
        or dense.ndim != 2
        or not all(key in parent for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"))
    ):
        return finish(parent)
    rows, channels = map(int, dense.shape)
    diagnostics["rows"] = rows
    if channels % _BLOCK != 0:
        return finish(parent)
    blocks = channels // _BLOCK
    h = gram64.to(device=dense.device, dtype=torch.float32)
    if tuple(h.shape) != (blocks, _BLOCK, _BLOCK):
        return finish(parent)

    q_initial = _dequantize_hif4(parent).to(torch.float32)
    if tuple(q_initial.shape) != (rows, channels):
        return finish(parent)
    x = dense.to(torch.float32).reshape(rows, blocks, 8, 2, 4)
    q = q_initial.reshape(rows, blocks, 8, 2, 4).clone()
    scale_factor = parent["scale_factor"].reshape(rows, blocks, 1, 1, 1)[
        ..., 0, 0, 0
    ].to(torch.float32)
    lv2 = parent["scale_lv2"].reshape(rows, blocks, 8, 1, 1)[..., 0, 0].to(
        torch.float32
    ).clone()
    lv3 = parent["scale_lv3"].reshape(rows, blocks, 8, 2, 1)[..., 0].to(
        torch.float32
    ).clone()
    sign = parent["sign"].reshape(rows, blocks, 8, 2, 4).clone()
    mant = parent["mant"].reshape(rows, blocks, 8, 2, 4).clone()
    error = q - x
    losses = torch.einsum(
        "rbi,bij,rbj->rb",
        error.reshape(rows, blocks, _BLOCK),
        h,
        error.reshape(rows, blocks, _BLOCK),
    )
    count = min(blocks, max(1, int(max_blocks)))
    chosen = torch.topk(losses, k=count, dim=1).indices
    diagnostics["selected_blocks"] = int(rows * count)

    levels = (1.0, 2.0)
    for row in range(rows):
        for block in chosen[row].tolist():
            block = int(block)
            x_block = x[row, block]
            q_work = q[row, block].reshape(_BLOCK).clone()
            e_work = error[row, block].reshape(_BLOCK).clone()
            lv2_work = lv2[row, block].clone()
            lv3_work = lv3[row, block].clone()
            sign_work = sign[row, block].clone()
            mant_work = mant[row, block].clone()
            gram = h[block]
            sf = scale_factor[row, block]
            changed_block = False

            def encode(
                trial_lv2: torch.Tensor,
                trial_lv3: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                denominator = sf * trial_lv2[:, None, None] * trial_lv3[:, :, None]
                trial_mant = torch.round(
                    x_block.abs() * (4.0 / denominator.clamp_min(_EPS))
                ).clamp(0.0, 7.0) * 0.25
                trial_sign = torch.sign(x_block)
                trial_sign = torch.where(
                    trial_mant == 0.0, torch.zeros_like(trial_sign), trial_sign
                )
                trial_q = (trial_sign * trial_mant * denominator).reshape(_BLOCK)
                return trial_q, trial_sign, trial_mant

            def try_coordinate(
                coordinate: tuple[str, int, int | None],
            ) -> None:
                nonlocal q_work, e_work, lv2_work, lv3_work
                nonlocal sign_work, mant_work, changed_block
                kind, first, second = coordinate
                current = (
                    lv2_work[first]
                    if kind == "lv2"
                    else lv3_work[first, int(second)]
                )
                best_delta = torch.zeros((), device=dense.device, dtype=torch.float32)
                best_value: float | None = None
                best_q: torch.Tensor | None = None
                best_sign: torch.Tensor | None = None
                best_mant: torch.Tensor | None = None
                for value in levels:
                    if abs(float(current.item()) - value) <= _EPS:
                        continue
                    trial_lv2 = lv2_work.clone()
                    trial_lv3 = lv3_work.clone()
                    if kind == "lv2":
                        trial_lv2[first] = value
                    else:
                        trial_lv3[first, int(second)] = value
                    trial_q, trial_sign, trial_mant = encode(trial_lv2, trial_lv3)
                    step = trial_q - q_work
                    gram_step = gram.mv(step)
                    delta = 2.0 * torch.dot(e_work, gram_step) + torch.dot(
                        step, gram_step
                    )
                    if torch.isfinite(delta) and delta < best_delta:
                        best_delta = delta
                        best_value = value
                        best_q = trial_q
                        best_sign = trial_sign
                        best_mant = trial_mant
                if (
                    best_value is None
                    or best_q is None
                    or best_sign is None
                    or best_mant is None
                ):
                    return
                if kind == "lv2":
                    lv2_work[first] = best_value
                else:
                    lv3_work[first, int(second)] = best_value
                q_work = best_q
                e_work = q_work - x_block.reshape(_BLOCK)
                sign_work = best_sign
                mant_work = best_mant
                changed_block = True
                diagnostics["accepted_coordinates"] += 1

            for _ in range(max(1, int(sweeps))):
                for group in range(8):
                    try_coordinate(("lv2", group, None))
                for group in range(8):
                    for subgroup in range(2):
                        try_coordinate(("lv3", group, subgroup))
            if changed_block:
                diagnostics["accepted_blocks"] += 1
                q[row, block] = q_work.reshape(8, 2, 4)
                lv2[row, block] = lv2_work
                lv3[row, block] = lv3_work
                sign[row, block] = sign_work
                mant[row, block] = mant_work

    result = _clone_params(parent)
    result["scale_lv2"] = lv2.reshape_as(parent["scale_lv2"])
    result["scale_lv3"] = lv3.reshape_as(parent["scale_lv3"])
    result["sign"] = sign.reshape_as(parent["sign"])
    result["mant"] = mant.reshape_as(parent["mant"])
    return finish(result)


@torch.no_grad()
def _structured_activation_lrh(
    deployment_gram: torch.Tensor | None,
    *,
    components: int = _ACT_STRUCTURED_LRH_COMPONENTS,
    max_channels: int = _ACT_STRUCTURED_LRH_MAX_CHANNELS,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compress cross-block Gram structure into block-circulant kernels.

    For each circular block distance ``d`` the block pairs of the exact
    deployed Gram are averaged, then the resulting ``B x 64 x 64`` sequence is
    fitted with a rank-``components`` SVD.  The state therefore stores only
    ``components`` kernels and ``B`` scalar coefficient vectors, never the
    dense block-pair matrix.  Distance zero is explicitly omitted because the
    established ``gram64`` state already supplies the block diagonal.
    """

    if deployment_gram is None or deployment_gram.ndim != 2:
        return None
    channels = int(deployment_gram.shape[0])
    if (
        tuple(deployment_gram.shape) != (channels, channels)
        or channels <= _BLOCK
        or channels > int(max_channels)
        or channels % _BLOCK != 0
        or int(components) <= 0
    ):
        return None
    blocks = channels // _BLOCK
    gram = deployment_gram.to(torch.float32)
    grouped = gram.reshape(blocks, _BLOCK, blocks, _BLOCK).permute(0, 2, 1, 3)
    distances = torch.zeros(
        blocks, _BLOCK, _BLOCK, device=gram.device, dtype=torch.float32
    )
    block_index = torch.arange(blocks, device=gram.device)
    for distance in range(1, blocks):
        paired = grouped[block_index, (block_index + distance) % blocks]
        distances[distance] = paired.mean(dim=0)
    try:
        left, singular, right = torch.linalg.svd(
            distances[1:].reshape(blocks - 1, -1), full_matrices=False
        )
    except RuntimeError:
        return None
    rank = min(int(components), int(right.shape[0]))
    if rank <= 0:
        return None
    kernels = right[:rank].reshape(rank, _BLOCK, _BLOCK)
    coefficients = torch.zeros(
        blocks, rank, device=gram.device, dtype=torch.float32
    )
    coefficients[1:] = left[:, :rank] * singular[:rank].reshape(1, -1)
    if not torch.isfinite(kernels).all() or not torch.isfinite(coefficients).all():
        return None
    return kernels, coefficients


@torch.no_grad()
def _structured_gram_matmul(
    error: torch.Tensor,
    kernels: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Multiply a row-batched block vector by a block-circulant approximation."""

    rows, blocks, width = map(int, error.shape)
    if width != _BLOCK:
        return torch.zeros_like(error)
    transformed = torch.einsum("rbi,sij->rsbj", error, kernels)
    result = torch.zeros_like(error)
    for distance in range(blocks):
        coefficient = coefficients[distance]
        if bool(torch.any(coefficient.abs() > _EPS)):
            result = result + torch.roll(
                transformed,
                shifts=distance,
                dims=2,
            ).mul(coefficient.reshape(1, -1, 1, 1)).sum(dim=1)
    return result


@torch.no_grad()
def _refine_activation_structured_reference(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    deployment_gram: torch.Tensor | None,
    structured_state: dict[str, torch.Tensor] | None,
    *,
    max_blocks: int = _ACT_STRUCTURED_LRH_BLOCKS,
) -> dict[str, torch.Tensor]:
    """Generate a structured cross-block proposal and gate it by exact ``G_q``."""

    if (
        dense.ndim != 2
        or gram64 is None
        or deployment_gram is None
        or not isinstance(structured_state, dict)
    ):
        return parent
    kernels = structured_state.get("kernels")
    coefficients = structured_state.get("coefficients")
    if not torch.is_tensor(kernels) or not torch.is_tensor(coefficients):
        return parent
    rows, channels = map(int, dense.shape)
    if channels % _BLOCK != 0 or tuple(deployment_gram.shape) != (channels, channels):
        return parent
    blocks = channels // _BLOCK
    h = gram64.to(device=dense.device, dtype=torch.float32)
    kernels = kernels.to(device=dense.device, dtype=torch.float32)
    coefficients = coefficients.to(device=dense.device, dtype=torch.float32)
    if (
        kernels.ndim != 3
        or tuple(kernels.shape[1:]) != (_BLOCK, _BLOCK)
        or coefficients.shape != (blocks, kernels.shape[0])
        or not torch.isfinite(kernels).all()
        or not torch.isfinite(coefficients).all()
    ):
        return parent
    q_initial = _dequantize_hif4(parent).to(torch.float32)
    if tuple(q_initial.shape) != (rows, channels):
        return parent
    x = dense.to(torch.float32)
    q = q_initial.reshape(rows, blocks, _BLOCK).clone()
    error = q - x.reshape_as(q)
    local_loss = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    count = min(blocks, max(1, int(max_blocks)))
    chosen = torch.topk(local_loss, k=count, dim=1).indices
    structured_gradient = _structured_gram_matmul(error, kernels, coefficients)
    den = _denominator(parent).to(torch.float32).reshape(rows, blocks, _BLOCK)
    levels = torch.as_tensor(_SIGNED_LEVELS, device=dense.device)
    codes = torch.round(q * 4.0 / den.clamp_min(_EPS)).clamp(-7.0, 7.0)
    for row in range(rows):
        for block in chosen[row].tolist():
            block = int(block)
            q_work = q[row, block].clone()
            e_work = error[row, block].clone()
            h_work = h[block]
            for coordinate in range(_BLOCK):
                current = q_work[coordinate]
                options = den[row, block, coordinate] * levels
                step = options - current
                local_gradient = h_work[coordinate].matmul(e_work)
                cross_gradient = structured_gradient[row, block, coordinate]
                diagonal = h_work[coordinate, coordinate]
                change = 2.0 * step * (local_gradient + cross_gradient) + diagonal * step.square()
                best_change, best_index = change.min(dim=-1)
                if bool(torch.isfinite(best_change) and best_change < -_EPS):
                    accepted = step[best_index]
                    q_work[coordinate] += accepted
                    e_work[coordinate] += accepted
                    codes[row, block, coordinate] = torch.round(
                        q_work[coordinate] * 4.0 / den[row, block, coordinate].clamp_min(_EPS)
                    ).clamp(-7.0, 7.0)
                    # The structured gradient is intentionally a frozen
                    # proposal gradient; the exact deployed-Gram row gate
                    # below decides whether the whole row survives.
            q[row, block] = q_work
    candidate = _write_codes(parent, codes.reshape(rows, channels))
    return _select_activation_by_deployment_gram(
        dense,
        parent,
        candidate,
        deployment_gram.to(device=dense.device, dtype=torch.float32),
    )


@torch.no_grad()
def _structured_coordinate_batch(
    q_work: torch.Tensor,
    e_work: torch.Tensor,
    h_work: torch.Tensor,
    cross_work: torch.Tensor,
    den_work: torch.Tensor,
    levels: torch.Tensor,
    codes: torch.Tensor,
    row_index: torch.Tensor,
    block_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the ordered 64-coordinate/15-level update to a batch of blocks."""

    for coordinate in range(_BLOCK):
        current = q_work[:, coordinate]
        options = den_work[:, coordinate, None] * levels[None, :]
        step = options - current[:, None]
        local_gradient = torch.bmm(
            h_work[:, coordinate, :].unsqueeze(1), e_work.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)
        cross_gradient = cross_work[:, coordinate]
        diagonal = h_work[:, coordinate, coordinate]
        change = (
            2.0 * step * (local_gradient + cross_gradient)[:, None]
            + diagonal[:, None] * step.square()
        )
        best_change, best_index = change.min(dim=-1)
        accepted_mask = torch.isfinite(best_change) & (best_change < -_EPS)
        accepted = step.gather(1, best_index[:, None]).squeeze(1)
        accepted = torch.where(accepted_mask, accepted, torch.zeros_like(accepted))
        q_work[:, coordinate] += accepted
        e_work[:, coordinate] += accepted
        new_codes = torch.round(
            q_work[:, coordinate] * 4.0 / den_work[:, coordinate].clamp_min(_EPS)
        ).clamp(-7.0, 7.0)
        old_codes = codes[row_index, block_index, coordinate]
        codes[row_index, block_index, coordinate] = torch.where(
            accepted_mask, new_codes, old_codes
        )
    return q_work, e_work


@torch.no_grad()
def _refine_activation_structured_vectorized(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    deployment_gram: torch.Tensor | None,
    structured_state: dict[str, torch.Tensor] | None,
    *,
    max_blocks: int = _ACT_STRUCTURED_LRH_BLOCKS,
    refresh_mode: str | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorized C1a structured proposal with reference-equivalent ordering.

    With refresh mode ``none`` this is the v118-equivalent C1a path.  The C1b
    ``block`` and ``sweep2`` modes reuse the same ordered coordinate kernel but
    refresh the structured gradient after each selected block (once or twice
    over the block list).  All modes retain the final exact ``G_q`` gate.
    """

    if (
        dense.ndim != 2
        or gram64 is None
        or deployment_gram is None
        or not isinstance(structured_state, dict)
    ):
        return parent
    kernels = structured_state.get("kernels")
    coefficients = structured_state.get("coefficients")
    if not torch.is_tensor(kernels) or not torch.is_tensor(coefficients):
        return parent
    rows, channels = map(int, dense.shape)
    if channels % _BLOCK != 0 or tuple(deployment_gram.shape) != (channels, channels):
        return parent
    blocks = channels // _BLOCK
    h = gram64.to(device=dense.device, dtype=torch.float32)
    kernels = kernels.to(device=dense.device, dtype=torch.float32)
    coefficients = coefficients.to(device=dense.device, dtype=torch.float32)
    if (
        kernels.ndim != 3
        or tuple(kernels.shape[1:]) != (_BLOCK, _BLOCK)
        or coefficients.shape != (blocks, kernels.shape[0])
        or not torch.isfinite(kernels).all()
        or not torch.isfinite(coefficients).all()
    ):
        return parent
    q_initial = _dequantize_hif4(parent).to(torch.float32)
    if tuple(q_initial.shape) != (rows, channels):
        return parent
    x = dense.to(torch.float32)
    q = q_initial.reshape(rows, blocks, _BLOCK).clone()
    error = q - x.reshape_as(q)
    local_loss = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    count = min(blocks, max(1, int(max_blocks)))
    chosen = torch.topk(local_loss, k=count, dim=1).indices
    den = _denominator(parent).to(torch.float32).reshape(rows, blocks, _BLOCK)
    levels = torch.as_tensor(_SIGNED_LEVELS, device=dense.device, dtype=torch.float32)
    codes = torch.round(q * 4.0 / den.clamp_min(_EPS)).clamp(-7.0, 7.0)
    refresh_mode = (
        _ACT_STRUCTURED_LRH_REFRESH_MODE if refresh_mode is None else refresh_mode
    )
    if refresh_mode not in {"none", "block", "sweep2"}:
        return parent
    if refresh_mode == "none":
        # Top-k blocks do not interact once the structured gradient is frozen.
        # Flattening in row-major/top-k order retains the v118 traversal set
        # while allowing all rows/blocks to share each coordinate-level kernel.
        structured_gradient = _structured_gram_matmul(error, kernels, coefficients)
        row_index = torch.arange(rows, device=dense.device).reshape(-1, 1).expand(
            rows, count
        ).reshape(-1)
        block_index = chosen.reshape(-1)
        q_work = q[row_index, block_index].clone()
        e_work = error[row_index, block_index].clone()
        q_work, e_work = _structured_coordinate_batch(
            q_work,
            e_work,
            h[block_index],
            structured_gradient[row_index, block_index],
            den[row_index, block_index],
            levels,
            codes,
            row_index,
            block_index,
        )
        q[row_index, block_index] = q_work
    else:
        # C1b refresh modes process the same top-k list by rank.  A refresh is
        # done after each rank so subsequent blocks see the accepted proposal
        # in their structured cross-block gradient.
        rounds = 2 if refresh_mode == "sweep2" else 1
        for _ in range(rounds):
            structured_gradient = _structured_gram_matmul(error, kernels, coefficients)
            row_index = torch.arange(rows, device=dense.device)
            for rank in range(count):
                block_index = chosen[:, rank]
                q_work = q[row_index, block_index].clone()
                e_work = error[row_index, block_index].clone()
                q_work, e_work = _structured_coordinate_batch(
                    q_work,
                    e_work,
                    h[block_index],
                    structured_gradient[row_index, block_index],
                    den[row_index, block_index],
                    levels,
                    codes,
                    row_index,
                    block_index,
                )
                q[row_index, block_index] = q_work
                error[row_index, block_index] = e_work
                structured_gradient = _structured_gram_matmul(
                    error, kernels, coefficients
                )
    candidate = _write_codes(parent, codes.reshape(rows, channels))
    return _select_activation_by_deployment_gram(
        dense,
        parent,
        candidate,
        deployment_gram.to(device=dense.device, dtype=torch.float32),
    )


@torch.no_grad()
def _refine_activation_structured(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    deployment_gram: torch.Tensor | None,
    structured_state: dict[str, torch.Tensor] | None,
    *,
    max_blocks: int = _ACT_STRUCTURED_LRH_BLOCKS,
) -> dict[str, torch.Tensor]:
    """Dispatch the C1a batched proposal or the v118 reference path."""

    if _ACT_STRUCTURED_LRH_VECTORIZED:
        return _refine_activation_structured_vectorized(
            dense,
            parent,
            gram64,
            deployment_gram,
            structured_state,
            max_blocks=max_blocks,
        )
    return _refine_activation_structured_reference(
        dense,
        parent,
        gram64,
        deployment_gram,
        structured_state,
        max_blocks=max_blocks,
    )


def _refine_activation_global_lrh(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    deployment_gram: torch.Tensor | None,
    global_u: torch.Tensor | None,
    *,
    max_blocks: int = _ACT_GLOBAL_LRH_BLOCKS,
    sweeps: int = _ACT_HSDQ_SWEEPS,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Propose global-LRH activation edits and gate them with ``G_q``.

    The local block Gram and low-rank residual are used only to generate a
    bounded proposal.  The final discrete candidate is decoded again and
    compared row-by-row with the exact Gram of the deployed quantized weight,
    so an approximation can never replace a parent on the true quadratic
    objective.  ``return_diagnostics`` is evaluator-only and does not alter
    the public API or state tree.
    """

    diagnostics = {
        "rows": 0,
        "proposal_rows": 0,
        "gram_accept_rows": 0,
        "mse_accept_rows": 0,
        "accepted_rows": 0,
        "gram_mse_conflict_rows": 0,
    }

    def finish(
        value: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, int]]:
        if return_diagnostics:
            return value, diagnostics
        return value

    if (
        gram64 is None
        or deployment_gram is None
        or global_u is None
        or dense.ndim != 2
    ):
        return finish(parent)
    rows, channels = map(int, dense.shape)
    diagnostics["rows"] = rows
    if channels % _BLOCK != 0:
        return finish(parent)
    blocks = channels // _BLOCK
    h = gram64.to(device=dense.device, dtype=torch.float32)
    g = deployment_gram.to(device=dense.device, dtype=torch.float32)
    u = global_u.to(device=dense.device, dtype=torch.float32)
    if (
        tuple(h.shape) != (blocks, _BLOCK, _BLOCK)
        or tuple(g.shape) != (channels, channels)
        or u.ndim != 2
        or tuple(u.shape[:1]) != (channels,)
        or int(u.shape[1]) <= 0
    ):
        return finish(parent)

    q_initial = _dequantize_hif4(parent).to(torch.float32)
    if tuple(q_initial.shape) != (rows, channels):
        return finish(parent)
    q = q_initial.reshape(rows, blocks, _BLOCK).clone()
    x = dense.to(torch.float32).reshape_as(q)
    den = _denominator(parent).reshape_as(q)
    codes = torch.round(q * 4.0 / den.clamp_min(_EPS)).clamp(-7.0, 7.0)
    error = q - x
    losses = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    count = min(blocks, max(1, int(max_blocks)))
    chosen = torch.topk(losses, k=count, dim=1).indices
    row_ids = (
        torch.arange(rows, device=dense.device)[:, None]
        .expand(rows, count)
        .reshape(-1)
    )
    block_ids = chosen.reshape(-1)
    q_work = q[row_ids, block_ids].clone()
    x_work = x[row_ids, block_ids]
    den_work = den[row_ids, block_ids]
    h_work = h.index_select(0, block_ids)
    levels = torch.as_tensor(_SIGNED_LEVELS, device=dense.device)
    diagonal = h_work.diagonal(dim1=-2, dim2=-1).clamp_min(_EPS)
    lowrank_projection = (q_initial - dense.to(torch.float32)).mm(u)
    for _ in range(max(1, int(sweeps))):
        gradient = torch.einsum("nij,nj->ni", h_work, q_work - x_work)
        for coordinate in range(_BLOCK):
            current = q_work[:, coordinate]
            options = den_work[:, coordinate, None] * levels[None, :]
            step = options - current[:, None]
            channel_ids = block_ids * _BLOCK + coordinate
            u_column = u.index_select(0, channel_ids)
            global_gradient = (
                lowrank_projection.index_select(0, row_ids) * u_column
            ).sum(dim=-1)
            global_diagonal = u_column.square().sum(dim=-1)
            change = (
                2.0
                * step
                * (gradient[:, coordinate] + global_gradient)[:, None]
                + (diagonal[:, coordinate] + global_diagonal)[:, None]
                * step.square()
            )
            best_change, best_index = change.min(dim=-1)
            improve = torch.isfinite(best_change) & (best_change < -_EPS)
            accepted = step.gather(-1, best_index[:, None]).squeeze(-1)
            accepted = torch.where(improve, accepted, torch.zeros_like(accepted))
            q_work[:, coordinate] += accepted
            gradient.add_(accepted[:, None] * h_work[:, :, coordinate])
            lowrank_projection.index_add_(
                0, row_ids, accepted[:, None] * u_column
            )
    q[row_ids, block_ids] = q_work
    codes[row_ids, block_ids] = torch.round(
        q_work * 4.0 / den_work.clamp_min(_EPS)
    ).clamp(-7.0, 7.0)
    candidate = _write_codes(parent, codes.reshape(rows, channels))
    q_candidate = _dequantize_hif4(candidate).to(torch.float32)

    parent_error = q_initial - dense.to(torch.float32)
    candidate_error = q_candidate - dense.to(torch.float32)
    parent_gram = (parent_error.mm(g) * parent_error).sum(dim=1)
    candidate_gram = (candidate_error.mm(g) * candidate_error).sum(dim=1)
    parent_mse = parent_error.square().sum(dim=1)
    candidate_mse = candidate_error.square().sum(dim=1)
    proposal = (q_candidate - q_initial).abs().amax(dim=1) > _EPS
    gram_accept = torch.isfinite(candidate_gram) & (
        candidate_gram <= parent_gram + _EPS
    )
    mse_accept = torch.isfinite(candidate_mse) & (
        candidate_mse <= parent_mse + _EPS
    )
    keep = proposal & gram_accept
    conflict = proposal & (gram_accept != mse_accept)
    diagnostics["proposal_rows"] = int(proposal.sum().item())
    diagnostics["gram_accept_rows"] = int((proposal & gram_accept).sum().item())
    diagnostics["mse_accept_rows"] = int((proposal & mse_accept).sum().item())
    diagnostics["accepted_rows"] = int(keep.sum().item())
    diagnostics["gram_mse_conflict_rows"] = int(conflict.sum().item())

    result: dict[str, torch.Tensor] = {}
    for key, candidate_value in candidate.items():
        mask = keep.reshape(rows, *([1] * (candidate_value.ndim - 1)))
        result[key] = torch.where(mask, candidate_value, parent[key])
    return finish(result)


def _gals_candidate_offsets(dense: torch.Tensor) -> torch.Tensor:
    """Generate a compact analytical E6M2 offset set per row and block."""

    rows, channels = map(int, dense.shape)
    blocks = channels // _BLOCK
    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * 7.0,
        neginf=-_E6M2_MAX * 7.0,
    ).reshape(rows, blocks, _BLOCK)
    absolute = x.abs()
    standard_code, _ = _standard_scale(absolute.amax(dim=-1))
    mantissa = torch.as_tensor(
        (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        device=x.device,
        dtype=torch.float32,
    )
    exponent = torch.as_tensor(
        (1.0, 2.0, 4.0), device=x.device, dtype=torch.float32
    )
    critical_scale = absolute[..., :, None, None] / (
        mantissa[None, None, None, :, None]
        * exponent[None, None, None, None, :]
    )
    projected = _e6m2_encode_nearest(critical_scale)
    projected = torch.cat((projected - 1, projected, projected + 1), dim=-1)
    projected = projected.clamp(0, 254).reshape(rows, blocks, -1)
    offsets = projected - standard_code[..., None].to(torch.int64)
    base = torch.as_tensor(
        _BASE_OFFSETS, device=x.device, dtype=torch.int64
    ).reshape(1, 1, -1)
    offsets = torch.cat((offsets, base.expand(rows, blocks, -1)), dim=-1)
    return torch.unique(offsets, dim=-1)


@torch.no_grad()
def _refine_activation_gals_final(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    *,
    max_blocks: int = 4,
    deployment_gram: torch.Tensor | None = None,
    return_gain: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], float]:
    """Small final-weight-Gram GALS search for high-headroom shapes.

    The analytical offset set is only evaluated for the largest-loss blocks;
    every hierarchy field is selected atomically by ``_encode_rows`` and the
    same deployment Gram block is used for the final comparison.  This helper
    is intentionally shape/statistics driven and has no role or model input.
    """

    if gram64 is None or dense.ndim != 2:
        return (parent, 0.0) if return_gain else parent
    rows, channels = map(int, dense.shape)
    if channels % _BLOCK != 0:
        return (parent, 0.0) if return_gain else parent
    blocks = channels // _BLOCK
    h = gram64.to(device=dense.device, dtype=torch.float32)
    if tuple(h.shape) != (blocks, _BLOCK, _BLOCK):
        return (parent, 0.0) if return_gain else parent
    q_parent = _dequantize_hif4(parent).to(torch.float32)
    error = (q_parent - dense.to(torch.float32)).reshape(rows, blocks, _BLOCK)
    parent_loss = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    count = min(blocks, max(1, int(max_blocks)))
    selected = torch.zeros_like(parent_loss, dtype=torch.bool)
    selected.scatter_(1, torch.topk(parent_loss, k=count, dim=1).indices, True)
    candidates = _gals_candidate_offsets(dense)
    best_loss = parent_loss.clone()
    best = _clone_params(parent)
    for offset in torch.unique(candidates).tolist():
        candidate = _encode_rows(dense, (int(offset),), gram64=h)
        candidate_error = (
            _dequantize_hif4(candidate).to(torch.float32)
            - dense.to(torch.float32)
        ).reshape(rows, blocks, _BLOCK)
        candidate_loss = torch.einsum(
            "rbi,bij,rbj->rb", candidate_error, h, candidate_error
        )
        allowed = selected & (candidates == int(offset)).any(dim=-1)
        better = allowed & torch.isfinite(candidate_loss) & (
            candidate_loss < best_loss
        )
        best_loss = torch.where(better, candidate_loss, best_loss)
        for key in best:
            condition = better.reshape(
                (rows, blocks) + (1,) * (best[key].ndim - 2)
            )
            best[key] = torch.where(condition, candidate[key], best[key])
    if deployment_gram is not None:
        best = _select_activation_by_deployment_gram(
            dense, parent, best, deployment_gram
        )
        q_best = _dequantize_hif4(best).to(torch.float32)
        full_gram = deployment_gram.to(
            device=dense.device, dtype=torch.float32
        )
        parent_error = q_parent - dense.to(torch.float32)
        best_error = q_best - dense.to(torch.float32)
        parent_full_loss = (parent_error.mm(full_gram) * parent_error).sum(
            dim=1
        )
        best_full_loss = (best_error.mm(full_gram) * best_error).sum(dim=1)
        gain = float((parent_full_loss - best_full_loss).sum().item())
    else:
        gain = float((parent_loss - best_loss).sum().item())
    if return_gain:
        return best, gain
    return best


@torch.no_grad()
def _select_activation_by_deployment_gram(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    deployment_gram: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Keep a final-Gram proposal only when its exact deployed-product loss wins.

    ``deployment_gram`` is formed from the decoded weight that is actually
    returned by the offline path.  The comparison is row-wise, so a proposal
    can improve only the rows where the full (including cross-block) quadratic
    loss decreases; every other row remains the established parent.
    """

    if deployment_gram is None or dense.ndim != 2:
        return parent
    rows, channels = map(int, dense.shape)
    gram = deployment_gram.to(device=dense.device, dtype=torch.float32)
    if tuple(gram.shape) != (channels, channels):
        return parent
    q_parent = _dequantize_hif4(parent).to(torch.float32)
    q_candidate = _dequantize_hif4(candidate).to(torch.float32)
    if tuple(q_parent.shape) != (rows, channels) or tuple(q_candidate.shape) != (
        rows,
        channels,
    ):
        return parent
    parent_error = q_parent - dense.to(torch.float32)
    candidate_error = q_candidate - dense.to(torch.float32)
    parent_loss = (parent_error.mm(gram) * parent_error).sum(dim=1)
    candidate_loss = (candidate_error.mm(gram) * candidate_error).sum(dim=1)
    proposal = (q_candidate - q_parent).abs().amax(dim=1) > _EPS
    keep = proposal & torch.isfinite(candidate_loss) & (
        candidate_loss <= parent_loss + _EPS
    )
    result: dict[str, torch.Tensor] = {}
    for key, candidate_value in candidate.items():
        mask = keep.reshape(rows, *([1] * (candidate_value.ndim - 1)))
        result[key] = torch.where(mask, candidate_value, parent[key])
    return result


def _cpu_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32).contiguous()


# ---------------------------------------------------------------------------
# Linear API
# ---------------------------------------------------------------------------


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale)
    calibration = [
        _dequantize_nvfp4_float32(pair[0], pair[1]).to(weight.device)
        for pair in calib_activation_list
    ]
    balance, seed, block_size = _choose_boat(weight, calibration)
    balance = _choose_expansive_cat_balance(
        weight, calibration, balance, seed, block_size
    )
    permutation = _choose_l5a_permutation(
        weight, calibration, balance, seed, block_size
    )
    weight_t = _apply_boat_rotation(
        weight * balance.reshape(1, -1),
        seed,
        block_size,
        permutation=permutation,
    )
    activation_t = [
        _apply_boat_rotation(
            sample / balance.reshape(1, -1),
            seed,
            block_size,
            permutation=permutation,
        )
        for sample in calibration
    ]
    weight_params = _dense_to_hif4(weight_t, offsets=_BASE_OFFSETS)
    weight_params = _crossfold_weight_hsdq(weight_t, weight_params, activation_t)

    gram_state = None
    deployment_gram64_state = None
    deployment_gram_state = None
    global_lrh_state = None
    structured_lrh_state = None
    final_gram_route = False
    gals_final_enabled = False
    if int(weight_t.shape[1]) <= _ACT_GRAM_MAX_CHANNELS:
        gram_float = _gram64(weight_t)
        gram_state = _cpu_tensor(gram_float)
        # The L3 gate must use the actual deployed weight, not the dense
        # calibration weight.  The first precision experiment intentionally
        # materializes this matrix for every Gram-enabled shape; later timing
        # work may route it to only the small exact-gate shapes.
        deployed_weight = _dequantize_hif4(weight_params).to(torch.float32)
        deployment_gram = deployed_weight.t().mm(deployed_weight)
        deployment_gram_state = _cpu_tensor(deployment_gram)
        if (
            int(weight_t.shape[1]) <= _L4_FINAL_GRAM_MAX_CHANNELS
            and int(weight_t.shape[0]) > int(weight_t.shape[1])
        ):
            final_gram_route = True
            deployment_gram64_state = _cpu_tensor(_gram64(deployed_weight))
        channels = int(weight_t.shape[1])
        if channels <= _ACT_GLOBAL_LRH_WIDE_MAX_CHANNELS:
            deployed_gram64 = _gram64(deployed_weight)
            global_rank = (
                _ACT_GLOBAL_LRH_RANK
                if channels <= _ACT_GLOBAL_LRH_MAX_CHANNELS
                else _ACT_GLOBAL_LRH_WIDE_RANK
            )
            global_lrh = _global_activation_lrh(
                deployed_weight,
                deployed_gram64,
                rank=global_rank,
                max_channels=_ACT_GLOBAL_LRH_WIDE_MAX_CHANNELS,
            )
            if global_lrh is not None:
                global_lrh_state = _cpu_tensor(
                    global_lrh * math.sqrt(_ACT_GLOBAL_LRH_MIX)
                )
            if (
                channels > _ACT_GLOBAL_LRH_MAX_CHANNELS
                and deployment_gram_state is not None
            ):
                structured = _structured_activation_lrh(deployment_gram)
                if structured is not None:
                    structured_lrh_state = {
                        "kernels": _cpu_tensor(structured[0]),
                        "coefficients": _cpu_tensor(structured[1]),
                    }
            if (
                _L4_GALS_FINAL_ENABLED
                and final_gram_route
                and deployment_gram64_state is not None
            ):
                probe_gram = deployment_gram64_state.to(weight_t.device)
                probe_gains: list[float] = []
                for sample in activation_t:
                    probe_params = _dense_to_hif4(
                        sample, offsets=_BASE_OFFSETS, gram64=probe_gram
                    )
                    probe_full_gram = (
                        deployment_gram_state.to(weight_t.device)
                        if deployment_gram_state is not None
                        else None
                    )
                    _, probe_gain = _refine_activation_gals_final(
                        sample,
                        probe_params,
                        probe_gram,
                        max_blocks=4,
                        deployment_gram=probe_full_gram,
                        return_gain=True,
                    )
                    probe_gains.append(float(probe_gain))
                gals_final_enabled = bool(
                    len(probe_gains) >= 2
                    and all(gain > _EPS for gain in probe_gains)
                )
    state: dict[str, Any] = {
        "smooth_inv": _cpu_tensor(balance.reciprocal()),
        "permutation": (
            permutation.detach().to(device="cpu", dtype=torch.int32)
            if permutation is not None
            else None
        ),
        "block_smooth_size": int(block_size),
        "block_smooth_seed": int(seed),
        "gram64": gram_state,
        "deployment_gram64": deployment_gram64_state,
        "final_gram_route": final_gram_route,
        "gals_final": gals_final_enabled,
        "deployment_gram": deployment_gram_state,
        "global_lrh": global_lrh_state,
        "structured_lrh": structured_lrh_state,
        "version": 1,
    }
    return {"weight_params": weight_params, "activation_state": state}


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(activation_quant, activation_scale)
    state = activation_state if isinstance(activation_state, dict) else {}
    smooth_inv = state.get("smooth_inv")
    if torch.is_tensor(smooth_inv):
        dense = dense * smooth_inv.to(dense.device).reshape(1, -1)
    permutation = state.get("permutation")
    seed = int(state.get("block_smooth_seed", -1))
    block_size = int(state.get("block_smooth_size", 0))
    if block_size > 0 or permutation is not None:
        dense = _apply_boat_rotation(
            dense,
            seed,
            block_size,
            permutation=permutation,
        )
    gram = state.get("gram64")
    gram_tensor = gram if torch.is_tensor(gram) else None
    if gram_tensor is not None:
        gram_tensor = gram_tensor.to(dense.device)
    deployment_gram64 = state.get("deployment_gram64")
    deployment_gram64_tensor = (
        deployment_gram64.to(dense.device)
        if torch.is_tensor(deployment_gram64)
        else None
    )
    final_gram_route = bool(state.get("final_gram_route", False))
    deployment_gram = state.get("deployment_gram")
    deployment_gram_tensor = (
        deployment_gram.to(dense.device)
        if torch.is_tensor(deployment_gram)
        else None
    )
    global_lrh = state.get("global_lrh")
    global_lrh_tensor = (
        global_lrh.to(dense.device) if torch.is_tensor(global_lrh) else None
    )
    structured_lrh = state.get("structured_lrh")
    if final_gram_route and deployment_gram64_tensor is not None:
        # Keep the established v107 proposal as the parent, then evaluate a
        # final-deployment-Gram proposal against the exact full deployed Gram.
        # This isolates L4a and prevents a block-diagonal surrogate from
        # regressing a row that the true product objective would reject.
        parent_params = _dense_to_hif4(
            dense, offsets=_BASE_OFFSETS, gram64=gram_tensor
        )
        parent_params = _refine_activation(dense, parent_params, gram_tensor)
        parent_params = _refine_activation_hierarchy_g64(
            dense, parent_params, gram_tensor
        )
        parent_refined = _refine_activation_global_lrh(
            dense,
            parent_params,
            gram_tensor,
            deployment_gram_tensor,
            global_lrh_tensor,
        )
        parent_params = (
            parent_refined[0]
            if isinstance(parent_refined, tuple)
            else parent_refined
        )
        final_params = _dense_to_hif4(
            dense, offsets=_BASE_OFFSETS, gram64=deployment_gram64_tensor
        )
        final_params = _refine_activation(
            dense, final_params, deployment_gram64_tensor
        )
        final_params = _refine_activation_hierarchy_g64(
            dense, final_params, deployment_gram64_tensor
        )
        final_refined = _refine_activation_global_lrh(
            dense,
            final_params,
            deployment_gram64_tensor,
            deployment_gram_tensor,
            global_lrh_tensor,
        )
        final_params = (
            final_refined[0]
            if isinstance(final_refined, tuple)
            else final_refined
        )
        params = _select_activation_by_deployment_gram(
            dense, parent_params, final_params, deployment_gram_tensor
        )
        params = _refine_activation_structured(
            dense,
            params,
            deployment_gram64_tensor,
            deployment_gram_tensor,
            structured_lrh if isinstance(structured_lrh, dict) else None,
        )
    else:
        params = _dense_to_hif4(
            dense, offsets=_BASE_OFFSETS, gram64=gram_tensor
        )
        params = _refine_activation(dense, params, gram_tensor)
        params = _refine_activation_hierarchy_g64(dense, params, gram_tensor)
        refined = _refine_activation_global_lrh(
            dense,
            params,
            gram_tensor,
            deployment_gram_tensor,
            global_lrh_tensor,
        )
        params = refined[0] if isinstance(refined, tuple) else refined
        params = _refine_activation_structured(
            dense,
            params,
            gram_tensor,
            deployment_gram_tensor,
            structured_lrh if isinstance(structured_lrh, dict) else None,
        )
    if (
        _L4_GALS_FINAL_ENABLED
        and bool(state.get("gals_final", False))
        and final_gram_route
        and deployment_gram64_tensor is not None
    ):
        params = _refine_activation_gals_final(
            dense,
            params,
            deployment_gram64_tensor,
            max_blocks=4,
            deployment_gram=deployment_gram_tensor,
        )
    return params


# ---------------------------------------------------------------------------
# Attention API
# ---------------------------------------------------------------------------


def _head_rotation_signs(
    kv_heads: int,
    head_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    return _rotation_signs(kv_heads * head_dim, seed, device, torch.float32).reshape(
        kv_heads, head_dim
    )


def _apply_head_transform(
    dense: torch.Tensor,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    multiplier: torch.Tensor | None,
    signs: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    x = dense.to(torch.float32).reshape(-1, num_heads, head_dim)
    group_ids = torch.div(
        torch.arange(num_heads, device=x.device) * kv_heads,
        num_heads,
        rounding_mode="floor",
    )
    if multiplier is not None:
        m = multiplier.to(x.device).index_select(0, group_ids)
        x = x * m.unsqueeze(0)
    if signs is not None and block_size > 0:
        s = signs.to(x.device).index_select(0, group_ids)
        x = _fwht_blocks((x * s.unsqueeze(0)).reshape(-1, head_dim), block_size).reshape_as(x)
    return x.reshape(-1, num_heads * head_dim)


def _apply_head_mixing(
    dense: torch.Tensor,
    num_heads: int,
    kv_heads: int,
    mixing: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a shared orthogonal block matrix in every GQA group."""
    if mixing is None or dense.ndim != 2:
        return dense
    if mixing.ndim != 4 or int(mixing.shape[0]) != kv_heads:
        return dense
    width = int(mixing.shape[-1])
    head_dim = int(dense.shape[-1]) // num_heads
    if width <= 0 or head_dim % width != 0 or tuple(mixing.shape[1:]) != (
        head_dim // width,
        width,
        width,
    ):
        return dense
    group = num_heads // kv_heads
    x = dense.to(torch.float32).reshape(-1, num_heads, head_dim)
    group_ids = torch.div(
        torch.arange(num_heads, device=x.device) * kv_heads,
        num_heads,
        rounding_mode="floor",
    )
    matrices = mixing.to(device=x.device, dtype=x.dtype).index_select(0, group_ids)
    blocks = x.reshape(-1, num_heads, head_dim // width, width)
    # Row-vector convention: each block is multiplied by M, hence Q and K
    # receive the same M when M is orthogonal and QK^T is invariant.
    blocks = torch.einsum("hbji,thbj->thbi", matrices, blocks)
    del group
    return blocks.reshape_as(x).reshape_as(dense)


def _gqrb_candidates(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    q_num_heads: int,
    kv_heads: int,
    head_dim: int,
) -> list[torch.Tensor | None]:
    """Construct small GQRB block-mixing candidates.

    Every matrix is orthogonal.  The covariance candidate uses the local
    eigensystem of Q/K second moments; fixed rotations provide deterministic
    alternatives when the eigensystem is nearly degenerate.
    """
    candidates: list[torch.Tensor | None] = [None]
    group = q_num_heads // kv_heads
    q = torch.cat(q_samples).reshape(-1, q_num_heads, head_dim)
    k = torch.cat(k_samples).reshape(-1, kv_heads, head_dim)
    for width in _ATTN_GQRB_WIDTHS:
        if width > head_dim or head_dim % width != 0:
            continue
        blocks = head_dim // width
        eye = torch.eye(width, device=q.device, dtype=torch.float32)
        hadamard = eye.clone()
        if width == 2:
            hadamard = torch.as_tensor(
                [[1.0, 1.0], [1.0, -1.0]], device=q.device
            ) / math.sqrt(2.0)
        elif width == 4:
            hadamard = torch.as_tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0, 1.0],
                ],
                device=q.device,
            ) / 2.0
        for angle in _ATTN_GQRB_ANGLES:
            rotation = eye.clone()
            if width == 2:
                c = math.cos(float(angle))
                s = math.sin(float(angle))
                rotation = torch.as_tensor(
                    [[c, s], [-s, c]], device=q.device, dtype=torch.float32
                )
            matrices = rotation.reshape(1, 1, width, width).expand(
                kv_heads, blocks, width, width
            ).clone()
            candidates.append(matrices)
        covariance = torch.empty(kv_heads, blocks, width, width, device=q.device)
        for kv_head in range(kv_heads):
            q_group = q[:, kv_head * group : (kv_head + 1) * group].reshape(-1, head_dim)
            k_head = k[:, kv_head]
            for block in range(blocks):
                lo = block * width
                hi = lo + width
                q_block = q_group[:, lo:hi]
                k_block = k_head[:, lo:hi]
                cov = q_block.t().mm(q_block) + k_block.t().mm(k_block)
                _, vectors = torch.linalg.eigh(cov + _EPS * eye)
                covariance[kv_head, block] = vectors
        candidates.append(covariance)
        # The fixed Hadamard is useful even when covariance eigenspaces are
        # unstable; it remains a legal orthogonal coordinate change.
        candidates.append(
            hadamard.reshape(1, 1, width, width).expand(
                kv_heads, blocks, width, width
            ).clone()
        )
    return candidates


def _attention_pair(item: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    pair = item[name]
    return pair[0], pair[1]


def _attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    tokens = int(q.shape[0])
    group = q_heads // kv_heads
    qh = q.reshape(tokens, q_heads, head_dim).transpose(0, 1)
    kh = k.reshape(tokens, kv_heads, head_dim).transpose(0, 1).repeat_interleave(
        group, dim=0
    )
    vh = v.reshape(tokens, kv_heads, head_dim).transpose(0, 1).repeat_interleave(
        group, dim=0
    )
    probability = torch.softmax(
        (qh @ kh.transpose(-1, -2)) / math.sqrt(float(head_dim)), dim=-1
    )
    return probability.bmm(vh).transpose(0, 1).reshape(tokens, q_heads * head_dim)


def _normalize_importance(value: torch.Tensor) -> torch.Tensor:
    result = value.to(torch.float32).clamp_min(_EPS)
    return (result / result.mean().clamp_min(_EPS)).clamp(0.05, 20.0)


def _qk_importance(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group = q_heads // kv_heads
    q_second = torch.cat(q_samples).reshape(-1, kv_heads, group, head_dim).square()
    k_second = torch.cat(k_samples).reshape(-1, kv_heads, head_dim).square()
    q_importance = k_second.mean(dim=0).repeat_interleave(group, dim=0).reshape(-1)
    k_importance = q_second.mean(dim=(0, 2)).reshape(-1)
    return _normalize_importance(q_importance), _normalize_importance(k_importance)


def _qk_gram64(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if head_dim % _BLOCK != 0:
        return None, None
    group = q_heads // kv_heads
    q = torch.cat(q_samples).reshape(-1, q_heads, head_dim)
    k = torch.cat(k_samples).reshape(-1, kv_heads, head_dim)
    q_grams: list[torch.Tensor] = []
    k_grams: list[torch.Tensor] = []
    for kv_head in range(kv_heads):
        q_group = q[:, kv_head * group : (kv_head + 1) * group]
        for local_head in range(group):
            for start in range(0, head_dim, _BLOCK):
                opposite = k[:, kv_head, start : start + _BLOCK]
                q_grams.append(opposite.t().mm(opposite))
        for start in range(0, head_dim, _BLOCK):
            opposite = q_group[:, :, start : start + _BLOCK].reshape(-1, _BLOCK)
            k_grams.append(opposite.t().mm(opposite))
    return torch.stack(q_grams), torch.stack(k_grams)


def _attention_candidate_score(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    v_hats: Sequence[torch.Tensor],
    references: Sequence[torch.Tensor],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    multiplier: torch.Tensor,
    signs: torch.Tensor | None,
    block_size: int,
    center: bool,
    refine: bool = False,
    mixing: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    transformed_q: list[torch.Tensor] = []
    transformed_k: list[torch.Tensor] = []
    for q, k in zip(q_samples, k_samples):
        transformed_q_item = _apply_head_transform(
            q, q_heads, kv_heads, head_dim, multiplier, signs, block_size
        )
        transformed_q.append(
            _apply_head_mixing(transformed_q_item, q_heads, kv_heads, mixing)
        )
        if center:
            kv = k.reshape(-1, kv_heads, head_dim)
            k = (kv - kv.mean(dim=0, keepdim=True)).reshape_as(k)
        transformed_k_item = _apply_head_transform(
            k, kv_heads, kv_heads, head_dim, multiplier.reciprocal(),
            signs, block_size,
        )
        transformed_k.append(
            _apply_head_mixing(transformed_k_item, kv_heads, kv_heads, mixing)
        )
    q_importance, k_importance = _qk_importance(
        transformed_q, transformed_k, q_heads, kv_heads, head_dim
    )
    q_gram = k_gram = None
    if refine:
        q_gram, k_gram = _qk_gram64(
            transformed_q, transformed_k, q_heads, kv_heads, head_dim
        )
    losses: list[float] = []
    for q, k, v_hat, reference in zip(
        transformed_q, transformed_k, v_hats, references
    ):
        q_params = _dense_to_hif4(
            q, offsets=_ATTN_OFFSETS, importance=q_importance,
            gram64=q_gram if refine else None,
        )
        k_params = _dense_to_hif4(
            k, offsets=_ATTN_OFFSETS, importance=k_importance,
            gram64=k_gram if refine else None,
        )
        if refine:
            q_params = _refine_activation(
                q, q_params, q_gram,
                max_blocks=max(1, int(q.shape[-1]) // _BLOCK), sweeps=3,
            )
            k_params = _refine_activation(
                k, k_params, k_gram,
                max_blocks=max(1, int(k.shape[-1]) // _BLOCK), sweeps=3,
            )
        q_hat = _dequantize_hif4(q_params).to(torch.float32)
        k_hat = _dequantize_hif4(k_params).to(torch.float32)
        output = _attention_forward(
            q_hat, k_hat, v_hat, q_heads, kv_heads, head_dim
        )
        losses.append(float((output - reference).square().mean()))
    mean = sum(losses) / len(losses)
    robust = mean + 0.25 * max(losses)
    return robust, q_importance, k_importance


def _attention_probability(
    q: torch.Tensor,
    k: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return non-causal attention probabilities for one calibration sample."""
    tokens = int(q.shape[0])
    group = q_heads // kv_heads
    qh = q.reshape(tokens, q_heads, head_dim).transpose(0, 1)
    kh = k.reshape(tokens, kv_heads, head_dim).transpose(0, 1).repeat_interleave(
        group, dim=0
    )
    return torch.softmax(
        (qh @ kh.transpose(-1, -2)) / math.sqrt(float(head_dim)), dim=-1
    )


def _build_pawv_metric(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Build length-keyed diagonals of the per-sample ``P.T @ P`` metric."""
    if not q_samples or not k_samples:
        return {}
    diagonal_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for q, k in zip(q_samples, k_samples):
        probability = _attention_probability(q, k, q_heads, kv_heads, head_dim)
        diagonal = probability.square().sum(dim=1).mean(dim=0).clamp_min(_EPS)
        key = str(int(k.shape[0]))
        if key in diagonal_sums:
            diagonal_sums[key] = diagonal_sums[key] + diagonal
            counts[key] += 1
        else:
            diagonal_sums[key] = diagonal
            counts[key] = 1
    return {
        key: diagonal_sum / float(counts[key])
        for key, diagonal_sum in diagonal_sums.items()
    }
def _refine_v(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    row_diagonal: torch.Tensor | None,
    row_lowrank: torch.Tensor | None,
    *,
    sweeps: int = _ATTN_PAWV_SWEEPS,
) -> dict[str, torch.Tensor]:
    """PAWV legal coordinate refinement under a token-row Hessian."""
    if row_diagonal is None or dense.ndim != 2:
        return parent
    rows, channels = map(int, dense.shape)
    if int(row_diagonal.numel()) != rows or channels % _BLOCK != 0:
        return parent
    diagonal = row_diagonal.to(device=dense.device, dtype=torch.float32).clamp_min(_EPS)
    lowrank = None
    if row_lowrank is not None:
        candidate = row_lowrank.to(device=dense.device, dtype=torch.float32)
        if candidate.ndim == 2 and int(candidate.shape[0]) == rows:
            lowrank = candidate
    q = _dequantize_hif4(parent).to(torch.float32).clone()
    x = dense.to(torch.float32)
    den = _denominator(parent).reshape(rows, channels)
    codes = torch.round(q * 4.0 / den.clamp_min(_EPS)).clamp(-7.0, 7.0)
    levels = torch.as_tensor(_SIGNED_LEVELS, device=dense.device)
    for _ in range(max(1, int(sweeps))):
        for block in range(channels // _BLOCK):
            lo = block * _BLOCK
            hi = lo + _BLOCK
            error = q[:, lo:hi] - x[:, lo:hi]
            for coordinate in range(_BLOCK):
                column_error = error[:, coordinate]
                gradient = diagonal * column_error
                if lowrank is not None:
                    gradient = gradient + lowrank.mv(lowrank.t().mv(column_error))
                curvature = diagonal
                if lowrank is not None:
                    curvature = curvature + lowrank.square().sum(dim=1)
                current = q[:, lo + coordinate]
                options = den[:, lo + coordinate, None] * levels[None, :]
                step = options - current[:, None]
                change = (
                    2.0 * step * gradient[:, None]
                    + curvature[:, None] * step.square()
                )
                best_change, best_index = change.min(dim=-1)
                accept = torch.isfinite(best_change) & (best_change < -_EPS)
                delta = step.gather(-1, best_index[:, None]).squeeze(-1)
                delta = torch.where(accept, delta, torch.zeros_like(delta))
                q[:, lo + coordinate] += delta
                error[:, coordinate] += delta
    codes = torch.round(q * 4.0 / den.clamp_min(_EPS)).clamp(-7.0, 7.0)
    return _write_codes(parent, codes)


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    q_samples = [
        _dequantize_nvfp4_float32(*_attention_pair(item, "q"))
        for item in calib_qkv_list
    ]
    k_samples = [
        _dequantize_nvfp4_float32(*_attention_pair(item, "k"))
        for item in calib_qkv_list
    ]
    if not q_samples or not k_samples:
        return {"q_state": {}, "k_state": {}, "v_state": {}}
    v_samples = [
        _dequantize_nvfp4_float32(*_attention_pair(item, "v"))
        for item in calib_qkv_list
    ]
    references = [
        _attention_forward(q, k, v, q_num_heads, kv_num_heads, head_dim)
        for q, k, v in zip(q_samples, k_samples, v_samples)
    ]
    row_diagonals = _build_pawv_metric(
        q_samples, k_samples, q_num_heads, kv_num_heads, head_dim
    )
    v_hats = []
    for v in v_samples:
        v_params = _dense_to_hif4(v, offsets=_ATTN_OFFSETS)
        v_params = _refine_v(
            v, v_params, row_diagonals.get(str(int(v.shape[0]))), None
        )
        v_hats.append(_dequantize_hif4(v_params).to(torch.float32))
    q_stack = torch.cat(q_samples).reshape(-1, q_num_heads, head_dim)
    k_stack = torch.cat(k_samples).reshape(-1, kv_num_heads, head_dim)
    group = q_num_heads // kv_num_heads
    q_rms = q_stack.square().mean(dim=0).reshape(
        kv_num_heads, group, head_dim
    ).mean(dim=1).add(_EPS).sqrt()
    k_rms = k_stack.square().mean(dim=0).add(_EPS).sqrt()
    ratio = (k_rms / q_rms).clamp(1.0 / 16.0, 16.0)

    best_score = math.inf
    best_multiplier = torch.ones_like(ratio)
    best_center = False
    best_block = 0
    best_signs = None
    best_mixing = None
    best_q_importance = torch.ones(q_num_heads * head_dim, device=q_stack.device)
    best_k_importance = torch.ones(kv_num_heads * head_dim, device=q_stack.device)
    candidate_pool: list[
        tuple[float, torch.Tensor, torch.Tensor | None, int, bool, torch.Tensor | None]
    ] = []

    # Stage 1 searches the exact reciprocal diagonal invariance and K shift.
    for alpha in _ATTN_SMOOTH_ALPHAS:
        multiplier = ratio.pow(float(alpha)).clamp(0.25, 4.0)
        for center in (False, True):
            score, q_importance, k_importance = _attention_candidate_score(
                q_samples, k_samples, v_hats, references,
                q_num_heads, kv_num_heads, head_dim,
                multiplier, None, 0, center,
            )
            candidate_pool.append((score, multiplier.clone(), None, 0, center, None))
            if score < best_score:
                best_score = score
                best_multiplier = multiplier.clone()
                best_center = center
                best_q_importance = q_importance
                best_k_importance = k_importance

    # Stage 2 adds a shared signed orthogonal transform around that winner.
    for block_size in _ATTN_ROTATION_SIZES[1:]:
        if block_size > head_dim or head_dim % block_size != 0:
            continue
        for seed in _ATTN_ROTATION_SEEDS:
            signs = _head_rotation_signs(
                kv_num_heads, head_dim, seed, q_stack.device
            )
            score, q_importance, k_importance = _attention_candidate_score(
                q_samples, k_samples, v_hats, references,
                q_num_heads, kv_num_heads, head_dim,
                best_multiplier, signs, block_size, best_center,
            )
            candidate_pool.append(
                (score, best_multiplier.clone(), signs.clone(), int(block_size), best_center, None)
            )
            if score < best_score:
                best_score = score
                best_block = int(block_size)
                best_signs = signs.clone()
                best_q_importance = q_importance
                best_k_importance = k_importance

    # Preserve the exact four-entry no-mixing proxy shortlist used by the
    # validated parent before adding any GQRB entries.
    base_proxy_shortlist = sorted(candidate_pool, key=lambda item: item[0])[:4]

    # B1 GQRB: a small block-orthogonal Q/K mixing shortlist.  The matrices
    # are shared inside each GQA group so the QK dot product is preserved.
    for mixing in _gqrb_candidates(
        q_samples, k_samples, q_num_heads, kv_num_heads, head_dim
    ):
        score, q_importance, k_importance = _attention_candidate_score(
            q_samples, k_samples, v_hats, references,
            q_num_heads, kv_num_heads, head_dim,
            best_multiplier, None, 0, best_center, mixing=mixing,
        )
        candidate_pool.append(
            (
                score,
                best_multiplier.clone(),
                None,
                0,
                best_center,
                None if mixing is None else mixing.clone(),
            )
        )
        if score < best_score:
            best_score = score
            best_mixing = None if mixing is None else mixing.clone()
            best_q_importance = q_importance
            best_k_importance = k_importance

    # The proxy scan is cheap; only its three strongest candidates are
    # re-ranked through the exact deployed Gram-HSDQ path.
    best_base_score = math.inf
    best_gqrb_score = math.inf
    best_base_entry: tuple[Any, ...] | None = None
    best_gqrb_entry: tuple[Any, ...] | None = None
    proxy_sorted = sorted(
        [item for item in candidate_pool if item[5] is not None],
        key=lambda item: item[0],
    )
    shortlist = base_proxy_shortlist + proxy_sorted[:4]
    for _, multiplier, signs, block_size, center, mixing in shortlist:
        score, q_importance, k_importance = _attention_candidate_score(
            q_samples, k_samples, v_hats, references,
            q_num_heads, kv_num_heads, head_dim,
            multiplier, signs, block_size, center, refine=True,
            mixing=mixing,
        )
        entry = (
            score,
            multiplier,
            signs,
            int(block_size),
            bool(center),
            mixing,
            q_importance,
            k_importance,
        )
        if mixing is None and score < best_base_score:
            best_base_score = score
            best_base_entry = entry
        if mixing is not None and score < best_gqrb_score:
            best_gqrb_score = score
            best_gqrb_entry = entry

    if best_base_entry is None:
        raise RuntimeError("attention shortlist lost its parent candidate")
    selected_entry = best_base_entry
    if (
        best_gqrb_entry is not None
        and best_gqrb_score < best_base_score * (1.0 - _ATTN_GQRB_MIN_GAIN)
    ):
        selected_entry = best_gqrb_entry
    _, best_multiplier, best_signs, best_block, best_center, best_mixing, best_q_importance, best_k_importance = selected_entry
    best_multiplier = best_multiplier.clone()
    best_signs = None if best_signs is None else best_signs.clone()
    best_mixing = None if best_mixing is None else best_mixing.clone()

    final_q: list[torch.Tensor] = []
    final_k: list[torch.Tensor] = []
    for q, k in zip(q_samples, k_samples):
        final_q_item = _apply_head_transform(
            q, q_num_heads, kv_num_heads, head_dim,
            best_multiplier, best_signs, best_block,
        )
        final_q.append(
            _apply_head_mixing(final_q_item, q_num_heads, kv_num_heads, best_mixing)
        )
        if best_center:
            view = k.reshape(-1, kv_num_heads, head_dim)
            k = (view - view.mean(dim=0, keepdim=True)).reshape_as(k)
        final_k_item = _apply_head_transform(
            k, kv_num_heads, kv_num_heads, head_dim,
            best_multiplier.reciprocal(), best_signs, best_block,
        )
        final_k.append(
            _apply_head_mixing(final_k_item, kv_num_heads, kv_num_heads, best_mixing)
        )
    q_gram, k_gram = _qk_gram64(
        final_q, final_k, q_num_heads, kv_num_heads, head_dim
    )
    q_state: dict[str, Any] = {
        "multiplier": _cpu_tensor(best_multiplier),
        "rotation_block": best_block,
        "rotation_signs": None if best_signs is None else _cpu_tensor(best_signs),
        "mixing": None if best_mixing is None else _cpu_tensor(best_mixing),
        "kv_heads": int(kv_num_heads),
        "importance": _cpu_tensor(best_q_importance),
        "gram64": None if q_gram is None else _cpu_tensor(q_gram),
    }
    k_state: dict[str, Any] = {
        "multiplier": _cpu_tensor(best_multiplier.reciprocal()),
        "rotation_block": best_block,
        "rotation_signs": None if best_signs is None else _cpu_tensor(best_signs),
        "mixing": None if best_mixing is None else _cpu_tensor(best_mixing),
        "kv_heads": int(kv_num_heads),
        "importance": _cpu_tensor(best_k_importance),
        "gram64": None if k_gram is None else _cpu_tensor(k_gram),
        "center": best_center,
    }
    return {
        "q_state": q_state,
        "k_state": k_state,
        "v_state": {
            "row_diagonals": {
                key: _cpu_tensor(diagonal)
                for key, diagonal in row_diagonals.items()
            },
            "row_lowrank": None,
        },
    }


def _dynamic_attention_operand(
    quant: torch.Tensor,
    scale: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: Any,
    *,
    center: bool,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(quant, scale)
    cfg = state if isinstance(state, dict) else {}
    kv_heads = int(cfg.get("kv_heads", num_heads))
    if center and bool(cfg.get("center", False)):
        view = dense.reshape(-1, num_heads, head_dim)
        dense = (view - view.mean(dim=0, keepdim=True)).reshape_as(dense)
    multiplier = cfg.get("multiplier")
    signs = cfg.get("rotation_signs")
    dense = _apply_head_transform(
        dense,
        num_heads,
        kv_heads,
        head_dim,
        multiplier if torch.is_tensor(multiplier) else None,
        signs if torch.is_tensor(signs) else None,
        int(cfg.get("rotation_block", 0)),
    )
    mixing = cfg.get("mixing")
    dense = _apply_head_mixing(
        dense,
        num_heads,
        kv_heads,
        mixing if torch.is_tensor(mixing) else None,
    )
    importance = cfg.get("importance")
    gram = cfg.get("gram64")
    gram_tensor = gram.to(dense.device) if torch.is_tensor(gram) else None
    params = _dense_to_hif4(
        dense,
        offsets=_ATTN_OFFSETS,
        importance=(importance.to(dense.device) if torch.is_tensor(importance) else None),
        gram64=gram_tensor,
    )
    return _refine_activation(
        dense,
        params,
        gram_tensor,
        max_blocks=max(1, int(dense.shape[-1]) // _BLOCK),
        sweeps=3,
    )


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention_operand(
        q_quant, q_scale, q_num_heads, head_dim, q_state, center=False
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention_operand(
        k_quant, k_scale, kv_num_heads, head_dim, k_state, center=True
    )


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(v_quant, v_scale)
    state = v_state if isinstance(v_state, dict) else {}
    row_diagonals = state.get("row_diagonals")
    row_diagonal = (
        row_diagonals.get(str(int(dense.shape[0])))
        if isinstance(row_diagonals, dict)
        else state.get("row_diagonal")
    )
    row_lowrank = state.get("row_lowrank")
    params = _dense_to_hif4(dense, offsets=_ATTN_OFFSETS)
    return _refine_v(
        dense,
        params,
        row_diagonal if torch.is_tensor(row_diagonal) else None,
        row_lowrank if torch.is_tensor(row_lowrank) else None,
    )


__all__ = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
