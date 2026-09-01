"""Clean HiF4 entry: BOAT + cross-fold HSDQ/LRH.

The file intentionally contains only the six competition APIs and the small
set of primitives needed by the active algorithm.  Historical C1--C88
experiments remain recoverable from Git and ``solutions/``; rejected flags and
dormant research branches do not live in this submission file.

Linear calibration has three stages:

1. BOAT selects an invertible diagonal + signed-Hadamard input transform from
   operand-local quantization errors.  It never constructs a Linear output.
2. The transformed weight is quantized to the legal HiF4 hierarchy.
3. Cross-fold HSDQ uses exact low-rank Hessians ``A.T @ A`` to polish Q(W).
   Products are calibration-local and can only change ``weight_params``.

Online Q(A) uses the frozen BOAT transform and a bounded block-Hessian HSDQ
whose state contains only static transformed-weight Gram blocks.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Union

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
_ROAB_PAIR_RIDGE = 1.0e-6
_ROAB_MAX_SINGULAR = 4.0
_ROAB_MIN_SINGULAR = 0.25
_ROAB_MAX_SCORE_ROWS = 128

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
_OUTPUT_WEIGHT_HSDQ_BLOCKS = 4
_OUTPUT_WEIGHT_HSDQ_SWEEPS = 1
_OUTPUT_WEIGHT_HSDQ_MIN_GAIN = 1.0e-5

_ACT_HSDQ_BLOCKS = 128
_ACT_HSDQ_SWEEPS = 2
_ACT_GRAM_MAX_CHANNELS = 8192

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


def _apply_boat_rotation(
    x: torch.Tensor, seed: int, block_size: int = _BLOCK
) -> torch.Tensor:
    if int(seed) < 0 or int(block_size) <= 0:
        return x
    signs = _rotation_signs(int(x.shape[-1]), seed, x.device, x.dtype)
    return _fwht_blocks(x * signs, int(block_size))


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
    # v152 role control: retain BOAT but remove the expansive CAT balance.
    # This is a single-mechanism ablation motivated by the external fc
    # attribution; non-expansive shapes already bypass this function.
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


def _pair_transform(
    tensor: torch.Tensor,
    matrices: torch.Tensor,
) -> torch.Tensor:
    """Apply a row-vector block-diagonal 2x2 transform."""

    if tensor.ndim < 1 or matrices.ndim != 3 or int(tensor.shape[-1]) % 2:
        return tensor
    channels = int(tensor.shape[-1])
    pairs = channels // 2
    if tuple(matrices.shape) != (pairs, 2, 2):
        return tensor
    blocks = tensor.to(torch.float32).reshape(-1, pairs, 2)
    result = torch.einsum(
        "npi,pij->npj",
        blocks,
        matrices.to(device=blocks.device, dtype=blocks.dtype),
    )
    return result.reshape_as(tensor)


def _spd_sqrt_2x2(
    matrices: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Batched square root/inverse square root for symmetric 2x2 matrices."""

    sym = 0.5 * (matrices + matrices.transpose(-1, -2))
    eye = torch.eye(2, device=sym.device, dtype=sym.dtype).expand_as(sym)
    values, vectors = torch.linalg.eigh(sym + _ROAB_PAIR_RIDGE * eye)
    values = values.clamp_min(_ROAB_PAIR_RIDGE)
    if inverse:
        values = values.rsqrt()
    else:
        values = values.sqrt()
    return (vectors * values.unsqueeze(-2)).matmul(vectors.transpose(-1, -2))


def _learn_roab_pairs(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
) -> torch.Tensor | None:
    """Learn reciprocal 2x2 output-balanced transforms from calibration moments.

    If ``U`` is returned, activation uses ``A @ U`` and the static weight uses
    ``W @ U^{-T}``; the product is therefore unchanged before quantization.  A
    geometric-mean balancing makes the two output-error metrics equal within
    each pair, then an eigensystem rotates the common metric toward a diagonal
    form.  Only calibration-time moments are used here.
    """

    if not calibration or weight.ndim != 2 or int(weight.shape[1]) % 2:
        return None
    channels = int(weight.shape[1])
    pairs = channels // 2
    try:
        activation = torch.cat(
            [_sample_rows(item, 128).to(torch.float32) for item in calibration],
            dim=0,
        )
        weight_sample = _sample_rows(weight, 256).to(torch.float32)
        if activation.ndim != 2 or int(activation.shape[1]) != channels:
            return None
        a_blocks = activation.reshape(-1, pairs, 2)
        w_blocks = weight_sample.reshape(-1, pairs, 2)
        ga = torch.einsum("npi,npj->pij", a_blocks, a_blocks)
        gw = torch.einsum("npi,npj->pij", w_blocks, w_blocks)
        ga = ga / max(1, int(activation.shape[0]))
        gw = gw / max(1, int(weight_sample.shape[0]))
        a_sqrt = _spd_sqrt_2x2(ga)
        a_inv_sqrt = _spd_sqrt_2x2(ga, inverse=True)
        middle = a_sqrt.matmul(gw).matmul(a_sqrt)
        middle_sqrt = _spd_sqrt_2x2(middle)
        balance = a_inv_sqrt.matmul(middle_sqrt).matmul(a_inv_sqrt)
        balance_sqrt = _spd_sqrt_2x2(balance)
        common = balance_sqrt.matmul(ga).matmul(balance_sqrt)
        _, vectors = torch.linalg.eigh(
            0.5 * (common + common.transpose(-1, -2))
            + _ROAB_PAIR_RIDGE
            * torch.eye(2, device=common.device, dtype=common.dtype).expand_as(common)
        )
        matrices = balance_sqrt.matmul(vectors)
        # Keep the reciprocal state numerically well-conditioned.  Singular
        # values are clipped symmetrically; the inverse-transpose is formed
        # from this final matrix, so the product identity remains exact.
        left, singular, right_t = torch.linalg.svd(matrices)
        singular = singular.clamp(_ROAB_MIN_SINGULAR, _ROAB_MAX_SINGULAR)
        matrices = left.matmul(torch.diag_embed(singular)).matmul(right_t)
        if not bool(torch.isfinite(matrices).all()):
            return None
        return matrices.contiguous()
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def _roab_output_score(
    weight: torch.Tensor,
    weight_t: torch.Tensor,
    activation_t: Sequence[torch.Tensor],
) -> float:
    """Score a transform using legal plain HiF4 output reconstruction."""

    try:
        weight_raw = _sample_rows(weight, _ROAB_MAX_SCORE_ROWS).to(torch.float32)
        transformed_weight = _sample_rows(weight_t, _ROAB_MAX_SCORE_ROWS).to(torch.float32)
        weight_params = _dense_to_hif4(transformed_weight, offsets=_BASE_OFFSETS)
        quantized_weight = _dequantize_hif4(weight_params).to(torch.float32)
        losses: list[float] = []
        for transformed in activation_t[:2]:
            sample = _sample_rows(transformed, _ROAB_MAX_SCORE_ROWS).to(torch.float32)
            params = _dense_to_hif4(sample, offsets=_BASE_OFFSETS)
            quantized = _dequantize_hif4(params).to(torch.float32)
            target = sample.mm(transformed_weight.t())
            predicted = quantized.mm(quantized_weight.t())
            losses.append(
                float(
                    (predicted - target).square().mean()
                    / (target.square().mean() + _EPS)
                )
            )
        if not losses:
            return math.inf
        return sum(losses) / len(losses) + 0.25 * max(losses)
    except (RuntimeError, ValueError, FloatingPointError):
        return math.inf


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


def _output_product_loss(
    raw_activation: torch.Tensor,
    deployed_activation: torch.Tensor,
    weight: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> float:
    """Relative output residual for the actually deployed activation.

    ``raw_activation`` is the teacher-side calibration tensor while
    ``deployed_activation`` is the legal HiF4 tensor that the online
    activation API will emit.  Keeping the two separate is important: the
    constant ``E_A W`` term cannot be optimized away when the weight codes
    are selected.
    """
    q = _dequantize_hif4(params).to(torch.float32)
    raw = _sample_rows(raw_activation, _WEIGHT_HSDQ_MAX_ROWS).to(torch.float32)
    deployed = _sample_rows(
        deployed_activation, _WEIGHT_HSDQ_MAX_ROWS
    ).to(torch.float32)
    if raw.ndim != 2 or deployed.shape != raw.shape:
        return math.inf
    target = raw.mm(weight.t())
    residual = target - deployed.mm(q.t())
    return float(
        residual.square().mean() / (target.square().mean() + _EPS)
    )


def _polish_weight_output(
    weight: torch.Tensor,
    parent: dict[str, torch.Tensor],
    raw_activation: torch.Tensor,
    deployed_activation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Output-supervised legal coordinate refinement for a weight tensor.

    The residual is initialized against ``raw_activation @ weight.T`` and
    updated incrementally as each HiF4 coordinate changes.  Thus every
    accepted code is judged in the real ``Q(A) @ Q(W).T`` output space rather
    than by an operand-local weight MSE.
    """
    rows, channels = map(int, weight.shape)
    if channels < _WEIGHT_HSDQ_MIN_CHANNELS or channels % _BLOCK != 0:
        return parent
    raw = _sample_rows(raw_activation, _WEIGHT_HSDQ_MAX_ROWS).to(torch.float32)
    deployed = _sample_rows(
        deployed_activation, _WEIGHT_HSDQ_MAX_ROWS
    ).to(torch.float32)
    if (
        raw.ndim != 2
        or deployed.shape != raw.shape
        or int(raw.shape[1]) != channels
    ):
        return parent

    blocks = channels // _BLOCK
    q = _dequantize_hif4(parent).to(torch.float32).clone()
    den = _denominator(parent).reshape(rows, blocks, _BLOCK)
    codes = torch.round(
        q.reshape(rows, blocks, _BLOCK) * 4.0 / den.clamp_min(_EPS)
    ).clamp(-7.0, 7.0)
    residual = raw.mm(weight.t()) - deployed.mm(q.t())

    leverage = []
    for block in range(blocks):
        lo = block * _BLOCK
        hi = lo + _BLOCK
        leverage.append(
            deployed[:, lo:hi].t().mm(residual).square().sum()
        )
    count = min(blocks, max(1, int(_OUTPUT_WEIGHT_HSDQ_BLOCKS)))
    selected = torch.topk(torch.stack(leverage), k=count).indices.tolist()
    levels = torch.as_tensor(_SIGNED_LEVELS, device=weight.device)
    for block in selected:
        lo = int(block) * _BLOCK
        hi = lo + _BLOCK
        local_z = deployed[:, lo:hi]
        gram = local_z.t().mm(local_z)
        diagonal = gram.diagonal().clamp_min(_EPS)
        local_q = q[:, lo:hi]
        local_den = den[:, int(block)]
        local_codes = codes[:, int(block)]
        for _ in range(max(1, int(_OUTPUT_WEIGHT_HSDQ_SWEEPS))):
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
                    local_q[:, coordinate] * 4.0
                    / local_den[:, coordinate].clamp_min(_EPS)
                ).clamp(-7.0, 7.0)
                residual.add_(-local_z[:, coordinate, None] * accepted[None, :])
                correlation.add_(-accepted[:, None] * gram[coordinate][None, :])
        q[:, lo:hi] = local_q
        codes[:, int(block)] = local_codes
    return _write_codes(parent, codes.reshape(rows, channels))


def _crossfold_weight_output(
    weight: torch.Tensor,
    parent: dict[str, torch.Tensor],
    raw_calibration: Sequence[torch.Tensor],
    deployed_calibration: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Cross-fold selector for output-supervised weight candidates."""
    if not raw_calibration or len(raw_calibration) != len(deployed_calibration):
        return parent
    if len(raw_calibration) < 2:
        return _polish_weight_output(
            weight, parent, raw_calibration[0], deployed_calibration[0]
        )
    raw_folds = [item.to(torch.float32) for item in raw_calibration[:2]]
    deployed_folds = [item.to(torch.float32) for item in deployed_calibration[:2]]
    candidates = [parent]
    cand0 = _polish_weight_output(
        weight, parent, raw_folds[0], deployed_folds[0]
    )
    cand1 = _polish_weight_output(
        weight, parent, raw_folds[1], deployed_folds[1]
    )
    parent_losses = [
        _output_product_loss(raw, deployed, weight, parent)
        for raw, deployed in zip(raw_folds, deployed_folds)
    ]
    if _output_product_loss(raw_folds[1], deployed_folds[1], weight, cand0) < parent_losses[1]:
        candidates.append(cand0)
    if _output_product_loss(raw_folds[0], deployed_folds[0], weight, cand1) < parent_losses[0]:
        candidates.append(cand1)
    best = parent
    best_score = sum(parent_losses) / 2.0 + _WEIGHT_HSDQ_ROBUST_MIX * max(parent_losses)
    for candidate in candidates[1:]:
        losses = [
            _output_product_loss(raw, deployed, weight, candidate)
            for raw, deployed in zip(raw_folds, deployed_folds)
        ]
        score = sum(losses) / 2.0 + _WEIGHT_HSDQ_ROBUST_MIX * max(losses)
        if score < best_score * (1.0 - _OUTPUT_WEIGHT_HSDQ_MIN_GAIN):
            best = candidate
            best_score = score
    return best


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
    return _block_cross64(weight, weight)


def _block_cross64(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return contiguous 64-channel blocks of ``left.T @ right``.

    The previous implementation formed the full ``channels x channels`` Gram
    matrix and discarded its off-block entries.  The deployed objective only
    consumes the block diagonal, so a batched block product is mathematically
    equivalent while avoiding the quadratic full matrix.  ``left`` and
    ``right`` are ``[rows, channels]`` tensors and the result is
    ``[channels // 64, 64, 64]``.
    """

    if left.ndim != 2 or right.ndim != 2 or tuple(left.shape) != tuple(right.shape):
        raise ValueError("block cross operands must have identical 2D shapes")
    rows, channels = map(int, left.shape)
    if channels % _BLOCK != 0:
        raise ValueError("block cross channels must be divisible by 64")
    blocks = channels // _BLOCK
    left_blocks = left.to(torch.float32).reshape(rows, blocks, _BLOCK)
    right_blocks = right.to(torch.float32).reshape(rows, blocks, _BLOCK)
    return torch.bmm(
        left_blocks.permute(1, 2, 0).contiguous(),
        right_blocks.permute(1, 0, 2).contiguous(),
    )


def _refine_activation(
    dense: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
    *,
    max_blocks: int = _ACT_HSDQ_BLOCKS,
    sweeps: int = _ACT_HSDQ_SWEEPS,
    output_cross64: torch.Tensor | None = None,
    output_target: torch.Tensor | None = None,
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
    cross = None
    target = None
    output_mode = output_cross64 is not None and output_target is not None
    if output_mode:
        candidate_cross = output_cross64.to(
            device=dense.device, dtype=torch.float32
        )
        candidate_target = output_target.to(
            device=dense.device, dtype=torch.float32
        )
        if tuple(candidate_cross.shape) != (blocks, _BLOCK, _BLOCK):
            output_mode = False
        elif (
            candidate_target.ndim != 2
            or tuple(candidate_target.shape) != (rows, channels)
        ):
            output_mode = False
        else:
            cross = candidate_cross
            target = candidate_target
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
    target_work = None
    cross_work = None
    if output_mode and cross is not None and target is not None:
        target_blocks = target.reshape(rows, blocks, _BLOCK)
        target_work = target_blocks[row_ids, block_ids]
        cross_work = cross.index_select(0, block_ids)
    levels = torch.as_tensor(_SIGNED_LEVELS, device=dense.device)
    diagonal = h_work.diagonal(dim1=-2, dim2=-1).clamp_min(_EPS)
    for _ in range(max(1, int(sweeps))):
        gradient = torch.einsum("nij,nj->ni", h_work, q_work)
        if output_mode and target_work is not None and cross_work is not None:
            gradient = gradient - torch.bmm(
                cross_work, target_work.unsqueeze(-1)
            ).squeeze(-1)
        else:
            gradient = gradient - torch.einsum(
                "nij,nj->ni", h_work, x_work
            )
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
    base_weight_t = _apply_boat_rotation(
        weight * balance.reshape(1, -1), seed, block_size
    )
    base_activation_t = [
        _apply_boat_rotation(
            sample / balance.reshape(1, -1), seed, block_size
        )
        for sample in calibration
    ]
    pair_transform = _learn_roab_pairs(weight, calibration)
    weight_t = base_weight_t
    activation_t = base_activation_t
    if pair_transform is not None:
        try:
            pair_weight_transform = torch.linalg.inv(pair_transform).transpose(-1, -2)
            roab_weight_t = _pair_transform(weight, pair_weight_transform)
            roab_activation_t = [
                _pair_transform(sample, pair_transform) for sample in calibration
            ]
            base_score = _roab_output_score(
                weight, base_weight_t, base_activation_t
            )
            roab_score = _roab_output_score(
                weight, roab_weight_t, roab_activation_t
            )
            if math.isfinite(roab_score) and roab_score < base_score:
                weight_t = roab_weight_t
                activation_t = roab_activation_t
        except (RuntimeError, ValueError, FloatingPointError):
            pair_transform = None
    if weight_t is base_weight_t:
        pair_transform = None
    weight_params = _dense_to_hif4(weight_t, offsets=_BASE_OFFSETS)
    weight_params = _crossfold_weight_hsdq(weight_t, weight_params, activation_t)

    deployed_weight = _dequantize_hif4(weight_params).to(torch.float32)
    output_gain = (
        (weight_t * deployed_weight).sum(dim=0)
        / deployed_weight.square().sum(dim=0).clamp_min(_EPS)
    )
    output_gain = torch.nan_to_num(output_gain, nan=1.0, posinf=2.0, neginf=0.5)
    output_gain = output_gain.clamp(0.5, 2.0)

    gram_tensor = None
    output_cross_tensor = None
    if int(weight_t.shape[1]) <= _ACT_GRAM_MAX_CHANNELS:
        # The online activation objective is evaluated after this weight has
        # been quantized.  Use the deployed Q(W) curvature instead of the
        # raw-W curvature so the block HSDQ step follows the real operator.
        gram_tensor = _gram64(deployed_weight)
        # L2: retain the block diagonal of Wq.T @ W.  The dynamic activation
        # API can then optimize the actual output objective against the raw
        # transformed activation without storing the full weight matrix.
        output_cross_tensor = _block_cross64(deployed_weight, weight_t)
    # Recreate the activation tensor that the online API will deploy, then
    # use it as the left operand of an output-supervised W refinement.  The
    # teacher target remains raw ``A_t @ W_t.T`` so the weight codes can
    # compensate the fixed activation error instead of merely fitting W_t.
    deployed_calibration: list[torch.Tensor] = []
    for sample in activation_t:
        quant_dense = sample * output_gain.reshape(1, -1)
        activation_params = _dense_to_hif4(
            quant_dense, offsets=_BASE_OFFSETS, gram64=gram_tensor
        )
        activation_params = _refine_activation(
            quant_dense,
            activation_params,
            gram_tensor,
            output_cross64=output_cross_tensor,
            output_target=sample,
        )
        deployed_calibration.append(
            _dequantize_hif4(activation_params).to(torch.float32)
        )
    weight_params = _crossfold_weight_output(
        weight_t, weight_params, activation_t, deployed_calibration
    )

    # The output-supervised W pass may change Q(W).  Refresh both block
    # statistics so the online L2 activation pass uses the final deployed
    # operator rather than the pre-refinement parent.
    if gram_tensor is not None:
        deployed_weight = _dequantize_hif4(weight_params).to(torch.float32)
        gram_tensor = _gram64(deployed_weight)
        output_cross_tensor = _block_cross64(deployed_weight, weight_t)

    gram_state = None
    if gram_tensor is not None:
        gram_state = _cpu_tensor(gram_tensor)
    output_cross_state = None
    if output_cross_tensor is not None:
        output_cross_state = _cpu_tensor(output_cross_tensor)
    state: dict[str, Any] = {
        "smooth_inv": _cpu_tensor(balance.reciprocal()),
        "block_smooth_size": int(block_size),
        "block_smooth_seed": int(seed),
        "roab_pairs": (
            _cpu_tensor(pair_transform)
            if pair_transform is not None
            else None
        ),
        "gram64": gram_state,
        "output_cross64": output_cross_state,
        "output_gain": _cpu_tensor(output_gain),
        "version": 3,
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
    roab_pairs = state.get("roab_pairs")
    if torch.is_tensor(roab_pairs):
        dense = _pair_transform(dense, roab_pairs.to(dense.device))
    else:
        smooth_inv = state.get("smooth_inv")
        if torch.is_tensor(smooth_inv):
            dense = dense * smooth_inv.to(dense.device).reshape(1, -1)
        seed = int(state.get("block_smooth_seed", -1))
        block_size = int(state.get("block_smooth_size", 0))
        if block_size > 0:
            dense = _apply_boat_rotation(dense, seed, block_size)
    # Keep the transformed pre-gain activation as the teacher target for the
    # output-space objective.  ``output_gain`` is only the deployed activation
    # reweighting and must not change the reference ``A_t @ W_t.T``.
    teacher_dense = dense
    output_gain = state.get("output_gain")
    if torch.is_tensor(output_gain):
        gain = output_gain.to(dense.device, dtype=torch.float32).reshape(1, -1)
        if int(gain.shape[1]) == int(dense.shape[1]):
            dense = dense * gain
    gram = state.get("gram64")
    gram_tensor = gram if torch.is_tensor(gram) else None
    if gram_tensor is not None:
        gram_tensor = gram_tensor.to(dense.device)
    output_cross = state.get("output_cross64")
    output_cross_tensor = output_cross if torch.is_tensor(output_cross) else None
    if output_cross_tensor is not None:
        output_cross_tensor = output_cross_tensor.to(dense.device)
    params = _dense_to_hif4(
        dense, offsets=_BASE_OFFSETS, gram64=gram_tensor
    )
    return _refine_activation(
        dense,
        params,
        gram_tensor,
        output_cross64=output_cross_tensor,
        output_target=teacher_dense,
    )

# ---------------------------------------------------------------------------
# v86 Attention implementation copied directly into this file.
# Conflicting private helpers are named _attention_* so they cannot alter the
# v140 Linear path; the four public Attention APIs retain their official names.
# ---------------------------------------------------------------------------
_NVFP4_BLOCK_SIZE = 16

_HIF4_BLOCK_SIZE = 64

_attention_E6M2_MIN = 2.0**-48

_attention_E6M2_MAX = 49152.0

_HIF4_MAX_INNER = 7.0

_attention_BF16_ONE_SEVENTH = 0.142578125

_attention_EPS = 1.0e-12

_attention_ATTN_STATS_TOKENS = 4096

_ATTN_EVAL_TOKENS = 128

_BLOCK_SMOOTH_ALLOWED_SIZES = (4, 8, 16, 32, 64)

_CAT64_BLOCK_SIZE = 64

_LINEAR_R64_BLOCK = 64

_WEIGHT_FULL64_SIGNED_CODES = tuple(
    round(code * 0.25, 2) for code in range(-7, 8)
)

_QK_SMOOTH_ALPHAS = (0.25, 0.50)

_QK_SMOOTH_RMS = True

_ATTN_CENTER_MODES = (0, 2)

_ATTN_BLOCK_SMOOTH_ENABLED = True

_ATTN_BLOCK_SMOOTH_SIZES = (4, 8, 16)

_ATTN_BLOCK_SMOOTH_SEEDS = (0,)

_ATTN_BLOCK_SMOOTH_MIN_GAIN = 1.0e-3

_ATTN_BLOCK_SMOOTH_WORST_TOLERANCE = 0.01

_ATTN_BLOCK_SMOOTH_FINAL_QUANTIZER = True

_ATTN_BLOCK_SMOOTH_REFINE_RATIO = 0.50

_ATTN_BLOCK_SMOOTH_REFINE_BLOCKS = 24_576

_ATTN_SCALE_AWARE_CENTER = True

_ATTN_SCALE_AWARE_CENTER_GQA = False

_ATTN_CENTER_ALTERNATIONS = 3

_ATTN_OUTPUT_SELECTOR = True

_ATTN_A1_MAX_TOKENS = 256

_ATTN_OUTPUT_HEADWISE_PERMUTATION = False

_ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES = 4

_ATTN_FISHER_IMPORTANCE = False

_ATTN_FISHER_MIN_GAIN = 0.001

_ATTN_FISHER_WORST_TOLERANCE = 0.03

_ATTN_FISHER_BLEND_VALUES = (0.25, 0.50, 1.00)

_ATTN_OUTPUT_HEAD_SCALE = False

_ATTN_OUTPUT_HEAD_SCALE_FACTORS = (0.50, 0.75, 1.25, 1.50, 2.00)

_ATTN_OUTPUT_EXTRA_SMOOTH_ALPHAS = ()

_A1_GATE_MIN_IMPROVEMENT = 0.005

_A1_GATE_WORST_TOLERANCE = 0.02

_ATTN_H64 = False

_ATTN_H64_SEEDS = (0, 1)

_ATTN_H64_BLOCK = 64

_ATTN_ROTATION_ENABLED = True

_ATTN_ROTATION_BLOCKS = (16, 32, 64)

_attention_ATTN_ROTATION_SEEDS = (0, 1, 2, 3)

_ATTN_ROTATION_GQA_ONLY = True

_V_IMPORTANCE_CANDIDATES = False

_L1_DATA_DRIVEN_SCALE = False

_L1_TRIM_QUANTILES = (0.90, 0.95)

_L1_ADJACENT_CODE_DELTAS = (-1, 0, 1)

_REFINE_RANK_BY_ABSOLUTE = True

_ATTN_REFINE_ERROR_THRESHOLD = 1.0e-7

_Q_REFINE_ACCEPT_MARGIN = 0.03

_Q_REFINE_MAX_RATIO = 0.60

_Q_REFINE_MAX_BLOCKS = 16_384

_K_REFINE_ACCEPT_MARGIN = 0.03

_K_REFINE_MAX_RATIO = 0.70

_K_REFINE_MAX_BLOCKS = 24_576

_V_REFINE_ACCEPT_MARGIN = 0.01

_V_REFINE_MAX_RATIO = 0.60

_V_REFINE_MAX_BLOCKS = 24_576

_QK_SMOOTH_MIN = 1.0 / 16.0

_QK_SMOOTH_MAX = 16.0

_IMPORTANCE_FLOOR = 0.05

_DYNAMIC_OFFSETS = (-1, 1, 2, 3)

_REFINE_EDGE_EXTENSION = True

_REFINE_EDGE_EXTEND_STEPS = 2

_DATA_DRIVEN_RATIO = True

_RATIO_CAPTURE_TARGET = 1.0

_RATIO_MIN = 0.10

_WEIGHT_QUADRATIC8_MAX_RATIO = 0.05

_WEIGHT_QUADRATIC8_MAX_GROUPS = 8192

_WEIGHT_QUADRATIC8_SWEEPS = 2

_WEIGHT_QUADRATIC8_ACCEPT_MARGIN = 1.0e-5

_ACTIVATION_QUADRATIC_MAX_FEATURES = 4096

_ACTIVATION_QUADRATIC8_MAX_RATIO = 0.08

_ACTIVATION_QUADRATIC8_MAX_GROUPS = 4096

_ACTIVATION_QUADRATIC8_SWEEPS = 1

_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN = 1.0e-5

_ACTIVATION_QUADRATIC16 = False  # REJECTED 2026-08-28: -3.43pp real-data

_ACTIVATION_QUADRATIC16_MAX_FEATURES = 1024

_ACTIVATION_QUADRATIC16_MAX_RATIO = 0.10

_ACTIVATION_QUADRATIC16_MAX_GROUPS = 4096

_ACTIVATION_QUADRATIC16_SWEEPS = 1

_ACTIVATION_QUADRATIC16_ACCEPT_MARGIN = 1.0e-5

_ACTIVATION_GRAM64 = True

_ACTIVATION_GRAM64_MAX_FEATURES = 8192

_ACTIVATION_GRAM64_MAX_RATIO = 1.0

_ACTIVATION_GRAM64_MAX_BLOCKS = 128

_ACTIVATION_GRAM64_SWEEPS = 5

_ACTIVATION_GRAM64_ACCEPT_MARGIN = 1.0e-5

_ACTIVATION_GRAM64_HIERARCHY = True

_ACTIVATION_GRAM64_HIERARCHY_OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)

_ACTIVATION_SOURCE_SCALE_STATS = ("median", "q75", "max")

_ACTIVATION_SOURCE_SCALE_TO_HIF4_AMAX = 6.0 / 7.0

_ACTIVATION_SOURCE_SCALE_MARKER = 1_000_000

_ACTIVATION_SAMPLE_IMPORTANCE = False  # REJECTED 2026-08-28: Linear bit-identical, Attention -2.7pp

_PERMUTATION_BASES = True

_V_ATTENTION_IMPORTANCE = True

_V_ATTENTION_IMPORTANCE_SHRINK = 1.0

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

def _attention_dequantize_nvfp4_float32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
) -> torch.Tensor:
    """Match the supplied BF16 dequantizer, then use FP32 for optimization."""

    return dequantize_nvfp4(quant_float, scale_float).to(torch.float32)

def _attention_sample_rows(x: torch.Tensor, limit: int) -> torch.Tensor:
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
    ).clamp_min(_attention_EPS)

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
    if float(mean) <= _attention_EPS:
        return torch.ones_like(w)
    return (w / mean).clamp_min(_IMPORTANCE_FLOOR)

def _standard_block_losses(
    dense: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-block importance-weighted squared error of standard HiF4."""

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
        posinf=_attention_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_attention_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    max4 = x_abs.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    _, standard_scale = _standard_e6m2_scale(amax)

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

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
    else:
        weighted_error = (
            (x_abs - mantissa * denominator).square()
            * channel_importance.reshape(*([1] * len(prefix)), blocks, 8, 2, 4)
        )
    return weighted_error.sum(dim=(-1, -2, -3)).reshape(-1)

def _loss_capture_ratio(
    losses: torch.Tensor,
    *,
    target: float,
    ratio_min: float,
) -> float:
    """Smallest fraction of the largest-loss blocks covering ``target`` of the
    total loss.  This converts the per-block loss tail into a refine budget."""

    losses = losses.detach().to(torch.float32).reshape(-1)
    total = float(losses.sum())
    if total <= _attention_EPS:
        return float(ratio_min)
    sorted_descending = torch.sort(losses, descending=True).values
    cumulative = torch.cumsum(sorted_descending, dim=0)
    k = int((cumulative < float(target) * total).sum()) + 1
    return float(
        min(1.0, max(float(ratio_min), k / max(1, int(losses.numel()))))
    )

def _refine_weight_groups8(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    group_gram8: torch.Tensor,
    *,
    max_ratio: float = _WEIGHT_QUADRATIC8_MAX_RATIO,
    max_groups: int = _WEIGHT_QUADRATIC8_MAX_GROUPS,
    sweeps: int = _WEIGHT_QUADRATIC8_SWEEPS,
    accept_margin: float = _WEIGHT_QUADRATIC8_ACCEPT_MARGIN,
) -> dict[str, torch.Tensor]:
    """Coordinate-refine top-loss 8-channel groups using incremental H*e."""

    if dense.ndim != 2:
        return params
    rows, channels = map(int, dense.shape)
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % 8 != 0:
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    expected_grams = blocks * 8
    if tuple(group_gram8.shape) != (expected_grams, 8, 8):
        return params

    dense8 = dense.reshape(rows, blocks, 8, 8).reshape(-1, 8)
    quantized8 = _attention_dequantize_hif4(params).to(torch.float32).reshape(
        rows, blocks, 8, 8
    ).reshape(-1, 8)
    grams = group_gram8.unsqueeze(0).expand(rows, -1, -1, -1).reshape(
        -1, 8, 8
    )
    error = quantized8 - dense8
    losses = torch.einsum("ni,nij,nj->n", error, grams, error)
    finite = torch.isfinite(losses) & (losses > _attention_EPS)
    candidates = torch.nonzero(finite, as_tuple=False).reshape(-1)
    if int(candidates.numel()) == 0:
        return params
    cap = max(
        1,
        int(
            math.ceil(
                int(losses.numel()) * float(max_ratio)
            )
        ),
    )
    cap = min(cap, int(max_groups), int(candidates.numel()))
    if int(candidates.numel()) > cap:
        order = torch.topk(
            losses.index_select(0, candidates), k=cap, largest=True
        ).indices
        candidates = candidates.index_select(0, order)

    x_selected = dense8.index_select(0, candidates)
    q_selected = quantized8.index_select(0, candidates).clone()
    gram_selected = grams.index_select(0, candidates)
    error_selected = q_selected - x_selected
    he = torch.einsum("nij,nj->ni", gram_selected, error_selected)
    initial_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )

    scale = params["scale_factor"].reshape(rows, blocks, 1).expand(
        rows, blocks, 8
    )
    lv2 = params["scale_lv2"].reshape(rows, blocks, 8)
    lv3 = params["scale_lv3"].reshape(rows, blocks, 8, 2)
    denominator = (
        (
            scale[..., None]
            * lv2[..., None]
            * lv3.repeat_interleave(4, dim=-1)
        ).reshape(-1, 8)
    ).index_select(0, candidates)
    signed_codes = torch.arange(
        -7, 8, dtype=torch.float32, device=dense.device
    ) * 0.25

    for _ in range(int(sweeps)):
        for coordinate in range(8):
            possible = denominator[:, coordinate, None] * signed_codes[None, :]
            delta = possible - q_selected[:, coordinate, None]
            diagonal = gram_selected[:, coordinate, coordinate].clamp_min(_attention_EPS)
            change = (
                2.0 * delta * he[:, coordinate, None]
                + delta.square() * diagonal[:, None]
            )
            best = change.argmin(dim=1)
            row_ids = torch.arange(int(candidates.numel()), device=dense.device)
            best_delta = delta[row_ids, best]
            improve = change[row_ids, best] < -_attention_EPS
            best_delta = torch.where(
                improve, best_delta, torch.zeros_like(best_delta)
            )
            q_selected[:, coordinate] += best_delta
            error_selected[:, coordinate] += best_delta
            he += best_delta[:, None] * gram_selected[:, :, coordinate]

    final_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )
    improve = final_loss < initial_loss * (
        1.0 - float(accept_margin)
    )
    improved_indices = candidates[improve]
    if int(improved_indices.numel()) == 0:
        return params
    improved_q = q_selected[improve]
    improved_denominator = denominator[improve]
    improved_codes = torch.round(
        improved_q / improved_denominator.clamp_min(_attention_EPS) * 4.0
    ).clamp(-7.0, 7.0)

    refined = dict(params)
    sign8 = params["sign"].clone().reshape(-1, 8)
    mant8 = params["mant"].clone().reshape(-1, 8)
    sign8.index_copy_(0, improved_indices, torch.sign(improved_codes))
    mant8.index_copy_(0, improved_indices, improved_codes.abs() * 0.25)
    refined["sign"] = sign8.reshape_as(params["sign"])
    refined["mant"] = mant8.reshape_as(params["mant"])
    return refined

def _refine_activation_groups16(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    group_gram16: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """C34: coordinate-refine top-loss 16-channel activation groups.

    Mirrors the weight-side 16-group refiner, driven by the activation
    Hessian logits instead of the weight ones.  Activation refinement runs
    in the per-sample dynamic path, so the ratio is kept low to bound the
    dynamic cost; wide layers (channels > max_features) skip by design.
    """

    rows, channels = map(int, (dense.shape[0], dense.shape[1]))
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % 16 != 0:
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    expected_grams = blocks * 4
    if tuple(group_gram16.shape) != (expected_grams, 16, 16):
        return params

    dense16 = dense.reshape(rows, blocks, 4, 16).reshape(-1, 16)
    quantized16 = _attention_dequantize_hif4(params).to(torch.float32).reshape(
        rows, blocks, 4, 16
    ).reshape(-1, 16)
    grams = group_gram16.unsqueeze(0).expand(rows, -1, -1, -1).reshape(
        -1, 16, 16
    )
    error = quantized16 - dense16
    losses = torch.einsum("ni,nij,nj->n", error, grams, error)
    candidates = torch.nonzero(
        torch.isfinite(losses) & (losses > _attention_EPS), as_tuple=False
    ).reshape(-1)
    if int(candidates.numel()) == 0:
        return params
    cap = max(
        1,
        int(
            math.ceil(
                int(losses.numel()) * _ACTIVATION_QUADRATIC16_MAX_RATIO
            )
        ),
    )
    cap = min(
        cap, _ACTIVATION_QUADRATIC16_MAX_GROUPS, int(candidates.numel())
    )
    if int(candidates.numel()) > cap:
        order = torch.topk(
            losses.index_select(0, candidates), k=cap, largest=True
        ).indices
        candidates = candidates.index_select(0, order)

    x_selected = dense16.index_select(0, candidates)
    q_selected = quantized16.index_select(0, candidates).clone()
    gram_selected = grams.index_select(0, candidates)
    error_selected = q_selected - x_selected
    he = torch.einsum("nij,nj->ni", gram_selected, error_selected)
    initial_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )

    scale = params["scale_factor"].reshape(rows, blocks, 1).expand(
        rows, blocks, 8
    )
    lv2 = params["scale_lv2"].reshape(rows, blocks, 8)
    lv3 = params["scale_lv3"].reshape(rows, blocks, 8, 2)
    denominator8 = (
        scale[..., None]
        * lv2[..., None]
        * lv3.repeat_interleave(4, dim=-1)
    )
    denominator = denominator8.reshape(rows, blocks, 4, 16).reshape(
        -1, 16
    ).index_select(0, candidates)
    signed_codes = torch.arange(
        -7, 8, dtype=torch.float32, device=dense.device
    ) * 0.25

    for _ in range(_ACTIVATION_QUADRATIC16_SWEEPS):
        for coordinate in range(16):
            possible = denominator[:, coordinate, None] * signed_codes[None, :]
            delta = possible - q_selected[:, coordinate, None]
            diagonal = gram_selected[:, coordinate, coordinate].clamp_min(_attention_EPS)
            change = (
                2.0 * delta * he[:, coordinate, None]
                + delta.square() * diagonal[:, None]
            )
            best = change.argmin(dim=1)
            row_ids = torch.arange(int(candidates.numel()), device=dense.device)
            best_delta = delta[row_ids, best]
            improve = change[row_ids, best] < -_attention_EPS
            best_delta = torch.where(
                improve, best_delta, torch.zeros_like(best_delta)
            )
            q_selected[:, coordinate] += best_delta
            error_selected[:, coordinate] += best_delta
            he += best_delta[:, None] * gram_selected[:, :, coordinate]

    final_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )
    improve = final_loss < initial_loss * (
        1.0 - _ACTIVATION_QUADRATIC16_ACCEPT_MARGIN
    )
    improved_indices = candidates[improve]
    if int(improved_indices.numel()) == 0:
        return params
    improved_q = q_selected[improve]
    improved_denominator = denominator[improve]
    improved_codes = torch.round(
        improved_q / improved_denominator.clamp_min(_attention_EPS) * 4.0
    ).clamp(-7.0, 7.0)

    refined = dict(params)
    sign16 = params["sign"].clone().reshape(-1, 16)
    mant16 = params["mant"].clone().reshape(-1, 16)
    sign16.index_copy_(0, improved_indices, torch.sign(improved_codes))
    mant16.index_copy_(0, improved_indices, improved_codes.abs() * 0.25)
    refined["sign"] = sign16.reshape_as(params["sign"])
    refined["mant"] = mant16.reshape_as(params["mant"])
    return refined

@torch.no_grad()
def _refine_activation_blocks64(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    gram64: torch.Tensor,
    *,
    max_ratio: float = _ACTIVATION_GRAM64_MAX_RATIO,
    max_blocks: int = _ACTIVATION_GRAM64_MAX_BLOCKS,
    sweeps: int = _ACTIVATION_GRAM64_SWEEPS,
    accept_margin: float = _ACTIVATION_GRAM64_ACCEPT_MARGIN,
) -> dict[str, torch.Tensor]:
    """C75.2: refine selected activation 64-groups under full ``W.T@W``.

    ``gram64`` is block diagonal by construction, so a dynamic sample can
    select its highest-loss 64-groups and solve those groups independently.
    The existing batched coordinate solver enumerates the legal signed
    mantissa lattice and accepts only exact negative quadratic changes.  All
    unselected groups and failed/non-finite groups keep the parent fields.
    """

    if not _ACTIVATION_GRAM64:
        return params
    if dense.ndim != 2 or gram64.ndim != 3:
        return params
    rows, channels = map(int, dense.shape)
    if rows <= 0 or channels % _HIF4_BLOCK_SIZE != 0:
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    if tuple(int(v) for v in gram64.shape) != (
        blocks,
        _HIF4_BLOCK_SIZE,
        _HIF4_BLOCK_SIZE,
    ):
        return params
    h = gram64.detach().to(device=dense.device, dtype=torch.float32)
    if not bool(torch.isfinite(h).all()):
        return params

    x = dense.detach().to(torch.float32).reshape(rows, blocks, _HIF4_BLOCK_SIZE)
    q = _attention_dequantize_hif4(params).to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, _HIF4_BLOCK_SIZE)
    error = q - x
    losses = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    losses = torch.where(
        torch.isfinite(losses), losses, torch.full_like(losses, -torch.inf)
    )
    if not bool((losses > _attention_EPS).any()):
        return params
    ratio_count = max(
        1,
        int(math.ceil(blocks * max(0.0, min(float(max_ratio), 1.0)))),
    )
    selected_count = min(blocks, max(1, min(ratio_count, int(max_blocks))))
    selected = torch.topk(losses, k=selected_count, dim=1, largest=True).indices
    flat_selected = selected.reshape(-1)
    row_index = torch.arange(rows, device=dense.device).unsqueeze(1).expand(
        rows, selected_count
    ).reshape(-1)
    valid = losses[row_index, flat_selected] > _attention_EPS
    if not bool(valid.any()):
        return params
    row_index = row_index[valid]
    block_index = flat_selected[valid]

    q_selected = q[row_index, block_index].unsqueeze(1)
    x_selected = x[row_index, block_index].unsqueeze(1)
    h_selected = h.index_select(0, block_index)
    scale = params["scale_factor"].to(torch.float32).reshape(rows, blocks)
    lv2 = params["scale_lv2"].to(torch.float32).reshape(rows, blocks, 8)
    lv3 = params["scale_lv3"].to(torch.float32).reshape(rows, blocks, 8, 2)
    denominator = (
        scale[row_index, block_index].reshape(-1, 1, 1, 1)
        * lv2[row_index, block_index, :, None, None]
        * lv3[row_index, block_index, :, :, None]
    ).repeat_interleave(4, dim=-1).reshape(-1, 1, _HIF4_BLOCK_SIZE)
    q_work = q_selected[:, 0].clone()
    x_work = x_selected[:, 0]
    den_work = denominator[:, 0]
    signed_codes = torch.tensor(
        _WEIGHT_FULL64_SIGNED_CODES,
        dtype=torch.float32,
        device=dense.device,
    )
    for _ in range(max(1, int(sweeps))):
        gradient = torch.einsum("nij,nj->ni", h_selected, q_work - x_work)
        diagonal = h_selected.diagonal(dim1=-2, dim2=-1).clamp_min(_attention_EPS)
        for coordinate in range(_HIF4_BLOCK_SIZE):
            current = q_work[:, coordinate]
            candidates = den_work[:, coordinate, None] * signed_codes[None, :]
            delta = candidates - current[:, None]
            change = (
                2.0 * delta * gradient[:, coordinate, None]
                + diagonal[:, coordinate, None] * delta.square()
            )
            best_change, best_index = change.min(dim=-1)
            improve = torch.isfinite(best_change) & (best_change < -_attention_EPS)
            step = delta.gather(-1, best_index[:, None]).squeeze(-1)
            step = torch.where(improve, step, torch.zeros_like(step))
            q_work[:, coordinate] = current + step
            gradient.add_(step[:, None] * h_selected[:, :, coordinate])
    q_selected = q_work.unsqueeze(1)
    improved = q_selected[:, 0] - q[row_index, block_index]
    if not bool(torch.isfinite(improved).all()):
        return params
    # The coordinate solver already accepts only negative changes.  A final
    # exact loss check keeps the acceptance margin explicit and protects
    # against roundoff in a near-zero Hessian block.
    old_loss = torch.einsum(
        "ni,nij,nj->n",
        q[row_index, block_index] - x[row_index, block_index],
        h_selected,
        q[row_index, block_index] - x[row_index, block_index],
    )
    new_loss = torch.einsum(
        "ni,nij,nj->n",
        q_selected[:, 0] - x_selected[:, 0],
        h_selected,
        q_selected[:, 0] - x_selected[:, 0],
    )
    accept = new_loss <= old_loss * (1.0 - max(0.0, float(accept_margin)))
    if not bool(accept.any()):
        return params
    full_denominator = (
        params["scale_factor"].to(torch.float32).reshape(rows, blocks, 1, 1, 1)
        * params["scale_lv2"].to(torch.float32).reshape(rows, blocks, 8, 1, 1)
        * params["scale_lv3"].to(torch.float32).reshape(rows, blocks, 8, 2, 1)
    ).repeat_interleave(4, dim=-1).reshape(rows, blocks, _HIF4_BLOCK_SIZE)
    accepted_rows = row_index[accept]
    accepted_blocks = block_index[accept]
    accepted_denominator = full_denominator[accepted_rows, accepted_blocks]
    codes = torch.round(
        q_selected[accept, 0] * 4.0 / accepted_denominator.clamp_min(_attention_EPS)
    ).clamp(-7.0, 7.0)
    refined = dict(params)
    sign_flat = refined["sign"].to(torch.float32).reshape(
        rows, blocks, _HIF4_BLOCK_SIZE
    ).clone()
    mant_flat = refined["mant"].to(torch.float32).reshape(
        rows, blocks, _HIF4_BLOCK_SIZE
    ).clone()
    sign_flat[accepted_rows, accepted_blocks] = torch.where(
        codes == 0.0, torch.zeros_like(codes), torch.sign(codes)
    )
    mant_flat[accepted_rows, accepted_blocks] = codes.abs().mul(0.25)
    refined["sign"] = sign_flat.reshape_as(params["sign"])
    refined["mant"] = mant_flat.reshape_as(params["mant"])
    return refined

@torch.no_grad()
def _refine_activation_hierarchy64(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    gram64: torch.Tensor,
    *,
    max_ratio: float = _ACTIVATION_GRAM64_MAX_RATIO,
    max_blocks: int = _ACTIVATION_GRAM64_MAX_BLOCKS,
    offsets: Sequence[int] = _ACTIVATION_GRAM64_HIERARCHY_OFFSETS,
    accept_margin: float = _ACTIVATION_GRAM64_ACCEPT_MARGIN,
) -> dict[str, torch.Tensor]:
    """C75.5: full-H hierarchy beam for selected activation blocks.

    The existing C75.2 refiner changes mantissa coordinates while keeping the
    parent scale hierarchy fixed.  This companion searches a small legal
    E6M2 neighbourhood, solves lv2/lv3 for each scale, and accepts only a
    negative exact full-H quadratic change.  The candidate tensor is local to
    one dynamic call; only the five legal HiF4 fields are returned.
    """

    if not _ACTIVATION_GRAM64_HIERARCHY:
        return params
    if dense.ndim != 2 or gram64.ndim != 3:
        return params
    rows, channels = map(int, dense.shape)
    if rows <= 0 or channels % _HIF4_BLOCK_SIZE != 0:
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    if tuple(int(v) for v in gram64.shape) != (
        blocks,
        _HIF4_BLOCK_SIZE,
        _HIF4_BLOCK_SIZE,
    ):
        return params
    offset_values = tuple(int(value) for value in offsets)
    if not offset_values:
        return params
    h = gram64.detach().to(device=dense.device, dtype=torch.float32)
    if not bool(torch.isfinite(h).all()):
        return params
    x = dense.detach().to(torch.float32).reshape(
        rows, blocks, _HIF4_BLOCK_SIZE
    )
    q = _attention_dequantize_hif4(params).to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, _HIF4_BLOCK_SIZE)
    error = q - x
    losses = torch.einsum("rbi,bij,rbj->rb", error, h, error)
    losses = torch.where(
        torch.isfinite(losses), losses, torch.full_like(losses, -torch.inf)
    )
    ratio_count = max(
        1,
        int(math.ceil(blocks * max(0.0, min(float(max_ratio), 1.0)))),
    )
    selected_count = min(blocks, max(1, min(ratio_count, int(max_blocks))))
    selected = torch.topk(losses, k=selected_count, dim=1, largest=True).indices
    row_index = (
        torch.arange(rows, device=dense.device)
        .unsqueeze(1)
        .expand(rows, selected_count)
        .reshape(-1)
    )
    block_index = selected.reshape(-1)
    valid = losses[row_index, block_index] > _attention_EPS
    if not bool(valid.any()):
        return params
    row_index = row_index[valid]
    block_index = block_index[valid]
    x_selected = x[row_index, block_index]
    old_selected = q[row_index, block_index]
    h_selected = h.index_select(0, block_index)
    sign_selected = params["sign"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 4)[row_index, block_index]
    x_abs = x_selected.abs().reshape(-1, 8, 2, 4)
    standard_code, _ = _standard_e6m2_scale(
        x_abs.amax(dim=(-1, -2, -3))
    )
    current_scale = params["scale_factor"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks)[row_index, block_index]
    current_code = _attention_e6m2_encode_nearest(current_scale)
    delta_codes = torch.as_tensor(
        offset_values, dtype=torch.int64, device=dense.device
    )
    candidate_codes = (
        standard_code.to(torch.int64).unsqueeze(0) + delta_codes[:, None]
    ).clamp(min=0, max=254)
    candidate_codes = torch.cat(
        (candidate_codes, current_code.to(torch.int64).unsqueeze(0)), dim=0
    )
    candidate_scales = _attention_e6m2_decode(candidate_codes)
    old_gradient = torch.einsum(
        "nij,nj->ni", h_selected, old_selected - x_selected
    )
    best_change = torch.zeros(
        int(row_index.numel()), dtype=torch.float32, device=dense.device
    )
    best_index = torch.full_like(best_change, candidate_codes.shape[0] - 1, dtype=torch.int64)
    best_scale = current_scale.clone()
    best_lv2 = params["scale_lv2"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 8)[row_index, block_index].clone()
    best_lv3 = params["scale_lv3"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2)[row_index, block_index].clone()
    best_mantissa = params["mant"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 4)[row_index, block_index].clone()
    for candidate_index in range(int(candidate_scales.shape[0])):
        scale = candidate_scales[candidate_index]
        _, level2, level3, mantissa = _solve_exact_hierarchy(
            x_abs,
            scale,
            None,
            sign_selected,
            None,
        )
        candidate = (
            sign_selected
            * mantissa
            * level3[..., None]
            * level2[..., None, None]
            * scale[..., None, None, None]
        ).reshape(-1, _HIF4_BLOCK_SIZE)
        delta = candidate - old_selected
        change = (
            2.0 * (old_gradient * delta).sum(dim=1)
            + torch.einsum("ni,nij,nj->n", delta, h_selected, delta)
        )
        better = torch.isfinite(change) & (change < best_change)
        best_change = torch.where(better, change, best_change)
        best_index = torch.where(
            better,
            torch.full_like(best_index, candidate_index),
            best_index,
        )
        best_scale = torch.where(better, scale, best_scale)
        best_lv2 = torch.where(better[:, None], level2, best_lv2)
        best_lv3 = torch.where(better[:, None, None], level3, best_lv3)
        best_mantissa = torch.where(
            better[:, None, None, None], mantissa, best_mantissa
        )
    new_selected = (
        sign_selected
        * best_mantissa
        * best_lv3[..., None]
        * best_lv2[..., None, None]
        * best_scale[..., None, None, None]
    ).reshape(-1, _HIF4_BLOCK_SIZE)
    old_loss = torch.einsum(
        "ni,nij,nj->n", old_selected - x_selected, h_selected, old_selected - x_selected
    )
    new_loss = torch.einsum(
        "ni,nij,nj->n", new_selected - x_selected, h_selected, new_selected - x_selected
    )
    accept = torch.isfinite(new_loss) & (
        new_loss <= old_loss * (1.0 - max(0.0, float(accept_margin)))
    )
    if not bool(accept.any()):
        return params
    refined = {
        key: value.detach().to(device=dense.device).clone()
        for key, value in params.items()
    }
    scale_view = refined["scale_factor"].reshape(rows, blocks)
    lv2_view = refined["scale_lv2"].reshape(rows, blocks, 8)
    lv3_view = refined["scale_lv3"].reshape(rows, blocks, 8, 2)
    mant_view = refined["mant"].reshape(rows, blocks, 8, 2, 4)
    sign_view = refined["sign"].reshape(rows, blocks, 8, 2, 4)
    scale_view[row_index, block_index] = torch.where(
        accept, best_scale, scale_view[row_index, block_index]
    )
    lv2_view[row_index, block_index] = torch.where(
        accept[:, None], best_lv2, lv2_view[row_index, block_index]
    )
    lv3_view[row_index, block_index] = torch.where(
        accept[:, None, None], best_lv3, lv3_view[row_index, block_index]
    )
    mant_view[row_index, block_index] = torch.where(
        accept[:, None, None, None], best_mantissa, mant_view[row_index, block_index]
    )
    sign_view[row_index, block_index] = torch.where(
        accept[:, None, None, None] & (best_mantissa != 0.0),
        sign_selected,
        torch.where(
            accept[:, None, None, None],
            torch.zeros_like(sign_selected),
            sign_view[row_index, block_index],
        ),
    )
    return refined

def _headwise_range_permutation(ranges: torch.Tensor) -> torch.Tensor:
    """Per-head argsort of log ranges (ranges: [heads, head_dim])."""

    log_r = torch.log2(ranges.to(torch.float32).clamp_min(_attention_EPS))
    log_r = log_r - log_r.median(dim=-1, keepdim=True).values
    spread = log_r.amax(dim=-1) - log_r.amin(dim=-1)
    identity = torch.arange(
        int(ranges.shape[-1]), dtype=torch.int64, device=ranges.device
    ).expand_as(ranges)
    ordered = torch.argsort(log_r, dim=-1, descending=True)
    return torch.where(spread[:, None] >= 0.25, ordered, identity)

def _headwise_hierarchy_permutation(
    q_range: torch.Tensor,
    k_range: torch.Tensor,
) -> torch.Tensor:
    """Return a local feature permutation for each paired Q/KV head."""

    if q_range.ndim != 2 or tuple(q_range.shape) != tuple(k_range.shape):
        raise ValueError("Headwise Q/K ranges must have shape [heads, head_dim]")
    q_log = torch.log2(q_range.to(torch.float32).clamp_min(_attention_EPS))
    k_log = torch.log2(k_range.to(torch.float32).clamp_min(_attention_EPS))
    q_log = q_log - q_log.median(dim=-1, keepdim=True).values
    k_log = k_log - k_log.median(dim=-1, keepdim=True).values
    pressure = torch.maximum(q_log, k_log)
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

@torch.no_grad()
def _solve_k_center_scale_aware(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rounds: int,
) -> torch.Tensor:
    """C41: solve a quantization-aware K center by fixed-point iteration.

    ``K' = K - 1 c^T`` is an exact softmax invariance for every choice of
    ``c``, so the center can be optimized purely against HiF4 reconstruction
    error.  With the quantized codes held fixed, the MSE-optimal center is

        c = mean_tokens(K - dequant(Q(K - c)))

    which yields a simple fixed-point iteration.  Starting from ``c = 0``
    keeps the identity candidate admissible, so this can never be worse than
    the uncentered path when the gate also keeps the incumbent.
    """

    heads = int(num_heads)
    width = int(head_dim)
    if dense.ndim != 2 or int(dense.shape[0]) <= 0:
        raise ValueError("scale-aware centering expects a non-empty 2D tensor")
    if int(dense.shape[1]) != heads * width:
        raise ValueError("Invalid dimensions for scale-aware centering")
    grouped = dense.reshape(-1, heads, width).to(torch.float32)
    center = torch.zeros(
        (heads, width), dtype=torch.float32, device=grouped.device
    )
    for _ in range(max(1, int(rounds))):
        shifted = (grouped - center).reshape(-1, heads * width)
        params = _attention_dense_to_hif4(shifted, search_offsets=())
        rebuilt = _attention_dequantize_hif4(params).reshape(-1, heads, width)
        updated = (grouped - rebuilt).mean(dim=0)
        if not torch.isfinite(updated).all():
            return torch.zeros(
                (heads, width), dtype=torch.float32, device="cpu"
            )
        delta = float((updated - center).abs().max())
        center = updated
        if delta <= 1.0e-6:
            break
    return center.detach().to(device="cpu").contiguous()

def _center_attention_k(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    center_mode: int,
    center_value: Optional[torch.Tensor] = None,
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
    if mode == 2:
        center = 0.5 * (
            grouped.amax(dim=0, keepdim=True)
            + grouped.amin(dim=0, keepdim=True)
        )
    elif mode == 4:
        if center_value is None:
            raise ValueError("scale-aware centering requires a center vector")
        center = center_value.detach().to(
            device=grouped.device, dtype=grouped.dtype
        )
        if tuple(int(size) for size in center.shape) != (
            int(num_heads),
            int(head_dim),
        ):
            raise ValueError("Invalid scale-aware center shape")
        center = center.reshape(1, int(num_heads), int(head_dim))
    else:
        raise ValueError("Unsupported attention center mode")
    return (grouped - center).reshape_as(dense)

def _attention_forward(
    q_dense: torch.Tensor,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    """Real attention output for (seq, channels) Q/K/V, evaluator-equivalent."""

    seq = int(q_dense.shape[0])
    group = q_num_heads // kv_num_heads
    q = q_dense.reshape(seq, q_num_heads, head_dim).transpose(0, 1)
    k = (
        k_dense.reshape(seq, kv_num_heads, head_dim)
        .transpose(0, 1)
        .repeat_interleave(group, dim=0)
    )
    v = (
        v_dense.reshape(seq, kv_num_heads, head_dim)
        .transpose(0, 1)
        .repeat_interleave(group, dim=0)
    )
    logits = q @ k.transpose(-1, -2) / math.sqrt(float(head_dim))
    if causal:
        logits = logits + torch.triu(
            torch.full((seq, seq), float("-inf"), device=logits.device), 1
        )
    probs = torch.softmax(logits, dim=-1)
    return (probs @ v).transpose(0, 1).reshape(seq, q_num_heads * head_dim)

def _attention_deployed_mse(
    q_pairs: list,
    k_pairs: list,
    v_hats: list,
    refs: list,
    q_state: dict,
    k_state: dict,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple:
    """Per-sample real attention output MSE (causal, non-causal) through the
    deployed dynamic quantization path, mirroring the evaluator scoring."""

    causal_scores: list = []
    safety_scores: list = []
    for (q_quant, q_scale), (k_quant, k_scale), v_hat, (ref_c, ref_n) in zip(
        q_pairs, k_pairs, v_hats, refs
    ):
        q_hat = _attention_dequantize_hif4(
            hif4_dynamic_quantize_q(
                q_quant, q_scale, q_num_heads, head_dim, q_state
            )
        ).to(torch.float32)
        k_hat = _attention_dequantize_hif4(
            hif4_dynamic_quantize_k(
                k_quant, k_scale, kv_num_heads, head_dim, k_state
            )
        ).to(torch.float32)
        out_c = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
        )
        out_n = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
        )
        causal_scores.append(float((out_c - ref_c).square().mean()))
        safety_scores.append(float((out_n - ref_n).square().mean()))
    return causal_scores, safety_scores

def _a1_gate_passes(
    winner_causal: list,
    winner_safety: list,
    reference_causal: list,
    reference_safety: list,
    safety_tolerance: Optional[float] = None,
) -> bool:
    """终验门判定：A1 winner 在部署路径上相对 B0 proxy winner 无退化。

    ``safety_tolerance`` 允许对 non-causal 安全轨均值采用更严格的容忍
    （默认取 _A1_GATE_WORST_TOLERANCE；旋转等纯方差均衡机制传 0.0 ——
    若机制真实有效则不应使安全轨均值变差）。
    """

    if (
        not winner_causal
        or not winner_safety
        or not reference_causal
        or not reference_safety
    ):
        return False
    if len(winner_causal) != len(reference_causal):
        return False
    for value in (
        winner_causal + winner_safety + reference_causal + reference_safety
    ):
        if not math.isfinite(value):
            return False
    winner_mean = sum(winner_causal) / len(winner_causal)
    reference_mean = sum(reference_causal) / len(reference_causal)
    if winner_mean > max(reference_mean, 1.0e-12) * (
        1.0 - _A1_GATE_MIN_IMPROVEMENT
    ):
        return False
    tolerance = 1.0 + _A1_GATE_WORST_TOLERANCE
    for value, reference in zip(winner_causal, reference_causal):
        if value > max(reference, 1.0e-12) * tolerance:
            return False
    for value, reference in zip(winner_safety, reference_safety):
        if value > max(reference, 1.0e-12) * tolerance:
            return False
    if safety_tolerance is None:
        safety_tolerance = _A1_GATE_WORST_TOLERANCE
    winner_safety_mean = sum(winner_safety) / len(winner_safety)
    reference_safety_mean = sum(reference_safety) / len(reference_safety)
    if winner_safety_mean > max(
        reference_safety_mean, 1.0e-12
    ) * (1.0 + safety_tolerance):
        return False
    return True

def _attention_head_square_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple:
    """每个 KV head 的注意力质量（E[A] 与 E[A^2]）（因果 softmax）。

    V 输出误差被 A 加权：输出 MSE 的对角项由 ``E[A^2]`` 主导，块间
    偏差交叉项（均值误差）由 ``E[A]^2`` 主导。GQA 下每个 KV head 对应
    group 个 Q head，取组内平均。返回 ``(mean_mass, square_mass)``，
    形状均为 ``[kv_num_heads]``。
    """

    seq = int(q.shape[0])
    group = q_num_heads // kv_num_heads
    qh = q.reshape(seq, q_num_heads, head_dim)
    kh = k.reshape(seq, kv_num_heads, head_dim).repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->tsh", qh, kh) / math.sqrt(float(head_dim))
    mask = torch.triu(
        torch.full((seq, seq), float("-inf"), device=scores.device), 1
    ).unsqueeze(-1)
    probs = torch.softmax(scores + mask, dim=1)
    square_mass = probs.square().mean(dim=(0, 1))
    mean_mass = (
        probs.mean(dim=(0, 1)) if _V_IMPORTANCE_CANDIDATES else None
    )
    return (
        (
            None
            if mean_mass is None
            else mean_mass.reshape(kv_num_heads, group).mean(dim=1)
        ),
        square_mass.reshape(kv_num_heads, group).mean(dim=1),
    )

@torch.no_grad()
def _attention_qk_fisher_importance(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    v_samples: Sequence[torch.Tensor],
    d_kv: torch.Tensor,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    center_mode: int,
    center_value: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Estimate output-Fisher channel weights for transformed Q/K.

    For a logit ``l_ts = q_t @ k_s / sqrt(d)``, the attention output
    derivative is ``p_ts (v_s - o_t)``.  Squaring that derivative and
    contracting it with the opposite operand's squared channels gives a
    diagonal Fisher proxy for each transformed Q/K coordinate.  The estimate
    is computed for causal and non-causal masks and averaged; it is returned
    in the *pre-permutation* layout expected by ``_build_qk_states``.

    This helper is calibration-local.  Only its normalized one-dimensional
    vectors are copied into the ordinary Q/K state, so no token/output tensor
    can reach the deployed activation path.
    """

    if not q_samples or len(q_samples) != len(k_samples) or len(q_samples) != len(v_samples):
        return None
    q_heads = int(q_num_heads)
    kv_heads = int(kv_num_heads)
    width = int(head_dim)
    if q_heads <= 0 or kv_heads <= 0 or width <= 0 or q_heads % kv_heads != 0:
        return None
    group = q_heads // kv_heads
    q_channels = q_heads * width
    kv_channels = kv_heads * width
    device = d_kv.device
    d_q = d_kv.to(device=device, dtype=torch.float32).repeat_interleave(group)
    d_k = d_kv.to(device=device, dtype=torch.float32).reciprocal()
    q_order = q_permutation.to(device=device, dtype=torch.int64).reshape(-1)
    k_order = k_permutation.to(device=device, dtype=torch.int64).reshape(-1)
    if int(q_order.numel()) != q_channels or int(k_order.numel()) != kv_channels:
        return None
    q_acc = torch.zeros((q_channels,), dtype=torch.float32, device=device)
    k_acc = torch.zeros((kv_channels,), dtype=torch.float32, device=device)
    count = 0
    for q, k, v in zip(q_samples, k_samples, v_samples):
        try:
            q_raw = q.to(device=device, dtype=torch.float32)
            k_raw = k.to(device=device, dtype=torch.float32)
            v_raw = v.to(device=device, dtype=torch.float32)
            tokens = int(q_raw.shape[0])
            if (
                q_raw.ndim != 2
                or k_raw.ndim != 2
                or v_raw.ndim != 2
                or int(q_raw.shape[1]) != q_channels
                or int(k_raw.shape[1]) != kv_channels
                or int(v_raw.shape[1]) != kv_channels
                or int(k_raw.shape[0]) != tokens
                or int(v_raw.shape[0]) != tokens
                or tokens <= 0
            ):
                continue
            q_transformed = (q_raw * d_q.reshape(1, -1)).index_select(
                -1, q_order
            ).reshape(tokens, q_heads, width).transpose(0, 1)
            k_centered = _center_attention_k(
                k_raw, kv_heads, width, int(center_mode), center_value
            )
            k_transformed = (k_centered * d_k.reshape(1, -1)).index_select(
                -1, k_order
            ).reshape(tokens, kv_heads, width).transpose(0, 1)
            v_heads = v_raw.reshape(tokens, kv_heads, width).transpose(0, 1)
            k_for_q = k_transformed.repeat_interleave(group, dim=0)
            v_for_q = v_heads.repeat_interleave(group, dim=0)
            logits = q_transformed.matmul(k_for_q.transpose(-1, -2)) / math.sqrt(
                float(width)
            )
            for causal in (True, False):
                if causal:
                    logits_used = logits + torch.triu(
                        torch.full(
                            (tokens, tokens),
                            float("-inf"),
                            dtype=logits.dtype,
                            device=device,
                        ),
                        1,
                    )
                else:
                    logits_used = logits
                probs = torch.softmax(logits_used, dim=-1)
                output = probs.matmul(v_for_q)
                value_delta = v_for_q[:, None, :, :] - output[:, :, None, :]
                logit_fisher = probs.square() * value_delta.square().sum(dim=-1)
                q_local = torch.einsum(
                    "hts,hsd->hd", logit_fisher, k_for_q.square()
                )
                k_local_q = torch.einsum(
                    "hts,htd->hd", logit_fisher, q_transformed.square()
                )
                k_local = k_local_q.reshape(kv_heads, group, width).mean(dim=1)
                q_acc.add_(q_local.reshape(-1))
                k_acc.add_(k_local.reshape(-1))
                count += 1
        except (RuntimeError, ValueError, TypeError):
            continue
    if count <= 0:
        return None
    q_acc = torch.nan_to_num(q_acc / float(count), nan=0.0, posinf=0.0, neginf=0.0)
    k_acc = torch.nan_to_num(k_acc / float(count), nan=0.0, posinf=0.0, neginf=0.0)
    # Map transformed/post-permutation scores back to the raw input layout.
    q_raw_importance = torch.zeros_like(q_acc)
    k_raw_importance = torch.zeros_like(k_acc)
    q_raw_importance.index_copy_(0, q_order, q_acc)
    k_raw_importance.index_copy_(0, k_order, k_acc)
    if not bool(torch.isfinite(q_raw_importance).all()) or not bool(
        torch.isfinite(k_raw_importance).all()
    ):
        return None
    return q_raw_importance, k_raw_importance

def _attention_e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    """Encode non-negative FP32 values into finite unsigned E6M2 codes.

    Codes 0..254 are finite and monotonic.  Code 255 is NaN and is never
    produced.  Round-to-nearest-even is inherited from ``torch.round``.
    """

    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_attention_E6M2_MIN,
        posinf=_attention_E6M2_MAX,
        neginf=_attention_E6M2_MIN,
    ).clamp(min=_attention_E6M2_MIN, max=_attention_E6M2_MAX)

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

def _attention_e6m2_decode(code: torch.Tensor) -> torch.Tensor:
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
        amax.to(torch.bfloat16) * _attention_BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = _attention_e6m2_encode_nearest(high_precision_scale)
    return code, _attention_e6m2_decode(code)

def _source_scale_code_candidates(
    source_scale_float: Optional[torch.Tensor],
    dense_shape: Sequence[int],
    channels: int,
) -> Optional[torch.Tensor]:
    """Build per-64-block E6M2 scale proposals from the NVFP4 source.

    NVFP4 stores one E4M3 scale per 16-value block.  Four such blocks map to
    one HiF4 64-group.  In the log domain, the median, 75th percentile and
    maximum are stable summaries of the source dynamic range.  NVFP4's
    ``amax / 6`` scale is converted to the HiF4 ``amax / 7`` convention by
    multiplying by ``6/7`` and then encoded with the official E6M2 nearest
    encoder.  The result is only a candidate list; ``_solve_exact_hierarchy``
    still decides the legal hierarchy and the ordinary amax/offset candidate
    remains available.

    Returns ``[..., blocks, num_stats]`` codes, or ``None`` when the source
    shape is unavailable/incompatible.  No source tensor is retained in the
    returned activation state.
    """

    if source_scale_float is None:
        return None
    try:
        prefix = tuple(int(v) for v in dense_shape[:-1])
        width = int(channels)
        if width <= 0 or width % _HIF4_BLOCK_SIZE != 0:
            return None
        expected = prefix + (width // _NVFP4_BLOCK_SIZE,)
        if tuple(int(v) for v in source_scale_float.shape) != expected:
            return None
        blocks = width // _HIF4_BLOCK_SIZE
        grouped = source_scale_float.detach().to(torch.float32).reshape(
            *prefix, blocks, _HIF4_BLOCK_SIZE // _NVFP4_BLOCK_SIZE
        )
        grouped = torch.nan_to_num(
            grouped,
            nan=_attention_E6M2_MIN,
            posinf=_attention_E6M2_MAX,
            neginf=_attention_E6M2_MIN,
        ).clamp_min(_attention_E6M2_MIN)
        log2_scale = torch.log2(grouped)
        stats = []
        if "median" in _ACTIVATION_SOURCE_SCALE_STATS:
            stats.append(torch.median(log2_scale, dim=-1).values)
        if "q75" in _ACTIVATION_SOURCE_SCALE_STATS:
            # Four source blocks make the percentile inexpensive and
            # deterministic; interpolation is acceptable because the result
            # is rounded to the finite E6M2 lattice immediately afterwards.
            stats.append(torch.quantile(log2_scale, 0.75, dim=-1))
        if "max" in _ACTIVATION_SOURCE_SCALE_STATS:
            stats.append(log2_scale.amax(dim=-1))
        if not stats:
            return None
        values = torch.stack(stats, dim=-1).exp2()
        values = values * float(_ACTIVATION_SOURCE_SCALE_TO_HIF4_AMAX)
        return _attention_e6m2_encode_nearest(values)
    except (RuntimeError, ValueError, TypeError):
        return None

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
    sign: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly solve lv2/lv3 for fixed scales using three loss tables.

    Args:
        x_abs: ``[num_blocks, 8, 2, 4]`` absolute values.
        scale: ``[num_blocks]`` finite E6M2 values.
        importance: optional tensor with the same shape as ``x_abs``.
        sign: ``[num_blocks, 8, 2, 4]`` signs (required with ``group_gram``).
        group_gram: ``[num_blocks, 8, 2, 4, 4]`` per-group quadratic weights;
            when given, the loss is the quadratic form ``delta^T G delta``
            instead of the diagonal per-channel weighted squares.
    """

    losses: list[torch.Tensor] = []
    mantissas: list[torch.Tensor] = []

    for total_exponent in (0, 1, 2):
        local_scale = scale[..., None, None, None] * float(1 << total_exponent)
        mant_code = torch.round(x_abs * (4.0 / local_scale)).clamp_(0.0, 7.0)
        mantissa = mant_code * 0.25
        if group_gram is not None:
            delta = sign * (x_abs - mantissa * local_scale)
            losses.append(
                torch.einsum(
                    "...abi,...abij,...abj->...ab", delta, group_gram, delta
                )
            )
        else:
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

    # [..., 3, 4]，指数维固定在倒数第二维（批量时随输入维度自然后移）。
    mantissa_stack = torch.stack(mantissas, dim=-2)
    gather_index = total_exponent[..., None, None].expand(
        *total_exponent.shape, 1, 4
    )
    gather_dim = mantissa_stack.ndim - 2  # 指数维：4D 输入为 3，批量时随维度后移
    mantissa = torch.gather(
        mantissa_stack, gather_dim, gather_index
    ).squeeze(-2)

    scale_lv2 = 1.0 + e2.to(torch.float32)
    scale_lv3 = 1.0 + e3.to(torch.float32)
    return block_loss, scale_lv2, scale_lv3, mantissa

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

def _attention_dense_to_hif4(
    dense: torch.Tensor,
    *,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    source_scale_float: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor into valid HiF4 parameters."""

    if group_gram is not None:
        expected_gram_shape = dense.shape[:-1] + (
            dense.shape[-1] // 64,
            8,
            2,
            4,
            4,
        )
        if tuple(group_gram.shape) != tuple(expected_gram_shape):
            raise ValueError(
                f"group_gram shape {tuple(group_gram.shape)} does not match "
                f"expected {expected_gram_shape}"
            )

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
        posinf=_attention_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_attention_E6M2_MAX * _HIF4_MAX_INNER,
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
    source_codes = _source_scale_code_candidates(
        source_scale_float, x.shape, channels
    )
    if source_codes is not None and source_codes.device != x.device:
        source_codes = source_codes.to(device=x.device)
    refine_ratio = max(0.0, min(float(max_refine_ratio), 1.0))
    if refine_ratio <= 0.0 or len(offsets) == 0:
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
    if group_gram is not None:
        delta = sign * (x_abs - mantissa * denominator)
        weighted_error = torch.einsum(
            "...abi,...abij,...abj->...ab", delta, group_gram, delta
        )
        weighted_energy = x_abs.square()
        importance_view = None
    elif channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
        weighted_energy = x_abs.square()
        importance_view = None
    else:
        importance_view = channel_importance.reshape(
            *([1] * len(prefix)), blocks, 8, 2, 4
        )
        weighted_error = (x_abs - mantissa * denominator).square() * importance_view
        weighted_energy = x_abs.square() * importance_view

    loss_reduce_dims = (-1, -2) if group_gram is not None else (-1, -2, -3)
    standard_loss = weighted_error.sum(dim=loss_reduce_dims)
    energy = weighted_energy.sum(dim=(-1, -2, -3))
    normalized_error = standard_loss / (energy + _attention_EPS)

    flat_norm = normalized_error.reshape(-1)
    flat_loss = standard_loss.reshape(-1)
    hard_mask = flat_norm > float(error_threshold)
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

    total_blocks = int(flat_norm.numel())
    refine_cap = max(1, int(math.ceil(total_blocks * refine_ratio)))
    if max_refine_blocks is not None:
        refine_cap = min(refine_cap, max(1, int(max_refine_blocks)))
    if int(hard_indices.numel()) > refine_cap:
        if _REFINE_RANK_BY_ABSOLUTE:
            # Rank by the block's absolute (importance-weighted) reconstruction
            # error, i.e. its true contribution to the output MSE, instead of
            # the normalized error: under a fixed refinement budget this
            # greedily maximizes the total MSE reduction (and hence the
            # competition score).
            hard_indices = torch.topk(flat_loss, k=refine_cap, largest=True).indices
        else:
            hard_indices = torch.topk(flat_norm, k=refine_cap, largest=True).indices

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
    sign_hard = sign.reshape(-1, 8, 2, 4).index_select(0, hard_indices)
    group_gram_hard = (
        None
        if group_gram is None
        else group_gram.reshape(-1, 8, 2, 4, 4).index_select(0, hard_indices)
    )
    best_offset = torch.zeros(
        int(hard_indices.numel()), dtype=torch.int64, device=x.device
    )

    if channel_importance is None:
        importance_hard = None
    else:
        block_importance = channel_importance.reshape(blocks, 8, 2, 4)
        channel_block_ids = torch.remainder(hard_indices, blocks)
        importance_hard = block_importance.index_select(0, channel_block_ids)

    # 全部 scale proposal 一次性批量求解：把 [N] 块沿 candidate 维展开
    # 成 [K, N]，一次精确求解后按块取 argmin。标准 code（offset 0）必须
    # 保留在候选里：阈值式 lv2/lv3 与精确解不等价（真实数据约半数块
    # 有更低损失），offset 0 会把 hard 块的 lv2/lv3 升级为精确解。
    offset_values = torch.tensor(
        [int(o) for o in offsets], dtype=torch.int64, device=x.device
    )
    expanded_codes = (
        standard_code_hard.to(torch.int64).unsqueeze(0)
        + offset_values.unsqueeze(1)
    ).clamp(min=0, max=254)
    candidate_markers = offset_values
    if source_codes is not None:
        # ``hard_indices`` is flattened over prefix rows and 64-blocks.  Pick
        # only the source proposals for those blocks; no dense all-block
        # candidate tensor is materialized.
        source_flat = source_codes.reshape(-1, blocks, source_codes.shape[-1])
        row_ids = torch.div(hard_indices, blocks, rounding_mode="floor")
        block_ids = torch.remainder(hard_indices, blocks)
        source_hard = source_flat[row_ids, block_ids].transpose(0, 1)
        expanded_codes = torch.cat((expanded_codes, source_hard), dim=0)
        # Keep source proposals out of the offset-edge extension below.
        source_markers = torch.full(
            (int(source_hard.shape[0]),),
            _ACTIVATION_SOURCE_SCALE_MARKER,
            dtype=torch.int64,
            device=x.device,
        )
        candidate_markers = torch.cat((candidate_markers, source_markers), dim=0)
    candidate_scales = _attention_e6m2_decode(expanded_codes)
    num_candidates = int(expanded_codes.shape[0])
    x_expanded = x_hard.unsqueeze(0).expand(
        num_candidates, -1, -1, -1, -1
    )
    sign_expanded = sign_hard.unsqueeze(0).expand(
        num_candidates, -1, -1, -1, -1
    )
    importance_expanded = (
        None
        if importance_hard is None
        else importance_hard.unsqueeze(0).expand(
            num_candidates, -1, -1, -1, -1
        )
    )
    gram_expanded = (
        None
        if group_gram_hard is None
        else group_gram_hard.unsqueeze(0).expand(
            num_candidates, -1, -1, -1, -1, -1
        )
    )
    all_losses, all_lv2, all_lv3, all_mantissa = _solve_exact_hierarchy(
        x_expanded,
        candidate_scales,
        importance_expanded,
        sign_expanded,
        gram_expanded,
    )
    best_k = all_losses.argmin(dim=0)
    hard_arange = torch.arange(
        int(hard_indices.numel()), device=x.device
    )
    candidate_loss = all_losses[best_k, hard_arange]
    candidate_scale = candidate_scales[best_k, hard_arange]
    candidate_lv2 = all_lv2[best_k, hard_arange]
    candidate_lv3 = all_lv3[best_k, hard_arange]
    candidate_mantissa = all_mantissa[best_k, hard_arange]

    improve = candidate_loss < best_loss
    best_loss = torch.where(improve, candidate_loss, best_loss)
    best_scale = torch.where(improve, candidate_scale, best_scale)
    best_lv2 = torch.where(improve[:, None], candidate_lv2, best_lv2)
    best_lv3 = torch.where(improve[:, None, None], candidate_lv3, best_lv3)
    best_mantissa = torch.where(
        improve[:, None, None, None], candidate_mantissa, best_mantissa
    )
    best_offset = torch.where(improve, candidate_markers[best_k], best_offset)

    if _REFINE_EDGE_EXTENSION and len(offsets) > 1:
        lo_offset = int(offsets[0])
        hi_offset = int(offsets[-1])

        def extend_edge(edge: int, direction: int) -> None:
            nonlocal best_loss, best_scale
            nonlocal best_lv2, best_lv3, best_mantissa, best_offset
            mask = best_offset == edge
            for _ in range(_REFINE_EDGE_EXTEND_STEPS):
                if not bool(mask.any()):
                    return
                edge_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                target = edge + direction
                if target < -254 or target > 254:
                    return
                edge_code = (
                    standard_code_hard.index_select(0, edge_indices).to(
                        torch.int64
                    )
                    + target
                ).clamp(min=0, max=254)
                edge_scale = _attention_e6m2_decode(edge_code)
                edge_importance = (
                    None
                    if importance_hard is None
                    else importance_hard.index_select(0, edge_indices)
                )
                edge_loss, edge_lv2, edge_lv3, edge_mantissa = (
                    _solve_exact_hierarchy(
                        x_hard.index_select(0, edge_indices),
                        edge_scale,
                        edge_importance,
                        sign_hard.index_select(0, edge_indices),
                        (
                            None
                            if group_gram_hard is None
                            else group_gram_hard.index_select(0, edge_indices)
                        ),
                    )
                )
                improve = edge_loss < best_loss.index_select(0, edge_indices)
                improved = edge_indices[improve]
                if int(improved.numel()) == 0:
                    return
                best_loss.index_copy_(
                    0, improved, edge_loss[improve]
                )
                best_scale.index_copy_(0, improved, edge_scale[improve])
                best_lv2.index_copy_(0, improved, edge_lv2[improve])
                best_lv3.index_copy_(0, improved, edge_lv3[improve])
                best_mantissa.index_copy_(
                    0, improved, edge_mantissa[improve]
                )
                best_offset.index_copy_(
                    0,
                    improved,
                    torch.full_like(best_offset[improved], target),
                )
                edge = target
                mask = best_offset == target

        extend_edge(hi_offset, +1)
        extend_edge(lo_offset, -1)

    if _L1_DATA_DRIVEN_SCALE:
        # L1 数据驱动 scale 候选：锚定当前五字段 winner（best_* 来自
        # offset 搜索 + 边缘扩展），生成每块独立候选 code，全部经
        # _solve_exact_hierarchy 精确解后逐块 improve-mask 回退。
        if importance_hard is not None:
            l1_weights = importance_hard
        elif group_gram_hard is not None:
            l1_weights = torch.diagonal(
                group_gram_hard, dim1=-2, dim2=-1
            )
        else:
            l1_weights = None
        hierarchy = best_lv2[:, :, None, None] * best_lv3[:, :, :, None]
        model = hierarchy * best_mantissa
        if l1_weights is None:
            numerator = (model * x_hard).sum(dim=(1, 2, 3))
            denominator = (model * model).sum(dim=(1, 2, 3))
        else:
            numerator = (l1_weights * model * x_hard).sum(dim=(1, 2, 3))
            denominator = (l1_weights * model * model).sum(dim=(1, 2, 3))
        ls_scale = numerator / denominator.clamp_min(_attention_EPS)
        flat_abs = x_hard.reshape(int(x_hard.shape[0]), -1)
        base_codes = [_attention_e6m2_encode_nearest(ls_scale)]
        for trim_quantile in _L1_TRIM_QUANTILES:
            trim_scale = torch.quantile(
                flat_abs, float(trim_quantile), dim=1
            ) * (4.0 / 7.0)
            base_codes.append(_attention_e6m2_encode_nearest(trim_scale))
        code_deltas = torch.tensor(
            _L1_ADJACENT_CODE_DELTAS, dtype=torch.int64, device=x.device
        )
        candidate_codes = torch.stack(
            [
                (
                    base_code.to(torch.int64).unsqueeze(0)
                    + code_deltas.unsqueeze(1)
                ).clamp(min=0, max=254)
                for base_code in base_codes
            ],
            dim=0,
        ).reshape(-1, int(x_hard.shape[0]))
        l1_scales = _attention_e6m2_decode(candidate_codes)
        num_l1 = int(candidate_codes.shape[0])
        l1_losses, l1_lv2, l1_lv3, l1_mantissa = _solve_exact_hierarchy(
            x_hard.unsqueeze(0).expand(num_l1, -1, -1, -1, -1),
            l1_scales,
            (
                None
                if importance_hard is None
                else importance_hard.unsqueeze(0).expand(
                    num_l1, -1, -1, -1, -1
                )
            ),
            sign_hard.unsqueeze(0).expand(num_l1, -1, -1, -1, -1),
            (
                None
                if group_gram_hard is None
                else group_gram_hard.unsqueeze(0).expand(
                    num_l1, -1, -1, -1, -1, -1
                )
            ),
        )
        l1_best = l1_losses.argmin(dim=0)
        l1_loss = l1_losses[l1_best, hard_arange]
        l1_scale = l1_scales[l1_best, hard_arange]
        l1_lv2_best = l1_lv2[l1_best, hard_arange]
        l1_lv3_best = l1_lv3[l1_best, hard_arange]
        l1_mantissa_best = l1_mantissa[l1_best, hard_arange]
        improve_l1 = l1_loss < best_loss
        best_loss = torch.where(improve_l1, l1_loss, best_loss)
        best_scale = torch.where(improve_l1, l1_scale, best_scale)
        best_lv2 = torch.where(improve_l1[:, None], l1_lv2_best, best_lv2)
        best_lv3 = torch.where(
            improve_l1[:, None, None], l1_lv3_best, best_lv3
        )
        best_mantissa = torch.where(
            improve_l1[:, None, None, None], l1_mantissa_best, best_mantissa
        )

    margin = max(0.0, min(float(accept_margin), 0.99))
    # Source proposals are scored with the same operand-local quadratic
    # metric (when available) but can be useful below the historical 2%
    # reconstruction margin.  Let a strictly improving source winner pass;
    # ordinary amax/offset candidates keep the incumbent margin semantics.
    source_winner = best_offset >= int(_ACTIVATION_SOURCE_SCALE_MARKER)
    source_accept = best_loss < (standard_loss_hard - _attention_EPS)
    regular_accept = best_loss <= ((1.0 - margin) * standard_loss_hard)
    accept = torch.where(source_winner, source_accept, regular_accept)
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
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
    cat_transform: Optional[torch.Tensor] = None,
    center_mode: int = 0,
    center_num_heads: Optional[int] = None,
    center_head_dim: Optional[int] = None,
    center_value: Optional[torch.Tensor] = None,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    group_gram8: Optional[torch.Tensor] = None,
    group_gram16: Optional[torch.Tensor] = None,
    group_gram64: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
    source_scale_proposal: bool = False,
    attention_rotation: Optional[torch.Tensor] = None,
    rotation_num_heads: Optional[int] = None,
    attention_rotation_block: Optional[int] = None,
    attention_block_signs: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    dense = _attention_dequantize_nvfp4_float32(quant_float, scale_float)
    channels = int(dense.shape[-1])
    if int(center_mode) != 0:
        if center_num_heads is None or center_head_dim is None:
            raise ValueError("Attention centering requires head metadata")
        dense = _center_attention_k(
            dense,
            int(center_num_heads),
            int(center_head_dim),
            int(center_mode),
            center_value,
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
    if attention_rotation is not None:
        if rotation_num_heads is None:
            raise ValueError("Attention rotation requires head count")
        signs = attention_rotation.detach().to(device="cpu")
        head_dim = int(signs.shape[-1])
        if int(rotation_num_heads) * head_dim != channels:
            raise ValueError(
                "Attention rotation does not match tensor width"
            )
        if head_dim < 4 or (head_dim & (head_dim - 1)) != 0:
            raise ValueError("Attention rotation requires power-of-two head_dim")
        dense = _apply_attention_rotation(
            dense,
            int(rotation_num_heads),
            head_dim,
            signs,
            attention_rotation_block,
        )
    if int(block_smooth_size) != 0:
        if attention_block_signs is not None:
            if rotation_num_heads is None:
                raise ValueError("Attention block smoothing requires head count")
            block_signs = attention_block_signs.detach().to(device="cpu")
            if block_signs.ndim != 2 or int(block_signs.shape[1]) <= 0:
                raise ValueError("Attention block signs have invalid shape")
            dense = _apply_attention_rotation(
                dense,
                int(rotation_num_heads),
                int(block_signs.shape[1]),
                block_signs,
                int(block_smooth_size),
            )
        else:
            dense = _block_hadamard_transform(
                dense, int(block_smooth_size), int(block_smooth_seed)
            )
    if cat_transform is not None:
        dense = _apply_cat64_rows(dense, cat_transform, inverse=False)
    gram = None
    if group_gram is not None:
        gram = group_gram.detach().to(
            device=dense.device, dtype=torch.float32
        )
        expected = (channels // 4, 4, 4)
        if tuple(gram.shape) != expected:
            raise ValueError(
                f"group_gram shape {tuple(gram.shape)} does not match "
                f"expected {expected}"
            )
        blocks = channels // _HIF4_BLOCK_SIZE
        gram = gram.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
            int(dense.shape[0]), blocks, 8, 2, 4, 4
        )
    refine_importance = importance
    if _ACTIVATION_SAMPLE_IMPORTANCE and dense.ndim == 2:
        refine_importance = torch.sqrt(
            dense.square().mean(dim=0).clamp_min(_attention_EPS)
        )
    params = _attention_dense_to_hif4(
        dense,
        importance=refine_importance,
        group_gram=gram,
        source_scale_float=(scale_float if source_scale_proposal else None),
        search_offsets=search_offsets,
        error_threshold=error_threshold,
        accept_margin=accept_margin,
        max_refine_ratio=max_refine_ratio,
        max_refine_blocks=max_refine_blocks,
    )
    if group_gram8 is not None:
        gram8 = group_gram8.detach().to(
            device=dense.device, dtype=torch.float32
        )
        params = _refine_weight_groups8(
            dense,
            params,
            gram8,
            max_ratio=_ACTIVATION_QUADRATIC8_MAX_RATIO,
            max_groups=_ACTIVATION_QUADRATIC8_MAX_GROUPS,
            sweeps=_ACTIVATION_QUADRATIC8_SWEEPS,
            accept_margin=_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN,
        )
    if (
        _ACTIVATION_QUADRATIC16
        and group_gram16 is not None
        and channels <= _ACTIVATION_QUADRATIC16_MAX_FEATURES
    ):
        gram16 = group_gram16.detach().to(
            device=dense.device, dtype=torch.float32
        )
        params = _refine_activation_groups16(dense, params, gram16)
    if (
        _ACTIVATION_GRAM64
        and group_gram64 is not None
        and channels <= _ACTIVATION_GRAM64_MAX_FEATURES
    ):
        gram64 = group_gram64.detach().to(
            device=dense.device, dtype=torch.float32
        )
        if (
            _ACTIVATION_GRAM64_HIERARCHY
            and channels > _ACTIVATION_QUADRATIC_MAX_FEATURES
        ):
            params = _refine_activation_hierarchy64(
                dense,
                params,
                gram64,
                max_ratio=_ACTIVATION_GRAM64_MAX_RATIO,
                max_blocks=_ACTIVATION_GRAM64_MAX_BLOCKS,
                offsets=_ACTIVATION_GRAM64_HIERARCHY_OFFSETS,
                accept_margin=_ACTIVATION_GRAM64_ACCEPT_MARGIN,
            )
        params = _refine_activation_blocks64(dense, params, gram64)
    return params

def _attention_dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    dense = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return dense.flatten(start_dim=-4, end_dim=-1)

def _hadamard_matrix(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix (size 4/8/16)."""

    n = int(size)
    if n not in _BLOCK_SMOOTH_ALLOWED_SIZES:
        raise ValueError(
            "block_smooth_size must be one of "
            f"{_BLOCK_SMOOTH_ALLOWED_SIZES}, got {n}"
        )
    return _hadamard_matrix_unchecked(n, device, dtype)

def _hadamard_matrix_unchecked(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix for power-of-two sizes."""

    n = int(size)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while int(h.shape[0]) < n:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return h * (1.0 / math.sqrt(float(n)))

def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Butterfly fast Walsh-Hadamard transform along the last dimension.

    Equivalent to ``x @ H_n`` for the normalized Sylvester Hadamard matrix
    ``H_n`` (which is symmetric), but never materializes the dense matrix.
    The input is never modified in place; float32/bfloat16 and CPU/CUDA all
    run the same deterministic op sequence.
    """

    n = int(x.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT width must be a power of two, got {n}")
    lead = tuple(x.shape[:-1])
    y = x.reshape(-1, n).clone()
    width = 1
    while width < n:
        y = y.reshape(-1, n // (2 * width), 2, width)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        width *= 2
    y = y * (1.0 / math.sqrt(float(n)))
    return y.reshape(*lead, n)

def _linear_r64_signs(
    channels: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic per-channel sign vector shared by both Linear sides."""

    indices = torch.arange(int(channels), dtype=torch.int64, device=device)
    bits = (
        indices * 1_103_515_245 + int(seed) * 214_013 + 12_345
    ).bitwise_and(1 << 30)
    return torch.where(bits == 0, 1.0, -1.0).to(dtype=dtype)

def _apply_linear_r64(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply the signed orthogonal R64 incoherence transform.

    ``R64 = diag(signs) · H64`` applied on the last dimension via the
    butterfly FWHT (no dense [64, 64] matrix is ever built).  The transform
    is exactly orthogonal, so the inverse applies the FWHT and multiplies
    the same signs back.
    """

    channels = int(x.shape[-1])
    if channels % _LINEAR_R64_BLOCK != 0:
        raise ValueError(
            f"Feature width {channels} is not divisible by "
            f"{_LINEAR_R64_BLOCK}"
        )
    signs = _linear_r64_signs(channels, seed, x.device, x.dtype)
    grouped = x.reshape(
        *x.shape[:-1], channels // _LINEAR_R64_BLOCK, _LINEAR_R64_BLOCK
    )
    grouped = grouped * signs.reshape(
        channels // _LINEAR_R64_BLOCK, _LINEAR_R64_BLOCK
    )
    return _fwht_last_dim(grouped).reshape(*x.shape)

def _block_hadamard_transform(
    dense: torch.Tensor,
    block_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Apply a deterministic signed orthogonal transform to feature blocks.

    The signs avoid concentrating positively correlated channels in the DC
    Hadamard coefficient.  They are derived from the absolute feature index,
    so calibration and dynamic quantization only share ``block_size`` and a
    small integer ``seed``.  Size 64 routes through the butterfly FWHT; the
    smaller sizes keep the dense matrix product.
    """

    size = int(block_size)
    if size == 0:
        return dense
    if size == _LINEAR_R64_BLOCK:
        return _apply_linear_r64(dense, int(seed))
    channels = int(dense.shape[-1])
    if channels % size != 0:
        raise ValueError(
            f"Feature width {channels} is not divisible by block size {size}"
        )
    signs = _linear_r64_signs(channels, seed, dense.device, dense.dtype)
    grouped = dense.reshape(*dense.shape[:-1], channels // size, size)
    grouped = grouped * signs.reshape(channels // size, size)
    h = _hadamard_matrix(size, dense.device, dense.dtype)
    return torch.matmul(grouped, h).reshape_as(dense)

def _block_average(moment: torch.Tensor, size: int) -> torch.Tensor:
    """Broadcast each block's mean importance after an orthogonal mixing."""

    if int(size) <= 0:
        return moment
    width = int(moment.numel())
    if width % int(size) != 0:
        raise ValueError("Importance width is not divisible by block size")
    return moment.reshape(-1, int(size)).mean(dim=-1, keepdim=True).expand(
        -1, int(size)
    ).reshape(-1)

def _apply_cat64_rows(
    dense: torch.Tensor,
    transforms: Optional[torch.Tensor],
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a block-diagonal CAT transform to row-major data."""

    if transforms is None:
        return dense
    channels = int(dense.shape[-1])
    block = int(_CAT64_BLOCK_SIZE)
    if channels % block != 0:
        raise ValueError("CAT-64 transform width does not divide channels")
    matrix = transforms.detach().to(device=dense.device, dtype=torch.float32)
    expected = (channels // block, block, block)
    if tuple(matrix.shape) != expected:
        raise ValueError(
            f"CAT-64 transform shape {tuple(matrix.shape)} != {expected}"
        )
    if inverse:
        matrix = torch.linalg.inv(matrix)
    grouped = dense.to(dtype=torch.float32).reshape(
        *dense.shape[:-1], channels // block, block
    )
    transformed = torch.einsum("...bi,bji->...bj", grouped, matrix)
    return transformed.reshape_as(grouped).reshape_as(dense)

def _attention_rotation_signs(
    kv_num_heads: int,
    head_dim: int,
    seed: int,
) -> torch.Tensor:
    """Deterministic per-(KV group, channel) signs shared by Q and K.

    Signs derive from the flat index within the [kv_num_heads, head_dim]
    layout using the same integer hash as the Linear block smoothing, so
    calibration and dynamic quantization agree without extra state.
    """

    index = torch.arange(
        kv_num_heads * head_dim, dtype=torch.int64, device="cpu"
    )
    bits = (
        index * 1_103_515_245 + int(seed) * 214_013 + 12_345
    ).bitwise_and(1 << 30)
    signs = torch.where(bits == 0, 1.0, -1.0)
    return signs.reshape(kv_num_heads, head_dim)

def _apply_attention_rotation(
    dense: torch.Tensor,
    rotation_num_heads: int,
    head_dim: int,
    signs: torch.Tensor,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply the group-aligned signed Hadamard rotation to head blocks.

    ``signs`` has shape [kv_num_heads, head_dim].  For K (or MHA) the heads
    map one-to-one; for Q the heads of the same KV group share their K
    rotation, which keeps Q·K dot products exactly invariant.
    """

    kv_num_heads = int(signs.shape[0])
    group_size = int(rotation_num_heads) // kv_num_heads
    if group_size * kv_num_heads != int(rotation_num_heads):
        raise ValueError("Rotation head count is not a GQA multiple")
    signs_f = signs.detach().to(
        device=dense.device, dtype=dense.dtype
    ).reshape(kv_num_heads, head_dim)
    if group_size > 1:
        signs_f = signs_f.repeat_interleave(group_size, dim=0)
    x = dense.reshape(*dense.shape[:-1], int(rotation_num_heads), head_dim)
    x = x * signs_f.reshape(int(rotation_num_heads), head_dim)
    requested_block = int(
        _ATTN_H64_BLOCK if block_size is None else block_size
    )
    block = requested_block if requested_block > 0 and head_dim % requested_block == 0 else head_dim
    blocks = head_dim // block
    h = _hadamard_matrix_unchecked(block, dense.device, dense.dtype)
    x = x.reshape(*x.shape[:-2], int(rotation_num_heads), blocks, block)
    x = torch.matmul(x, h)
    return x.reshape(*dense.shape)

def _cpu_state_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        x.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()

def _smooth_qk_scale(
    q_peak: torch.Tensor,
    k_peak: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (k_peak + _attention_EPS).pow(alpha) / (q_peak + _attention_EPS).pow(1.0 - alpha)
    return torch.nan_to_num(
        d, nan=1.0, posinf=_QK_SMOOTH_MAX, neginf=_QK_SMOOTH_MIN
    ).clamp(min=_QK_SMOOTH_MIN, max=_QK_SMOOTH_MAX)

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
    a1_context: Optional[dict] = None,
    center_value: Optional[torch.Tensor] = None,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
    use_final_quantizer: bool = False,
) -> tuple[float, tuple[float, ...]]:
    """Q/K quantization proxy with GQA-aligned equivalent transforms.

    With ``a1_context`` the score becomes the real attention output error
    (A1): causal MSE is the primary selector, non-causal MSE interleaves into
    the per-case tuple so ``_candidate_is_safe`` protects both masks.  The
    V quantization is held fixed across candidates, isolating the Q/K
    transform choice exactly like the proxy did.
    """

    group_size = q_num_heads // kv_num_heads
    d_q = d_kv.repeat_interleave(group_size, dim=0)
    d_k = d_kv.reciprocal()
    q_order = q_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)
    k_order = k_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)
    block_signs = None
    if int(block_smooth_size) != 0:
        block_signs = _attention_rotation_signs(
            kv_num_heads, head_dim, int(block_smooth_seed)
        )

    if a1_context is not None:
        causal_scores: list[float] = []
        safety_scores: list[float] = []
        identity_cases = a1_context["identity"]
        final_q_importance = None
        final_k_importance = None
        if use_final_quantizer:
            q_second_kv = q_second_moment.reshape(
                kv_num_heads, group_size, head_dim
            ).mean(dim=1)
            h_k = k_effective_second_moment * d_k.square()
            h_q = q_second_kv * d_kv.square()
            final_q_importance = _normalize_importance(
                h_k.repeat_interleave(group_size, dim=0)
                .reshape(-1)
                .index_select(0, q_order),
                q_num_heads * head_dim,
            )
            final_k_importance = _normalize_importance(
                h_q.reshape(-1).index_select(0, k_order),
                kv_num_heads * head_dim,
            )
            if (
                final_q_importance is None
                or final_k_importance is None
            ):
                return 1.0e30, (1.0e30,)
            if int(block_smooth_size) != 0:
                final_q_importance = _block_average(
                    final_q_importance, int(block_smooth_size)
                )
                final_k_importance = _block_average(
                    final_k_importance, int(block_smooth_size)
                )
        for index, (q_full, k_full, v_hat, (ref_c, ref_n)) in enumerate(
            zip(
                a1_context["q_full"],
                a1_context["k_full"],
                a1_context["v_hat"],
                a1_context["refs"],
            )
        ):
            q_smooth = (q_full * d_q.reshape(1, -1)).index_select(-1, q_order)
            k_centered = _center_attention_k(
                k_full, kv_num_heads, head_dim, center_mode, center_value
            )
            k_smooth = (k_centered * d_k.reshape(1, -1)).index_select(
                -1, k_order
            )
            if int(block_smooth_size) != 0:
                q_smooth = _apply_attention_rotation(
                    q_smooth,
                    q_num_heads,
                    head_dim,
                    block_signs,
                    int(block_smooth_size),
                )
                k_smooth = _apply_attention_rotation(
                    k_smooth,
                    kv_num_heads,
                    head_dim,
                    block_signs,
                    int(block_smooth_size),
                )
            if use_final_quantizer:
                q_params = _attention_dense_to_hif4(
                    q_smooth,
                    importance=final_q_importance,
                    search_offsets=_DYNAMIC_OFFSETS,
                    error_threshold=_ATTN_REFINE_ERROR_THRESHOLD,
                    accept_margin=_Q_REFINE_ACCEPT_MARGIN,
                    max_refine_ratio=_ATTN_BLOCK_SMOOTH_REFINE_RATIO,
                    max_refine_blocks=_ATTN_BLOCK_SMOOTH_REFINE_BLOCKS,
                )
                k_params = _attention_dense_to_hif4(
                    k_smooth,
                    importance=final_k_importance,
                    search_offsets=_DYNAMIC_OFFSETS,
                    error_threshold=_ATTN_REFINE_ERROR_THRESHOLD,
                    accept_margin=_K_REFINE_ACCEPT_MARGIN,
                    max_refine_ratio=_ATTN_BLOCK_SMOOTH_REFINE_RATIO,
                    max_refine_blocks=_ATTN_BLOCK_SMOOTH_REFINE_BLOCKS,
                )
            else:
                q_params = _attention_dense_to_hif4(q_smooth)
                k_params = _attention_dense_to_hif4(k_smooth)
            q_hat = _attention_dequantize_hif4(q_params)
            k_hat = _attention_dequantize_hif4(k_params)
            out_c = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
            )
            out_n = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
            )
            err_c = float((out_c - ref_c).square().mean())
            err_n = float((out_n - ref_n).square().mean())
            id_c, id_n = identity_cases[index]
            ratio_c = err_c / max(id_c, 1.0e-12)
            ratio_n = err_n / max(id_n, 1.0e-12)
            if not (math.isfinite(ratio_c) and math.isfinite(ratio_n)):
                return 1.0e30, (1.0e30,)
            causal_scores.append(ratio_c)
            safety_scores.append(ratio_n)
        if not causal_scores:
            return 1.0e30, (1.0e30,)
        primary = sum(causal_scores) / len(causal_scores)
        if not math.isfinite(primary):
            return 1.0e30, (1.0e30,)
        cases = [
            value
            for pair in zip(causal_scores, safety_scores)
            for value in pair
        ]
        cases.append(sum(safety_scores) / len(safety_scores))
        return primary, tuple(cases)

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
    if int(block_smooth_size) != 0:
        h_k_for_q = _block_average(h_k_for_q, int(block_smooth_size))
        h_q_for_k = _block_average(h_q_for_k, int(block_smooth_size))

    case_scores: list[float] = []
    for q_sample, k_sample in zip(q_samples, k_samples):
        q_smooth = (q_sample * d_q.reshape(1, -1)).index_select(
            -1, q_order
        )
        k_centered = _center_attention_k(
            k_sample, kv_num_heads, head_dim, center_mode, center_value
        )
        k_smooth = (k_centered * d_k.reshape(1, -1)).index_select(
            -1, k_order
        )
        if int(block_smooth_size) != 0:
            q_smooth = _apply_attention_rotation(
                q_smooth,
                q_num_heads,
                head_dim,
                block_signs,
                int(block_smooth_size),
            )
            k_smooth = _apply_attention_rotation(
                k_smooth,
                kv_num_heads,
                head_dim,
                block_signs,
                int(block_smooth_size),
            )
        q_hat = _attention_dequantize_hif4(_attention_dense_to_hif4(q_smooth))
        k_hat = _attention_dequantize_hif4(_attention_dense_to_hif4(k_smooth))

        q_error = (
            (q_smooth - q_hat).square() * h_k_for_q.reshape(1, -1)
        ).sum()
        q_energy = (q_smooth.square() * h_k_for_q.reshape(1, -1)).sum()
        k_error = (
            (k_smooth - k_hat).square() * h_q_for_k.reshape(1, -1)
        ).sum()
        k_energy = (k_smooth.square() * h_q_for_k.reshape(1, -1)).sum()
        score = torch.nan_to_num(
            q_error / (q_energy + _attention_EPS) + k_error / (k_energy + _attention_EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)

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

    q_sum_square = torch.zeros(q_num_heads, head_dim, dtype=torch.float32)
    k_sum_square = torch.zeros(kv_num_heads, head_dim, dtype=torch.float32)
    k_mid_sum_square = torch.zeros_like(k_sum_square)
    q_peak_square = torch.zeros_like(q_sum_square)
    k_peak_square = torch.zeros_like(k_sum_square)
    k_mid_peak_square = torch.zeros_like(k_sum_square)
    k_sac_sum_square = torch.zeros_like(k_sum_square)
    k_sac_peak_square = torch.zeros_like(k_sum_square)
    # C41: solve the quantization-aware K center once, before the statistics
    # loop, so that mode 4 has its own second moment / peak estimates.
    sac_center = None
    if _ATTN_SCALE_AWARE_CENTER and (
        _ATTN_SCALE_AWARE_CENTER_GQA or q_num_heads == kv_num_heads
    ):
        sac_pieces = []
        for sample in calib_qkv_list:
            k_dense = _attention_dequantize_nvfp4_float32(sample["k"][0], sample["k"][1])
            sac_pieces.append(_attention_sample_rows(k_dense, _attention_ATTN_STATS_TOKENS))
        if sac_pieces:
            sac_center = _solve_k_center_scale_aware(
                torch.cat(sac_pieces, dim=0),
                kv_num_heads,
                head_dim,
                _ATTN_CENTER_ALTERNATIONS,
            )
    q_token_count = 0
    k_token_count = 0
    sample_count = 0
    v_head_mass = torch.zeros(kv_num_heads, dtype=torch.float32)
    v_head_mean_mass = torch.zeros(kv_num_heads, dtype=torch.float32)
    q_samples: list[torch.Tensor] = []
    k_samples: list[torch.Tensor] = []
    v_samples: list[torch.Tensor] = []
    a1_q: list[torch.Tensor] = []
    a1_k: list[torch.Tensor] = []
    a1_v: list[torch.Tensor] = []
    a1_q_pairs: list = []
    a1_k_pairs: list = []
    a1_v_pairs: list = []

    for sample in calib_qkv_list:
        if not isinstance(sample, dict) or set(sample.keys()) != {"q", "k", "v"}:
            raise ValueError("Each attention calibration sample must contain q/k/v")
        q = _attention_dequantize_nvfp4_float32(*sample["q"])
        k = _attention_dequantize_nvfp4_float32(*sample["k"])
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

        if q_sum_square.device != q.device:
            q_sum_square = q_sum_square.to(q.device)
            k_sum_square = k_sum_square.to(q.device)
            k_mid_sum_square = k_mid_sum_square.to(q.device)
            q_peak_square = q_peak_square.to(q.device)
            k_peak_square = k_peak_square.to(q.device)
            k_mid_peak_square = k_mid_peak_square.to(q.device)
            k_sac_sum_square = k_sac_sum_square.to(q.device)
            k_sac_peak_square = k_sac_peak_square.to(q.device)
            v_head_mass = v_head_mass.to(q.device)
            v_head_mean_mass = v_head_mean_mass.to(q.device)

        if _V_ATTENTION_IMPORTANCE:
            head_mean_mass, head_square_mass = _attention_head_square_mass(
                q, k, q_num_heads, kv_num_heads, head_dim
            )
            v_head_mass += head_square_mass
            if head_mean_mass is not None:
                v_head_mean_mass += head_mean_mass

        q_stats = _attention_sample_rows(q, _attention_ATTN_STATS_TOKENS).reshape(
            -1, q_num_heads, head_dim
        )
        k_stats = _attention_sample_rows(k, _attention_ATTN_STATS_TOKENS).reshape(
            -1, kv_num_heads, head_dim
        )
        k_mid_stats = _center_attention_k(
            k_stats.reshape(-1, kv_channels),
            kv_num_heads,
            head_dim,
            2,
        ).reshape(-1, kv_num_heads, head_dim)
        q_sum_square += q_stats.square().sum(dim=0)
        k_sum_square += k_stats.square().sum(dim=0)
        k_mid_sum_square += k_mid_stats.square().sum(dim=0)
        q_peak_square += q_stats.abs().amax(dim=0).square()
        k_peak_square += k_stats.abs().amax(dim=0).square()
        k_mid_peak_square += k_mid_stats.abs().amax(dim=0).square()
        if sac_center is not None:
            k_sac_stats = _center_attention_k(
                k_stats.reshape(-1, kv_channels),
                kv_num_heads,
                head_dim,
                4,
                sac_center,
            ).reshape(-1, kv_num_heads, head_dim)
            k_sac_sum_square += k_sac_stats.square().sum(dim=0)
            k_sac_peak_square += k_sac_stats.abs().amax(dim=0).square()
        q_token_count += int(q_stats.shape[0])
        k_token_count += int(k_stats.shape[0])
        sample_count += 1
        q_samples.append(_attention_sample_rows(q, _ATTN_EVAL_TOKENS).clone())
        k_samples.append(_attention_sample_rows(k, _ATTN_EVAL_TOKENS).clone())
        v_dense = _attention_dequantize_nvfp4_float32(v_quant, v_scale)
        v_samples.append(_attention_sample_rows(v_dense, _ATTN_EVAL_TOKENS).clone())
        if _ATTN_OUTPUT_SELECTOR:
            prefix = min(int(q.shape[0]), _ATTN_A1_MAX_TOKENS)
            a1_q.append(q[:prefix].clone())
            a1_k.append(k[:prefix].clone())
            a1_v.append(v_dense[:prefix].clone())
            a1_q_pairs.append(
                (sample["q"][0][:prefix], sample["q"][1][:prefix])
            )
            a1_k_pairs.append(
                (sample["k"][0][:prefix], sample["k"][1][:prefix])
            )
            a1_v_pairs.append((v_quant[:prefix], v_scale[:prefix]))

    a1_context = None
    if _ATTN_OUTPUT_SELECTOR and a1_q:
        v_hats = [_attention_dequantize_hif4(_attention_dense_to_hif4(v)) for v in a1_v]
        refs = []
        identity_cases = []
        for q, k, v, v_hat in zip(a1_q, a1_k, a1_v, v_hats):
            ref_c = _attention_forward(
                q, k, v, q_num_heads, kv_num_heads, head_dim, True
            )
            ref_n = _attention_forward(
                q, k, v, q_num_heads, kv_num_heads, head_dim, False
            )
            refs.append((ref_c, ref_n))
            q_hat = _attention_dequantize_hif4(_attention_dense_to_hif4(q))
            k_hat = _attention_dequantize_hif4(_attention_dense_to_hif4(k))
            id_c = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
            )
            id_n = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
            )
            identity_cases.append(
                (
                    float((id_c - ref_c).square().mean()),
                    float((id_n - ref_n).square().mean()),
                )
            )
        a1_context = {
            "q_full": a1_q,
            "k_full": a1_k,
            "v_hat": v_hats,
            "refs": refs,
            "identity": identity_cases,
        }

    v_importance = None
    v_importance_candidates: dict = {}
    if _V_ATTENTION_IMPORTANCE and sample_count > 0:
        head_importance = v_head_mass / float(max(sample_count, 1))
        head_mean_importance = v_head_mean_mass / float(
            max(sample_count, 1)
        )
        head_importance = head_importance / head_importance.mean().clamp_min(
            _attention_EPS
        )
        head_mean_importance = head_mean_importance / (
            head_mean_importance.mean().clamp_min(_attention_EPS)
        )
        if _V_ATTENTION_IMPORTANCE_SHRINK < 1.0:
            head_importance = 1.0 + _V_ATTENTION_IMPORTANCE_SHRINK * (
                head_importance - 1.0
            )
            head_mean_importance = 1.0 + _V_ATTENTION_IMPORTANCE_SHRINK * (
                head_mean_importance - 1.0
            )
        v_importance = _normalize_importance(
            head_importance.repeat_interleave(head_dim).reshape(-1),
            kv_channels,
        )
        if _V_IMPORTANCE_CANDIDATES and a1_context is not None:
            # A3 候选：一阶矩 E[A] 与 E[A^2] + E[A]^2（均值交叉项）。
            # 仅改变 head 级 importance 向量，V 坐标系不变。
            first_moment = _normalize_importance(
                head_mean_importance.repeat_interleave(head_dim).reshape(-1),
                kv_channels,
            )
            combined_head = head_importance + head_mean_importance.square()
            combined = _normalize_importance(
                combined_head.repeat_interleave(head_dim).reshape(-1),
                kv_channels,
            )
            v_importance_candidates["first_moment"] = first_moment
            v_importance_candidates["mean_cross"] = combined

    q_second_moment = q_sum_square / float(max(q_token_count, 1))
    k_second_moment = k_sum_square / float(max(k_token_count, 1))
    k_mid_second_moment = k_mid_sum_square / float(max(k_token_count, 1))
    q_peak = torch.sqrt(q_peak_square / float(max(sample_count, 1)))
    k_peak = torch.sqrt(k_peak_square / float(max(sample_count, 1)))
    k_mid_peak = torch.sqrt(k_mid_peak_square / float(max(sample_count, 1)))
    k_sac_second_moment = k_sac_sum_square / float(max(k_token_count, 1))
    k_sac_peak = torch.sqrt(k_sac_peak_square / float(max(sample_count, 1)))

    group_size = q_num_heads // kv_num_heads
    q_peak_kv = q_peak.reshape(kv_num_heads, group_size, head_dim).amax(dim=1)
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

    def _run_selection(use_a1: bool):
        context = a1_context if use_a1 else None
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
            context,
        )
        best_metrics = baseline_metrics
        best_d = identity_d
        best_center_mode = 0
        best_q_perm = q_identity_perm
        best_k_perm = k_identity_perm
        best_block_smooth_size = 0
        best_block_smooth_seed = 0

        # Midrange K-centering is an exact softmax invariance.  First select
        # the centering/smoothing pair with identity ordering, then test one
        # hierarchy-aware ordering for the selected pair to bound calibration
        # time.
        for center_mode in _ATTN_CENTER_MODES:
            if center_mode == 4:
                if not _ATTN_SCALE_AWARE_CENTER:
                    continue
                if (
                    not _ATTN_SCALE_AWARE_CENTER_GQA
                    and q_num_heads != kv_num_heads
                ):
                    continue
            if center_mode in (2, 3):
                effective_second = k_mid_second_moment
                effective_peak = k_mid_peak
            elif center_mode == 4:
                effective_second = k_sac_second_moment
                effective_peak = k_sac_peak
            else:
                effective_second = k_second_moment
                effective_peak = k_peak
            q_rms_kv = torch.sqrt(
                q_second_moment.reshape(
                    kv_num_heads, group_size, head_dim
                ).mean(dim=1).clamp_min(_attention_EPS)
            )
            k_rms = torch.sqrt(effective_second.clamp_min(_attention_EPS))
            smooth_candidates = [identity_d]
            alpha_values = _QK_SMOOTH_ALPHAS
            if context is not None and _ATTN_OUTPUT_SELECTOR:
                alpha_values = tuple(
                    dict.fromkeys(
                        (*_QK_SMOOTH_ALPHAS, *_ATTN_OUTPUT_EXTRA_SMOOTH_ALPHAS)
                    )
                )
            for alpha in alpha_values:
                smooth_candidates.append(
                    _smooth_qk_scale(q_peak_kv, effective_peak, alpha)
                )
                if _QK_SMOOTH_RMS:
                    smooth_candidates.append(
                        _smooth_qk_scale(q_rms_kv, k_rms, alpha)
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
                    context,
                    sac_center if center_mode == 4 else None,
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

        # C76.3 output-aware reciprocal temperature: scan a few global
        # factors around the selected per-channel SmoothQuant scale.  Since
        # Q is multiplied by ``factor`` and K by its reciprocal, the
        # continuous dot product is unchanged; only the two quantizers'
        # lattice errors move.  This stage is A1-only and keeps the exact
        # identity candidate available through the parent/proxy track.
        if context is not None and _ATTN_OUTPUT_HEAD_SCALE:
            parent_d = best_d
            if best_center_mode == 4:
                scale_second = k_sac_second_moment
            elif best_center_mode in (2, 3):
                scale_second = k_mid_second_moment
            else:
                scale_second = k_second_moment
            for factor in _ATTN_OUTPUT_HEAD_SCALE_FACTORS:
                candidate_d = (
                    parent_d * float(factor)
                ).clamp(min=_QK_SMOOTH_MIN, max=_QK_SMOOTH_MAX)
                metrics = _attention_candidate_metrics(
                    q_samples,
                    k_samples,
                    candidate_d,
                    q_second_moment,
                    scale_second,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    q_identity_perm,
                    k_identity_perm,
                    best_center_mode,
                    context,
                    sac_center if best_center_mode == 4 else None,
                )
                if (
                    metrics[0] < best_metrics[0]
                    and _candidate_is_safe(
                        metrics,
                        baseline_metrics,
                        min_mean_improvement=0.0,
                        worst_tolerance=0.02,
                    )
                ):
                    best_metrics = metrics
                    best_d = candidate_d

        if best_center_mode == 4:
            selected_k_peak = k_sac_peak
            selected_k_second = k_sac_second_moment
        elif best_center_mode in (2, 3):
            selected_k_peak = k_mid_peak
            selected_k_second = k_mid_second_moment
        else:
            selected_k_peak = k_peak
            selected_k_second = k_second_moment
        local_permutation = _headwise_hierarchy_permutation(
            q_peak_kv * best_d,
            selected_k_peak * best_d.reciprocal(),
        )
        candidate_k_perm = _flatten_head_permutation(local_permutation)
        candidate_q_perm = _flatten_head_permutation(
            local_permutation.repeat_interleave(group_size, dim=0)
        )
        if not torch.equal(candidate_k_perm, k_identity_perm):
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
                context,
                sac_center if best_center_mode == 4 else None,
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

        # 置换基扩展：单侧排序（Q-only / K-only）常优于 max(log range) 组合。
        if _PERMUTATION_BASES:
            basis_ranges = {
                "q_amax": q_peak_kv * best_d,
                "k_amax": selected_k_peak * best_d.reciprocal(),
            }
            seen = {tuple(best_k_perm.tolist())}
            for bname, b_range in basis_ranges.items():
                b_local = _headwise_range_permutation(b_range)
                b_k_perm = _flatten_head_permutation(b_local)
                if torch.equal(b_k_perm, k_identity_perm):
                    continue
                if tuple(b_k_perm.tolist()) in seen:
                    continue
                seen.add(tuple(b_k_perm.tolist()))
                b_q_perm = _flatten_head_permutation(
                    b_local.repeat_interleave(group_size, dim=0)
                )
                b_metrics = _attention_candidate_metrics(
                    q_samples,
                    k_samples,
                    best_d,
                    q_second_moment,
                    selected_k_second,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    b_q_perm,
                    b_k_perm,
                    best_center_mode,
                    context,
                    sac_center if best_center_mode == 4 else None,
                )
                if (
                    b_metrics[0] < best_metrics[0]
                    and _candidate_is_safe(
                        b_metrics,
                        baseline_metrics,
                        min_mean_improvement=0.02,
                        worst_tolerance=0.005,
                    )
                ):
                    best_metrics = b_metrics
                    best_q_perm = b_q_perm
                    best_k_perm = b_k_perm

        # C76.1 output-aware decoupling: the tied hierarchy permutation is
        # optimal only for the *continuous* Q·K invariant.  Once Q and K are
        # quantized independently, a Q-only, K-only, or independently paired
        # head ordering can reduce the real attention error.  Evaluate a small
        # deterministic pool on the A1 output context; the proxy track never
        # sees these candidates, and the deployed A1 gate still compares its
        # winner against the proxy winner on both causal and non-causal folds.
        if (
            context is not None
            and _ATTN_OUTPUT_HEADWISE_PERMUTATION
            and int(_ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES) > 0
        ):
            q_range = q_peak_kv * best_d
            k_range = selected_k_peak * best_d.reciprocal()
            q_local = _headwise_range_permutation(
                q_range.repeat_interleave(group_size, dim=0)
            )
            k_local = _headwise_range_permutation(k_range)
            q_independent = _flatten_head_permutation(q_local)
            k_independent = _flatten_head_permutation(k_local)
            independent_candidates = [
                (q_independent, k_identity_perm),
                (q_identity_perm, k_independent),
                (q_independent, k_independent),
            ]
            seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
            for q_candidate, k_candidate in independent_candidates:
                key = (
                    tuple(q_candidate.tolist()),
                    tuple(k_candidate.tolist()),
                )
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if len(seen_pairs) > int(_ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES):
                    break
                if (
                    torch.equal(q_candidate, q_identity_perm)
                    and torch.equal(k_candidate, k_identity_perm)
                ):
                    continue
                independent_metrics = _attention_candidate_metrics(
                    q_samples,
                    k_samples,
                    best_d,
                    q_second_moment,
                    selected_k_second,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    q_candidate,
                    k_candidate,
                    best_center_mode,
                    context,
                    sac_center if best_center_mode == 4 else None,
                )
                if (
                    independent_metrics[0] < best_metrics[0]
                    and _candidate_is_safe(
                        independent_metrics,
                        baseline_metrics,
                        min_mean_improvement=0.0,
                        worst_tolerance=0.02,
                    )
                ):
                    best_metrics = independent_metrics
                    best_q_perm = q_candidate
                    best_k_perm = k_candidate

        # C86: after d/center/permutation are selected, search a shared
        # head-local Hadamard block transform.  The transform is exactly
        # QK-preserving before quantization and is scored with the same proxy
        # or A1 output metric as the parent candidate.  Keeping this stage
        # last avoids evaluating block candidates against a stale ordering.
        if _ATTN_BLOCK_SMOOTH_ENABLED and _ATTN_OUTPUT_SELECTOR:
            for block_size in _ATTN_BLOCK_SMOOTH_SIZES:
                block = int(block_size)
                if block <= 0 or head_dim % block != 0:
                    continue
                for block_seed in _ATTN_BLOCK_SMOOTH_SEEDS:
                    seed = int(block_seed)
                    block_metrics = _attention_candidate_metrics(
                        q_samples,
                        k_samples,
                        best_d,
                        q_second_moment,
                        selected_k_second,
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                        best_q_perm,
                        best_k_perm,
                        best_center_mode,
                        context,
                        sac_center if best_center_mode == 4 else None,
                        block,
                        seed,
                        _ATTN_BLOCK_SMOOTH_FINAL_QUANTIZER,
                    )
                    if (
                        block_metrics[0] < best_metrics[0]
                        and _candidate_is_safe(
                            block_metrics,
                            baseline_metrics,
                            min_mean_improvement=_ATTN_BLOCK_SMOOTH_MIN_GAIN,
                            worst_tolerance=_ATTN_BLOCK_SMOOTH_WORST_TOLERANCE,
                        )
                    ):
                        best_metrics = block_metrics
                        best_block_smooth_size = block
                        best_block_smooth_seed = seed

        return (
            best_d,
            best_center_mode,
            best_q_perm,
            best_k_perm,
            best_block_smooth_size,
            best_block_smooth_seed,
        )

    # 双轨选择：A1 轨用真实 attention 输出误差（朴素 HiF4 代理量化），
    # proxy 轨复刻当前 Champion（B0）的 Q/K 重建 proxy 选择逻辑。终验门
    # 在部署路径上对比两个 winner，A1 无明确优势时回退 B0 选择。
    if a1_context is not None:
        (
            a1_d,
            a1_center,
            a1_q_perm,
            a1_k_perm,
            a1_block_size,
            a1_block_seed,
        ) = _run_selection(True)
        (
            proxy_d,
            proxy_center,
            proxy_q_perm,
            proxy_k_perm,
            proxy_block_size,
            proxy_block_seed,
        ) = _run_selection(False)
    else:
        (
            proxy_d,
            proxy_center,
            proxy_q_perm,
            proxy_k_perm,
            proxy_block_size,
            proxy_block_seed,
        ) = _run_selection(False)
        a1_d, a1_center, a1_q_perm, a1_k_perm, a1_block_size, a1_block_seed = (
            proxy_d,
            proxy_center,
            proxy_q_perm,
            proxy_k_perm,
            proxy_block_size,
            proxy_block_seed,
        )

    def _build_v_state(importance) -> dict:
        if _DATA_DRIVEN_RATIO:
            v_ratio = _loss_capture_ratio(
                torch.cat(
                    [
                        _standard_block_losses(s, importance)
                        for s in v_samples
                    ]
                ),
                target=_RATIO_CAPTURE_TARGET,
                ratio_min=_RATIO_MIN,
            )
        else:
            v_ratio = _V_REFINE_MAX_RATIO
        return {
            "offsets": torch.tensor(
                _DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"
            ),
            "importance": (
                None if importance is None else _cpu_state_tensor(importance)
            ),
            "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
            "accept_margin": _V_REFINE_ACCEPT_MARGIN,
            "max_refine_ratio": float(v_ratio),
            "max_refine_blocks": _V_REFINE_MAX_BLOCKS,
            "num_heads": int(kv_num_heads),
            "head_dim": int(head_dim),
            "version": 2,
        }

    v_state = _build_v_state(v_importance)

    def _build_qk_states(
        d: torch.Tensor,
        center_mode: int,
        q_perm: torch.Tensor,
        k_perm: torch.Tensor,
        block_smooth_size: int = 0,
        block_smooth_seed: int = 0,
        rotation: Optional[torch.Tensor] = None,
        rotation_block: Optional[int] = None,
        q_importance_raw: Optional[torch.Tensor] = None,
        k_importance_raw: Optional[torch.Tensor] = None,
    ) -> tuple:
        d_q = d.repeat_interleave(group_size, dim=0)
        d_k = d.reciprocal()
        q_second_kv = q_second_moment.reshape(
            kv_num_heads, group_size, head_dim
        ).mean(dim=1)
        if int(center_mode) == 4:
            eff_k_second = k_sac_second_moment
        elif int(center_mode) in (2, 3):
            eff_k_second = k_mid_second_moment
        else:
            eff_k_second = k_second_moment
        if q_importance_raw is None:
            h_k = eff_k_second * d_k.square()
            h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
        else:
            h_k_for_q = q_importance_raw.to(
                device=d.device, dtype=torch.float32
            ).reshape(-1)
        if k_importance_raw is None:
            h_q = q_second_kv * d.square()
            h_q_for_k = h_q.reshape(-1)
        else:
            h_q_for_k = k_importance_raw.to(
                device=d.device, dtype=torch.float32
            ).reshape(-1)
        h_k_for_q = _normalize_importance(
            h_k_for_q.index_select(0, q_perm), q_channels
        )
        h_q_for_k = _normalize_importance(
            h_q_for_k.index_select(0, k_perm), kv_channels
        )
        if h_k_for_q is None:
            h_k_for_q = torch.ones(
                q_channels, dtype=torch.float32, device=d_q.device
            )
        if h_q_for_k is None:
            h_q_for_k = torch.ones(
                kv_channels, dtype=torch.float32, device=d_k.device
            )
        if int(block_smooth_size) != 0:
            h_k_for_q = _block_average(h_k_for_q, int(block_smooth_size))
            h_q_for_k = _block_average(h_q_for_k, int(block_smooth_size))
        block_signs = (
            None
            if int(block_smooth_size) == 0
            else _attention_rotation_signs(
                kv_num_heads, head_dim, int(block_smooth_seed)
            )
        )
        q_flat = d_q.reshape(-1)
        k_flat = d_k.reshape(-1)

        def q_transform(sample: torch.Tensor) -> torch.Tensor:
            transformed = (sample * q_flat.reshape(1, -1)).index_select(
                -1, q_perm
            )
            if rotation is not None:
                transformed = _apply_attention_rotation(
                    transformed,
                    q_num_heads,
                    head_dim,
                    rotation,
                    rotation_block,
                )
            if int(block_smooth_size) != 0:
                transformed = _apply_attention_rotation(
                    transformed,
                    q_num_heads,
                    head_dim,
                    block_signs,
                    int(block_smooth_size),
                )
            return transformed

        def k_transform(sample: torch.Tensor) -> torch.Tensor:
            transformed = (
                _center_attention_k(
                    sample,
                    kv_num_heads,
                    head_dim,
                    int(center_mode),
                    sac_center if int(center_mode) == 4 else None,
                )
                * k_flat.reshape(1, -1)
            ).index_select(-1, k_perm)
            if rotation is not None:
                transformed = _apply_attention_rotation(
                    transformed,
                    kv_num_heads,
                    head_dim,
                    rotation,
                    rotation_block,
                )
            if int(block_smooth_size) != 0:
                transformed = _apply_attention_rotation(
                    transformed,
                    kv_num_heads,
                    head_dim,
                    block_signs,
                    int(block_smooth_size),
                )
            return transformed

        if _DATA_DRIVEN_RATIO:
            q_ratio = _loss_capture_ratio(
                torch.cat(
                    [_standard_block_losses(q_transform(s), h_k_for_q)
                     for s in q_samples]
                ),
                target=_RATIO_CAPTURE_TARGET,
                ratio_min=_RATIO_MIN,
            )
            k_ratio = _loss_capture_ratio(
                torch.cat(
                    [_standard_block_losses(k_transform(s), h_q_for_k)
                     for s in k_samples]
                ),
                target=_RATIO_CAPTURE_TARGET,
                ratio_min=_RATIO_MIN,
            )
        else:
            q_ratio = _Q_REFINE_MAX_RATIO
            k_ratio = _K_REFINE_MAX_RATIO

        q_permutation_state = None
        k_permutation_state = None
        if not torch.equal(k_perm, k_identity_perm):
            q_permutation_state = q_perm.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
            k_permutation_state = k_perm.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
        q_multiplier_state = None
        k_multiplier_state = None
        if not torch.equal(d, identity_d):
            q_multiplier_state = _cpu_state_tensor(d_q.reshape(-1))
            k_multiplier_state = _cpu_state_tensor(d_k.reshape(-1))

        rotation_state = (
            None
            if rotation is None
            else rotation.detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        )
        q_state = {
            "multiplier": q_multiplier_state,
            "permutation": q_permutation_state,
            "importance": _cpu_state_tensor(h_k_for_q),
            "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
            "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
            "accept_margin": _Q_REFINE_ACCEPT_MARGIN,
            "max_refine_ratio": float(q_ratio),
            "max_refine_blocks": _Q_REFINE_MAX_BLOCKS,
            "num_heads": int(q_num_heads),
            "head_dim": int(head_dim),
            "version": 2,
        }
        k_state = {
            "multiplier": k_multiplier_state,
            "permutation": k_permutation_state,
            "center_mode": int(center_mode),
            "importance": _cpu_state_tensor(h_q_for_k),
            "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
            "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
            "accept_margin": _K_REFINE_ACCEPT_MARGIN,
            "max_refine_ratio": float(k_ratio),
            "max_refine_blocks": _K_REFINE_MAX_BLOCKS,
            "num_heads": int(kv_num_heads),
            "head_dim": int(head_dim),
            "version": 2,
        }
        if int(block_smooth_size) != 0:
            q_state["block_smooth_size"] = int(block_smooth_size)
            q_state["block_smooth_seed"] = int(block_smooth_seed)
            q_state["block_smooth_signs"] = _cpu_state_tensor(block_signs)
            k_state["block_smooth_size"] = int(block_smooth_size)
            k_state["block_smooth_seed"] = int(block_smooth_seed)
            k_state["block_smooth_signs"] = _cpu_state_tensor(block_signs)
        # C41: only carry the center vector when mode 4 is actually selected,
        # so the state key set stays identical to the parent otherwise.
        if int(center_mode) == 4 and sac_center is not None:
            k_state["center_value"] = _cpu_state_tensor(sac_center)
        if rotation_state is not None:
            q_state["rotation"] = rotation_state
            k_state["rotation"] = rotation_state
            q_state["rotation_block"] = int(
                _ATTN_H64_BLOCK if rotation_block is None else rotation_block
            )
            k_state["rotation_block"] = int(
                _ATTN_H64_BLOCK if rotation_block is None else rotation_block
            )
        return q_state, k_state

    q_state, k_state = _build_qk_states(
        a1_d,
        int(a1_center),
        a1_q_perm,
        a1_k_perm,
        int(a1_block_size),
        int(a1_block_seed),
    )
    final_d, final_center = a1_d, int(a1_center)
    final_q_perm, final_k_perm = a1_q_perm, a1_k_perm
    a1_v_hats = None
    base_causal = None
    base_safety = None

    # A1 终验门：A1 选择基于朴素 HiF4 代理，部署路径（offset 搜索 +
    # importance 精修）与隐藏 test 分布上的排序都可能错位。用完整
    # hif4_dynamic_quantize_q/k/v 路径在 calibration 前缀上重算真实
    # attention 输出误差（causal 主轨 + non-causal 安全轨，V 部署路径
    # 固定以隔离 Q/K 变换选择）；A1 winner 相对 B0 proxy winner（当前
    # Champion 的选择）无明确改善或安全轨退化时，回退 B0 选择。
    same_winner = (
        torch.equal(a1_d, proxy_d)
        and int(a1_center) == int(proxy_center)
        and torch.equal(a1_q_perm, proxy_q_perm)
        and torch.equal(a1_k_perm, proxy_k_perm)
        and int(a1_block_size) == int(proxy_block_size)
        and int(a1_block_seed) == int(proxy_block_seed)
    )
    if a1_context is not None and not same_winner:
        proxy_q_state, proxy_k_state = _build_qk_states(
            proxy_d,
            int(proxy_center),
            proxy_q_perm,
            proxy_k_perm,
            int(proxy_block_size),
            int(proxy_block_seed),
        )
        a1_v_hats = [
            _attention_dequantize_hif4(
                hif4_dynamic_quantize_v(
                    v_quant, v_scale, kv_num_heads, head_dim, v_state
                )
            ).to(torch.float32)
            for v_quant, v_scale in a1_v_pairs
        ]
        winner_causal, winner_safety = _attention_deployed_mse(
            a1_q_pairs,
            a1_k_pairs,
            a1_v_hats,
            a1_context["refs"],
            q_state,
            k_state,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        proxy_causal, proxy_safety = _attention_deployed_mse(
            a1_q_pairs,
            a1_k_pairs,
            a1_v_hats,
            a1_context["refs"],
            proxy_q_state,
            proxy_k_state,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        if _a1_gate_passes(
            winner_causal, winner_safety, proxy_causal, proxy_safety
        ):
            base_causal, base_safety = winner_causal, winner_safety
        else:
            q_state, k_state = proxy_q_state, proxy_k_state
            final_d, final_center = proxy_d, int(proxy_center)
            final_q_perm, final_k_perm = proxy_q_perm, proxy_k_perm
            base_causal, base_safety = proxy_causal, proxy_safety

    # C76.2 output-Fisher importance: Q/K channel error is not weighted only
    # by the opposite operand's second moment.  Use the local attention
    # Jacobian (probability times value deviation) to form a diagonal Fisher
    # proxy, then compare Q-only, K-only and joint importance states through
    # the real deployed quantizer.  The candidate is static one-dimensional
    # state; all token/output tensors remain calibration-local.
    if _ATTN_FISHER_IMPORTANCE and a1_context is not None:
        if a1_v_hats is None:
            a1_v_hats = [
                _attention_dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, v_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
        if base_causal is None or base_safety is None:
            base_causal, base_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
        fisher = _attention_qk_fisher_importance(
            q_samples,
            k_samples,
            v_samples,
            final_d,
            final_q_perm,
            final_k_perm,
            q_num_heads,
            kv_num_heads,
            head_dim,
            int(final_center),
            sac_center if int(final_center) == 4 else None,
        )
        if fisher is not None:
            q_fisher, k_fisher = fisher
            q_second_kv = q_second_moment.reshape(
                kv_num_heads, group_size, head_dim
            ).mean(dim=1)
            if int(final_center) == 4:
                effective_k_second = k_sac_second_moment
            elif int(final_center) in (2, 3):
                effective_k_second = k_mid_second_moment
            else:
                effective_k_second = k_second_moment
            baseline_q_raw = (
                effective_k_second * final_d.reciprocal().square()
            ).repeat_interleave(group_size, dim=0).reshape(-1)
            baseline_k_raw = (q_second_kv * final_d.square()).reshape(-1)
            baseline_q_norm = _normalize_importance(
                baseline_q_raw, q_channels
            )
            baseline_k_norm = _normalize_importance(
                baseline_k_raw, kv_channels
            )
            fisher_q_norm = _normalize_importance(q_fisher, q_channels)
            fisher_k_norm = _normalize_importance(k_fisher, kv_channels)
            if (
                baseline_q_norm is None
                or baseline_k_norm is None
                or fisher_q_norm is None
                or fisher_k_norm is None
            ):
                fisher_candidates = ()
            else:
                fisher_candidates = []
                for blend in _ATTN_FISHER_BLEND_VALUES:
                    beta = max(0.0, min(float(blend), 1.0))
                    q_mix = baseline_q_norm + beta * (
                        fisher_q_norm - baseline_q_norm
                    )
                    k_mix = baseline_k_norm + beta * (
                        fisher_k_norm - baseline_k_norm
                    )
                    fisher_candidates.extend(
                        (
                            (q_mix, None),
                            (None, k_mix),
                            (q_mix, k_mix),
                        )
                    )
            best_fisher_mean = sum(base_causal) / max(len(base_causal), 1)
            for q_importance_raw, k_importance_raw in fisher_candidates:
                fisher_q_state, fisher_k_state = _build_qk_states(
                    final_d,
                    int(final_center),
                    final_q_perm,
                    final_k_perm,
                    int(q_state.get("block_smooth_size", 0)),
                    int(q_state.get("block_smooth_seed", 0)),
                    q_importance_raw=q_importance_raw,
                    k_importance_raw=k_importance_raw,
                )
                candidate_causal, candidate_safety = _attention_deployed_mse(
                    a1_q_pairs,
                    a1_k_pairs,
                    a1_v_hats,
                    a1_context["refs"],
                    fisher_q_state,
                    fisher_k_state,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                if not candidate_causal or not candidate_safety:
                    continue
                candidate_mean = sum(candidate_causal) / len(candidate_causal)
                base_mean = sum(base_causal) / max(len(base_causal), 1)
                if not math.isfinite(candidate_mean) or candidate_mean >= (
                    base_mean * (1.0 - float(_ATTN_FISHER_MIN_GAIN))
                ):
                    continue
                tolerance = 1.0 + float(_ATTN_FISHER_WORST_TOLERANCE)
                if any(
                    current > reference * tolerance
                    for current, reference in zip(candidate_causal, base_causal)
                ) or any(
                    current > reference * tolerance
                    for current, reference in zip(candidate_safety, base_safety)
                ):
                    continue
                candidate_safety_mean = sum(candidate_safety) / len(candidate_safety)
                base_safety_mean = sum(base_safety) / max(len(base_safety), 1)
                if candidate_safety_mean > base_safety_mean * tolerance:
                    continue
                if candidate_mean < best_fisher_mean:
                    best_fisher_mean = candidate_mean
                    q_state, k_state = fisher_q_state, fisher_k_state
                    base_causal, base_safety = candidate_causal, candidate_safety

    # A2 固定 H64：对最终 Q/K winner 施加组对齐 signed Hadamard(64)
    # 旋转（同组 Q heads 与 K head 共享旋转，Q·K 点积严格不变），首版只
    # 比较 2 个确定性 sign seed。旋转候选须通过同一真实 attention 输出
    # 门控（causal 主轨 + non-causal 安全轨），否则保持无旋转 winner。
    if (
        _ATTN_H64
        and a1_context is not None
        and head_dim >= _ATTN_H64_BLOCK
        and head_dim % _ATTN_H64_BLOCK == 0
    ):
        if a1_v_hats is None:
            a1_v_hats = [
                _attention_dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, v_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
        if base_causal is None:
            base_causal, base_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
        best_rotation_states = None
        best_rotation_mean = None
        for seed in _ATTN_H64_SEEDS:
            signs = _attention_rotation_signs(kv_num_heads, head_dim, seed)
            rotation_q_state, rotation_k_state = _build_qk_states(
                final_d,
                int(final_center),
                final_q_perm,
                final_k_perm,
                int(q_state.get("block_smooth_size", 0)),
                int(q_state.get("block_smooth_seed", 0)),
                rotation=signs,
            )
            rotation_causal, rotation_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                rotation_q_state,
                rotation_k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            if not _a1_gate_passes(
                rotation_causal,
                rotation_safety,
                base_causal,
                base_safety,
                safety_tolerance=0.0,
            ):
                continue
            rotation_mean = sum(rotation_causal) / len(rotation_causal)
            if best_rotation_mean is None or rotation_mean < best_rotation_mean:
                best_rotation_mean = rotation_mean
                best_rotation_states = (
                    rotation_q_state,
                    rotation_k_state,
                )
        if best_rotation_states is not None:
            q_state, k_state = best_rotation_states

    # C76.4 variable head-local rotations.  The old A2 arm only tested a
    # fixed H64 transform and was disabled after it migrated on some GQA
    # tails.  Search smaller H16/H32 blocks as well; all candidates preserve
    # the Q·K dot product exactly before quantization because the same signed
    # orthogonal transform is applied to each Q/KV group.
    if (
        _ATTN_ROTATION_ENABLED
        and a1_context is not None
        and head_dim >= 4
        and (
            not _ATTN_ROTATION_GQA_ONLY
            or q_num_heads != kv_num_heads
        )
    ):
        if a1_v_hats is None:
            a1_v_hats = [
                _attention_dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, v_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
        if base_causal is None or base_safety is None:
            base_causal, base_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
        base_rotation_mean = sum(base_causal) / max(len(base_causal), 1)
        best_rotation_states = None
        best_rotation_mean = base_rotation_mean
        for block_size in _ATTN_ROTATION_BLOCKS:
            block = int(block_size)
            if block < 4 or head_dim % block != 0:
                continue
            for seed in _attention_ATTN_ROTATION_SEEDS:
                signs = _attention_rotation_signs(kv_num_heads, head_dim, int(seed))
                rotation_q_state, rotation_k_state = _build_qk_states(
                    final_d,
                    int(final_center),
                    final_q_perm,
                    final_k_perm,
                    int(q_state.get("block_smooth_size", 0)),
                    int(q_state.get("block_smooth_seed", 0)),
                    rotation=signs,
                    rotation_block=block,
                )
                rotation_causal, rotation_safety = _attention_deployed_mse(
                    a1_q_pairs,
                    a1_k_pairs,
                    a1_v_hats,
                    a1_context["refs"],
                    rotation_q_state,
                    rotation_k_state,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                if not _a1_gate_passes(
                    rotation_causal,
                    rotation_safety,
                    base_causal,
                    base_safety,
                    safety_tolerance=0.0,
                ):
                    continue
                rotation_mean = sum(rotation_causal) / len(rotation_causal)
                if rotation_mean < best_rotation_mean:
                    best_rotation_mean = rotation_mean
                    best_rotation_states = (
                        rotation_q_state,
                        rotation_k_state,
                    )
        if best_rotation_states is not None:
            q_state, k_state = best_rotation_states

    # A3 V importance 候选：Q/K state 已定稿（A1 终验门 + A2 旋转），
    # 仅更换 V 的 head 级 importance（当前 E[A^2] vs 一阶矩 E[A] vs
    # E[A^2]+E[A]^2），用完整 hif4_dynamic_quantize_v 部署路径在
    # calibration 前缀上重算真实 attention 输出误差；候选须通过同一
    # 门控（causal 主轨 + non-causal 安全轨），否则保持当前 importance。
    if v_importance_candidates and a1_context is not None:
        if a1_v_hats is None:
            a1_v_hats = [
                _attention_dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, v_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
        if base_causal is None:
            base_causal, base_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
        best_v_state = None
        best_v_mean = None
        for candidate_importance in v_importance_candidates.values():
            candidate_state = _build_v_state(candidate_importance)
            candidate_v_hats = [
                _attention_dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, candidate_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
            candidate_causal, candidate_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                candidate_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            if not _a1_gate_passes(
                candidate_causal, candidate_safety, base_causal, base_safety
            ):
                continue
            candidate_mean = sum(candidate_causal) / len(candidate_causal)
            if best_v_mean is None or candidate_mean < best_v_mean:
                best_v_mean = candidate_mean
                best_v_state = candidate_state
        if best_v_state is not None:
            v_state = best_v_state
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
    return _nvfp4_to_hif4(
        q_quant,
        q_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        block_smooth_size=int(state.get("block_smooth_size", 0)),
        block_smooth_seed=int(state.get("block_smooth_seed", 0)),
        attention_rotation=state.get("rotation"),
        rotation_num_heads=int(q_num_heads),
        attention_rotation_block=state.get("rotation_block"),
        attention_block_signs=state.get("block_smooth_signs"),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
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
    return _nvfp4_to_hif4(
        k_quant,
        k_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        block_smooth_size=int(state.get("block_smooth_size", 0)),
        block_smooth_seed=int(state.get("block_smooth_seed", 0)),
        attention_rotation=state.get("rotation"),
        rotation_num_heads=int(kv_num_heads),
        attention_rotation_block=state.get("rotation_block"),
        attention_block_signs=state.get("block_smooth_signs"),
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        center_value=state.get("center_value"),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
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
    return _nvfp4_to_hif4(
        v_quant,
        v_scale,
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )

if _ATTN_SCALE_AWARE_CENTER:
    _ATTN_CENTER_MODES = (0, 2, 4)

# ---------------------------------------------------------------------------
# One fixed A3/block residual increment on the v140 Linear path.
# ---------------------------------------------------------------------------
def _active_transformed_calibration(weight_quant, weight_scale, calib_activation_list, state):
    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(torch.float32)
    samples = [_dequantize_nvfp4_float32(item[0], item[1]).to(device=weight.device, dtype=torch.float32) for item in calib_activation_list]
    pair = state.get("roab_pairs")
    if torch.is_tensor(pair):
        pair = pair.to(device=weight.device, dtype=torch.float32)
        inverse = torch.linalg.inv(pair)
        return _pair_transform(weight, inverse.transpose(-1, -2)), [_pair_transform(sample, pair) for sample in samples]
    smooth_inv = state.get("smooth_inv")
    if not torch.is_tensor(smooth_inv):
        raise ValueError("v140 state has no Linear transform")
    smooth_inv = smooth_inv.to(device=weight.device, dtype=torch.float32).reshape(-1)
    balance = smooth_inv.reciprocal()
    seed = int(state.get("block_smooth_seed", -1)); block_size = int(state.get("block_smooth_size", 0))
    return _apply_boat_rotation(weight * balance.reshape(1, -1), seed, block_size), [_apply_boat_rotation(sample * smooth_inv.reshape(1, -1), seed, block_size) for sample in samples]

def _active_deploy_from_state(weight_params, activation_t, state):
    gram = state.get("gram64"); cross = state.get("output_cross64"); gain = state.get("output_gain")
    if not torch.is_tensor(gram) or not torch.is_tensor(gain):
        raise ValueError("v140 state has no deployed-weight curvature")
    device = activation_t[0].device; gram = gram.to(device=device, dtype=torch.float32)
    if torch.is_tensor(cross): cross = cross.to(device=device, dtype=torch.float32)
    gain = gain.to(device=device, dtype=torch.float32).reshape(1, -1)
    deployed=[]
    for sample in activation_t:
        dense = sample.to(torch.float32) * gain
        params = _dense_to_hif4(dense, offsets=_BASE_OFFSETS, gram64=gram)
        params = _refine_activation(dense, params, gram, output_cross64=cross, output_target=sample)
        deployed.append(_dequantize_hif4(params).to(torch.float32))
    return deployed

def _active_rebuild_state(weight_t, weight_params, state, activation_t):
    deployed_weight = _dequantize_hif4(weight_params).to(torch.float32)
    gain = (weight_t * deployed_weight).sum(dim=0) / deployed_weight.square().sum(dim=0).clamp_min(_EPS)
    gain = torch.nan_to_num(gain, nan=1.0, posinf=2.0, neginf=0.5).clamp(0.5, 2.0)
    next_state = dict(state)
    next_state["output_gain"] = _cpu_tensor(gain)
    next_state["gram64"] = _cpu_tensor(_gram64(deployed_weight))
    next_state["output_cross64"] = _cpu_tensor(_block_cross64(deployed_weight, weight_t))
    next_state["version"] = 4
    return next_state, _active_deploy_from_state(weight_params, activation_t, next_state)

def _active_joint_loss(weight_t, weight_params, activation_t, deployed_activation):
    deployed_weight = _dequantize_hif4(weight_params).to(torch.float32); losses=[]
    for raw, deployed in zip(activation_t[:2], deployed_activation[:2]):
        target = raw.to(torch.float32).mm(weight_t.to(torch.float32).t()); prediction = deployed.to(torch.float32).mm(deployed_weight.t())
        losses.append(float((target-prediction).square().mean()/(target.square().mean()+_EPS)))
    return losses

def _active_robust_score(losses):
    return math.inf if not losses else sum(losses)/len(losses) + _WEIGHT_HSDQ_ROBUST_MIX*max(losses)

# v86 Attention API is copied directly above; one public binding per API.
__all__ = [
    "hif4_calibration_and_quantize_weight", "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention", "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k", "hif4_dynamic_quantize_v",
]
