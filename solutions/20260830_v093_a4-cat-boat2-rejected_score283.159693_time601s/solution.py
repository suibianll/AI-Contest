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

# A4 coordinate-system candidate.  The local block search is deliberately
# small enough to leave identity/global BOAT as a cheap fallback.
_BOAT2_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_BOAT2_PERMUTATION_NAMES = ("identity", "activation", "weight", "geometric", "interleave")
_BOAT2_HOUSEHOLDER_REG = 1.0e-4

_WEIGHT_HSDQ_BLOCKS = 2
_WEIGHT_HSDQ_SWEEPS = 1
_WEIGHT_HSDQ_MIN_CHANNELS = 256
_WEIGHT_HSDQ_MAX_ROWS = 256
_WEIGHT_HSDQ_ROBUST_MIX = 0.5
_WEIGHT_HSDQ_MIN_GAIN = 1.0e-5

_ACT_HSDQ_BLOCKS = 128
_ACT_HSDQ_SWEEPS = 2
_ACT_GRAM_MAX_CHANNELS = 8192

_ATTN_OFFSETS = (-2, -1, 0, 1, 2)
_ATTN_ROTATION_SIZES = (0, 16, 32, 64)
_ATTN_SMOOTH_ALPHAS = (0.0, 0.25, 0.5, 0.75)
_ATTN_ROTATION_SEEDS = (0, 1)

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


def _apply_householder_blocks(
    x: torch.Tensor,
    vectors: torch.Tensor | None,
) -> torch.Tensor:
    """Apply self-inverse Householder reflectors independently per 64 block."""
    if vectors is None or x.ndim < 1:
        return x
    blocks = int(x.shape[-1]) // _BLOCK
    if int(x.shape[-1]) % _BLOCK != 0 or tuple(vectors.shape[-1:]) != (_BLOCK,):
        return x
    if int(vectors.shape[0]) != blocks:
        return x
    y = x.reshape(-1, blocks, _BLOCK).clone()
    for reflector in range(int(vectors.shape[1])):
        v = vectors[:, reflector].to(device=y.device, dtype=y.dtype)
        norm = v.square().sum(dim=-1, keepdim=True)
        active = norm.squeeze(-1) > _EPS
        if not bool(active.any()):
            continue
        projection = (y * v[None, :, :]).sum(dim=-1, keepdim=True)
        update = 2.0 * projection * v[None, :, :] / norm.clamp_min(_EPS)[None, :, :]
        y = torch.where(active[None, :, None], y - update, y)
    return y.reshape_as(x)


def _apply_boat2_transform(
    x: torch.Tensor,
    balance: torch.Tensor,
    permutation: torch.Tensor | None,
    householder: torch.Tensor | None,
    seed: int,
    block_size: int,
) -> torch.Tensor:
    """Apply the A4 right-coordinate transform to an operand."""
    y = x.to(torch.float32) * balance.to(x.device, torch.float32).reshape(1, -1)
    if permutation is not None:
        y = y.index_select(-1, permutation.to(device=y.device, dtype=torch.long))
    y = _apply_householder_blocks(y, householder)
    return _apply_boat_rotation(y, int(seed), int(block_size))


def _block_permutation(kind: str, activation_rms: torch.Tensor, weight_rms: torch.Tensor) -> torch.Tensor:
    """Return a deterministic hierarchy-aware order for one 64-value block."""
    if kind == "identity":
        return torch.arange(_BLOCK, device=activation_rms.device)
    if kind == "activation":
        return torch.argsort(activation_rms, stable=True)
    if kind == "weight":
        return torch.argsort(weight_rms, stable=True)
    if kind == "geometric":
        return torch.argsort((activation_rms * weight_rms).clamp_min(_EPS), stable=True)
    # Interleave high/low magnitudes so that the fixed lv2/lv3 groups do not
    # receive all extremes in the same hierarchy bucket.
    order = torch.argsort((activation_rms * weight_rms).clamp_min(_EPS), stable=True)
    low = order[: (_BLOCK + 1) // 2]
    high = order.flip(0)[: _BLOCK // 2]
    result = torch.empty(_BLOCK, dtype=torch.long, device=order.device)
    result[0::2] = high
    result[1::2] = low[: result[1::2].numel()]
    return result


def _cat_householder_vector(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Get one CAT-inspired generalized covariance direction for a block."""
    n = max(int(activation.shape[0]), 1)
    m = max(int(weight.shape[0]), 1)
    eye = torch.eye(_BLOCK, device=activation.device, dtype=torch.float32)
    act_cov = activation.t().mm(activation) / float(n)
    weight_cov = weight.t().mm(weight) / float(m)
    scale = act_cov.diagonal().mean().clamp_min(_EPS)
    reg = float(_BOAT2_HOUSEHOLDER_REG) * scale
    try:
        evals, evecs = torch.linalg.eigh(act_cov + reg * eye)
        inv_sqrt = evecs @ torch.diag(evals.clamp_min(_EPS).rsqrt()) @ evecs.t()
        cat_cov = inv_sqrt.mm(weight_cov).mm(inv_sqrt)
        _, cat_vecs = torch.linalg.eigh(cat_cov + reg * eye)
        vector = inv_sqrt.mm(cat_vecs[:, -1])
    except RuntimeError:
        return torch.zeros(_BLOCK, device=activation.device, dtype=torch.float32)
    norm = vector.norm()
    if not torch.isfinite(norm) or float(norm) <= _EPS:
        return torch.zeros(_BLOCK, device=activation.device, dtype=torch.float32)
    return vector / norm


def _boat2_proxy_score(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    balance: torch.Tensor,
    permutation: torch.Tensor,
    householder: torch.Tensor | None,
    seed: int,
    block_size: int,
) -> float:
    """Operand-local BOAT-2 proxy; it never forms a mixed product."""
    transformed_weight = _apply_boat2_transform(
        _sample_rows(weight, _BOAT_PROXY_WEIGHT_ROWS),
        balance,
        permutation,
        householder,
        seed,
        block_size,
    )
    weight_error = _relative_quant_error(transformed_weight, _PROXY_OFFSETS)
    activation_errors = []
    for sample in calibration:
        transformed = _apply_boat2_transform(
            _sample_rows(sample, 128) / balance.reshape(1, -1),
            torch.ones_like(balance),
            permutation,
            householder,
            seed,
            block_size,
        )
        activation_errors.append(_relative_quant_error(transformed, _PROXY_OFFSETS))
    if not activation_errors:
        return math.inf
    act_mean = sum(activation_errors) / len(activation_errors)
    act_worst = max(activation_errors)
    return (
        math.sqrt(max(weight_error, 0.0))
        + math.sqrt(max(act_mean, 0.0))
        + math.sqrt(max(weight_error * act_mean, 0.0))
        + 0.25 * math.sqrt(max(act_worst, 0.0))
    )


def _choose_boat2(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    """Search the A4 blockwise balance/permutation/CAT-reflector candidate."""
    channels = int(weight.shape[1])
    if not calibration or channels % _BLOCK != 0:
        return torch.ones(channels, device=weight.device), torch.arange(
            channels, device=weight.device
        ), None, -1, 0
    # Keep the already validated global BOAT candidate as an explicit parent.
    base_balance, base_seed, base_block = _choose_boat(weight, calibration)
    best_balance = base_balance.clone()
    best_permutation = torch.arange(channels, device=weight.device)
    best_householder: torch.Tensor | None = None
    best_score = _boat2_proxy_score(
        weight, calibration, best_balance, best_permutation, None, base_seed, base_block
    )

    joined = torch.cat([_sample_rows(item, 128) for item in calibration], dim=0).to(torch.float32)
    a_rms = joined.square().mean(dim=0).add(_EPS).sqrt()
    w_rms = weight.to(torch.float32).square().mean(dim=0).add(_EPS).sqrt()
    block_count = channels // _BLOCK
    balance = torch.empty_like(a_rms)
    permutation_parts: list[torch.Tensor] = []
    for block in range(block_count):
        lo = block * _BLOCK
        hi = lo + _BLOCK
        ratio = (a_rms[lo:hi] / w_rms[lo:hi]).clamp(
            _BOAT_SCALE_MIN, _BOAT_SCALE_MAX
        )
        # A small local search over alpha uses only operand-local quantization
        # errors.  The selected scales are normalized globally afterwards.
        local_best = math.inf
        local_balance = torch.ones(_BLOCK, device=weight.device)
        for alpha in _BOAT2_ALPHAS:
            candidate = ratio.pow(float(alpha)).clamp(_BOAT_SCALE_MIN, _BOAT_SCALE_MAX)
            candidate = candidate / candidate.log().mean().exp().clamp_min(_EPS)
            w_block = _sample_rows(weight[:, lo:hi], _BOAT_PROXY_WEIGHT_ROWS) * candidate
            a_blocks = [
                _sample_rows(sample[:, lo:hi], 128) / candidate for sample in calibration
            ]
            score = _relative_quant_error(w_block, _PROXY_OFFSETS)
            score += sum(_relative_quant_error(item, _PROXY_OFFSETS) for item in a_blocks) / len(a_blocks)
            if score < local_best:
                local_best = score
                local_balance = candidate
        balance[lo:hi] = local_balance
        # The hierarchy-aware orders are scored later globally; geometric order
        # is a safe initial choice because it couples both operand RMS values.
        permutation_parts.append(
            _block_permutation("geometric", a_rms[lo:hi], w_rms[lo:hi]) + lo
        )
    balance = balance / torch.exp(torch.log(balance).mean()).clamp_min(_EPS)
    permutation = torch.cat(permutation_parts)
    score = _boat2_proxy_score(
        weight, calibration, balance, permutation, None, base_seed, base_block
    )
    # Test the other legal hierarchy orders as complete blockwise candidates.
    for kind in _BOAT2_PERMUTATION_NAMES:
        parts = []
        for block in range(block_count):
            lo = block * _BLOCK
            hi = lo + _BLOCK
            parts.append(_block_permutation(kind, a_rms[lo:hi], w_rms[lo:hi]) + lo)
        candidate_perm = torch.cat(parts)
        candidate_score = _boat2_proxy_score(
            weight, calibration, balance, candidate_perm, None, base_seed, base_block
        )
        if candidate_score < score:
            score = candidate_score
            permutation = candidate_perm
    if score < best_score:
        best_score = score
        best_balance = balance
        best_permutation = permutation

    # CAT-inspired Householder directions are tested per block.  A zero vector
    # means identity and keeps the state compact/fully serializable.
    vectors = torch.zeros(block_count, 1, _BLOCK, device=weight.device)
    transformed_weight = weight.to(torch.float32)
    transformed_activation = [item.to(torch.float32) for item in calibration]
    # Evaluate each reflector against the current blockwise frame so a bad
    # direction cannot poison all roles.
    for block in range(block_count):
        lo = block * _BLOCK
        hi = lo + _BLOCK
        perm_block = permutation[lo:hi]
        w_block = transformed_weight[:, perm_block] * balance[perm_block]
        a_blocks = [item[:, perm_block] / balance[perm_block] for item in transformed_activation]
        vector = _cat_householder_vector(torch.cat(a_blocks), w_block)
        if not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= _EPS:
            continue
        trial = vectors.clone()
        trial[block, 0] = vector
        identity_score = _boat2_proxy_score(
            weight, calibration, balance, permutation, vectors, base_seed, base_block
        )
        trial_score = _boat2_proxy_score(
            weight, calibration, balance, permutation, trial, base_seed, base_block
        )
        if trial_score < identity_score:
            vectors = trial
    householder = vectors if bool((vectors.square().sum(dim=-1) > _EPS).any()) else None
    candidate_score = _boat2_proxy_score(
        weight, calibration, balance, permutation, householder, base_seed, base_block
    )
    if candidate_score < best_score:
        return balance, permutation, householder, base_seed, base_block
    return best_balance, best_permutation, best_householder, base_seed, base_block


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

    del permutation
    d = balance.to(device=tensor.device, dtype=torch.float32).reshape(1, -1)
    transformed = tensor.to(torch.float32) * (d if weight_side else d.reciprocal())
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
    balance, permutation, householder, seed, block_size = _choose_boat2(
        weight, calibration
    )
    weight_t = _apply_boat2_transform(
        weight, balance, permutation, householder, seed, block_size
    )
    activation_t = [
        _apply_boat2_transform(
            sample / balance.reshape(1, -1),
            torch.ones_like(balance),
            permutation,
            householder,
            seed,
            block_size,
        )
        for sample in calibration
    ]
    weight_params = _dense_to_hif4(weight_t, offsets=_BASE_OFFSETS)
    weight_params = _crossfold_weight_hsdq(weight_t, weight_params, activation_t)

    gram_state = None
    if int(weight_t.shape[1]) <= _ACT_GRAM_MAX_CHANNELS:
        gram_state = _cpu_tensor(_gram64(weight_t))
    state: dict[str, Any] = {
        "smooth_inv": _cpu_tensor(balance.reciprocal()),
        "permutation": _cpu_tensor(permutation.to(torch.float32)).to(torch.int64),
        "householder": (
            _cpu_tensor(householder)
            if householder is not None
            else None
        ),
        "block_smooth_size": int(block_size),
        "block_smooth_seed": int(seed),
        "gram64": gram_state,
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
    if not torch.is_tensor(permutation):
        permutation = torch.arange(dense.shape[-1], device=dense.device)
    householder = state.get("householder")
    if not torch.is_tensor(householder):
        householder = None
    seed = int(state.get("block_smooth_seed", -1))
    block_size = int(state.get("block_smooth_size", 0))
    dense = _apply_boat2_transform(
        dense,
        torch.ones(dense.shape[-1], device=dense.device),
        permutation,
        householder,
        seed,
        block_size,
    )
    gram = state.get("gram64")
    gram_tensor = gram if torch.is_tensor(gram) else None
    if gram_tensor is not None:
        gram_tensor = gram_tensor.to(dense.device)
    params = _dense_to_hif4(
        dense, offsets=_BASE_OFFSETS, gram64=gram_tensor
    )
    return _refine_activation(dense, params, gram_tensor)


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
) -> tuple[float, torch.Tensor, torch.Tensor]:
    transformed_q: list[torch.Tensor] = []
    transformed_k: list[torch.Tensor] = []
    for q, k in zip(q_samples, k_samples):
        transformed_q.append(
            _apply_head_transform(
                q, q_heads, kv_heads, head_dim, multiplier, signs, block_size
            )
        )
        if center:
            kv = k.reshape(-1, kv_heads, head_dim)
            k = (kv - kv.mean(dim=0, keepdim=True)).reshape_as(k)
        transformed_k.append(
            _apply_head_transform(
                k, kv_heads, kv_heads, head_dim, multiplier.reciprocal(),
                signs, block_size,
            )
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
    v_hats = [
        _dequantize_hif4(_dense_to_hif4(v, offsets=_ATTN_OFFSETS)).to(torch.float32)
        for v in v_samples
    ]
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
    best_q_importance = torch.ones(q_num_heads * head_dim, device=q_stack.device)
    best_k_importance = torch.ones(kv_num_heads * head_dim, device=q_stack.device)
    candidate_pool: list[
        tuple[float, torch.Tensor, torch.Tensor | None, int, bool]
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
            candidate_pool.append((score, multiplier.clone(), None, 0, center))
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
                (score, best_multiplier.clone(), signs.clone(), int(block_size), best_center)
            )
            if score < best_score:
                best_score = score
                best_block = int(block_size)
                best_signs = signs.clone()
                best_q_importance = q_importance
                best_k_importance = k_importance

    # The proxy scan is cheap; only its three strongest candidates are
    # re-ranked through the exact deployed Gram-HSDQ path.
    best_score = math.inf
    for _, multiplier, signs, block_size, center in sorted(
        candidate_pool, key=lambda item: item[0]
    )[:4]:
        score, q_importance, k_importance = _attention_candidate_score(
            q_samples, k_samples, v_hats, references,
            q_num_heads, kv_num_heads, head_dim,
            multiplier, signs, block_size, center, refine=True,
        )
        if score < best_score:
            best_score = score
            best_multiplier = multiplier.clone()
            best_signs = None if signs is None else signs.clone()
            best_block = int(block_size)
            best_center = bool(center)
            best_q_importance = q_importance
            best_k_importance = k_importance

    final_q: list[torch.Tensor] = []
    final_k: list[torch.Tensor] = []
    for q, k in zip(q_samples, k_samples):
        final_q.append(
            _apply_head_transform(
                q, q_num_heads, kv_num_heads, head_dim,
                best_multiplier, best_signs, best_block,
            )
        )
        if best_center:
            view = k.reshape(-1, kv_num_heads, head_dim)
            k = (view - view.mean(dim=0, keepdim=True)).reshape_as(k)
        final_k.append(
            _apply_head_transform(
                k, kv_num_heads, kv_num_heads, head_dim,
                best_multiplier.reciprocal(), best_signs, best_block,
            )
        )
    q_gram, k_gram = _qk_gram64(
        final_q, final_k, q_num_heads, kv_num_heads, head_dim
    )
    q_state: dict[str, Any] = {
        "multiplier": _cpu_tensor(best_multiplier),
        "rotation_block": best_block,
        "rotation_signs": None if best_signs is None else _cpu_tensor(best_signs),
        "kv_heads": int(kv_num_heads),
        "importance": _cpu_tensor(best_q_importance),
        "gram64": None if q_gram is None else _cpu_tensor(q_gram),
    }
    k_state: dict[str, Any] = {
        "multiplier": _cpu_tensor(best_multiplier.reciprocal()),
        "rotation_block": best_block,
        "rotation_signs": None if best_signs is None else _cpu_tensor(best_signs),
        "kv_heads": int(kv_num_heads),
        "importance": _cpu_tensor(best_k_importance),
        "gram64": None if k_gram is None else _cpu_tensor(k_gram),
        "center": best_center,
    }
    return {"q_state": q_state, "k_state": k_state, "v_state": {}}


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
    del kv_num_heads, head_dim, v_state
    dense = _dequantize_nvfp4_float32(v_quant, v_scale)
    return _dense_to_hif4(dense, offsets=_ATTN_OFFSETS)


__all__ = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
