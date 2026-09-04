"""Clean-room robust NVFP4 -> HiF4 operator quantizer.

This file is intentionally self-contained.  It implements the six competition
APIs without importing any repository code.  High-dimensional calibration
statistics are reduced to a few identity-anchored decisions; dynamic paths use
only fixed transforms and a bounded HiF4 scale neighbourhood.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


_NV_BLOCK = 16
_HIF_BLOCK = 64
_E6M2_MIN = 2.0 ** -48
_E6M2_MAX = 49152.0
_BF16_ONE_SEVENTH = 0.142578125
_EPS = 1.0e-12

_BASE_OFFSETS = (0, -1, 1, 2, 3)
_PLUS4_OFFSETS = (0, -1, 1, 2, 3, 4)
_WEIGHT_REFINE_RATIO = 0.12
_ACTIVATION_REFINE_RATIO = 0.18
_ATTENTION_REFINE_RATIO = 0.20
_PROBE_REFINE_RATIO = 0.06
_PROBE_TOKENS = 128
_PROBE_WEIGHT_ROWS = 128
_SENSITIVITY_SHRINK = 0.25
_SENSITIVITY_MIN = 0.5
_SENSITIVITY_MAX = 2.0


def _dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
) -> torch.Tensor:
    channels = int(quant_float.shape[-1])
    if channels % _NV_BLOCK != 0:
        raise ValueError("NVFP4 last dimension must be divisible by 16")
    grouped = quant_float.unflatten(-1, (-1, _NV_BLOCK))
    return (grouped * scale_float.unsqueeze(-1)).flatten(-2, -1).to(torch.bfloat16)


def _e6m2_encode(value: torch.Tensor) -> torch.Tensor:
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
    return (exponent_field * 4 + mantissa).clamp(0, 254).to(torch.int64)


def _e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    code64 = code.to(torch.int64).clamp(0, 254)
    exponent = torch.bitwise_right_shift(code64, 2).to(torch.float32) - 48.0
    mantissa = torch.bitwise_and(code64, 3).to(torch.float32)
    return torch.pow(2.0, exponent) * (1.0 + 0.25 * mantissa)


def _standard_scale_code(blocks: torch.Tensor) -> torch.Tensor:
    amax = blocks.abs().amax(dim=-1)
    scale = (amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH).to(torch.float32)
    return _e6m2_encode(scale)


def _solve_for_scale(
    blocks: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    absolute = blocks.abs().reshape(-1, 8, 2, 4)
    grouped_weights = weights.reshape(-1, 8, 2, 4)
    scale_view = scale[:, None, None, None]

    losses: list[torch.Tensor] = []
    for exponent in (0, 1, 2):
        local_scale = scale_view * float(1 << exponent)
        mantissa = torch.round(absolute * (4.0 / local_scale)).clamp(0.0, 7.0) * 0.25
        error = absolute - mantissa * local_scale
        losses.append((grouped_weights * error.square()).sum(dim=-1))

    loss0, loss1, loss2 = losses
    choose01 = loss1 < loss0
    choose12 = loss2 < loss1
    cost_l2_1 = torch.minimum(loss0, loss1).sum(dim=-1)
    cost_l2_2 = torch.minimum(loss1, loss2).sum(dim=-1)
    use_l2_2 = cost_l2_2 < cost_l2_1
    use_l3_2 = torch.where(use_l2_2[..., None], choose12, choose01)

    lv2 = 1.0 + use_l2_2.to(torch.float32)
    lv3 = 1.0 + use_l3_2.to(torch.float32)
    denominator = scale_view * lv2[..., None, None] * lv3[..., None]
    mantissa = torch.round(absolute * (4.0 / denominator)).clamp(0.0, 7.0) * 0.25
    sign = torch.sign(blocks.reshape(-1, 8, 2, 4))
    sign = torch.where(mantissa == 0.0, torch.zeros_like(sign), sign)
    reconstructed = sign * mantissa * denominator
    loss = (grouped_weights * (reconstructed - blocks.reshape(-1, 8, 2, 4)).square()).sum(
        dim=(1, 2, 3)
    )
    return loss, {
        "scale_factor": scale[:, None, None, None],
        "scale_lv2": lv2[..., None, None],
        "scale_lv3": lv3[..., None],
        "sign": sign,
        "mant": mantissa,
    }


def _solve_blocks(
    blocks: torch.Tensor,
    weights: torch.Tensor,
    offsets: Sequence[int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base_code = _standard_scale_code(blocks)
    best_loss = torch.full(
        (blocks.shape[0],),
        float("inf"),
        dtype=torch.float32,
        device=blocks.device,
    )
    best: dict[str, torch.Tensor] | None = None
    for offset in offsets:
        scale = _e6m2_decode((base_code + int(offset)).clamp(0, 254))
        loss, params = _solve_for_scale(blocks, weights, scale)
        update = loss < best_loss
        if best is None:
            best = {name: value for name, value in params.items()}
        else:
            for name, value in params.items():
                mask = update.reshape((-1,) + (1,) * (value.ndim - 1))
                best[name] = torch.where(mask, value, best[name])
        best_loss = torch.where(update, loss, best_loss)
    if best is None:
        raise RuntimeError("empty HiF4 scale candidate set")
    return best_loss, best


def _expanded_importance(dense: torch.Tensor, importance: torch.Tensor | None) -> torch.Tensor:
    if importance is None:
        return torch.ones_like(dense, dtype=torch.float32)
    value = importance.detach().to(device=dense.device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.reshape(*([1] * (dense.ndim - 1)), dense.shape[-1]).expand_as(dense)
    elif tuple(value.shape) != tuple(dense.shape):
        value = value.expand_as(dense)
    mean = value.mean(dim=-1, keepdim=True).clamp_min(_EPS)
    return (value / mean).clamp(0.05, 20.0)


def _encode_hif4(
    dense: torch.Tensor,
    importance: torch.Tensor | None = None,
    *,
    refine_ratio: float,
    plus4: bool = False,
) -> dict[str, torch.Tensor]:
    if dense.ndim < 1 or int(dense.shape[-1]) % _HIF_BLOCK != 0:
        raise ValueError("HiF4 last dimension must be divisible by 64")
    x = torch.nan_to_num(dense.detach().to(torch.float32))
    prefix = tuple(int(v) for v in x.shape[:-1])
    channels = int(x.shape[-1])
    num_channel_blocks = channels // _HIF_BLOCK
    flat_blocks = x.reshape(-1, _HIF_BLOCK)
    flat_weights = _expanded_importance(x, importance).reshape(-1, _HIF_BLOCK)

    base_loss, params = _solve_blocks(flat_blocks, flat_weights, (0,))
    if refine_ratio > 0.0 and flat_blocks.shape[0] > 0:
        count = max(1, min(flat_blocks.shape[0], int(math.ceil(flat_blocks.shape[0] * refine_ratio))))
        selected = torch.topk(base_loss, k=count, largest=True, sorted=False).indices
        offsets = _PLUS4_OFFSETS if plus4 else _BASE_OFFSETS
        _, refined = _solve_blocks(flat_blocks[selected], flat_weights[selected], offsets)
        for name in params:
            params[name][selected] = refined[name]

    return {
        "scale_factor": params["scale_factor"].reshape(*prefix, num_channel_blocks, 1, 1, 1),
        "scale_lv2": params["scale_lv2"].reshape(*prefix, num_channel_blocks, 8, 1, 1),
        "scale_lv3": params["scale_lv3"].reshape(*prefix, num_channel_blocks, 8, 2, 1),
        "sign": params["sign"].reshape(*prefix, num_channel_blocks, 8, 2, 4),
        "mant": params["mant"].reshape(*prefix, num_channel_blocks, 8, 2, 4),
    }


def _decode_hif4(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    value = (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
    )
    return value.flatten(start_dim=-4, end_dim=-1)


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32)


def _sample_rows(value: torch.Tensor, limit: int) -> torch.Tensor:
    rows = value.reshape(-1, value.shape[-1])
    if rows.shape[0] <= limit:
        return rows
    indices = torch.linspace(0, rows.shape[0] - 1, limit, device=rows.device).round().to(torch.long)
    return rows[indices]


def _normalized_importance(value: torch.Tensor) -> torch.Tensor:
    return (value / value.mean().clamp_min(_EPS)).clamp(0.05, 20.0)


def _accept_candidate(
    parent_losses: Sequence[float],
    candidate_losses: Sequence[float],
    *,
    min_gain: float,
    worst_tolerance: float,
) -> bool:
    if not parent_losses or len(parent_losses) != len(candidate_losses):
        return False
    deltas = torch.tensor(
        [
            (parent - candidate) / max(parent, _EPS)
            for parent, candidate in zip(parent_losses, candidate_losses)
        ],
        dtype=torch.float64,
    )
    return bool(deltas.median() > min_gain and deltas.min() > -worst_tolerance)


def _linear_probe_losses(
    weight: torch.Tensor,
    activations: Sequence[torch.Tensor],
    multiplier: torch.Tensor,
) -> list[float]:
    probe_weight = _sample_rows(weight, _PROBE_WEIGHT_ROWS)
    transformed_weight = probe_weight / multiplier
    transformed_activations = [_sample_rows(x, _PROBE_TOKENS) * multiplier for x in activations]
    weight_importance = _normalized_importance(
        torch.stack([x.square().mean(dim=0) for x in transformed_activations]).median(dim=0).values
    )
    weight_params = _encode_hif4(
        transformed_weight,
        weight_importance,
        refine_ratio=_PROBE_REFINE_RATIO,
    )
    quant_weight = _decode_hif4(weight_params)
    activation_importance = _normalized_importance(quant_weight.square().mean(dim=0))
    losses: list[float] = []
    for original, transformed in zip(activations, transformed_activations):
        original_rows = _sample_rows(original, _PROBE_TOKENS)
        activation_params = _encode_hif4(
            transformed,
            activation_importance,
            refine_ratio=_PROBE_REFINE_RATIO,
        )
        quant_activation = _decode_hif4(activation_params)
        reference = original_rows @ probe_weight.T
        player = quant_activation @ quant_weight.T
        losses.append(float((player - reference).square().mean()))
    return losses


def _fit_linear_multiplier(weight: torch.Tensor, activations: Sequence[torch.Tensor]) -> torch.Tensor:
    sampled = [_sample_rows(x, 256) for x in activations]
    activation_second = torch.stack([x.square().mean(dim=0) for x in sampled]).median(dim=0).values
    weight_second = _sample_rows(weight, 512).square().mean(dim=0)
    log_multiplier = 0.125 * (
        torch.log(weight_second.clamp_min(_EPS)) - torch.log(activation_second.clamp_min(_EPS))
    )
    multiplier = torch.exp(log_multiplier)
    grouped = multiplier.reshape(-1, _HIF_BLOCK)
    grouped = grouped / torch.exp(torch.log(grouped).mean(dim=-1, keepdim=True))
    candidate = grouped.reshape(-1).clamp(0.5, 2.0)
    identity = torch.ones_like(candidate)
    parent_losses = _linear_probe_losses(weight, sampled, identity)
    candidate_losses = _linear_probe_losses(weight, sampled, candidate)
    if _accept_candidate(
        parent_losses,
        candidate_losses,
        min_gain=0.002,
        worst_tolerance=0.005,
    ):
        return candidate
    return identity


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = _dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
    activations = [
        _dequantize_nvfp4(item[0], item[1]).to(torch.float32)
        for item in calib_activation_list
    ]
    multiplier = _fit_linear_multiplier(weight, activations)
    transformed_weight = weight / multiplier
    transformed_samples = [_sample_rows(x, 256) * multiplier for x in activations]
    weight_importance = _normalized_importance(
        torch.stack([x.square().mean(dim=0) for x in transformed_samples]).median(dim=0).values
    )
    weight_params = _encode_hif4(
        transformed_weight,
        weight_importance,
        refine_ratio=_WEIGHT_REFINE_RATIO,
    )
    quant_weight = _decode_hif4(weight_params)
    activation_importance = _normalized_importance(quant_weight.square().mean(dim=0))
    return {
        "weight_params": weight_params,
        "activation_state": {
            "channels": int(weight.shape[-1]),
            "multiplier": _cpu_tensor(multiplier),
            "importance": _cpu_tensor(activation_importance),
        },
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4(activation_quant, activation_scale).to(torch.float32)
    state = activation_state
    multiplier = state["multiplier"].to(device=dense.device, dtype=torch.float32)
    importance = state["importance"].to(device=dense.device, dtype=torch.float32)
    return _encode_hif4(
        dense * multiplier,
        importance,
        refine_ratio=_ACTIVATION_REFINE_RATIO,
    )


def _qkv_item(item: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if isinstance(item, Mapping):
        return item["q"], item["k"], item["v"]
    return item[0], item[1], item[2]


def _head_view(value: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return value.reshape(-1, heads, head_dim)


def _apply_qk_transform(
    q: torch.Tensor,
    k: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    center_k: bool,
    balance: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qh = _head_view(q, q_heads, head_dim)
    kh = _head_view(k, kv_heads, head_dim)
    if center_k:
        kh = kh - kh.mean(dim=0, keepdim=True)
    group = q_heads // kv_heads
    q_balance = balance.repeat_interleave(group, dim=0)
    q_gain = gamma.pow(0.35).repeat_interleave(group).reshape(1, q_heads, 1)
    k_gain = gamma.pow(0.65).reshape(1, kv_heads, 1)
    qh = qh * q_balance.reshape(1, q_heads, head_dim) * q_gain
    kh = kh / balance.reshape(1, kv_heads, head_dim) * k_gain
    return qh.reshape_as(q), kh.reshape_as(k)


def _attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    causal: bool,
) -> torch.Tensor:
    qh = _head_view(q, q_heads, head_dim).transpose(0, 1)
    group = q_heads // kv_heads
    kh = _head_view(k, kv_heads, head_dim).transpose(0, 1).repeat_interleave(group, dim=0)
    vh = _head_view(v, kv_heads, head_dim).transpose(0, 1).repeat_interleave(group, dim=0)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(head_dim)
    if causal:
        tokens = logits.shape[-1]
        mask = torch.triu(
            torch.ones((tokens, tokens), dtype=torch.bool, device=logits.device),
            diagonal=1,
        )
        logits = logits.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities @ vh).transpose(0, 1).reshape(q.shape[0], q_heads * head_dim)


def _attention_jacobian_sensitivity(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    qh = _head_view(q, q_heads, head_dim).transpose(0, 1)
    group = q_heads // kv_heads
    kh = _head_view(k, kv_heads, head_dim).transpose(0, 1).repeat_interleave(group, dim=0)
    vh = _head_view(v, kv_heads, head_dim).transpose(0, 1).repeat_interleave(group, dim=0)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(head_dim)
    if causal:
        tokens = logits.shape[-1]
        mask = torch.triu(
            torch.ones((tokens, tokens), dtype=torch.bool, device=logits.device),
            diagonal=1,
        )
        logits = logits.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    output = probabilities @ vh
    centered_v = vh[:, None, :, :] - output[:, :, None, :]

    q_jacobian = torch.einsum(
        "hij,hijo,hjc->hico", probabilities, centered_v, kh
    ) / math.sqrt(head_dim)
    q_sensitivity = q_jacobian.square().mean(dim=(1, 3))

    output_distance = centered_v.square().sum(dim=-1)
    k_coefficients = probabilities.square() * output_distance
    k_sensitivity = torch.einsum(
        "hij,hic->hjc", k_coefficients, qh.square()
    ).mean(dim=1) / float(head_dim)

    q_sensitivity = q_sensitivity.reshape(kv_heads, group, head_dim).mean(dim=1)
    k_sensitivity = k_sensitivity.reshape(kv_heads, group, head_dim).mean(dim=1)
    return q_sensitivity, k_sensitivity


def _attention_sensitivity_folds(
    calibration: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    center_k: bool,
    balance: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    q_folds: list[torch.Tensor] = []
    k_folds: list[torch.Tensor] = []
    for q_full, k_full, v_full in calibration:
        tokens = min(_PROBE_TOKENS, q_full.shape[0])
        q = q_full[:tokens]
        k = k_full[:tokens]
        v = v_full[:tokens]
        tq, tk = _apply_qk_transform(
            q,
            k,
            q_heads,
            kv_heads,
            head_dim,
            center_k=center_k,
            balance=balance,
            gamma=gamma,
        )
        q_causal, k_causal = _attention_jacobian_sensitivity(
            tq, tk, v, q_heads, kv_heads, head_dim, causal=True
        )
        q_full_view, k_full_view = _attention_jacobian_sensitivity(
            tq, tk, v, q_heads, kv_heads, head_dim, causal=False
        )
        q_value = 0.5 * (q_causal + q_full_view)
        k_value = 0.5 * (k_causal + k_full_view)
        q_folds.append(q_value / q_value.mean(dim=-1, keepdim=True).clamp_min(_EPS))
        k_folds.append(k_value / k_value.mean(dim=-1, keepdim=True).clamp_min(_EPS))
    return q_folds, k_folds


def _aggregate_attention_sensitivity(
    folds: Sequence[torch.Tensor],
) -> torch.Tensor:
    raw = torch.stack(tuple(folds)).median(dim=0).values.clamp_min(_EPS)
    shrunk = torch.exp(_SENSITIVITY_SHRINK * torch.log(raw))
    shrunk = shrunk / shrunk.mean(dim=-1, keepdim=True).clamp_min(_EPS)
    return shrunk.clamp(_SENSITIVITY_MIN, _SENSITIVITY_MAX)


def _fit_attention_balance(
    calibration: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    center_k: bool,
) -> torch.Tensor:
    group = q_heads // kv_heads
    q_seconds: list[torch.Tensor] = []
    k_seconds: list[torch.Tensor] = []
    for q, k, _ in calibration:
        qh = _head_view(q[:_PROBE_TOKENS], q_heads, head_dim).reshape(-1, kv_heads, group, head_dim)
        kh = _head_view(k[:_PROBE_TOKENS], kv_heads, head_dim)
        if center_k:
            kh = kh - kh.mean(dim=0, keepdim=True)
        q_seconds.append(qh.square().mean(dim=(0, 2)))
        k_seconds.append(kh.square().mean(dim=0))
    q2 = torch.stack(q_seconds).median(dim=0).values
    k2 = torch.stack(k_seconds).median(dim=0).values
    balance = torch.exp(0.125 * (torch.log(k2.clamp_min(_EPS)) - torch.log(q2.clamp_min(_EPS))))
    return balance.clamp(0.5, 2.0)


def _quantized_attention_loss(
    calibration: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    center_k: bool,
    balance: torch.Tensor,
    gamma: torch.Tensor,
    plus4: bool,
    q_importance: torch.Tensor | None = None,
    k_importance: torch.Tensor | None = None,
) -> list[float]:
    losses: list[float] = []
    for q_full, k_full, v_full in calibration:
        tokens = min(_PROBE_TOKENS, q_full.shape[0])
        q = q_full[:tokens]
        k = k_full[:tokens]
        v = v_full[:tokens]
        tq, tk = _apply_qk_transform(
            q,
            k,
            q_heads,
            kv_heads,
            head_dim,
            center_k=center_k,
            balance=balance,
            gamma=gamma,
        )
        group = q_heads // kv_heads
        q_weights = None
        if q_importance is not None:
            q_weights = q_importance.repeat_interleave(group, dim=0).reshape(-1)
        k_weights = None if k_importance is None else k_importance.reshape(-1)
        qhat = _decode_hif4(
            _encode_hif4(
                tq, q_weights, refine_ratio=_PROBE_REFINE_RATIO, plus4=plus4
            )
        )
        khat = _decode_hif4(
            _encode_hif4(
                tk, k_weights, refine_ratio=_PROBE_REFINE_RATIO, plus4=plus4
            )
        )
        vhat = _decode_hif4(
            _encode_hif4(v, refine_ratio=_PROBE_REFINE_RATIO, plus4=plus4)
        )
        for causal in (False, True):
            reference = _attention_output(
                q, k, v, q_heads, kv_heads, head_dim, causal=causal
            )
            player = _attention_output(
                qhat, khat, vhat, q_heads, kv_heads, head_dim, causal=causal
            )
            losses.append(float((player - reference).square().mean()))
    return losses


def _fit_logit_gain(
    calibration: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    center_k: bool,
    balance: torch.Tensor,
) -> torch.Tensor:
    group = q_heads // kv_heads
    fold_gains: list[torch.Tensor] = []
    unit_gamma = torch.ones(kv_heads, dtype=torch.float32, device=balance.device)
    for q_full, k_full, _ in calibration:
        tokens = min(_PROBE_TOKENS, q_full.shape[0])
        q = q_full[:tokens]
        k = k_full[:tokens]
        tq, tk = _apply_qk_transform(
            q,
            k,
            q_heads,
            kv_heads,
            head_dim,
            center_k=center_k,
            balance=balance,
            gamma=unit_gamma,
        )
        qhat = _decode_hif4(_encode_hif4(tq, refine_ratio=_PROBE_REFINE_RATIO))
        khat = _decode_hif4(_encode_hif4(tk, refine_ratio=_PROBE_REFINE_RATIO))
        q_ref = _head_view(q, q_heads, head_dim).transpose(0, 1)
        k_ref = _head_view(k, kv_heads, head_dim).transpose(0, 1)
        q_quant = _head_view(qhat, q_heads, head_dim).transpose(0, 1)
        k_quant = _head_view(khat, kv_heads, head_dim).transpose(0, 1)
        gains: list[torch.Tensor] = []
        for head in range(kv_heads):
            q_slice = slice(head * group, (head + 1) * group)
            ref_logits = q_ref[q_slice] @ k_ref[head].T / math.sqrt(head_dim)
            quant_logits = q_quant[q_slice] @ k_quant[head].T / math.sqrt(head_dim)
            ref_logits = ref_logits - ref_logits.mean(dim=-1, keepdim=True)
            quant_logits = quant_logits - quant_logits.mean(dim=-1, keepdim=True)
            numerator = (ref_logits * quant_logits).sum()
            denominator = quant_logits.square().sum().clamp_min(_EPS)
            gains.append((numerator / denominator).clamp(0.75, 4.0 / 3.0))
        fold_gains.append(torch.stack(gains))
    raw = torch.stack(fold_gains).median(dim=0).values
    return torch.exp(0.5 * torch.log(raw.clamp(0.75, 4.0 / 3.0)))


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    calibration: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for item in calib_qkv_list:
        q_pair, k_pair, v_pair = _qkv_item(item)
        calibration.append(
            (
                _dequantize_nvfp4(q_pair[0], q_pair[1]).to(torch.float32),
                _dequantize_nvfp4(k_pair[0], k_pair[1]).to(torch.float32),
                _dequantize_nvfp4(v_pair[0], v_pair[1]).to(torch.float32),
            )
        )

    device = calibration[0][0].device
    balance = torch.ones((kv_num_heads, head_dim), dtype=torch.float32, device=device)
    gamma = torch.ones(kv_num_heads, dtype=torch.float32, device=device)
    center_k = False
    plus4 = False
    current_losses = _quantized_attention_loss(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
        balance=balance,
        gamma=gamma,
        plus4=plus4,
    )

    centered_losses = _quantized_attention_loss(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=True,
        balance=balance,
        gamma=gamma,
        plus4=plus4,
    )
    if _accept_candidate(current_losses, centered_losses, min_gain=0.0005, worst_tolerance=0.01):
        center_k = True
        current_losses = centered_losses

    candidate_balance = _fit_attention_balance(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
    )
    balanced_losses = _quantized_attention_loss(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
        balance=candidate_balance,
        gamma=gamma,
        plus4=plus4,
    )
    if _accept_candidate(current_losses, balanced_losses, min_gain=0.001, worst_tolerance=0.01):
        balance = candidate_balance
        current_losses = balanced_losses

    candidate_gamma = _fit_logit_gain(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
        balance=balance,
    )
    gain_losses = _quantized_attention_loss(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
        balance=balance,
        gamma=candidate_gamma,
        plus4=plus4,
    )
    if _accept_candidate(current_losses, gain_losses, min_gain=0.001, worst_tolerance=0.01):
        gamma = candidate_gamma
        current_losses = gain_losses

    plus4_losses = _quantized_attention_loss(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
        balance=balance,
        gamma=gamma,
        plus4=True,
    )
    if _accept_candidate(current_losses, plus4_losses, min_gain=0.001, worst_tolerance=0.01):
        plus4 = True
        current_losses = plus4_losses

    q_sensitivity_folds, k_sensitivity_folds = _attention_sensitivity_folds(
        calibration,
        q_num_heads,
        kv_num_heads,
        head_dim,
        center_k=center_k,
        balance=balance,
        gamma=gamma,
    )
    sensitivity_losses: list[float] = []
    for held_out in range(len(calibration)):
        train_q = [
            value for index, value in enumerate(q_sensitivity_folds) if index != held_out
        ]
        train_k = [
            value for index, value in enumerate(k_sensitivity_folds) if index != held_out
        ]
        q_importance_fold = _aggregate_attention_sensitivity(
            train_q if train_q else q_sensitivity_folds
        )
        k_importance_fold = _aggregate_attention_sensitivity(
            train_k if train_k else k_sensitivity_folds
        )
        sensitivity_losses.extend(
            _quantized_attention_loss(
                [calibration[held_out]],
                q_num_heads,
                kv_num_heads,
                head_dim,
                center_k=center_k,
                balance=balance,
                gamma=gamma,
                plus4=plus4,
                q_importance=q_importance_fold,
                k_importance=k_importance_fold,
            )
        )
    use_sensitivity = _accept_candidate(
        current_losses,
        sensitivity_losses,
        min_gain=0.0005,
        worst_tolerance=0.01,
    )
    if use_sensitivity:
        q_importance = _aggregate_attention_sensitivity(q_sensitivity_folds)
        k_importance = _aggregate_attention_sensitivity(k_sensitivity_folds)
    else:
        q_importance = torch.ones(
            (kv_num_heads, head_dim), dtype=torch.float32, device=device
        )
        k_importance = torch.ones(
            (kv_num_heads, head_dim), dtype=torch.float32, device=device
        )

    shared = {
        "q_heads": int(q_num_heads),
        "kv_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "center_k": bool(center_k),
        "balance": _cpu_tensor(balance),
        "gamma": _cpu_tensor(gamma),
        "plus4": bool(plus4),
        "sensitivity_weighted": bool(use_sensitivity),
    }
    return {
        "q_state": {**shared, "importance": _cpu_tensor(q_importance)},
        "k_state": {**shared, "importance": _cpu_tensor(k_importance)},
        "v_state": {
            "kv_heads": int(kv_num_heads),
            "head_dim": int(head_dim),
            "plus4": bool(plus4),
        },
    }


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    q = _dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    state = q_state
    kv_heads = int(state["kv_heads"])
    group = q_num_heads // kv_heads
    balance = state["balance"].to(device=q.device, dtype=torch.float32)
    gamma = state["gamma"].to(device=q.device, dtype=torch.float32)
    qh = _head_view(q, q_num_heads, head_dim)
    qh = qh * balance.repeat_interleave(group, dim=0).reshape(1, q_num_heads, head_dim)
    qh = qh * gamma.pow(0.35).repeat_interleave(group).reshape(1, q_num_heads, 1)
    importance = state["importance"].to(device=q.device, dtype=torch.float32)
    importance = importance.repeat_interleave(group, dim=0).reshape(-1)
    return _encode_hif4(
        qh.reshape_as(q),
        importance,
        refine_ratio=_ATTENTION_REFINE_RATIO,
        plus4=bool(state["plus4"]),
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    k = _dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    state = k_state
    balance = state["balance"].to(device=k.device, dtype=torch.float32)
    gamma = state["gamma"].to(device=k.device, dtype=torch.float32)
    kh = _head_view(k, kv_num_heads, head_dim)
    if bool(state["center_k"]):
        kh = kh - kh.mean(dim=0, keepdim=True)
    kh = kh / balance.reshape(1, kv_num_heads, head_dim)
    kh = kh * gamma.pow(0.65).reshape(1, kv_num_heads, 1)
    importance = state["importance"].to(device=k.device, dtype=torch.float32).reshape(-1)
    return _encode_hif4(
        kh.reshape_as(k),
        importance,
        refine_ratio=_ATTENTION_REFINE_RATIO,
        plus4=bool(state["plus4"]),
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    v = _dequantize_nvfp4(v_quant, v_scale).to(torch.float32)
    return _encode_hif4(
        v,
        refine_ratio=_ATTENTION_REFINE_RATIO,
        plus4=bool(v_state["plus4"]),
    )


__all__ = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]

