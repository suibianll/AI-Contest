"""Fixed joint Q/K block-scale product training on the current root."""

import math
from typing import Any, Optional

import torch


_A23_TRAIN_STEPS = 32
_A23_TRAIN_LR = 0.01
_A23_TRAIN_CLIP = 1.0
_A23_REG_WEIGHT = 0.001
_A23_BETA1 = 0.9
_A23_BETA2 = 0.999
_A23_SPECTRAL_BOUND = math.log(2.0) / 2.0
_A23_FIT_WINDOWS = 3
_A23_GATE_WINDOWS = (3, 4)


def _a23_exp(s: torch.Tensor, sign: float = 1.0) -> tuple[torch.Tensor, tuple]:
    values, vectors = torch.linalg.eigh(s)
    ev = (float(sign) * values).exp()
    result = (vectors * ev.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return result, (values, vectors, ev, float(sign))


def _a23_exp_backward(grad: torch.Tensor, cache: tuple) -> torch.Tensor:
    values, vectors, ev, sign = cache
    diff = values.unsqueeze(-1) - values.unsqueeze(-2)
    near = diff.abs() < 1.0e-6
    ratio = (ev.unsqueeze(-1) - ev.unsqueeze(-2)) / torch.where(
        near, torch.ones_like(diff), diff
    )
    midpoint = (
        float(sign) * (values.unsqueeze(-1) + values.unsqueeze(-2)) * 0.5
    ).exp() * float(sign)
    divided = torch.where(near, midpoint, ratio)
    local = vectors.transpose(-1, -2) @ (
        (grad + grad.transpose(-1, -2)) * 0.5
    ) @ vectors
    return vectors @ (local * divided) @ vectors.transpose(-1, -2)


def _a23_project(s: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(
        (s + s.transpose(-1, -2)) * 0.5
    )
    bound = _A23_SPECTRAL_BOUND
    lo = values.amin(-1, keepdim=True) - bound
    hi = values.amax(-1, keepdim=True) + bound
    for _ in range(32):
        mid = (lo + hi) * 0.5
        positive = (
            (values - mid).clamp(-bound, bound).sum(-1, keepdim=True) > 0
        )
        next_lo = torch.where(positive, mid, lo)
        next_hi = torch.where(positive, hi, mid)
        lo, hi = next_lo, next_hi
    values = (values - (lo + hi) * 0.5).clamp(-bound, bound)
    return (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _a23_group_block_amax2(
    x: torch.Tensor, heads: int, groups: int
) -> torch.Tensor:
    head_dim = int(x.shape[-1]) // int(heads)
    if head_dim <= 0 or head_dim % 64 != 0:
        raise ValueError("joint product objective requires 64-divisible heads")
    blocks = x.reshape(
        -1, int(groups), int(heads) // int(groups), head_dim // 64, 64
    )
    return blocks.square().amax(-1).mean(dim=(0, 2))


def _a23_qk_product_loss_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    parent_q: torch.Tensor,
    parent_k: torch.Tensor,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return joint product loss and exact max-with-tie coordinate gradients."""

    groups = int(kv_heads)
    q_head_dim = int(q.shape[-1]) // int(q_heads)
    k_head_dim = int(k.shape[-1]) // int(kv_heads)
    if q_head_dim != k_head_dim or q_head_dim % 64 != 0:
        raise ValueError("Q/K head dimensions must match 64-block layout")
    q_per_group = int(q_heads) // groups
    block_count = q_head_dim // 64
    q_blocks = q.reshape(
        -1, groups, q_per_group, block_count, 64
    )
    k_blocks = k.reshape(-1, groups, block_count, 64)
    q_square = q_blocks.square()
    k_square = k_blocks.square()
    q_max = q_square.amax(-1, keepdim=True)
    k_max = k_square.amax(-1, keepdim=True)
    a_q = q_max.squeeze(-1).mean(dim=(0, 2))
    a_k = k_max.squeeze(-1).mean(dim=0)
    denominator = (parent_q * parent_k).clamp_min(1.0e-12)
    ratio = a_q * a_k / denominator
    loss = ratio.mean()
    normalizer = float(max(groups * block_count, 1))

    q_ties = q_square == q_max
    k_ties = k_square == k_max
    q_tie_count = q_ties.sum(-1, keepdim=True).clamp_min(1)
    k_tie_count = k_ties.sum(-1, keepdim=True).clamp_min(1)
    q_grad = 2.0 * q_blocks * q_ties / q_tie_count
    q_grad = q_grad * (
        (a_k / denominator / normalizer).view(1, groups, 1, block_count, 1)
    )
    q_grad = q_grad / float(max(q_blocks.shape[0] * q_per_group, 1))
    k_grad = 2.0 * k_blocks * k_ties / k_tie_count
    k_grad = k_grad * (
        (a_q / denominator / normalizer).view(1, groups, block_count, 1)
    )
    k_grad = k_grad / float(max(k_blocks.shape[0], 1))
    return (
        loss,
        q_grad.reshape_as(q),
        k_grad.reshape_as(k),
        a_q,
        a_k,
    )


def _a23_matrix_grad(
    x: torch.Tensor, grad: torch.Tensor, heads: int, groups: int
) -> torch.Tensor:
    dim = int(x.shape[-1]) // int(heads)
    return torch.einsum(
        "tghi,tghj->gij",
        x.reshape(-1, groups, int(heads) // groups, dim),
        grad.reshape(-1, groups, int(heads) // groups, dim),
    )


def _a23_parent_coordinate(
    dense: torch.Tensor,
    state: dict[str, Any],
    heads: int,
    head_dim: int,
    is_k: bool,
) -> torch.Tensor:
    """Apply the current root's complete pre-residual coordinate stack."""

    out = _attention_state_transform_dense(
        dense, state, int(heads), int(head_dim), is_k=bool(is_k)
    )
    rotation = state.get("learned_rotation")
    if rotation is not None:
        out = _a2_apply_group_rotation(out, int(heads), rotation)
    center = state.get("learned_center") if is_k else None
    if center is not None:
        lead = out.shape[:-1]
        out = (
            out.reshape(*lead, int(heads), int(head_dim))
            + center.to(device=out.device, dtype=torch.float32).reshape(
                *([1] * len(lead)), int(heads), int(head_dim)
            )
        ).reshape_as(out)
    return out


def _a23_parent_copy(states: dict[str, Any]) -> dict[str, Any]:
    out = dict(states)
    for role in ("q_state", "k_state", "v_state"):
        state = dict(states[role])
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().to(device="cpu").clone()
        out[role] = state
    return out


@torch.no_grad()
def _a23_train(
    windows: list[dict[str, Any]],
    states: dict[str, Any],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    force_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
    """Train one symmetric zero-trace S using the joint Q/K product target."""

    if int(head_dim) <= 0 or int(head_dim) % 64 != 0:
        raise ValueError("joint product training requires a 64-divisible head dimension")
    if int(q_heads) % int(kv_heads) != 0:
        raise ValueError("GQA heads must divide evenly")
    parent_q_rotation = states["q_state"].get("learned_rotation")
    parent_k_rotation = states["k_state"].get("learned_rotation")
    parent_center = states["k_state"].get("learned_center")
    if parent_q_rotation is None:
        rq = torch.eye(int(head_dim), device=device).expand(
            int(kv_heads), int(head_dim), int(head_dim)
        ).clone()
    else:
        rq = parent_q_rotation.to(device=device, dtype=torch.float32).clone()
    if parent_k_rotation is None:
        rk = torch.eye(int(head_dim), device=device).expand(
            int(kv_heads), int(head_dim), int(head_dim)
        ).clone()
    else:
        rk = parent_k_rotation.to(device=device, dtype=torch.float32).clone()
    has_center = parent_center is not None
    center = (
        parent_center.to(device=device, dtype=torch.float32).clone()
        if has_center else torch.zeros(int(kv_heads), int(head_dim), device=device)
    )
    s = torch.zeros(
        int(kv_heads), int(head_dim), int(head_dim),
        device=device, dtype=torch.float32,
    )
    prepared = []
    for item in windows:
        q_raw = _dequantize_nvfp4_float32(*item["q"]).to(
            device=device, dtype=torch.float32
        )
        k_raw = _dequantize_nvfp4_float32(*item["k"]).to(
            device=device, dtype=torch.float32
        )
        q_coordinate = _a23_parent_coordinate(
            q_raw, states["q_state"], q_heads, head_dim, False
        )
        k_coordinate = _a23_parent_coordinate(
            k_raw, states["k_state"], kv_heads, head_dim, True
        )
        prepared.append((
            q_coordinate,
            k_coordinate,
            _a23_group_block_amax2(q_coordinate, q_heads, kv_heads),
            _a23_group_block_amax2(k_coordinate, kv_heads, kv_heads),
        ))
    if not prepared:
        raise ValueError("joint product training has no calibration windows")

    exp_avg = torch.zeros_like(s)
    exp_avg_sq = torch.zeros_like(s)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, _A23_TRAIN_STEPS + 1):
        exp_plus, cache_plus = _a23_exp(s)
        exp_minus, cache_minus = _a23_exp(s, -1.0)
        grad_plus = torch.zeros_like(s)
        grad_minus = torch.zeros_like(s)
        loss = s.square().mean() * _A23_REG_WEIGHT
        for q_coordinate, k_coordinate, parent_q, parent_k in prepared:
            transformed_q = _a2_apply_group_rotation(
                q_coordinate, int(q_heads), exp_plus
            )
            transformed_k = _a2_apply_group_rotation(
                k_coordinate, int(kv_heads), exp_minus
            )
            value, grad_q, grad_k, _, _ = _a23_qk_product_loss_grad(
                transformed_q,
                transformed_k,
                parent_q,
                parent_k,
                q_heads,
                kv_heads,
            )
            loss = loss + value / float(len(prepared))
            grad_plus.add_(
                _a23_matrix_grad(
                    q_coordinate, grad_q, int(q_heads), int(kv_heads)
                ) / float(len(prepared))
            )
            grad_minus.add_(
                _a23_matrix_grad(
                    k_coordinate, grad_k, int(kv_heads), int(kv_heads)
                ) / float(len(prepared))
            )
        if step == 1:
            initial_loss = float(loss.item())
        if force_zero:
            continue
        gradient = (
            _a23_exp_backward(grad_plus, cache_plus)
            + _a23_exp_backward(grad_minus, cache_minus)
            + 2.0 * _A23_REG_WEIGHT * s / float(s.numel())
        )
        gradient = (gradient + gradient.transpose(-1, -2)) * 0.5
        diagonal_mean = gradient.diagonal(dim1=-2, dim2=-1).mean(
            -1, keepdim=True
        )
        gradient = gradient - torch.diag_embed(
            diagonal_mean.expand(int(kv_heads), int(head_dim))
        )
        norm = float(gradient.norm().item())
        if norm > _A23_TRAIN_CLIP:
            gradient = gradient * (_A23_TRAIN_CLIP / norm)
        exp_avg.mul_(_A23_BETA1).add_(gradient, alpha=1.0 - _A23_BETA1)
        exp_avg_sq.mul_(_A23_BETA2).addcmul_(
            gradient, gradient, value=1.0 - _A23_BETA2
        )
        bias1 = 1.0 - _A23_BETA1 ** step
        bias2 = 1.0 - _A23_BETA2 ** step
        update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1.0e-8)
        s = _a23_project(s - _A23_TRAIN_LR * update)

    if force_zero:
        exp_plus = exp_minus = torch.eye(
            int(head_dim), device=device, dtype=torch.float32
        ).expand(int(kv_heads), int(head_dim), int(head_dim))
    else:
        exp_plus, _ = _a23_exp(s)
        exp_minus, _ = _a23_exp(s, -1.0)
        final_loss = float(s.square().mean().item() * _A23_REG_WEIGHT)
        for q_coordinate, k_coordinate, parent_q, parent_k in prepared:
            value, _, _, _, _ = _a23_qk_product_loss_grad(
                _a2_apply_group_rotation(q_coordinate, int(q_heads), exp_plus),
                _a2_apply_group_rotation(k_coordinate, int(kv_heads), exp_minus),
                parent_q,
                parent_k,
                q_heads,
                kv_heads,
            )
            final_loss += float(value.item()) / float(len(prepared))

    transformed_q = rq @ exp_plus
    transformed_k = rk @ exp_minus
    center_new = (
        (center.unsqueeze(-2) @ exp_minus).squeeze(-2)
        if has_center else None
    )
    inverse_error = float(
        (
            transformed_q @ transformed_k.transpose(-1, -2)
            - rq @ rk.transpose(-1, -2)
        ).abs().max().item()
    )
    parent_product = sum(
        float((parent_q * parent_k).mean().item())
        for _, _, parent_q, parent_k in prepared
    ) / float(len(prepared))
    candidate_q_values = []
    candidate_k_values = []
    candidate_product = 0.0
    for q_coordinate, k_coordinate, parent_q, parent_k in prepared:
        candidate_q = _a23_group_block_amax2(
            _a2_apply_group_rotation(q_coordinate, int(q_heads), exp_plus),
            q_heads,
            kv_heads,
        )
        candidate_k = _a23_group_block_amax2(
            _a2_apply_group_rotation(k_coordinate, int(kv_heads), exp_minus),
            kv_heads,
            kv_heads,
        )
        candidate_q_values.append(float(candidate_q.mean().item()))
        candidate_k_values.append(float(candidate_k.mean().item()))
        candidate_product += float((candidate_q * candidate_k).mean().item())
    candidate_product /= float(len(prepared))
    info = {
        "a23_steps": _A23_TRAIN_STEPS,
        "a23_fit_windows": len(prepared),
        "a23_attempted_groups": int(kv_heads),
        "a23_initial_loss": initial_loss,
        "a23_final_loss": final_loss,
        "a23_parent_product_mean": parent_product,
        "a23_candidate_product_mean": candidate_product,
        "a23_product_ratio": candidate_product / max(parent_product, 1.0e-12),
        "a23_q_amax2_mean": sum(candidate_q_values) / len(candidate_q_values),
        "a23_k_amax2_mean": sum(candidate_k_values) / len(candidate_k_values),
        "a23_inverse_error": inverse_error,
        "a23_s_norm": float(s.norm().item()),
        "a23_s_max_abs": float(s.abs().max().item()),
        "a23_parent_arm": "rotation" if parent_q_rotation is not None else "identity",
        "a23_center_compiled": bool(has_center),
    }
    return (
        transformed_q.detach().to(device="cpu").contiguous(),
        transformed_k.detach().to(device="cpu").contiguous(),
        None if center_new is None else center_new.detach().to(device="cpu").contiguous(),
        info,
    )


@torch.no_grad()
def _a23_gate_loss(
    item: dict[str, Any],
    states: dict[str, Any],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> float:
    decoded = []
    reference = []
    for role, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
        api = {"q": _A23_PARENT_Q, "k": _A23_PARENT_K, "v": _A23_PARENT_V}[role]
        params = api(*item[role], heads, head_dim, states[role + "_state"])
        decoded.append(_dequantize_hif4(params).to(torch.float32)[None])
        reference.append(
            _dequantize_nvfp4_float32(*item[role]).to(torch.float32)[None]
        )
    actual = _a2_attention_forward(
        decoded[0], decoded[1], decoded[2], q_heads, kv_heads, head_dim
    )
    target = _a2_attention_forward(
        reference[0], reference[1], reference[2], q_heads, kv_heads, head_dim
    )
    return float((actual - target).square().mean().item())


def _a23_annotated_states(
    states: dict[str, Any], info: dict[str, Any]
) -> dict[str, Any]:
    out = _a23_parent_copy(states)
    out["q_state"].update(info)
    out["k_state"].update(info)
    return out


_A23_PARENT_CALIBRATION = hif4_calibration_attention
_A23_PARENT_Q = hif4_dynamic_quantize_q
_A23_PARENT_K = hif4_dynamic_quantize_k
_A23_PARENT_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    states = _A23_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    info = {
        "a23_arm": "fallback",
        "a23_attempted": 0,
        "a23_accepted": 0,
        "a23_fit_windows": 0,
        "a23_gate_windows": len(_A23_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < max(_A23_GATE_WINDOWS) + 1
        or int(q_num_heads) % int(kv_num_heads) != 0
    ):
        info["a23_arm"] = "ineligible"
        return _a23_annotated_states(states, info)
    try:
        fit_windows = calib_qkv_list[:len(calib_qkv_list) - len(_A23_GATE_WINDOWS)]
        train_device = calib_qkv_list[0]["q"][0].device
        tq, tk, center, train_info = _a23_train(
            fit_windows,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
            train_device,
        )
        candidate = _a23_parent_copy(states)
        candidate["q_state"]["learned_rotation"] = tq
        candidate["k_state"]["learned_rotation"] = tk
        if center is not None:
            candidate["k_state"]["learned_center"] = center
        gate_records = []
        accepted = True
        for index in _A23_GATE_WINDOWS:
            parent_loss = _a23_gate_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _a23_gate_loss(
                calib_qkv_list[index], candidate,
                q_num_heads, kv_num_heads, head_dim,
            )
            current_pass = candidate_loss < parent_loss
            accepted = accepted and current_pass
            gate_records.append({
                "window": int(index),
                "pass": bool(current_pass),
                "parent_mse": float(parent_loss),
                "candidate_mse": float(candidate_loss),
            })
        info.update(train_info)
        info["a23_attempted"] = 1
        info["a23_fit_windows"] = len(fit_windows)
        info["a23_gate"] = gate_records
        info["a23_gate_parent_mse"] = float(
            sum(item["parent_mse"] for item in gate_records) / len(gate_records)
        )
        info["a23_gate_candidate_mse"] = float(
            sum(item["candidate_mse"] for item in gate_records) / len(gate_records)
        )
        if accepted:
            info["a23_arm"] = "accepted"
            info["a23_accepted"] = 1
            return _a23_annotated_states(candidate, info)
        info["a23_arm"] = "parent"
        return _a23_annotated_states(states, info)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        info["a23_arm"] = "fallback"
        info["a23_error"] = f"{type(exc).__name__}: {exc}"
        return _a23_annotated_states(states, info)


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _A23_PARENT_Q(q_quant, q_scale, q_num_heads, head_dim, q_state)


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _A23_PARENT_K(k_quant, k_scale, kv_num_heads, head_dim, k_state)


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _A23_PARENT_V(v_quant, v_scale, kv_num_heads, head_dim, v_state)
