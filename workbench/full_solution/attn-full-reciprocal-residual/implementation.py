"""Fixed full-matrix reciprocal Q/K residual training on the current root."""

import math
from typing import Any, Optional

import torch


_A21_TRAIN_STEPS = 32
_A21_TRAIN_LR = 0.01
_A21_TRAIN_CLIP = 1.0
_A21_REG_WEIGHT = 0.001
_A21_BETA1 = 0.9
_A21_BETA2 = 0.999
_A21_SPECTRAL_BOUND = math.log(2.0) / 2.0
_A21_FIT_WINDOWS = 3
_A21_GATE_WINDOWS = (3, 4)


def _a21_exp(s: torch.Tensor, sign: float = 1.0) -> tuple[torch.Tensor, tuple]:
    values, vectors = torch.linalg.eigh(s)
    ev = (float(sign) * values).exp()
    result = (vectors * ev.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return result, (values, vectors, ev, float(sign))


def _a21_exp_backward(grad: torch.Tensor, cache: tuple) -> torch.Tensor:
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


def _a21_project(s: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(
        (s + s.transpose(-1, -2)) * 0.5
    )
    bound = _A21_SPECTRAL_BOUND
    lo = values.amin(-1, keepdim=True) - bound
    hi = values.amax(-1, keepdim=True) + bound
    # Exact fixed 32-step projection onto the spectral box and zero-trace
    # plane, matching the registered reference implementation.
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


def _a21_scale_loss_grad(
    x: torch.Tensor, denominator: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = x.reshape(-1, x.shape[-1] // 64, 64)
    amax = blocks.abs().amax(-1, keepdim=True)
    denom = denominator.clamp_min(1.0e-12)
    loss = (amax / denom).square().mean()
    ties = blocks.abs() == amax
    grad = 2.0 * amax / denom.square() / float(max(int(amax.numel()), 1))
    grad = grad * blocks.sign() * ties / ties.sum(-1, keepdim=True)
    return loss, grad.reshape_as(x)


def _a21_matrix_grad(
    x: torch.Tensor, grad: torch.Tensor, heads: int, groups: int
) -> torch.Tensor:
    dim = int(x.shape[-1]) // int(heads)
    return torch.einsum(
        "tghi,tghj->gij",
        x.reshape(-1, groups, int(heads) // groups, dim),
        grad.reshape(-1, groups, int(heads) // groups, dim),
    )


def _a21_parent_coordinate(
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


def _a21_parent_copy(states: dict[str, Any]) -> dict[str, Any]:
    out = dict(states)
    for role in ("q_state", "k_state", "v_state"):
        state = dict(states[role])
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().to(device="cpu").clone()
        out[role] = state
    return out


@torch.no_grad()
def _a22b_train(
    windows: list[dict[str, Any]],
    states: dict[str, Any],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    force_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
    """Train one symmetric zero-trace S in the complete current-root frame."""

    if int(head_dim) <= 0 or int(head_dim) % 64 != 0:
        raise ValueError("Residual training requires a 64-divisible head dimension")
    parent_q = states["q_state"].get("learned_rotation")
    parent_k = states["k_state"].get("learned_rotation")
    parent_center = states["k_state"].get("learned_center")
    if parent_q is None:
        rq = torch.eye(int(head_dim), device=device).expand(
            int(kv_heads), int(head_dim), int(head_dim)
        ).clone()
    else:
        rq = parent_q.to(device=device, dtype=torch.float32).clone()
    if parent_k is None:
        rk = torch.eye(int(head_dim), device=device).expand(
            int(kv_heads), int(head_dim), int(head_dim)
        ).clone()
    else:
        rk = parent_k.to(device=device, dtype=torch.float32).clone()
    has_center = parent_center is not None
    if has_center:
        center = parent_center.to(device=device, dtype=torch.float32).clone()
    else:
        center = torch.zeros(int(kv_heads), int(head_dim), device=device)

    s = torch.zeros(
        int(kv_heads), int(head_dim), int(head_dim),
        device=device, dtype=torch.float32,
    )
    prepared = []
    for item in windows:
        fold = []
        for role, heads in (("q", q_heads), ("k", kv_heads)):
            raw = _dequantize_nvfp4_float32(*item[role]).to(
                device=device, dtype=torch.float32
            )
            coordinate = _a21_parent_coordinate(
                raw, states[role + "_state"], heads, head_dim, role == "k"
            )
            denominator = coordinate.reshape(
                -1, int(coordinate.shape[-1]) // 64, 64
            ).abs().amax(-1, keepdim=True)
            fold.append((coordinate, denominator, heads))
        prepared.append(fold)
    if not prepared:
        raise ValueError("Residual training has no calibration windows")

    exp_avg = torch.zeros_like(s)
    exp_avg_sq = torch.zeros_like(s)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, _A21_TRAIN_STEPS + 1):
        exp_plus, cache_plus = _a21_exp(s)
        exp_minus, cache_minus = _a21_exp(s, -1.0)
        grad_plus = torch.zeros_like(s)
        grad_minus = torch.zeros_like(s)
        loss = s.square().mean() * _A21_REG_WEIGHT
        for fold in prepared:
            for (coordinate, denominator, heads), matrix, grad_matrix in zip(
                fold, (exp_plus, exp_minus), (grad_plus, grad_minus)
            ):
                transformed = _a2_apply_group_rotation(
                    coordinate, int(heads), matrix
                )
                value, dx = _a21_scale_loss_grad(transformed, denominator)
                loss = loss + value / float(len(prepared))
                grad_matrix.add_(
                    _a21_matrix_grad(coordinate, dx, int(heads), int(kv_heads))
                    / float(len(prepared))
                )
        if step == 1:
            initial_loss = float(loss.item())
        if force_zero:
            continue
        gradient = (
            _a21_exp_backward(grad_plus, cache_plus)
            + _a21_exp_backward(grad_minus, cache_minus)
            + 2.0 * _A21_REG_WEIGHT * s / float(s.numel())
        )
        gradient = (gradient + gradient.transpose(-1, -2)) * 0.5
        diagonal_mean = gradient.diagonal(dim1=-2, dim2=-1).mean(
            -1, keepdim=True
        )
        gradient = gradient - torch.diag_embed(
            diagonal_mean.expand(int(kv_heads), int(head_dim))
        )
        norm = float(gradient.norm().item())
        if norm > _A21_TRAIN_CLIP:
            gradient = gradient * (_A21_TRAIN_CLIP / norm)
        exp_avg.mul_(_A21_BETA1).add_(gradient, alpha=1.0 - _A21_BETA1)
        exp_avg_sq.mul_(_A21_BETA2).addcmul_(
            gradient, gradient, value=1.0 - _A21_BETA2
        )
        bias1 = 1.0 - _A21_BETA1 ** step
        bias2 = 1.0 - _A21_BETA2 ** step
        update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1.0e-8)
        s = _a21_project(s - _A21_TRAIN_LR * update)

    if force_zero:
        exp_plus = exp_minus = torch.eye(
            int(head_dim), device=device, dtype=torch.float32
        ).expand(int(kv_heads), int(head_dim), int(head_dim))
    else:
        exp_plus, _ = _a21_exp(s)
        exp_minus, _ = _a21_exp(s, -1.0)
        final_loss = float(
            s.square().mean().item() * _A21_REG_WEIGHT
        )
        for fold in prepared:
            for (coordinate, denominator, heads), matrix in zip(
                fold, (exp_plus, exp_minus)
            ):
                value, _ = _a21_scale_loss_grad(
                    _a2_apply_group_rotation(coordinate, int(heads), matrix),
                    denominator,
                )
                final_loss += float(value.item()) / float(len(prepared))

    # Compile into the existing root fields.  The center follows the same
    # exp(-S) transform so the additive K term stays in the same reciprocal
    # coordinate frame.
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
    info = {
        "a22b_steps": _A21_TRAIN_STEPS,
        "a22b_fit_windows": len(prepared),
        "a22b_attempted_groups": int(kv_heads),
        "a22b_initial_loss": initial_loss,
        "a22b_final_loss": final_loss,
        "a22b_q_scale_ratio2": 0.0 if not prepared else float(
            sum(
                float(_a21_scale_loss_grad(
                    _a2_apply_group_rotation(fold[0][0], int(fold[0][2]), exp_plus),
                    fold[0][1],
                )[0].item()
                ) for fold in prepared
            ) / float(len(prepared))
        ),
        "a22b_k_scale_ratio2": 0.0 if not prepared else float(
            sum(
                float(_a21_scale_loss_grad(
                    _a2_apply_group_rotation(fold[1][0], int(fold[1][2]), exp_minus),
                    fold[1][1],
                )[0].item()
                ) for fold in prepared
            ) / float(len(prepared))
        ),
        "a22b_inverse_error": inverse_error,
        "a22b_s_norm": float(s.norm().item()),
        "a22b_s_max_abs": float(s.abs().max().item()),
        "a22b_parent_arm": "rotation" if parent_q is not None else "identity",
        "a22b_center_compiled": bool(has_center),
    }
    return (
        transformed_q.detach().to(device="cpu").contiguous(),
        transformed_k.detach().to(device="cpu").contiguous(),
        None if center_new is None else center_new.detach().to(device="cpu").contiguous(),
        info,
    )


@torch.no_grad()
def _a21_gate_loss(
    item: dict[str, Any],
    states: dict[str, Any],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> float:
    decoded = []
    reference = []
    for role, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
        api = {
            "q": _A21_PARENT_Q,
            "k": _A21_PARENT_K,
            "v": _A21_PARENT_V,
        }[role]
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


def _a21_annotated_states(
    states: dict[str, Any], info: dict[str, Any]
) -> dict[str, Any]:
    out = _a21_parent_copy(states)
    out["q_state"].update(info)
    out["k_state"].update(info)
    return out


_A21_PARENT_CALIBRATION = hif4_calibration_attention
_A21_PARENT_Q = hif4_dynamic_quantize_q
_A21_PARENT_K = hif4_dynamic_quantize_k
_A21_PARENT_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    states = _A21_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    info = {
        "a22b_arm": "fallback",
        "a22b_attempted": 0,
        "a22b_accepted": 0,
        "a22b_fit_windows": 0,
        "a22b_gate_windows": len(_A21_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < max(_A21_GATE_WINDOWS) + 1
        or int(q_num_heads) % int(kv_num_heads) != 0
    ):
        info["a22b_arm"] = "ineligible"
        return _a21_annotated_states(states, info)
    try:
        fit_windows = calib_qkv_list[:len(calib_qkv_list) - len(_A21_GATE_WINDOWS)]
        train_device = calib_qkv_list[0]["q"][0].device
        tq, tk, center, train_info = _a22b_train(
            fit_windows,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
            train_device,
        )
        candidate = _a21_parent_copy(states)
        candidate["q_state"]["learned_rotation"] = tq
        candidate["k_state"]["learned_rotation"] = tk
        if center is not None:
            candidate["k_state"]["learned_center"] = center
        gate_records = []
        accepted = True
        for index in _A21_GATE_WINDOWS:
            parent_loss = _a21_gate_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _a21_gate_loss(
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
        info["a22b_attempted"] = 1
        info["a22b_fit_windows"] = len(fit_windows)
        info["a22b_gate"] = gate_records
        info["a22b_gate_parent_mse"] = float(
            sum(item["parent_mse"] for item in gate_records) / len(gate_records)
        )
        info["a22b_gate_candidate_mse"] = float(
            sum(item["candidate_mse"] for item in gate_records) / len(gate_records)
        )
        if accepted:
            info["a22b_arm"] = "accepted"
            info["a22b_accepted"] = 1
            return _a21_annotated_states(candidate, info)
        info["a22b_arm"] = "parent"
        return _a21_annotated_states(states, info)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        info["a22b_arm"] = "fallback"
        info["a22b_error"] = f"{type(exc).__name__}: {exc}"
        return _a21_annotated_states(states, info)


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _A21_PARENT_Q(
        q_quant, q_scale, q_num_heads, head_dim, q_state
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _A21_PARENT_K(
        k_quant, k_scale, kv_num_heads, head_dim, k_state
    )


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _A21_PARENT_V(
        v_quant, v_scale, kv_num_heads, head_dim, v_state
    )
