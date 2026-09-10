"""A-GR1 mechanism block, appended to the current-root copy by build step."""

AGR1_BLOCK = r'''

# ---------------------------------------------------------------------------
# A-GR1 (2026-09-10): general asymmetric reciprocal matrix residual.
#
# Single-variable generalization of v192 (the only officially positive member
# of the reciprocal-residual family, side-isolated official +22): v192 trained
# a symmetric zero-trace S with Q @ exp(S), K @ exp(-S), where exp(S) is
# necessarily SPD.  A-GR1 replaces it with a general M = I + N (N
# unconstrained); the K side receives M^{-T}.  Everything else mirrors v192:
# same fit windows (0,1,2), same gate windows (3,4), same 32-step Adam
# (lr 0.01, clip 1.0, reg 1e-3), same per-call parent coordinate, same
# amax scale-loss objective, same strict all-gate-windows improvement rule,
# same compile into the existing learned_rotation / learned_center fields
# (so the dynamic path stays a single matmul and never inverts a matrix).
# The v192 spectral box +-log(2)/2 on eigenvalues of S becomes a singular
# value clamp of M to [1/sqrt(2), sqrt(2)] after every step.
# The root's own calibration (v195 A2 rotation/center and its gate) runs
# first and is frozen; A-GR1 composes on top of whatever arm the root
# selected, exactly like v192 composed on its parent.
# ---------------------------------------------------------------------------

_AGR1_TRAIN_STEPS = 32
_AGR1_TRAIN_LR = 0.01
_AGR1_TRAIN_CLIP = 1.0
_AGR1_REG_WEIGHT = 0.001
_AGR1_BETA1 = 0.9
_AGR1_BETA2 = 0.999
_AGR1_SINGULAR_LO = 2.0 ** -0.5
_AGR1_SINGULAR_HI = 2.0 ** 0.5
_AGR1_FIT_WINDOWS = 3
_AGR1_GATE_WINDOWS = (3, 4)


def _agr1_scale_loss_grad(
    x: torch.Tensor, denominator: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    blocks = x.reshape(-1, x.shape[-1] // 64, 64)
    amax = blocks.abs().amax(-1, keepdim=True)
    denom = denominator.clamp_min(1.0e-12)
    loss = (amax / denom).square().mean()
    ties = blocks.abs() == amax
    grad = 2.0 * amax / denom.square() / float(max(int(amax.numel()), 1))
    grad = grad * blocks.sign() * ties / ties.sum(-1, keepdim=True)
    return loss, grad.reshape_as(x)


def _agr1_matrix_grad(
    x: torch.Tensor, grad: torch.Tensor, heads: int, groups: int
) -> torch.Tensor:
    dim = int(x.shape[-1]) // int(heads)
    return torch.einsum(
        "tghi,tghj->gij",
        x.reshape(-1, groups, int(heads) // groups, dim),
        grad.reshape(-1, groups, int(heads) // groups, dim),
    )


def _agr1_project(m: torch.Tensor) -> torch.Tensor:
    """Clamp singular values of M to [1/sqrt(2), sqrt(2)] (v192 spectral box)."""

    u, sv, vh = torch.linalg.svd(m)
    sv = sv.clamp(min=_AGR1_SINGULAR_LO, max=_AGR1_SINGULAR_HI)
    return (u * sv.unsqueeze(-2)) @ vh


def _agr1_parent_coordinate(
    dense: torch.Tensor,
    state: dict,
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


def _agr1_parent_copy(states: dict) -> dict:
    out = dict(states)
    for role in ("q_state", "k_state", "v_state"):
        state = dict(states[role])
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().to(device="cpu").clone()
        out[role] = state
    return out


@torch.no_grad()
def _agr1_train(
    windows: list,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    force_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """Train one general M = I + N per KV group in the frozen parent frame."""

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

    n = torch.zeros(
        int(kv_heads), int(head_dim), int(head_dim),
        device=device, dtype=torch.float32,
    )
    eye = torch.eye(int(head_dim), device=device, dtype=torch.float32)
    prepared = []
    for item in windows:
        fold = []
        for role, heads in (("q", q_heads), ("k", kv_heads)):
            raw = _dequantize_nvfp4_float32(*item[role]).to(
                device=device, dtype=torch.float32
            )
            coordinate = _agr1_parent_coordinate(
                raw, states[role + "_state"], heads, head_dim, role == "k"
            )
            denominator = coordinate.reshape(
                -1, int(coordinate.shape[-1]) // 64, 64
            ).abs().amax(-1, keepdim=True)
            fold.append((coordinate, denominator, heads))
        prepared.append(fold)
    if not prepared:
        raise ValueError("Residual training has no calibration windows")

    exp_avg = torch.zeros_like(n)
    exp_avg_sq = torch.zeros_like(n)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, _AGR1_TRAIN_STEPS + 1):
        m = eye + n
        m_inv = torch.linalg.inv(m)
        p = m_inv.transpose(-1, -2)
        grad_m = torch.zeros_like(n)
        grad_p = torch.zeros_like(n)
        loss = n.square().mean() * _AGR1_REG_WEIGHT
        for fold in prepared:
            for (coordinate, denominator, heads), matrix, grad_matrix in zip(
                fold, (m, p), (grad_m, grad_p)
            ):
                transformed = _a2_apply_group_rotation(
                    coordinate, int(heads), matrix
                )
                value, dx = _agr1_scale_loss_grad(transformed, denominator)
                loss = loss + value / float(len(prepared))
                grad_matrix.add_(
                    _agr1_matrix_grad(coordinate, dx, int(heads), int(kv_heads))
                    / float(len(prepared))
                )
        if step == 1:
            initial_loss = float(loss.item())
        if force_zero:
            continue
        gradient = (
            grad_m
            - p @ grad_p.transpose(-1, -2) @ p
            + 2.0 * _AGR1_REG_WEIGHT * n / float(n.numel())
        )
        norm = float(gradient.norm().item())
        if norm > _AGR1_TRAIN_CLIP:
            gradient = gradient * (_AGR1_TRAIN_CLIP / norm)
        exp_avg.mul_(_AGR1_BETA1).add_(gradient, alpha=1.0 - _AGR1_BETA1)
        exp_avg_sq.mul_(_AGR1_BETA2).addcmul_(
            gradient, gradient, value=1.0 - _AGR1_BETA2
        )
        bias1 = 1.0 - _AGR1_BETA1 ** step
        bias2 = 1.0 - _AGR1_BETA2 ** step
        update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1.0e-8)
        n = _agr1_project(eye + n - _AGR1_TRAIN_LR * update) - eye

    if force_zero:
        m = eye.expand(int(kv_heads), int(head_dim), int(head_dim))
        p = m
    else:
        m = eye + n
        p = torch.linalg.inv(m).transpose(-1, -2)
        final_loss = float(n.square().mean().item()) * _AGR1_REG_WEIGHT
        for fold in prepared:
            for (coordinate, denominator, heads), matrix in zip(fold, (m, p)):
                value, _ = _agr1_scale_loss_grad(
                    _a2_apply_group_rotation(coordinate, int(heads), matrix),
                    denominator,
                )
                final_loss += float(value.item()) / float(len(prepared))

    # Compile into the existing root fields, exactly like v192: the K center
    # follows the same M^{-T} transform so the additive K term stays in the
    # same reciprocal coordinate frame and logits are exactly preserved.
    transformed_q = rq @ m
    transformed_k = rk @ p
    center_new = (
        (center.unsqueeze(-2) @ p).squeeze(-2)
        if has_center else None
    )
    inverse_error = float(
        (
            transformed_q @ transformed_k.transpose(-1, -2)
            - rq @ rk.transpose(-1, -2)
        ).abs().max().item()
    )
    singular = torch.linalg.svdvals(m)
    info = {
        "agr1_steps": _AGR1_TRAIN_STEPS,
        "agr1_fit_windows": len(prepared),
        "agr1_attempted_groups": int(kv_heads),
        "agr1_initial_loss": initial_loss,
        "agr1_final_loss": final_loss,
        "agr1_q_scale_ratio2": 0.0 if not prepared else float(
            sum(
                float(_agr1_scale_loss_grad(
                    _a2_apply_group_rotation(fold[0][0], int(fold[0][2]), m),
                    fold[0][1],
                )[0].item())
                for fold in prepared
            ) / float(len(prepared))
        ),
        "agr1_k_scale_ratio2": 0.0 if not prepared else float(
            sum(
                float(_agr1_scale_loss_grad(
                    _a2_apply_group_rotation(fold[1][0], int(fold[1][2]), p),
                    fold[1][1],
                )[0].item())
                for fold in prepared
            ) / float(len(prepared))
        ),
        "agr1_inverse_error": inverse_error,
        "agr1_n_norm": float(n.norm().item()),
        "agr1_n_max_abs": float(n.abs().max().item()),
        "agr1_singular_min": float(singular.min().item()),
        "agr1_singular_max": float(singular.max().item()),
        "agr1_parent_arm": "rotation" if parent_q is not None else "identity",
        "agr1_center_compiled": bool(has_center),
    }
    return (
        transformed_q.detach().to(device="cpu").contiguous(),
        transformed_k.detach().to(device="cpu").contiguous(),
        None if center_new is None else center_new.detach().to(device="cpu").contiguous(),
        info,
    )


@torch.no_grad()
def _agr1_gate_loss(
    item: dict,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> float:
    decoded = []
    reference = []
    for role, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
        api = {
            "q": _AGR1_PARENT_Q,
            "k": _AGR1_PARENT_K,
            "v": _AGR1_PARENT_V,
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


def _agr1_annotated_states(states: dict, info: dict) -> dict:
    out = _agr1_parent_copy(states)
    out["q_state"].update(info)
    out["k_state"].update(info)
    return out


_AGR1_PARENT_CALIBRATION = hif4_calibration_attention
_AGR1_PARENT_Q = hif4_dynamic_quantize_q
_AGR1_PARENT_K = hif4_dynamic_quantize_k
_AGR1_PARENT_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict:
    states = _AGR1_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    info = {
        "agr1_arm": "fallback",
        "agr1_attempted": 0,
        "agr1_accepted": 0,
        "agr1_fit_windows": 0,
        "agr1_gate_windows": len(_AGR1_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < max(_AGR1_GATE_WINDOWS) + 1
        or int(q_num_heads) % int(kv_num_heads) != 0
    ):
        info["agr1_arm"] = "ineligible"
        return _agr1_annotated_states(states, info)
    try:
        fit_windows = calib_qkv_list[:len(calib_qkv_list) - len(_AGR1_GATE_WINDOWS)]
        train_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tq, tk, center, train_info = _agr1_train(
            fit_windows,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
            train_device,
        )
        candidate = _agr1_parent_copy(states)
        candidate["q_state"]["learned_rotation"] = tq
        candidate["k_state"]["learned_rotation"] = tk
        if center is not None:
            candidate["k_state"]["learned_center"] = center
        gate_records = []
        accepted = True
        for index in _AGR1_GATE_WINDOWS:
            parent_loss = _agr1_gate_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _agr1_gate_loss(
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
        info["agr1_attempted"] = 1
        info["agr1_fit_windows"] = len(fit_windows)
        info["agr1_gate_parent_mse"] = float(
            sum(item["parent_mse"] for item in gate_records) / len(gate_records)
        )
        info["agr1_gate_candidate_mse"] = float(
            sum(item["candidate_mse"] for item in gate_records) / len(gate_records)
        )
        if accepted:
            info["agr1_arm"] = "accepted"
            info["agr1_accepted"] = 1
            return _agr1_annotated_states(candidate, info)
        info["agr1_arm"] = "parent"
        return _agr1_annotated_states(states, info)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        info["agr1_arm"] = "fallback"
        info["agr1_error"] = f"{type(exc).__name__}: {exc}"
        return _agr1_annotated_states(states, info)
'''
