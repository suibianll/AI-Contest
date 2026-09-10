"""A-GR2 mechanism block, appended to the v231 archive copy by build.py."""

AGR2_BLOCK = r'''

# ---------------------------------------------------------------------------
# A-GR2 (2026-09-10): general asymmetric reciprocal matrix residual trained
# on the TRUE deployed-path objective.
#
# Single-variable change over A-GR1 (v234, side-isolated official +29): the
# training objective switches from the per-64-block amax scale-ratio proxy to
# the gate's own true deployed-path output MSE (full hif4_dynamic_quantize_q/k/v
# encoding path, STE backward through the quantizer -- the A-FIX1 engineering
# piece, but training only M; rotation/center stay frozen).  Everything else
# is byte-for-byte the A-GR1 design: M = I + N per KV group, Q @ M and
# K @ M^{-T} with the exact inverse computed once at calibration compile time,
# center compiled as c @ M^{-T}, fit windows 0,1,2 / gate windows 3,4,
# 32-step Adam (lr 0.01, clip 1.0, reg 1e-3, betas 0.9/0.999), singular-value
# clamp of M to [1/sqrt(2), sqrt(2)] after every step, and the per-layer
# all-gate-windows strict-improvement rule.  The dynamic path stays a single
# matmul and never inverts a matrix.
# ---------------------------------------------------------------------------

_AGR2_TRAIN_STEPS = 32
_AGR2_TRAIN_LR = 0.01
_AGR2_TRAIN_CLIP = 1.0
_AGR2_REG_WEIGHT = 0.001
_AGR2_BETA1 = 0.9
_AGR2_BETA2 = 0.999
_AGR2_SINGULAR_LO = 2.0 ** -0.5
_AGR2_SINGULAR_HI = 2.0 ** 0.5
_AGR2_FIT_WINDOWS = 3
_AGR2_GATE_WINDOWS = (3, 4)


def _agr2_matrix_grad(
    x: torch.Tensor, grad: torch.Tensor, heads: int, groups: int
) -> torch.Tensor:
    dim = int(x.shape[-1]) // int(heads)
    return torch.einsum(
        "tghi,tghj->gij",
        x.reshape(-1, groups, int(heads) // groups, dim),
        grad.reshape(-1, groups, int(heads) // groups, dim),
    )


def _agr2_project(m: torch.Tensor) -> torch.Tensor:
    """Clamp singular values of M to [1/sqrt(2), sqrt(2)] (v192 spectral box)."""

    u, sv, vh = torch.linalg.svd(m)
    sv = sv.clamp(min=_AGR2_SINGULAR_LO, max=_AGR2_SINGULAR_HI)
    return (u * sv.unsqueeze(-2)) @ vh


def _agr2_parent_coordinate(
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


def _agr2_parent_copy(states: dict) -> dict:
    out = dict(states)
    for role in ("q_state", "k_state", "v_state"):
        state = dict(states[role])
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().to(device="cpu").clone()
        out[role] = state
    return out


@torch.no_grad()
def _agr2_train(
    windows: list,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    force_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """Train one general M = I + N per KV group on the true deployed MSE."""

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
        q_pair = (
            item["q"][0].detach().to(device=device, dtype=torch.float32),
            item["q"][1].detach().to(device=device, dtype=torch.float32),
        )
        k_pair = (
            item["k"][0].detach().to(device=device, dtype=torch.float32),
            item["k"][1].detach().to(device=device, dtype=torch.float32),
        )
        v_pair = (
            item["v"][0].detach().to(device=device, dtype=torch.float32),
            item["v"][1].detach().to(device=device, dtype=torch.float32),
        )
        q_dense = _dequantize_nvfp4_float32(*q_pair).to(torch.float32)
        k_dense = _dequantize_nvfp4_float32(*k_pair).to(torch.float32)
        v_dense = _dequantize_nvfp4_float32(*v_pair).to(torch.float32)
        reference = _a2_attention_forward(
            q_dense[None], k_dense[None], v_dense[None], q_heads, kv_heads, head_dim
        )[0].detach()
        # Per-window normalizer (same convention as the root A2 trainer and
        # A-FIX1): the standard HiF4 path's MSE.  The gate itself stays raw
        # MSE; normalization only keeps the fixed Adam epsilon (1e-8) healthy
        # against tiny raw-MSE gradients.  Per-window positive constants do
        # not change the per-window minimizer.
        std_q = _dequantize_hif4(_dense_to_hif4(q_dense)).to(torch.float32)
        std_k = _dequantize_hif4(_dense_to_hif4(k_dense)).to(torch.float32)
        std_v = _dequantize_hif4(_dense_to_hif4(v_dense)).to(torch.float32)
        standard = _a2_attention_forward(
            std_q[None], std_k[None], std_v[None], q_heads, kv_heads, head_dim
        )[0].detach()
        mse_std = float((standard - reference).square().mean())
        # Parent-frame coordinates for the (linear) gradient chain into M.
        q_coord = _agr2_parent_coordinate(
            q_dense, states["q_state"], q_heads, head_dim, False
        ).detach()
        k_coord = _agr2_parent_coordinate(
            k_dense, states["k_state"], kv_heads, head_dim, True
        ).detach()
        # Deployed V is independent of N: quantize once per window.
        v_params = _AGR2_PARENT_V(
            v_pair[0], v_pair[1], kv_heads, head_dim, states["v_state"]
        )
        v_hat = _dequantize_hif4(v_params).to(device=device, dtype=torch.float32)
        prepared.append({
            "q_pair": q_pair, "k_pair": k_pair,
            "q_coord": q_coord, "k_coord": k_coord,
            "v_hat": v_hat, "reference": reference,
            "mse_std": max(mse_std, 1e-12),
        })
    if not prepared:
        raise ValueError("Residual training has no calibration windows")

    exp_avg = torch.zeros_like(n)
    exp_avg_sq = torch.zeros_like(n)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, _AGR2_TRAIN_STEPS + 1):
        m = eye + n
        m_inv = torch.linalg.inv(m)
        p = m_inv.transpose(-1, -2)
        tq = rq @ m
        tk = rk @ p
        center_t = (
            (center.unsqueeze(-2) @ p).squeeze(-2) if has_center else None
        )
        grad_m = torch.zeros_like(n)
        grad_p = torch.zeros_like(n)
        loss = n.square().mean() * _AGR2_REG_WEIGHT
        for item in prepared:
            player_q_state = dict(states["q_state"], learned_rotation=tq)
            player_k_state = dict(states["k_state"], learned_rotation=tk)
            if center_t is not None:
                player_k_state["learned_center"] = center_t
            q_params = _AGR2_PARENT_Q(
                item["q_pair"][0], item["q_pair"][1], q_heads, head_dim,
                player_q_state,
            )
            k_params = _AGR2_PARENT_K(
                item["k_pair"][0], item["k_pair"][1], kv_heads, head_dim,
                player_k_state,
            )
            q_hat = _dequantize_hif4(q_params).to(device=device, dtype=torch.float32)
            k_hat = _dequantize_hif4(k_params).to(device=device, dtype=torch.float32)
            output = _a2_attention_forward(
                q_hat[None], k_hat[None], item["v_hat"][None],
                q_heads, kv_heads, head_dim,
            )[0]
            residual = output - item["reference"]
            loss = loss + residual.square().mean() / item["mse_std"] / float(len(prepared))
            d_output = (
                2.0 * residual / float(residual.numel())
                / item["mse_std"] / float(len(prepared))
            )
            d_qhat, d_khat = _m_attention_backward(
                d_output[None], q_hat, k_hat, item["v_hat"],
                q_heads, kv_heads, head_dim,
            )
            grad_m.add_(_agr2_matrix_grad(item["q_coord"], d_qhat, q_heads, kv_heads))
            grad_p.add_(_agr2_matrix_grad(item["k_coord"], d_khat, kv_heads, kv_heads))
        if step == 1:
            initial_loss = float(loss.item())
        if force_zero:
            continue
        gradient = (
            grad_m
            - p @ grad_p.transpose(-1, -2) @ p
            + 2.0 * _AGR2_REG_WEIGHT * n / float(n.numel())
        )
        norm = float(gradient.norm().item())
        if norm > _AGR2_TRAIN_CLIP:
            gradient = gradient * (_AGR2_TRAIN_CLIP / norm)
        exp_avg.mul_(_AGR2_BETA1).add_(gradient, alpha=1.0 - _AGR2_BETA1)
        exp_avg_sq.mul_(_AGR2_BETA2).addcmul_(
            gradient, gradient, value=1.0 - _AGR2_BETA2
        )
        bias1 = 1.0 - _AGR2_BETA1 ** step
        bias2 = 1.0 - _AGR2_BETA2 ** step
        update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1.0e-8)
        n = _agr2_project(eye + n - _AGR2_TRAIN_LR * update) - eye

    if force_zero:
        m = eye.expand(int(kv_heads), int(head_dim), int(head_dim))
        p = m
        final_loss = initial_loss
    else:
        m = eye + n
        p = torch.linalg.inv(m).transpose(-1, -2)
        tq_f = rq @ m
        tk_f = rk @ p
        center_f = (
            (center.unsqueeze(-2) @ p).squeeze(-2) if has_center else None
        )
        final_loss = float(n.square().mean().item()) * _AGR2_REG_WEIGHT
        for item in prepared:
            player_q_state = dict(states["q_state"], learned_rotation=tq_f)
            player_k_state = dict(states["k_state"], learned_rotation=tk_f)
            if center_f is not None:
                player_k_state["learned_center"] = center_f
            q_params = _AGR2_PARENT_Q(
                item["q_pair"][0], item["q_pair"][1], q_heads, head_dim,
                player_q_state,
            )
            k_params = _AGR2_PARENT_K(
                item["k_pair"][0], item["k_pair"][1], kv_heads, head_dim,
                player_k_state,
            )
            q_hat = _dequantize_hif4(q_params).to(device=device, dtype=torch.float32)
            k_hat = _dequantize_hif4(k_params).to(device=device, dtype=torch.float32)
            output = _a2_attention_forward(
                q_hat[None], k_hat[None], item["v_hat"][None],
                q_heads, kv_heads, head_dim,
            )[0]
            residual = output - item["reference"]
            final_loss += (
                float(residual.square().mean().item())
                / item["mse_std"] / float(len(prepared))
            )

    # Compile into the existing root fields, exactly like v192/A-GR1: the K
    # center follows the same M^{-T} transform so the additive K term stays in
    # the same reciprocal coordinate frame and logits are exactly preserved.
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
        "agr2_steps": _AGR2_TRAIN_STEPS,
        "agr2_fit_windows": len(prepared),
        "agr2_attempted_groups": int(kv_heads),
        "agr2_initial_loss": initial_loss,
        "agr2_final_loss": final_loss,
        "agr2_inverse_error": inverse_error,
        "agr2_n_norm": float(n.norm().item()),
        "agr2_n_max_abs": float(n.abs().max().item()),
        "agr2_singular_min": float(singular.min().item()),
        "agr2_singular_max": float(singular.max().item()),
        "agr2_parent_arm": "rotation" if parent_q is not None else "identity",
        "agr2_center_compiled": bool(has_center),
    }
    return (
        transformed_q.detach().to(device="cpu").contiguous(),
        transformed_k.detach().to(device="cpu").contiguous(),
        None if center_new is None else center_new.detach().to(device="cpu").contiguous(),
        info,
    )


@torch.no_grad()
def _agr2_gate_loss(
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
            "q": _AGR2_PARENT_Q,
            "k": _AGR2_PARENT_K,
            "v": _AGR2_PARENT_V,
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


def _agr2_annotated_states(states: dict, info: dict) -> dict:
    out = _agr2_parent_copy(states)
    out["q_state"].update(info)
    out["k_state"].update(info)
    return out


_AGR2_PARENT_CALIBRATION = hif4_calibration_attention
_AGR2_PARENT_Q = hif4_dynamic_quantize_q
_AGR2_PARENT_K = hif4_dynamic_quantize_k
_AGR2_PARENT_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict:
    states = _AGR2_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    info = {
        "agr2_arm": "fallback",
        "agr2_attempted": 0,
        "agr2_accepted": 0,
        "agr2_fit_windows": 0,
        "agr2_gate_windows": len(_AGR2_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < max(_AGR2_GATE_WINDOWS) + 1
        or int(q_num_heads) % int(kv_num_heads) != 0
    ):
        info["agr2_arm"] = "ineligible"
        return _agr2_annotated_states(states, info)
    try:
        fit_windows = calib_qkv_list[:len(calib_qkv_list) - len(_AGR2_GATE_WINDOWS)]
        train_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tq, tk, center, train_info = _agr2_train(
            fit_windows,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
            train_device,
        )
        candidate = _agr2_parent_copy(states)
        candidate["q_state"]["learned_rotation"] = tq
        candidate["k_state"]["learned_rotation"] = tk
        if center is not None:
            candidate["k_state"]["learned_center"] = center
        gate_records = []
        accepted = True
        for index in _AGR2_GATE_WINDOWS:
            parent_loss = _agr2_gate_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _agr2_gate_loss(
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
        info["agr2_attempted"] = 1
        info["agr2_fit_windows"] = len(fit_windows)
        info["agr2_gate_parent_mse"] = float(
            sum(item["parent_mse"] for item in gate_records) / len(gate_records)
        )
        info["agr2_gate_candidate_mse"] = float(
            sum(item["candidate_mse"] for item in gate_records) / len(gate_records)
        )
        if accepted:
            info["agr2_arm"] = "accepted"
            info["agr2_accepted"] = 1
            return _agr2_annotated_states(candidate, info)
        info["agr2_arm"] = "parent"
        return _agr2_annotated_states(states, info)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        info["agr2_arm"] = "fallback"
        info["agr2_error"] = f"{type(exc).__name__}: {exc}"
        return _agr2_annotated_states(states, info)
'''
