# ---------------------------------------------------------------------------
# v198 attn-gqa-reciprocal-diag: GQA group-shared reciprocal diagonal
# reparameterization of the final continuous Q/K (design doc sections 8-16).
#
# One diagonal D_g = diag(exp(u_g)) per KV group, shared by all Q heads of
# the group (GQA correctness invariant).  Deployment is a pure encode-stage
# rescale of the parent final continuous tensors:
#   Q' = (Q_pre R) * D            (rotation, then diag)
#   K' = (K_pre R + c) * D^{-1}   (rotation, learned center, then diag)
# The learned center is therefore synchronized automatically: the deployed
# continuous K equals K_pre (R D^{-1}) + c D^{-1}, so Q'K'^T == QK^T exactly
# in continuous arithmetic.  V is strictly frozen.
#
# Stage A (no attention forward): analytic RMS init
#   u_j = 0.5 * log((k_j + eps) / (q_j + eps)), clamped to +/-log(2),
# with Q statistics aggregated across the Q heads of each KV group, then a
# 5-step smooth-max (log-sum-exp, tau=8) range objective with lambda=1e-3.
# Stage B: hard gate on the last calibration window through the true
# deployed HiF4 encode + attention forward; strict output-MSE improvement
# required, otherwise the parent state is returned untouched.
# ---------------------------------------------------------------------------

_RD_EPS = 1.0e-12
_RD_U_BOUND = math.log(2.0)
_RD_TAU = 8.0
_RD_LAMBDA = 1.0e-3
_RD_STEPS = 5
_RD_LR = 0.25
_RD_BLOCK = 64


def _rd_clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in state.items()
    }


def _rd_annotated_states(
    states: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "q_state": _rd_clone_state(states["q_state"]),
        "k_state": _rd_clone_state(states["k_state"]),
        "v_state": states["v_state"],
    }
    result["q_state"].update(audit)
    result["k_state"].update(audit)
    return result


def _rd_parent_dense(
    dense: torch.Tensor,
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
    *,
    is_k: bool,
) -> torch.Tensor:
    """Return the continuous parent tensor consumed by the HiF4 encoder."""

    transformed = _attention_state_transform_dense(
        dense, state, num_heads, head_dim, is_k=is_k
    )
    learned_rotation = state.get("learned_rotation")
    if learned_rotation is not None:
        transformed = _a2_apply_group_rotation(
            transformed, num_heads, learned_rotation
        )
    learned_center = state.get("learned_center")
    if learned_center is not None:
        lead = transformed.shape[:-1]
        center = learned_center.to(
            device=transformed.device, dtype=torch.float32
        ).reshape(*([1] * len(lead)), int(num_heads), int(head_dim))
        transformed = (
            transformed.reshape(*lead, int(num_heads), int(head_dim)) + center
        ).reshape_as(transformed)
    return transformed


@torch.no_grad()
def _rd_channel_rms(
    calib_qkv_list: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Per-(KV group, channel) RMS of the parent final continuous Q/K.

    Q statistics are aggregated across all Q heads inside each KV group.
    All calibration windows except the last (gate holdout) are used.
    """

    group = int(q_num_heads) // int(kv_num_heads)
    q_sum_square = None
    k_sum_square = None
    q_count = 0.0
    k_count = 0.0
    used = 0
    for sample in calib_qkv_list[:-1]:
        q_raw = _dequantize_nvfp4_float32(*sample["q"]).to(
            device=device, dtype=torch.float32
        )
        k_raw = _dequantize_nvfp4_float32(*sample["k"]).to(
            device=device, dtype=torch.float32
        )
        q_parent = _rd_parent_dense(
            q_raw, states["q_state"], q_num_heads, head_dim, is_k=False
        )
        k_parent = _rd_parent_dense(
            k_raw, states["k_state"], kv_num_heads, head_dim, is_k=True
        )
        tokens = min(int(q_parent.shape[0]), int(k_parent.shape[0]))
        if tokens <= 0:
            continue
        q_grouped = q_parent[:tokens].reshape(
            tokens, kv_num_heads, group, head_dim
        )
        k_grouped = k_parent[:tokens].reshape(tokens, kv_num_heads, head_dim)
        q_ss = q_grouped.square().sum(dim=(0, 2))
        k_ss = k_grouped.square().sum(dim=0)
        q_sum_square = q_ss if q_sum_square is None else q_sum_square + q_ss
        k_sum_square = k_ss if k_sum_square is None else k_sum_square + k_ss
        q_count += float(tokens * group)
        k_count += float(tokens)
        used += 1
    if used == 0 or q_sum_square is None or k_sum_square is None:
        return None
    q_rms = (q_sum_square / max(q_count, 1.0)).sqrt()
    k_rms = (k_sum_square / max(k_count, 1.0)).sqrt()
    q_rms = torch.nan_to_num(q_rms, nan=0.0, posinf=0.0, neginf=0.0)
    k_rms = torch.nan_to_num(k_rms, nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(torch.isfinite(q_rms).all() and torch.isfinite(k_rms).all()):
        return None
    return q_rms, k_rms, used


def _rd_range_loss(
    u: torch.Tensor, a: torch.Tensor, b: torch.Tensor, block: int
) -> float:
    """Smooth-max log-range objective L_range(u) = M_Q + M_K + lambda|u|^2."""

    kv, hd = int(u.shape[0]), int(u.shape[1])
    ub = u.reshape(kv, hd // block, block)
    ab = a.reshape(kv, hd // block, block)
    bb = b.reshape(kv, hd // block, block)
    m_q = torch.logsumexp(_RD_TAU * (ab + ub), dim=-1) / _RD_TAU
    m_k = torch.logsumexp(_RD_TAU * (bb - ub), dim=-1) / _RD_TAU
    loss = (m_q + m_k).sum() + _RD_LAMBDA * u.square().sum()
    return float(loss)


@torch.no_grad()
def _rd_smooth_max_refine(
    u: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Five analytic-gradient steps on the smooth-max range objective."""

    kv, hd = int(u.shape[0]), int(u.shape[1])
    block = _RD_BLOCK if hd % _RD_BLOCK == 0 else hd
    for _ in range(_RD_STEPS):
        ub = u.reshape(kv, hd // block, block)
        ab = a.reshape(kv, hd // block, block)
        bb = b.reshape(kv, hd // block, block)
        p = torch.softmax(_RD_TAU * (ab + ub), dim=-1)
        m = torch.softmax(_RD_TAU * (bb - ub), dim=-1)
        grad = (p - m).reshape(kv, hd) + 2.0 * _RD_LAMBDA * u
        u = (u - _RD_LR * grad).clamp(min=-_RD_U_BOUND, max=_RD_U_BOUND)
    return u


@torch.no_grad()
def _rd_true_output_mse(
    sample: dict,
    q_state: dict[str, Any],
    k_state: dict[str, Any],
    v_state: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> float:
    """Final attention output MSE through the true deployed HiF4 path."""

    q_quant, q_scale = sample["q"]
    k_quant, k_scale = sample["k"]
    v_quant, v_scale = sample["v"]
    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)
    q_params = hif4_dynamic_quantize_q(
        q_quant, q_scale, q_num_heads, head_dim, q_state
    )
    k_params = hif4_dynamic_quantize_k(
        k_quant, k_scale, kv_num_heads, head_dim, k_state
    )
    v_params = hif4_dynamic_quantize_v(
        v_quant, v_scale, kv_num_heads, head_dim, v_state
    )
    q_hat = _dequantize_hif4(q_params).to(torch.float32)
    k_hat = _dequantize_hif4(k_params).to(torch.float32)
    v_hat = _dequantize_hif4(v_params).to(torch.float32)
    reference = _a2_attention_forward(
        q_ref[None], k_ref[None], v_ref[None], q_num_heads, kv_num_heads, head_dim
    )[0]
    player = _a2_attention_forward(
        q_hat[None], k_hat[None], v_hat[None], q_num_heads, kv_num_heads, head_dim
    )[0]
    return float((player - reference).square().mean())


_RD_PARENT_CALIBRATION = hif4_calibration_attention


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Current complete parent plus the v198 reciprocal diagonal arm."""

    states = _RD_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < 2
    ):
        return states
    audit: dict[str, Any] = {
        "rd_arm": "fallback",
        "rd_attempted": 0,
        "rd_accepted": 0,
        "rd_fit_windows": 0,
    }
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stats = _rd_channel_rms(
            calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim, device
        )
        if stats is None:
            audit["rd_arm"] = "ineligible"
            return _rd_annotated_states(states, audit)
        q_rms, k_rms, fit_used = stats
        a = torch.log(q_rms + _RD_EPS)
        b = torch.log(k_rms + _RD_EPS)
        u_init = (0.5 * (b - a)).clamp(min=-_RD_U_BOUND, max=_RD_U_BOUND)
        u = _rd_smooth_max_refine(u_init, a, b)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        if not bool(torch.isfinite(u).all()):
            audit["rd_arm"] = "fallback"
            return _rd_annotated_states(states, audit)
        block = _RD_BLOCK if int(head_dim) % _RD_BLOCK == 0 else int(head_dim)
        q_factor = u.exp()
        k_factor = u.neg().exp()
        group = int(q_num_heads) // int(kv_num_heads)
        q_candidate = _rd_clone_state(states["q_state"])
        k_candidate = _rd_clone_state(states["k_state"])
        q_candidate["diag_scale"] = _cpu_state_tensor(
            q_factor.repeat_interleave(group, dim=0).reshape(-1)
        )
        k_candidate["diag_scale"] = _cpu_state_tensor(k_factor.reshape(-1))
        gate_window = calib_qkv_list[-1]
        parent_mse = _rd_true_output_mse(
            gate_window,
            states["q_state"],
            states["k_state"],
            states["v_state"],
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        candidate_mse = _rd_true_output_mse(
            gate_window,
            q_candidate,
            k_candidate,
            states["v_state"],
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        audit.update(
            rd_attempted=1,
            rd_fit_windows=int(fit_used),
            rd_parent_mse=float(parent_mse),
            rd_candidate_mse=float(candidate_mse),
            rd_u_max_abs=float(u.abs().max()),
            rd_u_mean_abs=float(u.abs().mean()),
            rd_u_nonzero=int((u != 0).sum()),
            rd_range_loss_init=_rd_range_loss(u_init, a, b, block),
            rd_range_loss_final=_rd_range_loss(u, a, b, block),
        )
        if (
            math.isfinite(candidate_mse)
            and math.isfinite(parent_mse)
            and candidate_mse < parent_mse
        ):
            audit["rd_arm"] = "accepted"
            audit["rd_accepted"] = 1
            return _rd_annotated_states(
                {"q_state": q_candidate, "k_state": k_candidate,
                 "v_state": states["v_state"]},
                audit,
            )
        audit["rd_arm"] = "parent"
        return _rd_annotated_states(states, audit)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        audit["rd_arm"] = "fallback"
        audit["rd_error"] = f"{type(exc).__name__}: {exc}"
        return _rd_annotated_states(states, audit)


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
        attention_pair_transform=state.get("pair_transform"),
        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(q_num_heads),
        attention_diag_scale=state.get("diag_scale"),
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
        attention_pair_transform=state.get("pair_transform"),
        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(kv_num_heads),
        learned_center=state.get("learned_center"),
        attention_diag_scale=state.get("diag_scale"),
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
