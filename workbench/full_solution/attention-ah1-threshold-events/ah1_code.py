# ---------------------------------------------------------------------------
# A-H1: quantization threshold event search along the normalized A2 direction.
#
# The root A2 trainer sums gradients across windows while tracking a mean
# loss.  A-H1 keeps the root trainer untouched for the deployed state, but
# re-derives the search direction at the point just before the final Adam
# update (theta0/center0) with the window-mean gradient (FIX-A2 fallback
# clause: FIX-A2 is not merged into the root, so the trainer arithmetic is
# the root one and only the direction is normalized by the window count).
# The search path is theta(t)=theta0+t*d_theta, center(t)=center0+t*d_center
# with d=-g/max(||g||,1e-12), restricted to t>0.
#
# Events are the linearized HiF4 magnitude-midpoint crossings of the
# transformed calibration Q/K values under the Cayley path, with the parent
# state's permutation/offset and the parent encode's scale/lv2/lv3 frozen.
# Every candidate t is then re-evaluated through the complete deployed
# Q/K encode/decode, the current V path, and the final attention output MSE.
# Selection uses only the calibration folds (case-equal normalized output
# MSE); the independent holdout window is recorded but never vetoes.
# ---------------------------------------------------------------------------

_AH1_EVENT_SLOTS = 8


def _ah1_pretransform_dense(
    dense: torch.Tensor,
    state: dict,
    num_heads: int,
    is_k: bool,
) -> torch.Tensor:
    """Deployed dynamic pipeline up to (but excluding) the learned rotation."""

    out = dense
    channels = int(out.shape[-1])
    if is_k and int(state.get("center_mode", 0)) != 0:
        out = _center_attention_k(
            out,
            int(num_heads),
            channels // int(num_heads),
            int(state.get("center_mode", 0)),
            state.get("center_value"),
        )
    multiplier = state.get("multiplier")
    if multiplier is not None:
        scale = _safe_positive_vector(multiplier, channels).to(out.device)
        out = out * scale.reshape(*([1] * (out.ndim - 1)), channels)
    permutation = state.get("permutation")
    if permutation is not None:
        order = permutation.detach().to(
            device=out.device, dtype=torch.int64
        ).reshape(-1)
        out = out.index_select(-1, order)
    signs = state.get("rotation")
    if signs is not None:
        out = _apply_attention_rotation(
            out,
            int(num_heads),
            int(signs.shape[-1]),
            signs,
            state.get("rotation_block"),
        )
    block_smooth_size = int(state.get("block_smooth_size", 0))
    if block_smooth_size != 0:
        block_signs = state.get("block_smooth_signs")
        if block_signs is not None:
            out = _apply_attention_rotation(
                out,
                int(num_heads),
                int(block_signs.shape[1]),
                block_signs,
                block_smooth_size,
            )
        else:
            out = _block_hadamard_transform(
                out, block_smooth_size, int(state.get("block_smooth_seed", 0))
            )
    pair = state.get("pair_transform")
    if pair is not None:
        out = _apply_attention_pair_transform(
            out, int(num_heads), channels // int(num_heads), pair
        )
    return out


def _ah1_boundary_times(
    x0: torch.Tensor,
    dx: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    """Linearized HiF4 magnitude-midpoint crossing times (frozen hierarchy).

    For each element with path value x0 and derivative dx at t=0, and each
    adjacent HiF4 magnitude midpoint b=(j+0.5)*0.25*denom (j=0..6), the
    boundary is t=(b-|x0|)/(sign(x0)*dx).  At x0==0 the derivative of |x| is
    |dx|.  Only finite, strictly positive times are kept.
    """

    offsets = (
        torch.arange(7, device=x0.device, dtype=torch.float32) * 0.25 + 0.125
    )
    mids = denom.to(torch.float32).unsqueeze(-1) * offsets
    absx = x0.abs().unsqueeze(-1)
    slope = torch.where(x0 != 0, torch.sign(x0) * dx, dx.abs()).unsqueeze(-1)
    t = (mids - absx) / slope
    valid = torch.isfinite(t) & (t > 0)
    return t[valid]


def _ah1_hierarchy_denominator(params: dict[str, torch.Tensor]) -> torch.Tensor:
    denom = (
        params["scale_factor"] * params["scale_lv2"] * params["scale_lv3"]
    ).expand(params["sign"].shape)
    return denom.flatten(start_dim=-4).to(torch.float32)


def _ah1_changed_codes(
    parent_params: dict[str, torch.Tensor],
    event_params: dict[str, torch.Tensor],
) -> int:
    shape = parent_params["sign"].shape
    differ = (parent_params["sign"] != event_params["sign"]) | (
        parent_params["mant"] != event_params["mant"]
    )
    for key in ("scale_factor", "scale_lv2", "scale_lv3"):
        differ = differ | (
            parent_params[key].expand(shape) != event_params[key].expand(shape)
        )
    return int(differ.sum())


def _ah1_true_path_mse(
    q_pair: tuple,
    k_pair: tuple,
    q_state: dict,
    k_state: dict,
    v_hat: torch.Tensor,
    reference: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Tuple[float, dict, dict]:
    """Full deployed Q/K encode/decode + current V path + output MSE."""

    q_params = hif4_dynamic_quantize_q(
        q_pair[0], q_pair[1], q_num_heads, head_dim, q_state
    )
    k_params = hif4_dynamic_quantize_k(
        k_pair[0], k_pair[1], kv_num_heads, head_dim, k_state
    )
    q_hat = _dequantize_hif4(q_params).to(torch.float32)
    k_hat = _dequantize_hif4(k_params).to(torch.float32)
    player = _a2_attention_forward(
        q_hat[None], k_hat[None], v_hat[None],
        q_num_heads, kv_num_heads, head_dim,
    )[0]
    mse = float((player - reference).square().mean())
    return mse, q_params, k_params


def _ah1_finite(value: float) -> float:
    result = float(value)
    return result if math.isfinite(result) else -1.0


def _ah1_threshold_event_search(
    calib_qkv_list: list,
    windows: list,
    states: dict[str, Any],
    info: dict,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[dict[str, Any], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Search threshold events along the normalized A2 direction.

    Returns the audit dict plus the selected (rotation, center) CPU tensors
    when an event strictly improves over the parent on the calibration folds.
    Never mutates ``states``; the caller applies the selection.
    """

    audit: dict[str, Any] = {
        "ah1_status": "unavailable-direction",
        "ah1_direction_norm_theta": -1.0,
        "ah1_direction_norm_center": -1.0,
        "ah1_events_raw": 0,
        "ah1_events_dedup": 0,
        "ah1_event_t": [],
        "ah1_event_t_eval": [],
        "ah1_event_changed_q": [],
        "ah1_event_changed_k": [],
        "ah1_fold_loss_parent": -1.0,
        "ah1_fold_loss_events": [],
        "ah1_accepted_event": 0,
        "ah1_holdout_loss_parent": -1.0,
        "ah1_holdout_loss_final": -1.0,
        "ah1_attempted": 0,
        "ah1_accepted": 0,
        "ah1_folds": 0,
    }
    theta0 = info.get("ah1_theta0")
    center0 = info.get("ah1_center0")
    g_theta = info.get("ah1_g_theta")
    g_center = info.get("ah1_g_center")
    if (
        not torch.is_tensor(theta0)
        or not torch.is_tensor(center0)
        or not torch.is_tensor(g_theta)
        or not torch.is_tensor(g_center)
    ):
        return audit

    groups = int(kv_num_heads)
    dim = int(head_dim)
    norm_theta = float(g_theta.norm())
    norm_center = float(g_center.norm())
    audit["ah1_direction_norm_theta"] = _ah1_finite(norm_theta)
    audit["ah1_direction_norm_center"] = _ah1_finite(norm_center)
    d_theta = -g_theta / max(norm_theta, 1e-12)
    d_center = -g_center / max(norm_center, 1e-12)

    base = _a2_hadamard_orthogonal(dim)
    if base is None:
        base = torch.eye(dim, dtype=torch.float32)
    base = base.to(device)
    eye = torch.eye(dim, device=device, dtype=torch.float32)

    # Analytic Cayley derivative at t=0: C=(I-S)(I+S)^{-1} with
    # S(t)=S0+t*dS gives dC/dt=-(I+C) dS (I+S)^{-1}; rotation=base@C per group.
    theta0 = theta0.to(device=device, dtype=torch.float32)
    center0 = center0.to(device=device, dtype=torch.float32)
    d_skew = d_theta - d_theta.transpose(-1, -2)
    c0, right0 = _m_cayley_pair(theta0)
    b_mat = (eye.unsqueeze(0) + c0) @ d_skew
    d_c = -torch.linalg.solve(
        right0.transpose(-1, -2), b_mat.transpose(-1, -2)
    ).transpose(-1, -2)
    rotation0 = torch.einsum("dk,gkl->gdl", base, c0)
    d_rotation = torch.einsum("dk,gkl->gdl", base, d_c)

    q_parent = dict(states["q_state"])
    k_parent = dict(states["k_state"])
    q_standard = {
        key: value for key, value in q_parent.items() if key != "learned_rotation"
    }
    k_standard = {
        key: value
        for key, value in k_parent.items()
        if key not in ("learned_rotation", "learned_center")
    }

    fold_count = len(calib_qkv_list) - 1
    audit["ah1_folds"] = int(fold_count)
    folds = []
    raw_events = 0
    distinct_parts = []
    for index in range(fold_count):
        item = calib_qkv_list[index]
        q_ref = windows[index]["q"]
        k_ref = windows[index]["k"]
        v_ref = windows[index]["v"]
        reference = _a2_attention_forward(
            q_ref[None], k_ref[None], v_ref[None],
            q_num_heads, kv_num_heads, head_dim,
        )[0]
        v_hat = _dequantize_hif4(
            hif4_dynamic_quantize_v(
                item["v"][0], item["v"][1],
                kv_num_heads, head_dim, states["v_state"],
            )
        ).to(torch.float32)
        standard_mse, _sq, _sk = _ah1_true_path_mse(
            item["q"], item["k"], q_standard, k_standard, v_hat, reference,
            q_num_heads, kv_num_heads, head_dim,
        )
        parent_mse, parent_q_params, parent_k_params = _ah1_true_path_mse(
            item["q"], item["k"], q_parent, k_parent, v_hat, reference,
            q_num_heads, kv_num_heads, head_dim,
        )

        # Event generation: x0/dx at t=0 on the Cayley path; the frozen
        # magnitude hierarchy comes from the parent state's deployed encode.
        u_q = _ah1_pretransform_dense(q_ref, q_parent, q_num_heads, False)
        u_k = _ah1_pretransform_dense(k_ref, k_parent, kv_num_heads, True)
        x_q0 = _a2_apply_group_rotation(u_q, q_num_heads, rotation0)
        dx_q = _a2_apply_group_rotation(u_q, q_num_heads, d_rotation)
        x_k0 = (
            _a2_apply_group_rotation(u_k, kv_num_heads, rotation0).reshape(
                -1, kv_num_heads, head_dim
            )
            + center0[None]
        ).reshape(k_ref.shape)
        dx_k = (
            _a2_apply_group_rotation(u_k, kv_num_heads, d_rotation).reshape(
                -1, kv_num_heads, head_dim
            )
            + d_center[None]
        ).reshape(k_ref.shape)
        denom_q = _ah1_hierarchy_denominator(parent_q_params)
        denom_k = _ah1_hierarchy_denominator(parent_k_params)
        t_q = _ah1_boundary_times(x_q0, dx_q, denom_q)
        t_k = _ah1_boundary_times(x_k0, dx_k, denom_k)
        raw_events += int(t_q.numel()) + int(t_k.numel())
        if int(t_q.numel()) > 0:
            distinct_parts.append(torch.unique(t_q).cpu())
        if int(t_k.numel()) > 0:
            distinct_parts.append(torch.unique(t_k).cpu())
        folds.append({
            "q_pair": item["q"],
            "k_pair": item["k"],
            "v_hat": v_hat,
            "reference": reference,
            "standard_mse": max(standard_mse, 1e-12),
            "parent_mse": parent_mse,
            "parent_q_params": parent_q_params,
            "parent_k_params": parent_k_params,
        })

    audit["ah1_events_raw"] = int(raw_events)
    if distinct_parts:
        all_distinct = torch.unique(torch.cat(distinct_parts))
    else:
        all_distinct = torch.empty(0, dtype=torch.float32)
    dedup_events = int(all_distinct.numel())
    audit["ah1_events_dedup"] = dedup_events

    parent_agg = sum(
        fold["parent_mse"] / fold["standard_mse"] for fold in folds
    ) / float(fold_count)
    audit["ah1_fold_loss_parent"] = _ah1_finite(parent_agg)

    def _holdout_losses(final_q_state: dict, final_k_state: dict) -> Tuple[float, float]:
        gate_item = calib_qkv_list[-1]
        gate_w = windows[-1]
        reference_h = _a2_attention_forward(
            gate_w["q"][None], gate_w["k"][None], gate_w["v"][None],
            q_num_heads, kv_num_heads, head_dim,
        )[0]
        v_hat_h = _dequantize_hif4(
            hif4_dynamic_quantize_v(
                gate_item["v"][0], gate_item["v"][1],
                kv_num_heads, head_dim, states["v_state"],
            )
        ).to(torch.float32)
        standard_h, _sq, _sk = _ah1_true_path_mse(
            gate_item["q"], gate_item["k"], q_standard, k_standard,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )
        parent_h, _pq, _pk = _ah1_true_path_mse(
            gate_item["q"], gate_item["k"], q_parent, k_parent,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )
        final_h, _fq, _fk = _ah1_true_path_mse(
            gate_item["q"], gate_item["k"], final_q_state, final_k_state,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )
        normalizer = max(standard_h, 1e-12)
        return parent_h / normalizer, final_h / normalizer

    if dedup_events == 0:
        audit["ah1_status"] = "unreachable"
        holdout_parent, holdout_final = _holdout_losses(q_parent, k_parent)
        audit["ah1_holdout_loss_parent"] = _ah1_finite(holdout_parent)
        audit["ah1_holdout_loss_final"] = _ah1_finite(holdout_final)
        print(
            f"[A-H1] status=unreachable raw={raw_events} dedup={dedup_events} "
            f"attempted=0 accepted=0",
            flush=True,
        )
        return audit, None

    slots = all_distinct[:_AH1_EVENT_SLOTS]
    eval_ts = torch.nextafter(slots, torch.full_like(slots, float("inf")))
    audit["ah1_event_t"] = [float(value) for value in slots]
    audit["ah1_event_t_eval"] = [float(value) for value in eval_ts]

    fold_losses = []
    changed_q = []
    changed_k = []
    best = None
    for slot_index in range(int(slots.numel())):
        t_eval = float(eval_ts[slot_index])
        theta_t = theta0 + t_eval * d_theta
        c_t, _right_t = _m_cayley_pair(theta_t)
        rotation_t = torch.einsum("dk,gkl->gdl", base, c_t)
        center_t = center0 + t_eval * d_center
        cpu_rotation = rotation_t.detach().cpu().to(torch.float32)
        cpu_center = center_t.detach().cpu().to(torch.float32)
        cand_q = dict(q_parent)
        cand_q["learned_rotation"] = cpu_rotation
        cand_k = dict(k_parent)
        cand_k["learned_rotation"] = cpu_rotation.clone()
        cand_k["learned_center"] = cpu_center
        agg = 0.0
        n_changed_q = 0
        n_changed_k = 0
        for fold in folds:
            mse, q_params, k_params = _ah1_true_path_mse(
                fold["q_pair"], fold["k_pair"], cand_q, cand_k,
                fold["v_hat"], fold["reference"],
                q_num_heads, kv_num_heads, head_dim,
            )
            agg += mse / fold["standard_mse"]
            n_changed_q += _ah1_changed_codes(fold["parent_q_params"], q_params)
            n_changed_k += _ah1_changed_codes(fold["parent_k_params"], k_params)
        agg /= float(fold_count)
        fold_losses.append(_ah1_finite(agg))
        changed_q.append(int(n_changed_q))
        changed_k.append(int(n_changed_k))
        if best is None or agg < best[1]:
            best = (slot_index, agg, cpu_rotation, cpu_center)

    audit["ah1_fold_loss_events"] = fold_losses
    audit["ah1_event_changed_q"] = changed_q
    audit["ah1_event_changed_k"] = changed_k
    audit["ah1_attempted"] = int(slots.numel())

    selected = None
    if best is not None and best[1] < parent_agg:
        selected = (best[2], best[3])
        audit["ah1_status"] = "accepted"
        audit["ah1_accepted"] = 1
        audit["ah1_accepted_event"] = int(best[0]) + 1
        final_q = dict(q_parent)
        final_q["learned_rotation"] = best[2]
        final_k = dict(k_parent)
        final_k["learned_rotation"] = best[2].clone()
        final_k["learned_center"] = best[3]
    else:
        audit["ah1_status"] = "no-improvement"
        final_q = q_parent
        final_k = k_parent

    holdout_parent, holdout_final = _holdout_losses(final_q, final_k)
    audit["ah1_holdout_loss_parent"] = _ah1_finite(holdout_parent)
    audit["ah1_holdout_loss_final"] = _ah1_finite(holdout_final)
    print(
        f"[A-H1] status={audit['ah1_status']} raw={raw_events} "
        f"dedup={dedup_events} attempted={audit['ah1_attempted']} "
        f"accepted={audit['ah1_accepted']} "
        f"accepted_event={audit['ah1_accepted_event']} "
        f"parent={parent_agg:.6f}",
        flush=True,
    )
    return audit, selected
