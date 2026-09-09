# ---------------------------------------------------------------------------
# A-H1R: quantization threshold events anchored at the *deployed* parent state.
#
# v223 A-H1 generated events from theta_pre/center_pre (the point just before
# the final Adam update), so t=0 was not the gate-selected deployed state and
# the observed code flips were dominated by rolling back that last update.
# A-H1R fixes the anchor:
#   1. read the actually deployed parent rotation R_parent / center_parent from
#      the gate-selected state (identity/zero when the gate chose identity);
#   2. recompute the calibration-fold mean output gradient AT the parent, in
#      the deployed pre-rotation coordinate system (center/multiplier/
#      permutation/C76.4/block-smooth/pair transform applied first, exactly as
#      _nvfp4_to_hif4 does);
#   3. use the orthogonal tangent direction S = skew(R_parent^T G_R) with path
#      R(t) = R_parent @ cayley(t S); center stays fixed at center_parent.
# Because d cayley/dt|_0 = -2 S, the path derivative at t=0 is -2 R_parent S,
# i.e. a descent direction for the raw gradient G_R.
#
# Events are linearized HiF4 magnitude-midpoint crossings of the transformed
# calibration Q/K under that path, with the parent encode's scale/lv2/lv3
# frozen.  Each candidate t is re-evaluated through the complete deployed Q/K
# encode/decode, the current V path and the final attention output MSE.
# Selection uses only calibration folds (case-equal normalized output MSE);
# the independent holdout window is recorded but never vetoes.
# ---------------------------------------------------------------------------

_AH1R_EVENT_SLOTS = 8


def _ah1r_pretransform_dense(
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


def _ah1r_cayley(skew: torch.Tensor) -> torch.Tensor:
    """Cayley transform (I - A)(I + A)^-1 of a skew-symmetric matrix."""

    dim = int(skew.shape[-1])
    eye = torch.eye(dim, device=skew.device, dtype=skew.dtype)
    left = eye - skew
    right = eye + skew
    return torch.linalg.solve(
        right.transpose(-1, -2), left.transpose(-1, -2)
    ).transpose(-1, -2)


def _ah1r_boundary_times(
    x0: torch.Tensor,
    dx: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    """Linearized HiF4 magnitude-midpoint crossing times (frozen hierarchy)."""

    offsets = (
        torch.arange(7, device=x0.device, dtype=torch.float32) * 0.25 + 0.125
    )
    mids = denom.to(torch.float32).unsqueeze(-1) * offsets
    absx = x0.abs().unsqueeze(-1)
    slope = torch.where(x0 != 0, torch.sign(x0) * dx, dx.abs()).unsqueeze(-1)
    t = (mids - absx) / slope
    valid = torch.isfinite(t) & (t > 0)
    return t[valid]


def _ah1r_hierarchy_denominator(params: dict[str, torch.Tensor]) -> torch.Tensor:
    denom = (
        params["scale_factor"] * params["scale_lv2"] * params["scale_lv3"]
    ).expand(params["sign"].shape)
    return denom.flatten(start_dim=-4).to(torch.float32)


def _ah1r_changed_codes(
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


def _ah1r_true_path_mse(
    q_pair: tuple,
    k_pair: tuple,
    q_state: dict,
    k_state: dict,
    v_hat: torch.Tensor,
    reference: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Tuple[float, dict, dict, torch.Tensor]:
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
    return mse, q_params, k_params, player


def _ah1r_finite(value: float) -> float:
    result = float(value)
    return result if math.isfinite(result) else -1.0


def _ah1r_parent_state(
    states: dict[str, Any],
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Deployed parent rotation/center, identity/zero when the gate chose identity."""

    q_parent = states["q_state"]
    k_parent = states["k_state"]
    rotation = q_parent.get("learned_rotation")
    center = k_parent.get("learned_center")
    eye = torch.eye(head_dim, device=device, dtype=torch.float32)
    if torch.is_tensor(rotation):
        r_parent = rotation.detach().to(device=device, dtype=torch.float32).clone()
        arm = "rotation"
    else:
        r_parent = eye.unsqueeze(0).expand(kv_num_heads, head_dim, head_dim).clone()
        arm = "identity"
    if torch.is_tensor(center):
        c_parent = center.detach().to(device=device, dtype=torch.float32).clone()
    else:
        c_parent = torch.zeros(
            kv_num_heads, head_dim, device=device, dtype=torch.float32
        )
    return r_parent, c_parent, arm


def _ah1r_prepare_folds(
    calib_qkv_list: list,
    windows: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Tuple[dict, dict, list]:
    """Per-fold deployed parent reference data in pre-rotation coordinates."""

    q_parent = dict(states["q_state"])
    k_parent = dict(states["k_state"])
    folds = []
    for index in range(len(calib_qkv_list) - 1):
        item = calib_qkv_list[index]
        window = windows[index]
        reference = _a2_attention_forward(
            window["q"][None], window["k"][None], window["v"][None],
            q_num_heads, kv_num_heads, head_dim,
        )[0]
        v_hat = _dequantize_hif4(
            hif4_dynamic_quantize_v(
                item["v"][0], item["v"][1],
                kv_num_heads, head_dim, states["v_state"],
            )
        ).to(torch.float32)
        std_q = _dequantize_hif4(_dense_to_hif4(window["q"])).to(torch.float32)
        std_k = _dequantize_hif4(_dense_to_hif4(window["k"])).to(torch.float32)
        standard = _a2_attention_forward(
            std_q[None], std_k[None], v_hat[None],
            q_num_heads, kv_num_heads, head_dim,
        )[0]
        standard_mse = max(float((standard - reference).square().mean()), 1e-12)
        parent_mse, parent_q_params, parent_k_params, player = _ah1r_true_path_mse(
            item["q"], item["k"], q_parent, k_parent, v_hat, reference,
            q_num_heads, kv_num_heads, head_dim,
        )
        u_q = _ah1r_pretransform_dense(window["q"], q_parent, q_num_heads, False)
        u_k = _ah1r_pretransform_dense(window["k"], k_parent, kv_num_heads, True)
        folds.append({
            "q_pair": item["q"],
            "k_pair": item["k"],
            "v_hat": v_hat,
            "reference": reference,
            "standard_mse": standard_mse,
            "parent_mse": parent_mse,
            "parent_q_params": parent_q_params,
            "parent_k_params": parent_k_params,
            "player": player,
            "u_q": u_q,
            "u_k": u_k,
        })
    return q_parent, k_parent, folds


def _ah1r_mean_gradient(
    folds: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Case-equal mean output gradient G_R at the deployed parent state."""

    groups = int(kv_num_heads)
    per_group = q_num_heads // groups
    grad = torch.zeros(
        groups, head_dim, head_dim,
        device=folds[0]["u_q"].device, dtype=torch.float32,
    )
    for fold in folds:
        residual = fold["player"] - fold["reference"]
        d_output = (
            2.0 * residual / float(residual.numel()) / fold["standard_mse"]
        )
        q_hat = _dequantize_hif4(fold["parent_q_params"]).to(torch.float32)
        k_hat = _dequantize_hif4(fold["parent_k_params"]).to(torch.float32)
        d_qhat, d_khat = _m_attention_backward(
            d_output[None], q_hat, k_hat, fold["v_hat"],
            q_num_heads, kv_num_heads, head_dim,
        )
        tokens_q = int(fold["u_q"].shape[0])
        tokens_k = int(fold["u_k"].shape[0])
        u_q3g = fold["u_q"].reshape(tokens_q, groups, per_group, head_dim)
        dq3g = d_qhat.reshape(tokens_q, groups, per_group, head_dim)
        u_k3 = fold["u_k"].reshape(tokens_k, kv_num_heads, head_dim)
        dk3 = d_khat.reshape(tokens_k, kv_num_heads, head_dim)
        grad = grad + torch.einsum("tghk,tghd->gkd", u_q3g, dq3g)
        grad = grad + torch.einsum("tgk,tgd->gkd", u_k3, dk3)
    return grad / float(len(folds))


def _ah1r_threshold_event_search(
    calib_qkv_list: list,
    windows: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[dict[str, Any], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Parent-anchored threshold event search.

    Returns the audit dict plus the selected (rotation, center) CPU tensors when
    an event strictly improves over the parent on the calibration folds.  Never
    mutates ``states``; the caller applies the selection.
    """

    audit: dict[str, Any] = {
        "ah1r_status": "unavailable-parent",
        "ah1r_parent_arm": "unknown",
        "ah1r_direction_norm": -1.0,
        "ah1r_events_raw": 0,
        "ah1r_events_dedup": 0,
        "ah1r_event_t": [],
        "ah1r_event_t_eval": [],
        "ah1r_event_changed_q": [],
        "ah1r_event_changed_k": [],
        "ah1r_fold_loss_parent": -1.0,
        "ah1r_fold_loss_events": [],
        "ah1r_accepted_event": 0,
        "ah1r_holdout_loss_parent": -1.0,
        "ah1r_holdout_loss_final": -1.0,
        "ah1r_attempted": 0,
        "ah1r_accepted": 0,
        "ah1r_folds": 0,
        "ah1r_t0_identical": 0,
    }
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return audit, None

    r_parent, c_parent, arm = _ah1r_parent_state(
        states, kv_num_heads, head_dim, device
    )
    audit["ah1r_parent_arm"] = arm
    q_parent, k_parent, folds = _ah1r_prepare_folds(
        calib_qkv_list, windows, states,
        q_num_heads, kv_num_heads, head_dim,
    )
    fold_count = len(folds)
    audit["ah1r_folds"] = int(fold_count)

    grad = _ah1r_mean_gradient(
        folds, q_num_heads, kv_num_heads, head_dim
    )
    skew = r_parent.transpose(-1, -2) @ grad
    skew = skew - skew.transpose(-1, -2)
    audit["ah1r_direction_norm"] = _ah1r_finite(float(skew.norm()))

    # Path derivative at t=0: R(t)=R_parent@cayley(t S), d cayley/dt|_0=-2 S.
    d_rotation = -2.0 * (r_parent @ skew)

    raw_events = 0
    distinct_parts = []
    for fold in folds:
        u_q = fold["u_q"]
        u_k = fold["u_k"]
        x_q0 = _a2_apply_group_rotation(u_q, q_num_heads, r_parent)
        dx_q = _a2_apply_group_rotation(u_q, q_num_heads, d_rotation)
        x_k0 = (
            _a2_apply_group_rotation(u_k, kv_num_heads, r_parent).reshape(
                -1, kv_num_heads, head_dim
            )
            + c_parent[None]
        ).reshape(u_k.shape)
        dx_k = _a2_apply_group_rotation(u_k, kv_num_heads, d_rotation)
        denom_q = _ah1r_hierarchy_denominator(fold["parent_q_params"])
        denom_k = _ah1r_hierarchy_denominator(fold["parent_k_params"])
        t_q = _ah1r_boundary_times(x_q0, dx_q, denom_q)
        t_k = _ah1r_boundary_times(x_k0, dx_k, denom_k)
        raw_events += int(t_q.numel()) + int(t_k.numel())
        if int(t_q.numel()) > 0:
            distinct_parts.append(torch.unique(t_q).cpu())
        if int(t_k.numel()) > 0:
            distinct_parts.append(torch.unique(t_k).cpu())
    audit["ah1r_events_raw"] = int(raw_events)

    if distinct_parts:
        all_distinct = torch.unique(torch.cat(distinct_parts))
    else:
        all_distinct = torch.empty(0, dtype=torch.float32)
    dedup_events = int(all_distinct.numel())
    audit["ah1r_events_dedup"] = dedup_events

    parent_agg = sum(
        fold["parent_mse"] / fold["standard_mse"] for fold in folds
    ) / float(fold_count)
    audit["ah1r_fold_loss_parent"] = _ah1r_finite(parent_agg)

    def _holdout_losses(final_q_state: dict, final_k_state: dict) -> Tuple[float, float]:
        gate_item = calib_qkv_list[-1]
        gate_window = windows[-1]
        reference_h = _a2_attention_forward(
            gate_window["q"][None], gate_window["k"][None], gate_window["v"][None],
            q_num_heads, kv_num_heads, head_dim,
        )[0]
        v_hat_h = _dequantize_hif4(
            hif4_dynamic_quantize_v(
                gate_item["v"][0], gate_item["v"][1],
                kv_num_heads, head_dim, states["v_state"],
            )
        ).to(torch.float32)
        std_q_h = _dequantize_hif4(_dense_to_hif4(gate_window["q"])).to(torch.float32)
        std_k_h = _dequantize_hif4(_dense_to_hif4(gate_window["k"])).to(torch.float32)
        standard_h = _a2_attention_forward(
            std_q_h[None], std_k_h[None], v_hat_h[None],
            q_num_heads, kv_num_heads, head_dim,
        )[0]
        normalizer = max(float((standard_h - reference_h).square().mean()), 1e-12)
        parent_h = _ah1r_true_path_mse(
            gate_item["q"], gate_item["k"], q_parent, k_parent,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )[0]
        final_h = _ah1r_true_path_mse(
            gate_item["q"], gate_item["k"], final_q_state, final_k_state,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )[0]
        return parent_h / normalizer, final_h / normalizer

    # t=0 must reproduce the deployed parent exactly: R(0)=R_parent.
    r_zero = (
        r_parent @ _ah1r_cayley(torch.zeros_like(skew))
    ).detach().cpu().to(torch.float32)
    c_zero = c_parent.detach().cpu().to(torch.float32)
    q_zero = dict(q_parent)
    q_zero["learned_rotation"] = r_zero
    k_zero = dict(k_parent)
    k_zero["learned_rotation"] = r_zero.clone()
    k_zero["learned_center"] = c_zero
    t0_mse, t0_q_params, t0_k_params, _t0_player = _ah1r_true_path_mse(
        folds[0]["q_pair"], folds[0]["k_pair"], q_zero, k_zero,
        folds[0]["v_hat"], folds[0]["reference"],
        q_num_heads, kv_num_heads, head_dim,
    )
    t0_identical = int(
        t0_mse == folds[0]["parent_mse"]
        and _ah1r_changed_codes(folds[0]["parent_q_params"], t0_q_params) == 0
        and _ah1r_changed_codes(folds[0]["parent_k_params"], t0_k_params) == 0
    )
    audit["ah1r_t0_identical"] = t0_identical

    if dedup_events == 0:
        audit["ah1r_status"] = "unreachable"
        holdout_parent, holdout_final = _holdout_losses(q_parent, k_parent)
        audit["ah1r_holdout_loss_parent"] = _ah1r_finite(holdout_parent)
        audit["ah1r_holdout_loss_final"] = _ah1r_finite(holdout_final)
        print(
            f"[A-H1R] status=unreachable arm={arm} raw={raw_events} "
            f"dedup={dedup_events} t0_identical={t0_identical}",
            flush=True,
        )
        return audit, None

    slots = all_distinct[:_AH1R_EVENT_SLOTS]
    eval_ts = torch.nextafter(slots, torch.full_like(slots, float("inf")))
    audit["ah1r_event_t"] = [float(value) for value in slots]
    audit["ah1r_event_t_eval"] = [float(value) for value in eval_ts]

    fold_losses = []
    changed_q = []
    changed_k = []
    best = None
    for slot_index in range(int(slots.numel())):
        t_eval = float(eval_ts[slot_index])
        r_t = r_parent @ _ah1r_cayley(t_eval * skew)
        cpu_rotation = r_t.detach().cpu().to(torch.float32)
        cpu_center = c_parent.detach().cpu().to(torch.float32)
        cand_q = dict(q_parent)
        cand_q["learned_rotation"] = cpu_rotation
        cand_k = dict(k_parent)
        cand_k["learned_rotation"] = cpu_rotation.clone()
        cand_k["learned_center"] = cpu_center
        agg = 0.0
        n_changed_q = 0
        n_changed_k = 0
        for fold in folds:
            mse, q_params, k_params, _player = _ah1r_true_path_mse(
                fold["q_pair"], fold["k_pair"], cand_q, cand_k,
                fold["v_hat"], fold["reference"],
                q_num_heads, kv_num_heads, head_dim,
            )
            agg += mse / fold["standard_mse"]
            n_changed_q += _ah1r_changed_codes(fold["parent_q_params"], q_params)
            n_changed_k += _ah1r_changed_codes(fold["parent_k_params"], k_params)
        agg /= float(fold_count)
        fold_losses.append(_ah1r_finite(agg))
        changed_q.append(int(n_changed_q))
        changed_k.append(int(n_changed_k))
        if best is None or agg < best[1]:
            best = (slot_index, agg, cpu_rotation, cpu_center)

    audit["ah1r_fold_loss_events"] = fold_losses
    audit["ah1r_event_changed_q"] = changed_q
    audit["ah1r_event_changed_k"] = changed_k
    audit["ah1r_attempted"] = int(slots.numel())

    selected = None
    if best is not None and best[1] < parent_agg:
        selected = (best[2], best[3])
        audit["ah1r_status"] = "accepted"
        audit["ah1r_accepted"] = 1
        audit["ah1r_accepted_event"] = int(best[0]) + 1
        final_q = dict(q_parent)
        final_q["learned_rotation"] = best[2]
        final_k = dict(k_parent)
        final_k["learned_rotation"] = best[2].clone()
        final_k["learned_center"] = best[3]
    else:
        audit["ah1r_status"] = "no-improvement"
        final_q = q_parent
        final_k = k_parent

    holdout_parent, holdout_final = _holdout_losses(final_q, final_k)
    audit["ah1r_holdout_loss_parent"] = _ah1r_finite(holdout_parent)
    audit["ah1r_holdout_loss_final"] = _ah1r_finite(holdout_final)
    print(
        f"[A-H1R] status={audit['ah1r_status']} arm={arm} raw={raw_events} "
        f"dedup={dedup_events} attempted={audit['ah1r_attempted']} "
        f"accepted={audit['ah1r_accepted']} "
        f"accepted_event={audit['ah1r_accepted_event']} "
        f"parent={parent_agg:.6f} t0_identical={t0_identical}",
        flush=True,
    )
    return audit, selected


def _ah1r_assert_dynamic_applies(
    calib_qkv_list: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> None:
    """Fail loudly if a stored learned rotation/center is silently ignored."""

    item = calib_qkv_list[0]
    checks = (
        ("q", states["q_state"], q_num_heads, item["q"], False, hif4_dynamic_quantize_q),
        ("k", states["k_state"], kv_num_heads, item["k"], True, hif4_dynamic_quantize_k),
    )
    for name, state, heads, pair, is_k, api in checks:
        rotation = state.get("learned_rotation")
        center = state.get("learned_center") if is_k else None
        if not torch.is_tensor(rotation) and not torch.is_tensor(center):
            continue
        dense = _dequantize_nvfp4_float32(pair[0], pair[1]).to(torch.float32)
        transformed = _ah1r_pretransform_dense(dense, state, heads, is_k)
        if torch.is_tensor(rotation):
            transformed = _a2_apply_group_rotation(transformed, heads, rotation)
        if torch.is_tensor(center):
            transformed = (
                transformed.reshape(-1, heads, head_dim)
                + center.to(transformed.device)[None]
            ).reshape(transformed.shape)
        direct = _dense_to_hif4(
            transformed,
            importance=state["importance"],
            search_offsets=state["offsets"],
            error_threshold=float(state["error_threshold"]),
            accept_margin=float(state["accept_margin"]),
            max_refine_ratio=float(state["max_refine_ratio"]),
            max_refine_blocks=int(state["max_refine_blocks"]),
        )
        deployed = api(pair[0], pair[1], heads, head_dim, state)
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
            if not bool(torch.equal(deployed[key], direct[key])):
                raise RuntimeError(
                    f"deployed {name} dynamic path did not apply learned transform"
                )
