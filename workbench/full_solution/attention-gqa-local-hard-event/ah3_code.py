# ---------------------------------------------------------------------------
# A-H3: GQA-group local hard-event coordinate update.
#
# R1 used one global tangent direction, so a few useful per-head code flips can
# be cancelled by many unrelated flips in other KV groups.  A-H3 instead:
#   1. starts from the deployed root Q/K state and freezes V;
#   2. walks KV groups in fixed index order exactly once.  Each group uses the
#      block of the deployed-parent output gradient that lives in its own
#      orthogonal tangent space, S_g = skew(R_g^T G_g), and finds the first
#      real Q/K hard-code threshold along both -S_g and +S_g;
#   3. each group compares only three states (current, +first event, -first
#      event) through the full deployed Attention output; it is accepted only
#      if it flips codes and strictly lowers the calibration-fold aggregate;
#   4. the accepted state becomes the parent for the next group.  One round,
#      fixed order, first event only, no second/other-sign expansion.
#
# The gradient is evaluated once at the deployed parent; each group's rotation
# is untouched by other groups until that group is processed, so its tangent
# block is still evaluated at the correct point.  This keeps calibration time
# close to R1 on a root that only has ~19s of official headroom.
# ---------------------------------------------------------------------------

_AH3_MAX_EVENTS = 1


def _ah3_pretransform_dense(
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


def _ah3_cayley(skew: torch.Tensor) -> torch.Tensor:
    """Cayley transform (I - A)(I + A)^-1 of a skew-symmetric matrix."""

    dim = int(skew.shape[-1])
    eye = torch.eye(dim, device=skew.device, dtype=skew.dtype)
    left = eye - skew
    right = eye + skew
    return torch.linalg.solve(
        right.transpose(-1, -2), left.transpose(-1, -2)
    ).transpose(-1, -2)


def _ah3_boundary_times(
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


def _ah3_hierarchy_denominator(params: dict[str, torch.Tensor]) -> torch.Tensor:
    denom = (
        params["scale_factor"] * params["scale_lv2"] * params["scale_lv3"]
    ).expand(params["sign"].shape)
    return denom.flatten(start_dim=-4).to(torch.float32)


def _ah3_changed_codes(
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


def _ah3_true_path_mse(
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


def _ah3_finite(value: float) -> float:
    result = float(value)
    return result if math.isfinite(result) else -1.0


def _ah3_parent_state(
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


def _ah3_prepare_folds(
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
        parent_mse, parent_q_params, parent_k_params, player = _ah3_true_path_mse(
            item["q"], item["k"], q_parent, k_parent, v_hat, reference,
            q_num_heads, kv_num_heads, head_dim,
        )
        u_q = _ah3_pretransform_dense(window["q"], q_parent, q_num_heads, False)
        u_k = _ah3_pretransform_dense(window["k"], k_parent, kv_num_heads, True)
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


def _ah3_forward_folds(
    folds: list,
    q_parent: dict,
    k_parent: dict,
    rotation: torch.Tensor,
    center: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> list:
    cpu_rotation = rotation.detach().cpu().to(torch.float32)
    cpu_center = center.detach().cpu().to(torch.float32)
    cand_q = dict(q_parent)
    cand_q["learned_rotation"] = cpu_rotation
    cand_k = dict(k_parent)
    cand_k["learned_rotation"] = cpu_rotation.clone()
    cand_k["learned_center"] = cpu_center
    forward = []
    for fold in folds:
        mse, q_params, k_params, player = _ah3_true_path_mse(
            fold["q_pair"], fold["k_pair"], cand_q, cand_k,
            fold["v_hat"], fold["reference"],
            q_num_heads, kv_num_heads, head_dim,
        )
        forward.append((mse, q_params, k_params, player))
    return forward


def _ah3_aggregate_loss(folds: list, forward: list) -> float:
    return sum(
        entry[0] / fold["standard_mse"] for fold, entry in zip(folds, forward)
    ) / float(len(folds))


def _ah3_changed_between(forward_a: list, forward_b: list) -> Tuple[int, int]:
    q_total = 0
    k_total = 0
    for (_m1, qa, ka, _p1), (_m2, qb, kb, _p2) in zip(forward_a, forward_b):
        q_total += _ah3_changed_codes(qa, qb)
        k_total += _ah3_changed_codes(ka, kb)
    return int(q_total), int(k_total)


def _ah3_gradient_from(
    folds: list,
    forward: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    groups = int(kv_num_heads)
    per_group = q_num_heads // groups
    grad = torch.zeros(
        groups, head_dim, head_dim,
        device=folds[0]["u_q"].device, dtype=torch.float32,
    )
    for fold, (_mse, q_params, k_params, player) in zip(folds, forward):
        residual = player - fold["reference"]
        d_output = (
            2.0 * residual / float(residual.numel()) / fold["standard_mse"]
        )
        q_hat = _dequantize_hif4(q_params).to(torch.float32)
        k_hat = _dequantize_hif4(k_params).to(torch.float32)
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


def _ah3_first_event(
    folds: list,
    rotation: torch.Tensor,
    center: torch.Tensor,
    group: int,
    skew: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Optional[float]:
    """Smallest strictly-positive Q/K magnitude-boundary time for one group."""

    d_rotation = torch.zeros_like(rotation)
    d_rotation[group] = -2.0 * (rotation[group] @ skew)
    best = None
    for fold in folds:
        u_q = fold["u_q"]
        u_k = fold["u_k"]
        x_q0 = _a2_apply_group_rotation(u_q, q_num_heads, rotation)
        dx_q = _a2_apply_group_rotation(u_q, q_num_heads, d_rotation)
        x_k0 = (
            _a2_apply_group_rotation(u_k, kv_num_heads, rotation).reshape(
                -1, kv_num_heads, head_dim
            )
            + center[None]
        ).reshape(u_k.shape)
        dx_k = _a2_apply_group_rotation(u_k, kv_num_heads, d_rotation)
        t_q = _ah3_boundary_times(
            x_q0, dx_q, _ah3_hierarchy_denominator(fold["parent_q_params"])
        )
        t_k = _ah3_boundary_times(
            x_k0, dx_k, _ah3_hierarchy_denominator(fold["parent_k_params"])
        )
        for times in (t_q, t_k):
            if int(times.numel()) > 0:
                candidate = float(times.min())
                if best is None or candidate < best:
                    best = candidate
    return best


def _ah3_group_event_search(
    calib_qkv_list: list,
    windows: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[dict[str, Any], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Per-GQA-group first-event update; never mutates ``states``."""

    audit: dict[str, Any] = {
        "ah3_status": "unavailable-parent",
        "ah3_parent_arm": "unknown",
        "ah3_groups": 0,
        "ah3_group_accepted": [],
        "ah3_group_sign": [],
        "ah3_group_t": [],
        "ah3_group_changed_q": [],
        "ah3_group_changed_k": [],
        "ah3_group_loss_before": [],
        "ah3_group_loss_after": [],
        "ah3_group_pos_t": [],
        "ah3_group_neg_t": [],
        "ah3_fold_loss_parent": -1.0,
        "ah3_fold_loss_final": -1.0,
        "ah3_holdout_loss_parent": -1.0,
        "ah3_holdout_loss_final": -1.0,
        "ah3_attempted": 0,
        "ah3_accepted": 0,
        "ah3_folds": 0,
        "ah3_t0_identical": 0,
    }
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return audit, None

    r_parent, c_parent, arm = _ah3_parent_state(
        states, kv_num_heads, head_dim, device
    )
    audit["ah3_parent_arm"] = arm
    q_parent, k_parent, folds = _ah3_prepare_folds(
        calib_qkv_list, windows, states,
        q_num_heads, kv_num_heads, head_dim,
    )
    fold_count = len(folds)
    audit["ah3_folds"] = int(fold_count)
    audit["ah3_groups"] = int(kv_num_heads)

    parent_forward = _ah3_forward_folds(
        folds, q_parent, k_parent, r_parent, c_parent,
        q_num_heads, kv_num_heads, head_dim,
    )
    parent_loss = _ah3_aggregate_loss(folds, parent_forward)
    audit["ah3_fold_loss_parent"] = _ah3_finite(parent_loss)
    grad = _ah3_gradient_from(
        folds, parent_forward, q_num_heads, kv_num_heads, head_dim
    )

    r_zero = (
        r_parent @ _ah3_cayley(torch.zeros_like(r_parent[0]))
    ).detach().cpu().to(torch.float32)
    zero_forward = _ah3_forward_folds(
        folds, q_parent, k_parent, r_zero.to(device), c_parent,
        q_num_heads, kv_num_heads, head_dim,
    )
    zero_changed = _ah3_changed_between(parent_forward, zero_forward)
    audit["ah3_t0_identical"] = int(
        _ah3_aggregate_loss(folds, zero_forward) == parent_loss
        and zero_changed == (0, 0)
    )

    r_work = r_parent.clone()
    forward_work = parent_forward
    current_loss = parent_loss
    accepted_total = 0
    for group in range(int(kv_num_heads)):
        skew = r_work[group].transpose(-1, -2) @ grad[group]
        skew = skew - skew.transpose(-1, -2)
        pos_t = _ah3_first_event(
            folds, r_work, c_parent, group, skew,
            q_num_heads, kv_num_heads, head_dim,
        )
        neg_t = _ah3_first_event(
            folds, r_work, c_parent, group, -skew,
            q_num_heads, kv_num_heads, head_dim,
        )
        best = None
        for sign, event_t in ((1, pos_t), (-1, neg_t)):
            if event_t is None:
                continue
            r_try = r_work.clone()
            r_try[group] = r_work[group] @ _ah3_cayley(event_t * sign * skew)
            forward_try = _ah3_forward_folds(
                folds, q_parent, k_parent, r_try, c_parent,
                q_num_heads, kv_num_heads, head_dim,
            )
            loss_try = _ah3_aggregate_loss(folds, forward_try)
            changed_q, changed_k = _ah3_changed_between(forward_work, forward_try)
            if best is None or loss_try < best["loss"]:
                best = {
                    "sign": sign,
                    "t": event_t,
                    "rotation": r_try,
                    "forward": forward_try,
                    "loss": loss_try,
                    "changed_q": changed_q,
                    "changed_k": changed_k,
                }
        loss_before = current_loss
        accepted = False
        if (
            best is not None
            and best["changed_q"] + best["changed_k"] > 0
            and best["loss"] < current_loss
        ):
            r_work = best["rotation"]
            forward_work = best["forward"]
            current_loss = best["loss"]
            accepted = True
            accepted_total += 1
        audit["ah3_group_accepted"].append(int(accepted))
        audit["ah3_group_sign"].append(int(best["sign"]) if accepted else 0)
        audit["ah3_group_t"].append(
            _ah3_finite(best["t"]) if accepted else -1.0
        )
        audit["ah3_group_changed_q"].append(
            int(best["changed_q"]) if accepted else 0
        )
        audit["ah3_group_changed_k"].append(
            int(best["changed_k"]) if accepted else 0
        )
        audit["ah3_group_loss_before"].append(_ah3_finite(loss_before))
        audit["ah3_group_loss_after"].append(_ah3_finite(current_loss))
        audit["ah3_group_pos_t"].append(_ah3_finite(pos_t) if pos_t else -1.0)
        audit["ah3_group_neg_t"].append(_ah3_finite(neg_t) if neg_t else -1.0)

    audit["ah3_fold_loss_final"] = _ah3_finite(current_loss)
    audit["ah3_attempted"] = int(kv_num_heads)
    audit["ah3_accepted"] = int(accepted_total)

    def _holdout_losses(final_rotation: torch.Tensor) -> Tuple[float, float]:
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
        parent_h = _ah3_true_path_mse(
            gate_item["q"], gate_item["k"], q_parent, k_parent,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )[0]
        cand_q = dict(q_parent)
        cand_q["learned_rotation"] = final_rotation.detach().cpu().to(torch.float32)
        cand_k = dict(k_parent)
        cand_k["learned_rotation"] = (
            final_rotation.detach().cpu().to(torch.float32).clone()
        )
        cand_k["learned_center"] = c_parent.detach().cpu().to(torch.float32)
        final_h = _ah3_true_path_mse(
            gate_item["q"], gate_item["k"], cand_q, cand_k,
            v_hat_h, reference_h, q_num_heads, kv_num_heads, head_dim,
        )[0]
        return parent_h / normalizer, final_h / normalizer

    holdout_parent, holdout_final = _holdout_losses(r_work)
    audit["ah3_holdout_loss_parent"] = _ah3_finite(holdout_parent)
    audit["ah3_holdout_loss_final"] = _ah3_finite(holdout_final)

    if accepted_total == 0:
        audit["ah3_status"] = "no-improvement"
        print(
            f"[A-H3] status=no-improvement arm={arm} groups={kv_num_heads} "
            f"t0_identical={audit['ah3_t0_identical']}",
            flush=True,
        )
        return audit, None

    audit["ah3_status"] = "accepted"
    print(
        f"[A-H3] status=accepted arm={arm} accepted={accepted_total}/"
        f"{kv_num_heads} parent={parent_loss:.6f} final={current_loss:.6f} "
        f"t0_identical={audit['ah3_t0_identical']}",
        flush=True,
    )
    return audit, (
        r_work.detach().cpu().to(torch.float32),
        c_parent.detach().cpu().to(torch.float32),
    )


def _ah3_assert_dynamic_applies(
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
        transformed = _ah3_pretransform_dense(dense, state, heads, is_k)
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
