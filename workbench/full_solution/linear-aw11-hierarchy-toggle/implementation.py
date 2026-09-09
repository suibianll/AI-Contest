# ---------------------------------------------------------------------------
# v214 / L-AW11: output-aware legal lv2 hierarchy-bit toggle.
#
# L-AW10 tested the finer four-value lv3 field and was a complete no-op.  This
# card moves one level up: keep the final Q(A), choose one natural eight-value
# group in each 64-value block, and test the legal lv2 1 <-> 2 toggle against
# the aggregate deployment-output residual.  Scale factor, lv3, mantissa/sign,
# and activation state remain unchanged.
# ---------------------------------------------------------------------------


_V214_GROUP_SIZE = 8
_V214_GROUPS_PER_BLOCK = _HIF4_BLOCK_SIZE // _V214_GROUP_SIZE


@torch.no_grad()
def _v214_reconstruct_deployment_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    state: dict[str, Any],
) -> Optional[torch.Tensor]:
    """Rebuild the pre-quantization weight in the parent's deployment frame."""

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale)
    if weight.ndim != 2:
        return None
    rows, channels = map(int, weight.shape)
    if channels % _HIF4_BLOCK_SIZE != 0:
        return None

    smooth_inv = state.get("smooth_inv")
    if smooth_inv is None:
        d = torch.ones(channels, dtype=torch.float32, device=weight.device)
    else:
        inverse = _safe_positive_vector(smooth_inv, channels).to(weight.device)
        d = inverse.reciprocal()

    permutation = state.get("permutation")
    if permutation is None:
        permutation = _identity_permutation(channels, weight.device)
    else:
        permutation = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
        if int(permutation.numel()) != channels:
            return None
        if not torch.equal(
            torch.sort(permutation).values,
            torch.arange(channels, dtype=torch.int64, device=weight.device),
        ):
            return None

    transformed = _linear_pair_transform(
        weight,
        d,
        permutation,
        int(state.get("block_smooth_size", 0)),
        int(state.get("block_smooth_seed", 0)),
        weight_side=True,
        cat_transform=state.get("cat_transform"),
    ).to(torch.float32)

    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=weight.device, dtype=torch.float32)
        v = residual_v.to(device=weight.device, dtype=torch.float32)
        transformed = transformed - (transformed @ v) @ u.transpose(0, 1)
    else:
        rank1_u = state.get("rank1_u")
        rank1_v = state.get("rank1_v")
        if rank1_u is not None and rank1_v is not None:
            u = rank1_u.to(device=weight.device, dtype=torch.float32).reshape(-1, 1)
            v = rank1_v.to(device=weight.device, dtype=torch.float32).reshape(-1, 1)
            if int(u.shape[0]) != channels or int(v.shape[0]) != channels:
                return None
            transformed = transformed - (transformed @ v) @ u.transpose(0, 1)
    if tuple(transformed.shape) != (rows, channels):
        return None
    return torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def _v214_hierarchy_toggle_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Toggle one shared natural lv2 group per block when product loss falls."""

    target_weight = _v214_reconstruct_deployment_weight(
        weight_quant, weight_scale, state
    )
    if target_weight is None:
        return None
    parent_weight = _dequantize_hif4(parent_params).to(
        device=target_weight.device, dtype=torch.float32
    )
    if parent_weight.ndim != 2 or tuple(parent_weight.shape) != tuple(target_weight.shape):
        return None
    rows, channels = map(int, parent_weight.shape)
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % _V214_GROUP_SIZE != 0:
        return None

    blocks = channels // _HIF4_BLOCK_SIZE
    groups_per_block = _V214_GROUPS_PER_BLOCK
    diagnostics.update(
        {
            "v214_attempted": 1,
            "v214_coordinate": "deployment-output-shared-lv2-toggle",
            "v214_group_size": _V214_GROUP_SIZE,
            "v214_groups_per_block": _V214_GROUPS_PER_BLOCK,
            "v214_blocks": blocks,
        }
    )

    activation_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    parent_loss = 0.0
    fit_rows = 0
    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        act_quant, act_scale = pair
        a_fp = _dequantize_nvfp4_float32(act_quant, act_scale).to(
            device=target_weight.device, dtype=torch.float32
        )
        if a_fp.ndim != 2 or int(a_fp.shape[1]) != channels:
            return None
        if int(a_fp.shape[0]) == 0:
            continue

        # Freeze the exact final activation deployment path before changing
        # static weight hierarchy bits.  No product tensor is retained in state.
        a_hat = _dequantize_hif4(
            hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        ).to(device=target_weight.device, dtype=torch.float32)
        z_ref = _static_actorder_dense_from_state(act_quant, act_scale, state).to(
            device=target_weight.device, dtype=torch.float32
        )
        if tuple(a_hat.shape) != tuple(a_fp.shape) or tuple(z_ref.shape) != tuple(a_fp.shape):
            return None
        teacher = z_ref.mm(target_weight.transpose(0, 1))
        residual = teacher - a_hat.mm(parent_weight.transpose(0, 1))
        if not bool(torch.isfinite(residual).all()):
            return None
        activation_parts.append(a_hat)
        residual_parts.append(residual)
        parent_loss += float(residual.square().sum())
        fit_rows += int(a_fp.shape[0])

    diagnostics["v214_fit_windows"] = len(activation_parts)
    diagnostics["v214_fit_rows"] = fit_rows
    diagnostics["v214_loss_parent"] = float(parent_loss)
    if not activation_parts or not math.isfinite(parent_loss):
        return None

    activation_all = torch.cat(activation_parts, dim=0)
    residual_all = torch.cat(residual_parts, dim=0)
    grouped = activation_all.reshape(
        int(activation_all.shape[0]), blocks, groups_per_block, _V214_GROUP_SIZE
    )
    block_grams = torch.einsum("tbgi,tbgj->bgij", grouped, grouped)
    initial_gradient = torch.einsum("tbgi,tr->bgir", grouped, residual_all)

    parent_grouped = parent_weight.reshape(
        rows, blocks, groups_per_block, _V214_GROUP_SIZE
    )
    original_lv2 = parent_params["scale_lv2"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, groups_per_block)
    lv2 = original_lv2.clone()
    legal = (lv2 == 1.0) | (lv2 == 2.0)
    new_lv2 = torch.where(legal, 3.0 - lv2, lv2)
    ratio = (new_lv2 / lv2.clamp_min(_EPS) - 1.0).unsqueeze(-1)
    delta_weight = parent_grouped * ratio
    delta_grouped = delta_weight.permute(1, 2, 0, 3)
    gradient_grouped = initial_gradient.permute(0, 1, 3, 2)
    initial_change = -2.0 * (gradient_grouped * delta_grouped).sum(dim=(2, 3))
    initial_change = initial_change + torch.einsum(
        "bgri,bgij,bgrj->bgr", delta_grouped, block_grams, delta_grouped
    ).sum(dim=-1)
    initial_change = torch.where(
        legal.all(dim=0), initial_change,
        torch.full_like(initial_change, float("inf")),
    )
    selected_groups = torch.argmin(initial_change, dim=1)

    accepted_groups = 0
    changed_lv2 = 0
    for block_index in range(blocks):
        local_group = int(selected_groups[block_index])
        local_z = grouped[:, block_index, local_group, :]
        gradient = local_z.transpose(0, 1).mm(residual_all)
        gram = block_grams[block_index, local_group]
        old = lv2[:, block_index, local_group]
        valid = (old == 1.0) | (old == 2.0)
        proposed_lv2 = torch.where(valid, 3.0 - old, old)
        local_ratio = proposed_lv2 / old.clamp_min(_EPS) - 1.0
        delta = parent_grouped[:, block_index, local_group, :] * local_ratio.unsqueeze(-1)
        if not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(gradient).all()):
            continue
        change = -2.0 * (gradient.transpose(0, 1) * delta).sum()
        change = change + torch.einsum("ri,ij,rj->", delta, gram, delta)
        if not bool(torch.isfinite(change)) or float(change) >= -_EPS:
            continue

        lv2[:, block_index, local_group] = proposed_lv2
        residual_all.sub_(local_z.mm(delta.transpose(0, 1)))
        accepted_groups += 1
        changed_lv2 += int((proposed_lv2 != old).sum())

    diagnostics["v214_selected_groups"] = blocks
    diagnostics["v214_accepted_groups"] = accepted_groups
    diagnostics["v214_changed_lv2"] = changed_lv2
    if accepted_groups == 0:
        return None

    candidate_params = {
        key: value.detach().to(device=target_weight.device).clone()
        for key, value in parent_params.items()
    }
    candidate_params["scale_lv2"] = lv2.reshape_as(
        parent_params["scale_lv2"]
    ).to(dtype=parent_params["scale_lv2"].dtype)
    candidate_loss = float(residual_all.square().sum())
    diagnostics["v214_loss_candidate"] = candidate_loss
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    diagnostics["v214_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v214_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v214_attempted": 0,
        "v214_accepted": 0,
        "v214_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v214_hierarchy_toggle_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v214_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v214_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V214_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v214: v202 parent plus one output-aware lv2 group toggle per block."""

    result = _V214_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v214_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )

