# ---------------------------------------------------------------------------
# v212 / L-AW9: low-freedom shared signed-mantissa code offset.
#
# v211 reached the legal code lattice but its one-group-per-output-row update
# overfit badly.  This card keeps the same frozen-Q(A) output objective while
# sharing one integer four-code offset across all output rows.  Exactly one
# natural four-element group is selected inside each 64-value block.  The
# offset is obtained from one code-space 4x4 normal equation, clipped to one
# lattice step, and accepted only when the aggregate calibration product loss
# decreases.  Scale/lv2/lv3 and activation state remain unchanged.
# ---------------------------------------------------------------------------

_V212_GROUP_SIZE = 4
_V212_GROUPS_PER_BLOCK = 1
_V212_RIDGE_RATIO = 1.0e-2
_V212_MAX_CODE_STEP = 1


@torch.no_grad()
def _v212_reconstruct_deployment_weight(
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
def _v212_shared_code_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Apply one shared integer code offset to one group in every 64-block."""

    target_weight = _v212_reconstruct_deployment_weight(
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
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % _V212_GROUP_SIZE != 0:
        return None

    blocks = channels // _HIF4_BLOCK_SIZE
    groups_per_block = _HIF4_BLOCK_SIZE // _V212_GROUP_SIZE
    diagnostics.update(
        {
            "v212_attempted": 1,
            "v212_coordinate": "deployment-output-shared-4-code-offset",
            "v212_group_size": _V212_GROUP_SIZE,
            "v212_groups_per_block": _V212_GROUPS_PER_BLOCK,
            "v212_blocks": blocks,
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

        # This is the exact final dynamic path, frozen before the static
        # weight-only code update.  No product tensor is retained in state.
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

    diagnostics["v212_fit_windows"] = len(activation_parts)
    diagnostics["v212_fit_rows"] = fit_rows
    diagnostics["v212_loss_parent"] = float(parent_loss)
    if not activation_parts or not math.isfinite(parent_loss):
        return None

    activation_all = torch.cat(activation_parts, dim=0)
    residual_all = torch.cat(residual_parts, dim=0)
    grouped = activation_all.reshape(
        int(activation_all.shape[0]), blocks, groups_per_block, _V212_GROUP_SIZE
    )
    # One vectorized score chooses exactly one group inside each 64-block.
    initial_gradient = torch.einsum(
        "tbgi,tr->bgir", grouped, residual_all
    )
    initial_scores = initial_gradient.square().sum(dim=(2, 3))
    selected_groups = torch.argmax(
        torch.nan_to_num(initial_scores, nan=0.0, posinf=0.0, neginf=0.0), dim=1
    )
    block_grams = torch.einsum("tbgi,tbgj->bgij", grouped, grouped)

    scale = parent_params["scale_factor"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, 1, 1, 1)
    lv2 = parent_params["scale_lv2"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 1, 1)
    lv3 = parent_params["scale_lv3"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 1)
    denominator = (scale * lv2 * lv3).repeat_interleave(4, dim=-1).reshape(
        rows, blocks, _HIF4_BLOCK_SIZE
    )
    codes = (
        parent_params["sign"].to(device=target_weight.device, dtype=torch.float32)
        * parent_params["mant"].to(device=target_weight.device, dtype=torch.float32)
        * 4.0
    ).reshape(rows, blocks, _HIF4_BLOCK_SIZE)
    codes = torch.round(codes).clamp(-7.0, 7.0)

    eye = torch.eye(_V212_GROUP_SIZE, dtype=torch.float32, device=target_weight.device)
    accepted_groups = 0
    changed_codes = 0
    # Blocks are visited once in natural order; every proposed offset is shared
    # by all output rows and is evaluated against the current exact residual.
    for block_index in range(blocks):
        local_group = int(selected_groups[block_index])
        block_lo = local_group * _V212_GROUP_SIZE
        lo = block_index * _HIF4_BLOCK_SIZE + block_lo
        hi = lo + _V212_GROUP_SIZE
        local_z = activation_all[:, lo:hi]
        gradient = local_z.transpose(0, 1).mm(residual_all)
        gram = block_grams[block_index, local_group]
        if not bool(torch.isfinite(gram).all() and torch.isfinite(gradient).all()):
            continue

        local_denominator = denominator[:, block_index, block_lo:hi]
        code_scale = 0.25 * local_denominator
        code_gram = code_scale.transpose(0, 1).mm(code_scale)
        system = gram * code_gram
        ridge = float(_V212_RIDGE_RATIO) * system.diagonal().mean().clamp_min(_EPS)
        system = system + ridge * eye
        rhs = (gradient * code_scale.transpose(0, 1)).sum(dim=1)
        try:
            desired_offset = torch.linalg.solve(system, rhs)
        except RuntimeError:
            continue
        offset = torch.nan_to_num(
            desired_offset, nan=0.0, posinf=float(_V212_MAX_CODE_STEP), neginf=-float(_V212_MAX_CODE_STEP)
        ).round().clamp(-float(_V212_MAX_CODE_STEP), float(_V212_MAX_CODE_STEP))
        current = codes[:, block_index, block_lo:hi]
        proposed = (current + offset.reshape(1, -1)).clamp(-7.0, 7.0)
        delta = (proposed - current) * code_scale
        change = -2.0 * (gradient.transpose(0, 1) * delta).sum()
        change = change + torch.einsum("ri,ij,rj->", delta, gram, delta)
        if not bool(torch.isfinite(change)) or float(change) >= -_EPS:
            continue

        codes[:, block_index, block_lo:hi] = proposed
        residual_all.sub_(local_z.mm(delta.transpose(0, 1)))
        accepted_groups += 1
        changed_codes += int((proposed != current).sum())

    diagnostics["v212_selected_groups"] = blocks
    diagnostics["v212_accepted_groups"] = accepted_groups
    diagnostics["v212_changed_codes"] = changed_codes
    if accepted_groups == 0:
        return None

    candidate_params = {
        key: value.detach().to(device=target_weight.device).clone()
        for key, value in parent_params.items()
    }
    flat_codes = codes.reshape(rows, channels)
    candidate_params["sign"] = torch.sign(flat_codes).reshape_as(parent_params["sign"])
    candidate_params["mant"] = (flat_codes.abs() * 0.25).reshape_as(parent_params["mant"])
    candidate_params["sign"] = torch.where(
        candidate_params["mant"] == 0.0,
        torch.zeros_like(candidate_params["sign"]),
        candidate_params["sign"],
    )
    candidate_loss = float(residual_all.square().sum())
    diagnostics["v212_loss_candidate"] = candidate_loss
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    diagnostics["v212_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v212_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v212_attempted": 0,
        "v212_accepted": 0,
        "v212_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v212_shared_code_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v212_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v212_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V212_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v212: v202 parent plus shared direct four-code offset."""

    result = _V212_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v212_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
