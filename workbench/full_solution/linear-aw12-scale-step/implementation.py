# ---------------------------------------------------------------------------
# v215 / L-AW12: output-aware one-step E6M2 scale-factor update.
#
# The lv3 and lv2 hierarchy toggles were both complete no-ops.  This card uses
# the remaining legal static field: for each output row and natural 64-value
# block, the sign of the exact output-residual directional term chooses one
# adjacent E6M2 code step.  The proposed state is accepted only when the exact
# aggregate product residual decreases.  Q(A), hierarchy bits, and mantissa
# codes stay fixed; no neighboring code scan is performed.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _v215_reconstruct_deployment_weight(
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
def _v215_scale_factor_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Apply one residual-directed adjacent E6M2 code step per row/block."""

    target_weight = _v215_reconstruct_deployment_weight(
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
    if channels % _HIF4_BLOCK_SIZE != 0:
        return None

    blocks = channels // _HIF4_BLOCK_SIZE
    diagnostics.update(
        {
            "v215_attempted": 1,
            "v215_coordinate": "deployment-output-row-block-e6m2-step",
            "v215_group_size": _HIF4_BLOCK_SIZE,
            "v215_groups_per_block": 1,
            "v215_blocks": blocks,
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
        # static scale-factor codes.  No product tensor is retained in state.
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

    diagnostics["v215_fit_windows"] = len(activation_parts)
    diagnostics["v215_fit_rows"] = fit_rows
    diagnostics["v215_loss_parent"] = float(parent_loss)
    if not activation_parts or not math.isfinite(parent_loss):
        return None

    activation_all = torch.cat(activation_parts, dim=0)
    residual_all = torch.cat(residual_parts, dim=0)
    grouped = activation_all.reshape(
        int(activation_all.shape[0]), blocks, _HIF4_BLOCK_SIZE
    )
    block_grams = torch.einsum("tbi,tbj->bij", grouped, grouped)
    parent_grouped = parent_weight.reshape(rows, blocks, _HIF4_BLOCK_SIZE)

    current_codes = _e6m2_encode_nearest(
        parent_params["scale_factor"].to(
            device=target_weight.device, dtype=torch.float32
        ).reshape(rows, blocks)
    ).to(torch.int64)
    accepted_steps = 0
    changed_codes = 0

    # Visit natural blocks once.  Each row gets one directionally selected
    # adjacent code, then the whole row/block proposal is evaluated exactly.
    for block_index in range(blocks):
        local_z = grouped[:, block_index, :]
        gradient = local_z.transpose(0, 1).mm(residual_all)
        gram = block_grams[block_index]
        current_code = current_codes[:, block_index]
        current_scale = _e6m2_decode(current_code)
        sensitivity = (gradient.transpose(0, 1) * parent_grouped[:, block_index, :]).sum(dim=-1)
        if not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(sensitivity).all()):
            continue

        direction = torch.where(sensitivity >= 0.0, 1, -1).to(torch.int64)
        trial_code = (current_code + direction).clamp(0, 254)
        trial_scale = _e6m2_decode(trial_code)
        ratio = trial_scale / current_scale.clamp_min(_EPS) - 1.0
        delta = parent_grouped[:, block_index, :] * ratio.unsqueeze(-1)
        change = -2.0 * (gradient.transpose(0, 1) * delta).sum(dim=-1)
        change = change + torch.einsum("ri,ij,rj->r", delta, gram, delta)
        accept = (
            (trial_code != current_code)
            & torch.isfinite(change)
            & (change < -_EPS)
        )
        if not bool(accept.any()):
            continue

        accepted_delta = torch.where(accept.unsqueeze(-1), delta, torch.zeros_like(delta))
        residual_all.sub_(local_z.mm(accepted_delta.transpose(0, 1)))
        current_codes[:, block_index] = torch.where(
            accept, trial_code, current_code
        )
        accepted_steps += int(accept.sum())
        changed_codes += int(accept.sum())

    diagnostics["v215_accepted_steps"] = accepted_steps
    diagnostics["v215_changed_codes"] = changed_codes
    if accepted_steps == 0:
        return None

    candidate_params = {
        key: value.detach().to(device=target_weight.device).clone()
        for key, value in parent_params.items()
    }
    candidate_params["scale_factor"] = _e6m2_decode(current_codes).reshape_as(
        parent_params["scale_factor"]
    ).to(dtype=parent_params["scale_factor"].dtype)
    candidate_loss = float(residual_all.square().sum())
    diagnostics["v215_loss_candidate"] = candidate_loss
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    diagnostics["v215_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v215_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v215_attempted": 0,
        "v215_accepted": 0,
        "v215_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v215_scale_factor_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v215_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v215_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V215_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v215: v202 parent plus one output-aware E6M2 scale code step."""

    result = _V215_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v215_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )

