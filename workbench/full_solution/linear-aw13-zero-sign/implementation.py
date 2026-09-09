# ---------------------------------------------------------------------------
# v220 / L-AW13: zero-to-smallest signed-code insertion.
#
# The old LC2 implementation only moved an existing positive mantissa by a
# synchronous +/- step and never changed a zero carrier into a negative code.
# This card tests one genuinely discrete direction that remained open: for
# each natural 64-value block, choose one zero-code weight element by frozen
# output leverage and try the two legal signs of the smallest nonzero
# mantissa.  The exact aggregate calibration product loss is the acceptance
# rule.  Hierarchy fields and all dynamic APIs remain unchanged.
# ---------------------------------------------------------------------------

_V220_GROUP_SIZE = _HIF4_BLOCK_SIZE
_V220_CODE_MAGNITUDE = 0.25


@torch.no_grad()
def _v220_reconstruct_deployment_weight(
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
def _v220_zero_signed_code_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Insert at most one smallest signed code in each natural 64-value block."""

    target_weight = _v220_reconstruct_deployment_weight(
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
    if channels % _V220_GROUP_SIZE != 0:
        return None
    blocks = channels // _V220_GROUP_SIZE

    diagnostics.update(
        {
            "v220_attempted": 1,
            "v220_coordinate": "deployment-output-zero-to-smallest-signed-code",
            "v220_group_size": _V220_GROUP_SIZE,
            "v220_blocks": blocks,
        }
    )

    activation_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
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

        # Freeze the exact deployed activation operand and the parent's
        # calibration-side dense reference.  Output tensors stay local here;
        # only scalar provenance is added to activation_state.
        a_params = hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        a_hat = _dequantize_hif4(a_params).to(
            device=target_weight.device, dtype=torch.float32
        )
        z_ref = _static_actorder_dense_from_state(act_quant, act_scale, state).to(
            device=target_weight.device, dtype=torch.float32
        )
        if tuple(a_hat.shape) != tuple(a_fp.shape) or tuple(z_ref.shape) != tuple(a_fp.shape):
            return None
        teacher = z_ref.mm(target_weight.transpose(0, 1))
        residual = teacher - a_hat.mm(parent_weight.transpose(0, 1))
        if not bool(torch.isfinite(a_hat).all() and torch.isfinite(residual).all()):
            return None
        activation_parts.append(a_hat)
        residual_parts.append(residual)
        fit_rows += int(a_fp.shape[0])

    diagnostics["v220_fit_windows"] = len(activation_parts)
    diagnostics["v220_fit_rows"] = fit_rows
    if not activation_parts:
        return None

    activation_all = torch.cat(activation_parts, dim=0)
    residual_all = torch.cat(residual_parts, dim=0)
    parent_loss = float(residual_all.square().sum())
    diagnostics["v220_loss_parent"] = parent_loss
    if not math.isfinite(parent_loss):
        return None

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
        rows, blocks, _V220_GROUP_SIZE
    )
    codes = (
        parent_params["sign"].to(device=target_weight.device, dtype=torch.float32)
        * parent_params["mant"].to(device=target_weight.device, dtype=torch.float32)
        / _V220_CODE_MAGNITUDE
    ).reshape(rows, blocks, _V220_GROUP_SIZE)
    codes = torch.round(codes).clamp(-7.0, 7.0)

    selected = 0
    accepted = 0
    changed_codes = 0
    for block_index in range(blocks):
        lo = block_index * _V220_GROUP_SIZE
        hi = lo + _V220_GROUP_SIZE
        x_block = activation_all[:, lo:hi]
        gradient = x_block.transpose(0, 1).mm(residual_all)
        energy = x_block.square().sum(dim=0)
        candidate_score = gradient.abs().transpose(0, 1) * denominator[:, block_index, :]
        zero_mask = codes[:, block_index, :] == 0.0
        candidate_score = torch.where(
            zero_mask,
            candidate_score,
            torch.full_like(candidate_score, -float("inf")),
        )
        flat_index = int(torch.argmax(candidate_score).item())
        row_index = flat_index // _V220_GROUP_SIZE
        local_index = flat_index % _V220_GROUP_SIZE
        if not bool(zero_mask[row_index, local_index]):
            continue
        selected += 1

        grad = gradient[local_index, row_index]
        step = denominator[row_index, block_index, local_index] * _V220_CODE_MAGNITUDE
        curvature = energy[local_index]
        plus_change = -2.0 * grad * step + curvature * step.square()
        minus_change = 2.0 * grad * step + curvature * step.square()
        use_plus = plus_change <= minus_change
        best_change = torch.minimum(plus_change, minus_change)
        best_sign = 1.0 if bool(use_plus) else -1.0
        if not bool(torch.isfinite(best_change) and best_change < -_EPS):
            continue

        delta = best_sign * step
        codes[row_index, block_index, local_index] = best_sign
        residual_all[:, row_index] -= x_block[:, local_index] * delta
        accepted += 1
        changed_codes += 1

    diagnostics["v220_selected"] = selected
    diagnostics["v220_accepted_codes"] = accepted
    diagnostics["v220_changed_codes"] = changed_codes
    if accepted == 0:
        return None

    candidate_params = {
        key: value.detach().to(device=target_weight.device).clone()
        for key, value in parent_params.items()
    }
    flat_codes = codes.reshape(rows, channels)
    candidate_params["sign"] = torch.sign(flat_codes).reshape_as(parent_params["sign"])
    candidate_params["mant"] = (
        flat_codes.abs() * _V220_CODE_MAGNITUDE
    ).reshape_as(parent_params["mant"])
    candidate_params["sign"] = torch.where(
        candidate_params["mant"] == 0.0,
        torch.zeros_like(candidate_params["sign"]),
        candidate_params["sign"],
    )
    candidate_loss = float(residual_all.square().sum())
    diagnostics["v220_loss_candidate"] = candidate_loss
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    diagnostics["v220_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v220_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v220_attempted": 0,
        "v220_accepted": 0,
        "v220_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v220_zero_signed_code_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v220_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v220_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V220_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v220: v202 parent plus one fixed zero-code signed insertion per block."""

    result = _V220_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v220_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
