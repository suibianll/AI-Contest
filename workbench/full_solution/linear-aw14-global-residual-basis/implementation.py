# ---------------------------------------------------------------------------
# v221 / L-AW14: shared output-residual basis A@W fit.
#
# The previous L23 implementation solved a separate rank-8 residual-cross
# subspace and SVD for every 64-column block.  This card keeps a fixed rank-8
# structured output subspace, but obtains it once from the full calibration
# residual and reuses it for every block.  Each block still receives one
# sequential legal HiF4 projection and exact full-output residual acceptance.
# No gain/additive parameterization, unstructured per-code greedy insertion,
# or dynamic API change is introduced.
# ---------------------------------------------------------------------------

_V221_BLOCK_SIZE = _HIF4_BLOCK_SIZE
_V221_RANK = 8


@torch.no_grad()
def _v221_reconstruct_deployment_weight(
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
def _v221_global_residual_basis_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Fit sequential legal block updates in one shared rank-8 output basis."""

    target_weight = _v221_reconstruct_deployment_weight(
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
    if channels % _V221_BLOCK_SIZE != 0:
        return None
    blocks = channels // _V221_BLOCK_SIZE

    diagnostics.update(
        {
            "v221_attempted": 1,
            "v221_coordinate": "deployment-output-shared-rank8-residual-basis",
            "v221_block_size": _V221_BLOCK_SIZE,
            "v221_blocks": blocks,
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

    diagnostics["v221_fit_windows"] = len(activation_parts)
    diagnostics["v221_fit_rows"] = fit_rows
    if not activation_parts:
        return None

    activation_all = torch.cat(activation_parts, dim=0)
    residual_all = torch.cat(residual_parts, dim=0)
    parent_loss = float(residual_all.square().sum())
    diagnostics["v221_loss_parent"] = parent_loss
    if not math.isfinite(parent_loss):
        return None

    try:
        _u, _s, vh = torch.linalg.svd(residual_all, full_matrices=False)
    except RuntimeError:
        return None
    basis_rank = min(_V221_RANK, int(vh.shape[0]))
    if basis_rank <= 0:
        return None
    output_basis = vh[:basis_rank].transpose(0, 1).contiguous()
    if not bool(torch.isfinite(output_basis).all()):
        return None
    projected_residual = residual_all.mm(output_basis)
    diagnostics["v221_basis_rank"] = basis_rank

    scale = parent_params["scale_factor"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, 1, 1, 1)
    lv2 = parent_params["scale_lv2"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 1, 1)
    lv3 = parent_params["scale_lv3"].to(
        device=target_weight.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 1)
    scale_full = (scale * lv2 * lv3).expand(
        rows, blocks, 8, 2, 4
    ).reshape(rows, blocks, _V221_BLOCK_SIZE)
    current_codes = (
        parent_params["sign"].to(device=target_weight.device, dtype=torch.float32)
        * parent_params["mant"].to(device=target_weight.device, dtype=torch.float32)
        / 0.25
    ).reshape(rows, blocks, _V221_BLOCK_SIZE)
    current_codes = torch.round(current_codes).clamp(-7.0, 7.0)
    code_values = (
        torch.arange(-7, 8, dtype=torch.float32, device=target_weight.device) * 0.25
    )

    regularization = (
        _WEIGHT_GPTQ_REGULARIZATION_WIDE
        if (rows >= _WIDE_LAYER_MIN_DIM or channels >= _WIDE_LAYER_MIN_DIM)
        else _WEIGHT_GPTQ_REGULARIZATION
    )
    eye = torch.eye(_V221_BLOCK_SIZE, dtype=torch.float32, device=target_weight.device)
    work_weight = parent_weight.clone()
    current_loss = parent_loss
    accepted_blocks = 0
    changed_codes = 0

    # The output basis is fixed once; block order and residual maintenance are
    # sequential so later proposals see every earlier accepted code change.
    for block_index in range(blocks):
        lo = block_index * _V221_BLOCK_SIZE
        hi = lo + _V221_BLOCK_SIZE
        x_block = activation_all[:, lo:hi]
        gram = x_block.transpose(0, 1).mm(x_block)
        gram = 0.5 * (gram + gram.transpose(0, 1))
        diagonal = gram.diagonal().mean().clamp_min(_EPS)
        system = gram + float(regularization) * diagonal * eye
        cross_basis = x_block.transpose(0, 1).mm(projected_residual)
        if not bool(torch.isfinite(system).all() and torch.isfinite(cross_basis).all()):
            continue
        try:
            coeff = torch.linalg.solve(system, cross_basis)
        except RuntimeError:
            continue
        continuous_update = coeff.mm(output_basis.transpose(0, 1))
        if not bool(torch.isfinite(continuous_update).all()):
            continue

        parent_block = work_weight[:, lo:hi].transpose(0, 1).contiguous()
        block_scale = scale_full[:, block_index, :].transpose(0, 1).clamp_min(_EPS)
        grid = code_values.reshape(-1, 1, 1) * block_scale.reshape(
            1, _V221_BLOCK_SIZE, rows
        )
        desired = parent_block + continuous_update
        proposed_index = (desired.unsqueeze(0) - grid).abs().argmin(dim=0)
        proposed = grid.gather(0, proposed_index.unsqueeze(0))[0]
        delta = proposed - parent_block
        candidate_residual = residual_all - x_block.mm(delta)
        candidate_loss = float(candidate_residual.square().sum())
        if not math.isfinite(candidate_loss) or not candidate_loss < current_loss:
            continue

        work_weight[:, lo:hi] = proposed.transpose(0, 1)
        residual_all = candidate_residual
        projected_residual = projected_residual - x_block.mm(
            delta.mm(output_basis)
        )
        current_codes[:, block_index, :] = (proposed_index.to(torch.float32) - 7.0).transpose(0, 1)
        changed_codes += int((current_codes[:, block_index, :] != (
            parent_params["sign"].to(device=target_weight.device, dtype=torch.float32)
            * parent_params["mant"].to(device=target_weight.device, dtype=torch.float32)
            / 0.25
        ).reshape(rows, blocks, _V221_BLOCK_SIZE)[:, block_index, :]).sum())
        accepted_blocks += 1
        current_loss = candidate_loss

    diagnostics["v221_accepted_blocks"] = accepted_blocks
    diagnostics["v221_changed_codes"] = changed_codes
    diagnostics["v221_loss_candidate"] = current_loss
    if accepted_blocks == 0:
        return None

    candidate_params = {
        key: value.detach().to(device=target_weight.device).clone()
        for key, value in parent_params.items()
    }
    flat_codes = current_codes.reshape(rows, channels)
    candidate_params["sign"] = torch.sign(flat_codes).reshape_as(parent_params["sign"])
    candidate_params["mant"] = (flat_codes.abs() * 0.25).reshape_as(parent_params["mant"])
    candidate_params["sign"] = torch.where(
        candidate_params["mant"] == 0.0,
        torch.zeros_like(candidate_params["sign"]),
        candidate_params["sign"],
    )
    diagnostics["v221_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v221_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v221_attempted": 0,
        "v221_accepted": 0,
        "v221_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v221_global_residual_basis_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v221_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v221_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V221_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v221: v202 parent plus one shared rank-8 residual output basis."""

    result = _V221_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v221_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
