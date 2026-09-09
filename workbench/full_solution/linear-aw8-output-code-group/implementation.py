# ---------------------------------------------------------------------------
# v211 / L-AW8: output-aware joint four-code-group update.
#
# v204-v210 fitted continuous gains or additive values and then projected the
# result with an independent encoder.  This card changes the discrete update:
# for one selected natural four-value group per output row, solve the actual
# frozen-Q(A) 4x4 output normal equation, round that step directly in the
# existing signed-mantissa code lattice, and accept it only by the exact
# calibration product loss.  The hierarchy fields stay fixed, so every write
# remains a legal HiF4 carrier and no calibration product enters activation
# state.
# ---------------------------------------------------------------------------

_V211_GROUP_SIZE = 4
_V211_GROUPS_PER_ROW = 1
_V211_RIDGE_RATIO = 1.0e-2


@torch.no_grad()
def _v211_reconstruct_deployment_weight(
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
def _v211_output_code_group_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Perform one fixed direct code-lattice update per output row."""

    target_weight = _v211_reconstruct_deployment_weight(
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
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % _V211_GROUP_SIZE != 0:
        return None

    blocks = channels // _HIF4_BLOCK_SIZE
    groups = channels // _V211_GROUP_SIZE
    diagnostics.update(
        {
            "v211_attempted": 1,
            "v211_coordinate": "deployment-output-joint-4-code-group",
            "v211_group_size": _V211_GROUP_SIZE,
            "v211_groups_per_row": _V211_GROUPS_PER_ROW,
            "v211_groups": groups,
        }
    )

    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
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

        # Use the exact final dynamic path as the frozen Q(A) operand.  The
        # output-derived tensors below stay local to this calibration call.
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
        if not bool(torch.isfinite(residual).all()):
            return None
        records.append((a_hat, teacher, residual))
        parent_loss += float(residual.square().sum())
        fit_rows += int(a_fp.shape[0])

    diagnostics["v211_fit_windows"] = len(records)
    diagnostics["v211_fit_rows"] = fit_rows
    diagnostics["v211_loss_parent"] = float(parent_loss)
    if not records or not math.isfinite(parent_loss):
        return None

    # Select one group independently for every output row from the initial
    # output gradient.  The fixed budget avoids a free block/row search while
    # exposing a different discrete direction from diagonal additive fitting.
    scores = torch.zeros((rows, groups), dtype=torch.float32, device=target_weight.device)
    for group_index in range(groups):
        lo = group_index * _V211_GROUP_SIZE
        hi = lo + _V211_GROUP_SIZE
        gradient = None
        for a_hat, _teacher, residual in records:
            local = a_hat[:, lo:hi].transpose(0, 1).mm(residual)
            gradient = local if gradient is None else gradient + local
        if gradient is not None:
            scores[:, group_index] = torch.nan_to_num(
                gradient.square().sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0
            )
    selected_groups = scores.argmax(dim=1)

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

    eye = torch.eye(_V211_GROUP_SIZE, dtype=torch.float32, device=target_weight.device)
    accepted_pairs = 0
    changed_codes = 0
    # Group order is fixed and independent of score magnitude.  Each output
    # row occurs in one group only, so residual updates are exact Gauss-Seidel
    # updates for the selected code groups.
    for group_index in range(groups):
        row_ids = torch.nonzero(selected_groups == group_index, as_tuple=False).reshape(-1)
        if int(row_ids.numel()) == 0:
            continue
        lo = group_index * _V211_GROUP_SIZE
        hi = lo + _V211_GROUP_SIZE
        gram = None
        gradient_parts = []
        for a_hat, _teacher, residual in records:
            local_z = a_hat[:, lo:hi]
            local_gram = local_z.transpose(0, 1).mm(local_z)
            gram = local_gram if gram is None else gram + local_gram
            gradient_parts.append(local_z.transpose(0, 1).mm(residual[:, row_ids]))
        if gram is None:
            continue
        gradient = torch.stack(gradient_parts, dim=0).sum(dim=0)
        if not bool(torch.isfinite(gram).all() and torch.isfinite(gradient).all()):
            continue
        ridge = float(_V211_RIDGE_RATIO) * gram.diagonal().mean().clamp_min(_EPS)
        system = gram + ridge * eye
        try:
            desired = torch.linalg.solve(system, gradient)
        except RuntimeError:
            continue
        # The group never crosses a 64-value HiF4 block because both sizes are
        # fixed divisors; use explicit block/local coordinates for the carrier.
        block_index = lo // _HIF4_BLOCK_SIZE
        block_lo = lo % _HIF4_BLOCK_SIZE
        current = codes[row_ids, block_index, block_lo : block_lo + _V211_GROUP_SIZE]
        local_denominator = denominator[row_ids, block_index, block_lo : block_lo + _V211_GROUP_SIZE]
        target_codes = current + desired.transpose(0, 1) / (
            0.25 * local_denominator.clamp_min(_EPS)
        )
        proposed = torch.nan_to_num(target_codes, nan=0.0, posinf=7.0, neginf=-7.0).round().clamp(-7.0, 7.0)
        delta = (proposed - current) * local_denominator * 0.25
        change = -2.0 * (gradient.transpose(0, 1) * delta).sum(dim=1)
        change = change + torch.einsum("ni,ij,nj->n", delta, gram, delta)
        accept = torch.isfinite(change) & (change < -_EPS)
        if not bool(accept.any()):
            continue
        accepted_rows = row_ids[accept]
        accepted_delta = delta[accept]
        accepted_codes = proposed[accept]
        codes[accepted_rows, block_index, block_lo : block_lo + _V211_GROUP_SIZE] = accepted_codes
        for a_hat, _teacher, residual in records:
            residual[:, accepted_rows] -= a_hat[:, lo:hi].mm(accepted_delta.transpose(0, 1))
        accepted_pairs += int(accepted_rows.numel())
        changed_codes += int((accepted_codes != current[accept]).sum())

    diagnostics["v211_accepted_pairs"] = accepted_pairs
    diagnostics["v211_changed_codes"] = changed_codes
    if accepted_pairs == 0:
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
    candidate_loss = sum(float(residual.square().sum()) for _a, _t, residual in records)
    diagnostics["v211_loss_candidate"] = float(candidate_loss)
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    diagnostics["v211_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v211_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v211_attempted": 0,
        "v211_accepted": 0,
        "v211_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v211_output_code_group_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v211_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v211_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V211_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v211: v202 parent plus direct output-aware four-code-group update."""

    result = _V211_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v211_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
