# ---------------------------------------------------------------------------
# L-JRB1: one round of joint activation/weight rounding-boundary coordinate
# descent for the deployed Linear quantizer pair.
#
# L-RB1 changed only the static weight mantissa decision and left every
# activation error untouched; A-RB1 did the same for Q/K.  This card adds the
# second deployed quantizer as a free variable and optimizes the product:
#
#     c = m         if frac(u) <  tau[m],
#     c = m + 1     if frac(u) >  tau[m],
#     c = round(u)  if frac(u) == tau[m],     m = floor(u),
#
# with a sign-shared activation table tau_x[1..6] (threaded into the live
# activation encoder) and L-RB1's sign-split weight table tau_w[s, 1..6]
# applied as a post-process on the returned static five fields.  The parent
# boundary tau = 0.5 restores both sides bit for bit.
#
# Fixed order, exactly one round:
#   1. solve the six activation boundaries once from the frozen parent A@W
#      residual.  Linear output is linear in the activation, so the exact
#      gradient/curvature pair is g_x = E W_hat and h_x[j] = sum_i W_hat[i,j]^2
#      (no Hutchinson probe, no backward pass);
#   2. merge the table, re-quantize every calibration window once, and accept
#      the whole table only when the case-equal real ||Q(A)Q(W)^T - A W^T||^2
#      strictly decreases;
#   3. on the resulting fixed X_hat/H solve the twelve weight boundaries with
#      L-RB1's exact quadratic and accept only a strictly negative DeltaL.
#
# There is no second round, no alternation, no per-row/per-block table and no
# model/role routing.
# ---------------------------------------------------------------------------

_JRB1_BUCKETS = 64
_JRB1_CODE_STEP = 0.25
_JRB1_CODE_MAX = 7
_JRB1_SHARED_MIN = 1
_JRB1_SHARED_MAX = 6
_JRB1_PARENT_BOUNDARY = 0.5
_JRB1_ACT_TABLE_LENGTH = _JRB1_SHARED_MAX + 1
_JRB1_W_TABLE_LENGTH = _JRB1_CODE_MAX + 1


def _jrb1_boundary_mantissa(
    x_abs: torch.Tensor,
    scale: torch.Tensor,
    boundaries: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a learned length-7 boundary table to ``u = 4|x|/scale``.

    ``boundaries is None`` (or a table shorter than seven entries) keeps the
    parent nearest rounding, and an explicit all-0.5 table is bit-identical to
    it because exact halves fall back to ``torch.round`` (half-to-even).
    """

    u = x_abs * (4.0 / scale)
    if boundaries is None:
        return torch.round(u).clamp_(0.0, 7.0) * _JRB1_CODE_STEP
    tau = boundaries.to(device=x_abs.device, dtype=torch.float32).reshape(-1)
    if int(tau.numel()) < _JRB1_ACT_TABLE_LENGTH:
        return torch.round(u).clamp_(0.0, 7.0) * _JRB1_CODE_STEP
    floor_code = torch.floor(u).clamp_(0.0, float(_JRB1_CODE_MAX - 1))
    frac = u - floor_code
    tau_view = tau.index_select(
        0, floor_code.to(torch.int64).reshape(-1)
    ).reshape(floor_code.shape)
    code = torch.where(
        frac < tau_view,
        floor_code,
        torch.where(frac > tau_view, floor_code + 1.0, torch.round(u)),
    )
    return code.clamp_(0.0, 7.0) * _JRB1_CODE_STEP


@torch.no_grad()
def _jrb1_reconstruct_deployment_weight(
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
        permutation = permutation.to(
            device=weight.device, dtype=torch.int64
        ).reshape(-1)
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
            u = rank1_u.to(
                device=weight.device, dtype=torch.float32
            ).reshape(-1, 1)
            v = rank1_v.to(
                device=weight.device, dtype=torch.float32
            ).reshape(-1, 1)
            if int(u.shape[0]) != channels or int(v.shape[0]) != channels:
                return None
            transformed = transformed - (transformed @ v) @ u.transpose(0, 1)
    if tuple(transformed.shape) != (rows, channels):
        return None
    return torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def _jrb1_deployment_codes(
    target: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return (floor, code, sign, frac, eligible, denominator) for a pair.

    ``target`` is the pre-quantization operand in the parent deployment frame;
    eligibility keeps only elements whose stored parent code equals the plain
    nearest rounding of that operand, so tau = 0.5 restores the parent codes.
    """

    rows, channels = map(int, target.shape)
    blocks = channels // _HIF4_BLOCK_SIZE
    scale = params["scale_factor"].to(
        device=target.device, dtype=torch.float32
    ).reshape(rows, blocks, 1, 1, 1)
    lv2 = params["scale_lv2"].to(
        device=target.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 1, 1)
    lv3 = params["scale_lv3"].to(
        device=target.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 1)
    denominator = (scale * lv2 * lv3).repeat_interleave(4, dim=-1).reshape(
        rows, channels
    )
    sign = params["sign"].to(
        device=target.device, dtype=torch.float32
    ).reshape(rows, channels)
    code = torch.round(
        params["mant"].to(device=target.device, dtype=torch.float32).reshape(
            rows, channels
        )
        / _JRB1_CODE_STEP
    ).clamp_(0.0, float(_JRB1_CODE_MAX))

    magnitude = 4.0 * target.abs() / denominator.clamp_min(_EPS)
    magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    nearest = torch.round(magnitude).clamp(0.0, float(_JRB1_CODE_MAX))
    floor_code = torch.floor(magnitude)
    frac = magnitude - floor_code
    eligible = (
        (code == nearest)
        & (floor_code >= float(_JRB1_SHARED_MIN))
        & (floor_code <= float(_JRB1_SHARED_MAX))
    )
    return floor_code, code, sign, frac, eligible, denominator


@torch.no_grad()
def _jrb1_pick_bucket(total_cost: torch.Tensor) -> int:
    """Argmin over the 65 boundary positions, ties toward 0.5 then lower."""

    order = torch.argsort(total_cost, stable=True)
    best_cost = total_cost[order[0]]
    tied = order[total_cost[order] == best_cost]
    distance = (tied - _JRB1_BUCKETS // 2).abs()
    return int(tied[int(torch.argmin(distance))])


@torch.no_grad()
def _jrb1_accumulate_bins(
    frac: torch.Tensor,
    cost_up: torch.Tensor,
    cost_down: torch.Tensor,
    up_mask: torch.Tensor,
    down_mask: torch.Tensor,
    up_bins: torch.Tensor,
    down_bins: torch.Tensor,
    scale: float,
) -> None:
    """Scatter per-element switch costs into the 65 fixed boundary buckets."""

    if bool(up_mask.any()):
        # An up-switching element crosses every tau < frac, i.e. buckets
        # k <= ceil(64*frac) - 1; exact zeros never switch.
        index = (
            torch.ceil(frac[up_mask] * _JRB1_BUCKETS) - 1.0
        ).clamp_(0.0, float(_JRB1_BUCKETS - 1)).to(torch.int64)
        up_bins.scatter_add_(0, index, cost_up[up_mask] * scale)
    if bool(down_mask.any()):
        # A down-switching element crosses every tau > frac, i.e. buckets
        # k >= floor(64*frac) + 1.
        index = (
            torch.floor(frac[down_mask] * _JRB1_BUCKETS) + 1.0
        ).clamp_(1.0, float(_JRB1_BUCKETS)).to(torch.int64)
        down_bins.scatter_add_(0, index, cost_down[down_mask] * scale)


@torch.no_grad()
def _jrb1_bucket_table(
    up_bins: torch.Tensor,
    down_bins: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    """Turn the per-class bucket costs into the learned sign-shared table.

    Also reports the best (most negative) total switch cost any class reached,
    so a zero-proposal outcome can be attributed to "no beneficial move"
    rather than to an empty eligible set.
    """

    table = torch.full(
        (_JRB1_ACT_TABLE_LENGTH,),
        _JRB1_PARENT_BOUNDARY,
        device=up_bins.device,
        dtype=torch.float32,
    )
    proposals = 0
    best_cost = 0.0
    for code in range(_JRB1_SHARED_MIN, _JRB1_SHARED_MAX + 1):
        suffix = up_bins[code].flip(0).cumsum(0).flip(0)
        prefix = down_bins[code].cumsum(0)
        total = suffix + prefix
        bucket = _jrb1_pick_bucket(total)
        table[code] = float(bucket) / float(_JRB1_BUCKETS)
        best_cost = min(best_cost, float(total.min()))
        if bucket != _JRB1_BUCKETS // 2:
            proposals += 1
    return table, proposals, best_cost


@torch.no_grad()
def _jrb1_quadratic_delta(
    gradient: torch.Tensor,
    delta_weight: torch.Tensor,
    cases: list,
    case_count: float,
    rows: int,
) -> float:
    """Exact A@W loss change of ``DeltaW`` on the frozen activation operand."""

    linear_term = 2.0 * float((gradient * delta_weight).sum())
    quadratic_term = 0.0
    for activation, tokens in cases:
        weight = 1.0 / (case_count * float(tokens * rows))
        projected = activation.mm(delta_weight.transpose(0, 1))
        quadratic_term += weight * float(projected.square().sum())
    return linear_term + quadratic_term


@torch.no_grad()
def _jrb1_weight_table(
    rows: int,
    channels: int,
    floor_code: torch.Tensor,
    code: torch.Tensor,
    sign: torch.Tensor,
    frac: torch.Tensor,
    eligible: torch.Tensor,
    quarter: torch.Tensor,
    gradient: torch.Tensor,
    h_diag: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """L-RB1's sign-split exact quadratic fit on one frozen activation operand.

    Returns (boundaries, proposals, new codes, changed mask).
    """

    device = gradient.device
    boundaries = torch.full(
        (2, _JRB1_W_TABLE_LENGTH),
        _JRB1_PARENT_BOUNDARY,
        device=device,
        dtype=torch.float32,
    )
    h_view = h_diag.unsqueeze(0).expand(rows, channels)
    proposals = 0
    for sign_index, sign_value in enumerate((-1.0, 1.0)):
        for boundary in range(_JRB1_SHARED_MIN, _JRB1_SHARED_MAX + 1):
            mask = (
                eligible
                & (sign == sign_value)
                & (floor_code == float(boundary))
            )
            if not bool(mask.any()):
                continue
            up_mask = mask & (code == float(boundary)) & (frac > 0.0)
            down_mask = mask & (code == float(boundary) + 1.0)
            delta_up = sign_value * quarter
            cost_up = 2.0 * gradient * delta_up + h_view * delta_up.square()
            delta_down = -sign_value * quarter
            cost_down = (
                2.0 * gradient * delta_down + h_view * delta_down.square()
            )
            up_bins = torch.zeros(
                _JRB1_BUCKETS + 1, device=device, dtype=torch.float32
            )
            down_bins = torch.zeros_like(up_bins)
            _jrb1_accumulate_bins(
                frac,
                cost_up,
                cost_down,
                up_mask,
                down_mask,
                up_bins,
                down_bins,
                1.0,
            )
            suffix = up_bins.flip(0).cumsum(0).flip(0)
            prefix = down_bins.cumsum(0)
            bucket = _jrb1_pick_bucket(suffix + prefix)
            boundaries[sign_index, boundary] = (
                float(bucket) / float(_JRB1_BUCKETS)
            )
            if bucket != _JRB1_BUCKETS // 2:
                proposals += 1

    new_code = code.clone()
    for sign_index, sign_value in enumerate((-1.0, 1.0)):
        for boundary in range(_JRB1_SHARED_MIN, _JRB1_SHARED_MAX + 1):
            mask = (
                eligible
                & (sign == sign_value)
                & (floor_code == float(boundary))
            )
            if not bool(mask.any()):
                continue
            tau = float(boundaries[sign_index, boundary])
            switch_up = mask & (code == float(boundary)) & (frac > tau)
            switch_down = mask & (code == float(boundary) + 1.0) & (frac < tau)
            if bool(switch_up.any()):
                new_code[switch_up] = float(boundary) + 1.0
            if bool(switch_down.any()):
                new_code[switch_down] = float(boundary)
    changed = (new_code != code) & eligible
    return boundaries, proposals, new_code, changed


@torch.no_grad()
def _jrb1_activation_pass(
    calib_activation_list: list,
    state: dict[str, Any],
    parent_weight: torch.Tensor,
    target_weight: torch.Tensor,
    rows: int,
    channels: int,
    h_x: torch.Tensor,
    reference_codes: Optional[list] = None,
) -> Optional[dict[str, Any]]:
    """One full sweep over the calibration windows in the deployment frame.

    Recomputes the activation bucket costs, the frozen weight-side A@W
    statistics and the case-equal real output MSE for ``state``.  When
    ``reference_codes`` is given, the real encoder mantissa deltas against that
    reference are counted as well.
    """

    device = parent_weight.device
    up_bins = torch.zeros(
        _JRB1_ACT_TABLE_LENGTH,
        _JRB1_BUCKETS + 1,
        device=device,
        dtype=torch.float32,
    )
    down_bins = torch.zeros_like(up_bins)
    gradient = torch.zeros(rows, channels, device=device, dtype=torch.float32)
    h_diag = torch.zeros(channels, device=device, dtype=torch.float32)
    mse_sum = 0.0
    changed_mantissa = 0
    cases: list = []
    codes: list = []
    windows = 0
    # Reachability accounting: how many deployment elements could move at all,
    # and whether any single move is actually beneficial.  Without this a
    # zero-proposal outcome cannot be told apart from an empty eligible set.
    elements = 0
    eligible_elements = 0
    switch_up_elements = 0
    switch_down_elements = 0
    negative_up = 0
    negative_down = 0
    min_switch_cost = float("inf")

    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        act_quant, act_scale = pair
        a_fp = _dequantize_nvfp4_float32(act_quant, act_scale).to(
            device=device, dtype=torch.float32
        )
        if a_fp.ndim != 2 or int(a_fp.shape[1]) != channels:
            return None
        tokens = int(a_fp.shape[0])
        if tokens == 0:
            continue
        a_params = hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        a_hat = _dequantize_hif4(a_params).to(device=device, dtype=torch.float32)
        z_ref = _static_actorder_dense_from_state(
            act_quant, act_scale, state
        ).to(device=device, dtype=torch.float32)
        if tuple(a_hat.shape) != tuple(a_fp.shape) or tuple(z_ref.shape) != tuple(
            a_fp.shape
        ):
            return None
        teacher = z_ref.mm(target_weight.transpose(0, 1))
        residual = a_hat.mm(parent_weight.transpose(0, 1)) - teacher
        if not bool(torch.isfinite(a_hat).all() and torch.isfinite(residual).all()):
            return None

        case_weight = 1.0 / float(tokens * rows)
        mse_sum += case_weight * float(residual.square().sum())
        gradient += case_weight * residual.transpose(0, 1).mm(a_hat)
        h_diag += case_weight * a_hat.square().sum(dim=0)
        cases.append((a_hat, tokens))

        code = torch.round(
            a_params["mant"].to(device=device, dtype=torch.float32)
            / _JRB1_CODE_STEP
        )
        if reference_codes is not None:
            reference = reference_codes[windows]
            if tuple(reference.shape) != tuple(code.shape):
                return None
            changed_mantissa += int(
                (code != reference.to(device=device, dtype=code.dtype)).sum()
            )
        codes.append(code.to(device="cpu", dtype=torch.uint8))

        a_floor, a_code, a_sign, a_frac, a_eligible, a_denominator = (
            _jrb1_deployment_codes(z_ref, a_params)
        )
        quarter = a_denominator * _JRB1_CODE_STEP
        g_x = residual.mm(parent_weight)
        delta_up = a_sign * quarter
        cost_up = 2.0 * g_x * delta_up + h_x * delta_up.square()
        delta_down = -a_sign * quarter
        cost_down = 2.0 * g_x * delta_down + h_x * delta_down.square()
        up_all = a_eligible & (a_code == a_floor) & (a_frac > 0.0)
        down_all = a_eligible & (a_code == a_floor + 1.0)
        elements += int(a_eligible.numel())
        eligible_elements += int(a_eligible.sum())
        switch_up_elements += int(up_all.sum())
        switch_down_elements += int(down_all.sum())
        if bool(up_all.any()):
            negative_up += int((cost_up[up_all] < 0.0).sum())
            min_switch_cost = min(min_switch_cost, float(cost_up[up_all].min()))
        if bool(down_all.any()):
            negative_down += int((cost_down[down_all] < 0.0).sum())
            min_switch_cost = min(
                min_switch_cost, float(cost_down[down_all].min())
            )
        for boundary in range(_JRB1_SHARED_MIN, _JRB1_SHARED_MAX + 1):
            up_mask = up_all & (a_floor == float(boundary))
            down_mask = down_all & (a_floor == float(boundary))
            if not bool(up_mask.any() or down_mask.any()):
                continue
            _jrb1_accumulate_bins(
                a_frac,
                cost_up,
                cost_down,
                up_mask,
                down_mask,
                up_bins[boundary],
                down_bins[boundary],
                case_weight,
            )
        windows += 1

    if windows == 0:
        return None
    gradient /= float(windows)
    h_diag /= float(windows)
    up_bins /= float(windows)
    down_bins /= float(windows)
    if min_switch_cost == float("inf"):
        min_switch_cost = 0.0
    return {
        "up_bins": up_bins,
        "down_bins": down_bins,
        "gradient": gradient,
        "h_diag": h_diag,
        "mse": mse_sum / float(windows),
        "cases": cases,
        "codes": codes,
        "changed_mantissa": changed_mantissa,
        "windows": windows,
        "elements": elements,
        "eligible_elements": eligible_elements,
        "switch_up_elements": switch_up_elements,
        "switch_down_elements": switch_down_elements,
        "negative_up": negative_up,
        "negative_down": negative_down,
        "min_switch_cost": min_switch_cost,
    }


@torch.no_grad()
def _jrb1_joint_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """One round of joint A/W boundary coordinate descent for one Linear call."""

    target_weight = _jrb1_reconstruct_deployment_weight(
        weight_quant, weight_scale, state
    )
    if target_weight is None:
        return None
    device = target_weight.device
    parent_weight = _dequantize_hif4(parent_params).to(
        device=device, dtype=torch.float32
    )
    if parent_weight.ndim != 2 or tuple(parent_weight.shape) != tuple(
        target_weight.shape
    ):
        return None
    rows, channels = map(int, parent_weight.shape)
    if channels % _HIF4_BLOCK_SIZE != 0:
        return None

    (
        weight_floor,
        weight_code,
        weight_sign,
        weight_frac,
        weight_eligible,
        weight_denominator,
    ) = _jrb1_deployment_codes(target_weight, parent_params)
    weight_quarter = weight_denominator * _JRB1_CODE_STEP
    h_x = parent_weight.square().sum(dim=0)

    diagnostics.update(
        {
            "jrb1_attempted": 1,
            "jrb1_coordinate": "joint-activation-weight-rounding-boundary",
            "jrb1_rows": rows,
            "jrb1_channels": channels,
            "jrb1_boundary_count": _JRB1_SHARED_MAX + 2 * _JRB1_SHARED_MAX,
            "jrb1_weight_eligible": int(weight_eligible.sum()),
            "jrb1_weight_ineligible": int((~weight_eligible).sum()),
            "jrb1_act_changed_mantissa": 0,
            "jrb1_w_changed_mantissa": 0,
        }
    )
    if state.get("gram") is not None:
        # in_features <= _ACTIVATION_QUADRATIC_MAX_FEATURES: the deployed
        # activation mantissa is solved by ``_adaround_mantissa``, so the
        # rounding-boundary table is never consulted.  The joint coordinate
        # does not exist on this layer, so return the parent rather than
        # degrade into a weight-only edit (that coordinate is L-RB1/v226).
        diagnostics["jrb1_act_path"] = "adaround-gram-unreachable"
        diagnostics["jrb1_arm"] = "activation-unreachable"
        return None
    diagnostics["jrb1_act_path"] = "rounding-boundary"
    if int(weight_eligible.sum()) == 0:
        diagnostics["jrb1_arm"] = "no-weight-eligible"
        return None

    parent_pass = _jrb1_activation_pass(
        calib_activation_list,
        state,
        parent_weight,
        target_weight,
        rows,
        channels,
        h_x,
    )
    if parent_pass is None:
        return None
    diagnostics["jrb1_fit_windows"] = int(parent_pass["windows"])
    diagnostics["jrb1_mse_parent"] = float(parent_pass["mse"])
    diagnostics["jrb1_act_elements"] = int(parent_pass["elements"])
    diagnostics["jrb1_act_eligible"] = int(parent_pass["eligible_elements"])
    diagnostics["jrb1_act_up_switchable"] = int(parent_pass["switch_up_elements"])
    diagnostics["jrb1_act_down_switchable"] = int(
        parent_pass["switch_down_elements"]
    )
    diagnostics["jrb1_act_negative_up"] = int(parent_pass["negative_up"])
    diagnostics["jrb1_act_negative_down"] = int(parent_pass["negative_down"])
    diagnostics["jrb1_act_min_switch_cost"] = float(
        parent_pass["min_switch_cost"]
    )

    activation_table, activation_proposals, activation_best_cost = (
        _jrb1_bucket_table(parent_pass["up_bins"], parent_pass["down_bins"])
    )
    diagnostics["jrb1_act_boundaries"] = activation_table.to(
        device="cpu", dtype=torch.float32
    )
    diagnostics["jrb1_act_proposals"] = int(activation_proposals)
    diagnostics["jrb1_act_best_total_cost"] = float(activation_best_cost)

    # Weight-only attribution reading: L-RB1's table on the frozen parent X_hat.
    weight_table, weight_proposals, new_code_parent, changed_parent = (
        _jrb1_weight_table(
            rows,
            channels,
            weight_floor,
            weight_code,
            weight_sign,
            weight_frac,
            weight_eligible,
            weight_quarter,
            parent_pass["gradient"],
            parent_pass["h_diag"],
        )
    )
    diagnostics["jrb1_w_boundaries"] = weight_table.to(
        device="cpu", dtype=torch.float32
    )
    diagnostics["jrb1_w_proposals"] = int(weight_proposals)
    delta_weight_parent = torch.zeros(
        rows, channels, device=device, dtype=torch.float32
    )
    delta_weight_parent[changed_parent] = (
        (new_code_parent[changed_parent] - weight_code[changed_parent])
        * weight_sign[changed_parent]
        * weight_quarter[changed_parent]
    )
    delta_loss_parent = _jrb1_quadratic_delta(
        parent_pass["gradient"],
        delta_weight_parent,
        parent_pass["cases"],
        float(parent_pass["windows"]),
        rows,
    )
    diagnostics["jrb1_w_only_changed_mantissa"] = int(changed_parent.sum())
    diagnostics["jrb1_w_only_delta_loss"] = float(delta_loss_parent)
    diagnostics["jrb1_w_only_mse"] = float(
        parent_pass["mse"] + delta_loss_parent
    )

    if activation_proposals == 0:
        diagnostics["jrb1_act_accepted"] = 0
        diagnostics["jrb1_arm"] = "no-activation-proposal"
        return None

    candidate_state = dict(state)
    candidate_state["activation_boundaries"] = activation_table.to(
        device="cpu", dtype=torch.float32
    )
    candidate_pass = _jrb1_activation_pass(
        calib_activation_list,
        candidate_state,
        parent_weight,
        target_weight,
        rows,
        channels,
        h_x,
        reference_codes=parent_pass["codes"],
    )
    if candidate_pass is None:
        return None
    diagnostics["jrb1_mse_act_only"] = float(candidate_pass["mse"])
    diagnostics["jrb1_act_changed_encoder"] = int(
        candidate_pass["changed_mantissa"]
    )
    activation_accepted = candidate_pass["mse"] < parent_pass["mse"] - _EPS
    diagnostics["jrb1_act_accepted"] = int(activation_accepted)
    if not activation_accepted:
        diagnostics["jrb1_arm"] = "activation-rejected"
        return None

    joint_table, joint_proposals, new_code_joint, changed_joint = (
        _jrb1_weight_table(
            rows,
            channels,
            weight_floor,
            weight_code,
            weight_sign,
            weight_frac,
            weight_eligible,
            weight_quarter,
            candidate_pass["gradient"],
            candidate_pass["h_diag"],
        )
    )
    diagnostics["jrb1_w_joint_boundaries"] = joint_table.to(
        device="cpu", dtype=torch.float32
    )
    diagnostics["jrb1_w_joint_proposals"] = int(joint_proposals)
    delta_weight_joint = torch.zeros(
        rows, channels, device=device, dtype=torch.float32
    )
    delta_weight_joint[changed_joint] = (
        (new_code_joint[changed_joint] - weight_code[changed_joint])
        * weight_sign[changed_joint]
        * weight_quarter[changed_joint]
    )
    delta_loss_joint = _jrb1_quadratic_delta(
        candidate_pass["gradient"],
        delta_weight_joint,
        candidate_pass["cases"],
        float(candidate_pass["windows"]),
        rows,
    )
    diagnostics["jrb1_w_changed_mantissa"] = int(changed_joint.sum())
    diagnostics["jrb1_w_delta_loss"] = float(delta_loss_joint)
    diagnostics["jrb1_joint_mse"] = float(
        candidate_pass["mse"] + delta_loss_joint
    )
    if int(changed_joint.sum()) == 0 or not delta_loss_joint < 0.0:
        diagnostics["jrb1_arm"] = "weight-rejected"
        return None

    diagnostics["jrb1_act_changed_mantissa"] = int(
        candidate_pass["changed_mantissa"]
    )
    if int(candidate_pass["changed_mantissa"]) == 0:
        diagnostics["jrb1_arm"] = "activation-absorbed"
        return None

    candidate_params = {
        key: value.detach().clone() for key, value in parent_params.items()
    }
    candidate_mant = (new_code_joint * _JRB1_CODE_STEP).reshape_as(
        parent_params["mant"]
    )
    candidate_sign = weight_sign.reshape_as(parent_params["sign"])
    candidate_params["sign"] = torch.where(
        candidate_mant == 0.0,
        torch.zeros_like(candidate_sign),
        candidate_sign,
    )
    candidate_params["mant"] = candidate_mant
    diagnostics["jrb1_arm"] = "joint"
    return {
        "weight_params": candidate_params,
        "activation_boundaries": candidate_state["activation_boundaries"],
    }


@torch.no_grad()
def _jrb1_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "jrb1_attempted": 0,
        "jrb1_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate = None
    try:
        candidate = _jrb1_joint_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as error:
        diagnostics["jrb1_error"] = f"{type(error).__name__}: {error}"
        candidate = None
    if candidate is not None:
        state["activation_boundaries"] = candidate["activation_boundaries"]
    state.update(diagnostics)
    if diagnostics.get("jrb1_attempted"):
        activation_table = diagnostics.get("jrb1_act_boundaries")
        weight_table = diagnostics.get("jrb1_w_boundaries")
        activation_text = (
            ",".join(f"{float(value):.3f}" for value in activation_table.tolist())
            if torch.is_tensor(activation_table)
            else "n/a"
        )
        weight_text = (
            ",".join(
                f"{float(value):.3f}" for value in weight_table.reshape(-1).tolist()
            )
            if torch.is_tensor(weight_table)
            else "n/a"
        )
        print(
            f"[L-JRB1] arm={diagnostics['jrb1_arm']} "
            f"rows={diagnostics.get('jrb1_rows')} "
            f"channels={diagnostics.get('jrb1_channels')} "
            f"act_path={diagnostics.get('jrb1_act_path')} "
            f"act_proposals={diagnostics.get('jrb1_act_proposals')} "
            f"act_elig={diagnostics.get('jrb1_act_eligible')}/"
            f"{diagnostics.get('jrb1_act_elements')} "
            f"act_switch={diagnostics.get('jrb1_act_up_switchable')}u/"
            f"{diagnostics.get('jrb1_act_down_switchable')}d "
            f"act_neg={diagnostics.get('jrb1_act_negative_up')}u/"
            f"{diagnostics.get('jrb1_act_negative_down')}d "
            f"act_min_cost={float(diagnostics.get('jrb1_act_min_switch_cost', 0.0)):.6e} "
            f"act_changed={diagnostics.get('jrb1_act_changed_mantissa')} "
            f"act_encoder={diagnostics.get('jrb1_act_changed_encoder')} "
            f"act_mse={float(diagnostics.get('jrb1_mse_parent', 0.0)):.6e}"
            f"->{float(diagnostics.get('jrb1_mse_act_only', 0.0)):.6e} "
            f"w_proposals={diagnostics.get('jrb1_w_proposals')} "
            f"w_changed={diagnostics.get('jrb1_w_changed_mantissa')} "
            f"w_delta_loss={float(diagnostics.get('jrb1_w_delta_loss', 0.0)):.6e} "
            f"act=[{activation_text}] w=[{weight_text}]",
            flush=True,
        )
    if candidate is None:
        return result
    return dict(result, weight_params=candidate["weight_params"])


_JRB1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """L-JRB1: v202 root plus one joint A/W boundary coordinate-descent round."""

    result = _JRB1_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _jrb1_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
