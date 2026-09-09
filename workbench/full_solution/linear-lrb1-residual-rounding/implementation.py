# ---------------------------------------------------------------------------
# L-RB1 / v226: output-residual shared rounding boundaries for the static
# weight quantizer.
#
# The retained root fixes the deployed weight coordinate and then encodes every
# element with a plain nearest-round mantissa decision
#
#     c = clamp(round(4 * |w| / D), 0, 7),   D = scale_factor * lv2 * lv3.
#
# This card replaces the fixed 0.5 threshold by one learned boundary per
# (sign, lower-code) class -- 12 shared scalars per calibration call -- and
# leaves the five HiF4 fields' semantics, the hierarchy candidates and every
# dynamic API untouched:
#
#     c = m         if frac(u) < tau[s, m],
#     c = m + 1     otherwise,        m = floor(u), s = sign(w).
#
# Boundaries are solved once from the frozen parent A@W output residual in 64
# fixed fractional buckets, merged into a single table, and accepted only
# when the exact parent-coordinate quadratic output loss strictly decreases.
#
# Only elements whose stored parent code equals the plain nearest rounding of
# the deployed-coordinate magnitude enter the fit, so the parent boundary
# tau = 0.5 restores the parent five fields bit for bit.
# ---------------------------------------------------------------------------

_LRB1_BUCKETS = 64
_LRB1_CODE_STEP = 0.25
_LRB1_CODE_MAX = 7
_LRB1_SHARED_MIN = 1
_LRB1_SHARED_MAX = 6
_LRB1_PARENT_BOUNDARY = 0.5


@torch.no_grad()
def _lrb1_reconstruct_deployment_weight(
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
def _lrb1_apply_boundaries(
    floor_code: torch.Tensor,
    code_abs: torch.Tensor,
    sign_field: torch.Tensor,
    frac: torch.Tensor,
    eligible: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a [2, 8] boundary table; return (new floor codes, switched mask).

    Class ``m`` is the lower code of the interval ``[m, m+1)``.  The parent
    boundary tau = 0.5 leaves every eligible element at its stored code,
    because an element with ``frac >= 0.5`` already carries parent code
    ``m + 1``.
    """

    new_code = code_abs.clone()
    switched = torch.zeros_like(eligible)
    for sign_index, sign_value in enumerate((-1.0, 1.0)):
        for code in range(_LRB1_SHARED_MIN, _LRB1_SHARED_MAX + 1):
            mask = (
                eligible
                & (sign_field == sign_value)
                & (floor_code == float(code))
            )
            if not bool(mask.any()):
                continue
            tau = float(boundaries[sign_index, code])
            switch = mask & (frac >= tau)
            if bool(switch.any()):
                new_code[switch] = floor_code[switch] + 1.0
                switched |= switch
    return new_code, switched


@torch.no_grad()
def _lrb1_pick_bucket(total_cost: torch.Tensor) -> int:
    """Argmin over 65 boundary positions, ties toward 0.5 then lower tau."""

    order = torch.argsort(total_cost, stable=True)
    best_cost = total_cost[order[0]]
    tied = order[total_cost[order] == best_cost]
    distance = (tied - _LRB1_BUCKETS // 2).abs()
    return int(tied[int(torch.argmin(distance))])


@torch.no_grad()
def _lrb1_boundary_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Learn 12 shared weight rounding boundaries from the parent A@W residual."""

    target_weight = _lrb1_reconstruct_deployment_weight(
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
    blocks = channels // _HIF4_BLOCK_SIZE

    diagnostics.update(
        {
            "lrb1_attempted": 1,
            "lrb1_coordinate": "deployment-output-shared-weight-rounding-boundary",
            "lrb1_rows": rows,
            "lrb1_channels": channels,
            "lrb1_boundary_count": 2 * _LRB1_SHARED_MAX,
        }
    )

    scale = parent_params["scale_factor"].to(
        device=device, dtype=torch.float32
    ).reshape(rows, blocks, 1, 1, 1)
    lv2 = parent_params["scale_lv2"].to(
        device=device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 1, 1)
    lv3 = parent_params["scale_lv3"].to(
        device=device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 1)
    denominator = (scale * lv2 * lv3).repeat_interleave(4, dim=-1).reshape(
        rows, channels
    )
    sign_field = parent_params["sign"].to(
        device=device, dtype=torch.float32
    ).reshape(rows, channels)
    code_abs = torch.round(
        parent_params["mant"].to(device=device, dtype=torch.float32).reshape(
            rows, channels
        )
        / _LRB1_CODE_STEP
    ).clamp_(0.0, float(_LRB1_CODE_MAX))

    magnitude = 4.0 * target_weight.abs() / denominator.clamp_min(_EPS)
    magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    nearest = torch.round(magnitude).clamp(0.0, float(_LRB1_CODE_MAX))
    floor_code = torch.floor(magnitude)
    frac = (magnitude - floor_code).clamp(0.0, 1.0)
    eligible = (
        (code_abs == nearest)
        & (floor_code >= float(_LRB1_SHARED_MIN))
        & (floor_code <= float(_LRB1_SHARED_MAX))
    )

    ineligible = ~eligible
    diagnostics["lrb1_eligible"] = int(eligible.sum())
    diagnostics["lrb1_ineligible"] = int(ineligible.sum())
    diagnostics["lrb1_ineligible_by_parent_code"] = [
        int(((code_abs == float(code)) & ineligible).sum())
        for code in range(_LRB1_CODE_MAX + 1)
    ]
    diagnostics["lrb1_sign_field_mismatch"] = int(
        ((sign_field * target_weight < 0.0) & eligible).sum()
    )
    if not bool(eligible.any()):
        return None

    h_diag = torch.zeros(channels, device=device, dtype=torch.float32)
    gradient = torch.zeros(rows, channels, device=device, dtype=torch.float32)
    cases: list[tuple[torch.Tensor, int]] = []
    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        act_quant, act_scale = pair
        a_fp = _dequantize_nvfp4_float32(act_quant, act_scale).to(
            device=device, dtype=torch.float32
        )
        if a_fp.ndim != 2 or int(a_fp.shape[1]) != channels:
            return None
        if int(a_fp.shape[0]) == 0:
            continue

        # Freeze the exact deployed activation operand and the parent's
        # calibration-side dense reference.  Output tensors stay local; only
        # scalar provenance is added to activation_state.
        a_params = hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        a_hat = _dequantize_hif4(a_params).to(device=device, dtype=torch.float32)
        z_ref = _static_actorder_dense_from_state(act_quant, act_scale, state).to(
            device=device, dtype=torch.float32
        )
        if tuple(a_hat.shape) != tuple(a_fp.shape) or tuple(z_ref.shape) != tuple(
            a_fp.shape
        ):
            return None
        teacher = z_ref.mm(target_weight.transpose(0, 1))
        residual = a_hat.mm(parent_weight.transpose(0, 1)) - teacher
        if not bool(torch.isfinite(a_hat).all() and torch.isfinite(residual).all()):
            return None
        case_weight = 1.0 / float(int(a_fp.shape[0]) * rows)
        h_diag += case_weight * a_hat.square().sum(dim=0)
        gradient += case_weight * residual.transpose(0, 1).mm(a_hat)
        cases.append((a_hat, int(a_fp.shape[0])))

    diagnostics["lrb1_fit_windows"] = len(cases)
    if not cases:
        return None
    case_count = float(len(cases))
    gradient /= case_count
    h_diag /= case_count

    boundaries = torch.full(
        (2, _LRB1_CODE_MAX + 1),
        _LRB1_PARENT_BOUNDARY,
        device=device,
        dtype=torch.float32,
    )
    h_view = h_diag.unsqueeze(0).expand(rows, channels)
    quarter = denominator * _LRB1_CODE_STEP
    proposed = 0
    for sign_index, sign_value in enumerate((-1.0, 1.0)):
        for code in range(_LRB1_SHARED_MIN, _LRB1_SHARED_MAX + 1):
            mask = (
                eligible
                & (sign_field == sign_value)
                & (floor_code == float(code))
            )
            if not bool(mask.any()):
                continue
            # Elements that already rounded up to m + 1 have a zero step and
            # therefore a zero switch cost, so the parent bucket stays free.
            step = float(code) + 1.0 - code_abs[mask]
            delta = sign_value * step * quarter[mask]
            cost = 2.0 * gradient[mask] * delta + h_view[mask] * delta.square()
            bucket_cost = torch.zeros(
                _LRB1_BUCKETS, device=device, dtype=torch.float32
            )
            bucket_cost.scatter_add_(
                0,
                torch.clamp(
                    torch.floor(frac[mask] * _LRB1_BUCKETS).to(torch.int64),
                    0,
                    _LRB1_BUCKETS - 1,
                ),
                cost,
            )
            # cost(k) = sum over elements with frac >= k/64; the parent
            # boundary k=32 is always free because every eligible element
            # with frac >= 0.5 already carries code m + 1 and has zero step.
            suffix = torch.cumsum(bucket_cost.flip(0), dim=0).flip(0)
            total = torch.cat(
                [suffix, torch.zeros(1, device=device, dtype=torch.float32)]
            )
            best_bucket = _lrb1_pick_bucket(total)
            boundaries[sign_index, code] = (
                float(best_bucket) / float(_LRB1_BUCKETS)
            )
            if best_bucket != _LRB1_BUCKETS // 2:
                proposed += 1

    diagnostics["lrb1_boundaries"] = boundaries.reshape(-1).to(
        device="cpu", dtype=torch.float32
    )
    diagnostics["lrb1_proposals"] = proposed

    new_code, switched = _lrb1_apply_boundaries(
        floor_code, code_abs, sign_field, frac, eligible, boundaries
    )
    changed = switched & (new_code != code_abs)
    changed_count = int(changed.sum())
    diagnostics["lrb1_changed_mantissa"] = changed_count
    if changed_count == 0:
        diagnostics["lrb1_table_accepted"] = 0
        diagnostics["lrb1_delta_loss"] = 0.0
        return None

    delta_w = torch.zeros(rows, channels, device=device, dtype=torch.float32)
    delta_w[changed] = (
        (new_code[changed] - code_abs[changed])
        * sign_field[changed]
        * quarter[changed]
    )
    linear_term = 2.0 * float((gradient * delta_w).sum())
    quadratic_term = 0.0
    for a_hat, rows_f in cases:
        case_weight = 1.0 / (case_count * float(rows_f * rows))
        projected = a_hat.mm(delta_w.transpose(0, 1))
        quadratic_term += case_weight * float(projected.square().sum())
    delta_loss = linear_term + quadratic_term
    diagnostics["lrb1_delta_loss_linear"] = linear_term
    diagnostics["lrb1_delta_loss_quadratic"] = quadratic_term
    diagnostics["lrb1_delta_loss"] = delta_loss
    if not math.isfinite(delta_loss) or not delta_loss < 0.0:
        diagnostics["lrb1_table_accepted"] = 0
        return None

    candidate_params = {
        key: value.detach().to(device=device).clone()
        for key, value in parent_params.items()
    }
    candidate_mant = (new_code * _LRB1_CODE_STEP).reshape_as(
        parent_params["mant"]
    )
    candidate_sign = sign_field.reshape_as(parent_params["sign"])
    candidate_params["sign"] = torch.where(
        candidate_mant == 0.0,
        torch.zeros_like(candidate_sign),
        candidate_sign,
    )
    candidate_params["mant"] = candidate_mant
    diagnostics["lrb1_table_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _lrb1_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "lrb1_attempted": 0,
        "lrb1_table_accepted": 0,
        "lrb1_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _lrb1_boundary_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["lrb1_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("lrb1_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if diagnostics.get("lrb1_attempted"):
        learned = diagnostics.get("lrb1_boundaries")
        boundary_text = (
            ",".join(f"{float(value):.3f}" for value in learned.reshape(-1).tolist())
            if torch.is_tensor(learned)
            else "n/a"
        )
        print(
            f"[L-RB1] arm={diagnostics['lrb1_arm']} "
            f"rows={diagnostics.get('lrb1_rows')} "
            f"channels={diagnostics.get('lrb1_channels')} "
            f"eligible={diagnostics.get('lrb1_eligible')} "
            f"proposals={diagnostics.get('lrb1_proposals')} "
            f"changed={diagnostics.get('lrb1_changed_mantissa')} "
            f"delta_loss={float(diagnostics.get('lrb1_delta_loss', 0.0)):.6e} "
            f"boundaries=[{boundary_text}]",
            flush=True,
        )
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_LRB1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """L-RB1: v202 root plus shared output-residual weight rounding boundaries."""

    result = _LRB1_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _lrb1_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
