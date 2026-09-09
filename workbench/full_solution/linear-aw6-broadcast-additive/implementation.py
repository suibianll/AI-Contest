# ---------------------------------------------------------------------------
# v209 / L-AW6: deployment-coordinate broadcast additive A@W fit.
#
# v204-v208 only rescaled already decoded weight values.  This card changes
# the parameterization: one additive value is fitted for each natural 4-element
# input group and broadcast across output rows.  The fit uses the actual
# calibration output residual, then performs one legal five-field re-encode
# and a hard-output gate against the retained parent.
# ---------------------------------------------------------------------------

_V209_GROUP_SIZE = 4
_V209_DELTA_MIN = -1.0
_V209_DELTA_MAX = 1.0
_V209_RIDGE_RATIO = 1.0e-3
_V209_FORCE_IDENTITY = False


@torch.no_grad()
def _v209_broadcast_additive_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Fit one additive value per input group, shared by all output rows."""

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        dtype=torch.float32
    )
    if weight.ndim != 2:
        return None
    out_features, in_features = map(int, weight.shape)
    if (
        out_features <= 0
        or in_features <= 0
        or in_features % _HIF4_BLOCK_SIZE != 0
        or in_features % _V209_GROUP_SIZE != 0
    ):
        return None
    blocks = in_features // _HIF4_BLOCK_SIZE
    groups = in_features // _V209_GROUP_SIZE

    parent_weight = _dequantize_hif4(parent_params).to(torch.float32)
    if tuple(parent_weight.shape) != (out_features, in_features):
        return None
    if not bool(torch.isfinite(parent_weight).all()):
        return None

    # The weight is already in the final natural deployment coordinate.  The
    # block order and permutation belong to activation execution only.
    order = state.get("gptq_block_order")
    if order is not None:
        if not torch.is_tensor(order) or int(order.numel()) != blocks:
            return None
        order64 = order.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
        if not torch.equal(
            torch.sort(order64).values,
            torch.arange(blocks, dtype=torch.int64),
        ):
            return None
    permutation = state.get("permutation")
    if permutation is not None and (
        not torch.is_tensor(permutation)
        or int(permutation.numel()) != in_features
    ):
        return None
    diagnostics["v209_coordinate"] = "deployment-natural-broadcast-4-groups"
    diagnostics["v209_gptq_order_used_for"] = "activation-only"
    diagnostics["v209_group_size"] = _V209_GROUP_SIZE

    device = weight.device
    gram_diag = torch.zeros(groups, dtype=torch.float32, device=device)
    rhs = torch.zeros(groups, dtype=torch.float32, device=device)
    records: list[tuple[torch.Tensor, torch.Tensor]] = []
    parent_loss = 0.0
    fit_windows = 0
    fit_rows = 0
    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        act_quant, act_scale = pair
        a_fp = _dequantize_nvfp4_float32(act_quant, act_scale).to(
            dtype=torch.float32
        )
        if a_fp.ndim != 2 or int(a_fp.shape[1]) != in_features:
            return None
        if int(a_fp.shape[0]) == 0:
            continue

        a_hat = _dequantize_hif4(
            hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        ).to(torch.float32)
        if tuple(a_hat.shape) != tuple(a_fp.shape):
            return None
        if not bool(torch.isfinite(a_hat).all()):
            return None

        y_ref = a_fp.mm(weight.t())
        y_parent = a_hat.mm(parent_weight.t())
        residual = y_ref - y_parent
        if not bool(
            torch.isfinite(y_ref).all()
            and torch.isfinite(y_parent).all()
            and torch.isfinite(residual).all()
        ):
            return None

        # An additive group parameter contributes the group-summed activation
        # to every output row.  The diagonal quadratic is kept deliberately
        # low-dimensional; all terms still come from the actual output residual.
        phi = a_hat.reshape(int(a_hat.shape[0]), groups, _V209_GROUP_SIZE).sum(
            dim=-1
        )
        gram_diag.add_(phi.square().sum(dim=0) * float(out_features))
        rhs.add_(phi.t().mv(residual.sum(dim=1)))

        records.append((a_hat, y_ref))
        parent_loss += float(residual.square().sum())
        fit_windows += 1
        fit_rows += int(a_fp.shape[0])

    diagnostics["v209_fit_windows"] = int(fit_windows)
    diagnostics["v209_fit_rows"] = int(fit_rows)
    diagnostics["v209_attempted"] = 1
    diagnostics["v209_groups"] = int(groups)
    diagnostics["v209_loss_parent"] = float(parent_loss)
    if fit_windows == 0:
        return None

    gram64 = gram_diag.to(torch.float64)
    rhs64 = rhs.to(torch.float64)
    ridge = _V209_RIDGE_RATIO * max(float(gram64.mean()), 0.0) + 1.0e-30
    deltas = rhs64 / (gram64 + ridge)
    deltas = torch.nan_to_num(
        deltas,
        nan=0.0,
        posinf=_V209_DELTA_MAX,
        neginf=_V209_DELTA_MIN,
    ).clamp(min=_V209_DELTA_MIN, max=_V209_DELTA_MAX)
    if _V209_FORCE_IDENTITY:
        deltas = torch.zeros_like(deltas)
    deltas32 = deltas.to(torch.float32)
    diagnostics["v209_delta_abs_mean"] = float(deltas32.abs().mean())
    diagnostics["v209_deltas"] = deltas32.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()

    continuous_parent = float(parent_loss)
    continuous_candidate = float(
        parent_loss
        - 2.0 * float((deltas * rhs64).sum())
        + float((deltas.square() * gram64).sum())
    )
    diagnostics["v209_loss_continuous_parent"] = continuous_parent
    diagnostics["v209_loss_continuous_candidate"] = continuous_candidate
    if not (
        math.isfinite(continuous_parent)
        and math.isfinite(continuous_candidate)
        and continuous_candidate < continuous_parent
    ):
        return None

    shifted = parent_weight + deltas32.repeat_interleave(
        _V209_GROUP_SIZE
    ).reshape(1, in_features)
    candidate_params = _dense_to_hif4(shifted)
    candidate_weight = _dequantize_hif4(candidate_params).to(torch.float32)
    if tuple(candidate_weight.shape) != tuple(parent_weight.shape):
        return None
    if not bool(torch.isfinite(candidate_weight).all()):
        return None

    candidate_loss = 0.0
    for a_hat, y_ref in records:
        candidate_loss += float(
            (y_ref - a_hat.mm(candidate_weight.t())).square().sum()
        )
    diagnostics["v209_loss_candidate"] = float(candidate_loss)
    changed_groups = int(
        (
            candidate_weight.reshape(
                out_features, groups, _V209_GROUP_SIZE
            )
            != parent_weight.reshape(
                out_features, groups, _V209_GROUP_SIZE
            )
        )
        .any(dim=(0, 2))
        .sum()
    )
    diagnostics["v209_changed_groups"] = changed_groups
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    if changed_groups == 0:
        return None
    diagnostics["v209_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v209_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v209_attempted": 0,
        "v209_accepted": 0,
        "v209_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v209_broadcast_additive_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v209_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v209_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V209_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v209: v202 parent plus broadcast additive hard-gated A@W."""

    result = _V209_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v209_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
