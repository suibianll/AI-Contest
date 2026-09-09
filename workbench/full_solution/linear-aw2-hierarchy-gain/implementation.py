# ---------------------------------------------------------------------------
# v205 / L-AW2: hierarchy-aligned 8-element-group A@W fit.
#
# v204 used one scalar for an entire 64-channel carrier block.  This card
# keeps the same direct output objective but aligns the low-dimensional
# parameters with HiF4's first internal hierarchy: one scalar for each
# 8-element group inside each deployed 64-channel block.  The final candidate
# is still encoded once into the legal five-field carrier and hard-gated on
# every supplied calibration row.
# ---------------------------------------------------------------------------

_V205_GROUP_SIZE = 8
_V205_GAIN_MIN = 0.5
_V205_GAIN_MAX = 2.0
_V205_RIDGE_RATIO = 1.0e-3
_V205_MAX_FULL_FIT_CHANNELS = 8192
_V205_FORCE_IDENTITY = False


@torch.no_grad()
def _v205_group_gain_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Fit one scalar per deployed 8-element HiF4 hierarchy group."""

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        dtype=torch.float32
    )
    if weight.ndim != 2:
        return None
    out_features, in_features = map(int, weight.shape)
    if (
        in_features <= 0
        or in_features % _HIF4_BLOCK_SIZE != 0
        or in_features % _V205_GROUP_SIZE != 0
    ):
        return None
    blocks = in_features // _HIF4_BLOCK_SIZE
    groups = in_features // _V205_GROUP_SIZE

    parent_weight = _dequantize_hif4(parent_params).to(torch.float32)
    if tuple(parent_weight.shape) != (out_features, in_features):
        return None
    if not bool(torch.isfinite(parent_weight).all()):
        return None

    # Weight blocks are already in final natural deployment coordinates.  The
    # block order and permutation belong to the activation execution path and
    # must be validated but never applied a second time here.
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
    diagnostics["v205_coordinate"] = "deployment-natural-8-channel-groups"
    diagnostics["v205_gptq_order_used_for"] = "activation-only"

    device = weight.device
    hessian: Optional[torch.Tensor]
    cross: Optional[torch.Tensor]
    group_hessian: Optional[torch.Tensor]
    group_cross: Optional[torch.Tensor]
    if in_features <= _V205_MAX_FULL_FIT_CHANNELS:
        hessian = torch.zeros(
            in_features, in_features, dtype=torch.float32, device=device
        )
        cross = torch.zeros(
            in_features, out_features, dtype=torch.float32, device=device
        )
        group_hessian = None
        group_cross = None
        diagnostics["v205_solver"] = "full-hierarchy-quadratic"
    else:
        hessian = None
        cross = None
        group_hessian = torch.zeros(
            groups, _V205_GROUP_SIZE, _V205_GROUP_SIZE,
            dtype=torch.float32, device=device,
        )
        group_cross = torch.zeros(
            groups, _V205_GROUP_SIZE, out_features,
            dtype=torch.float32, device=device,
        )
        diagnostics["v205_solver"] = "independent-hierarchy-quadratic"

    records: list[tuple[torch.Tensor, torch.Tensor]] = []
    yty = 0.0
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

        # Use the parent's compiled dynamic path.  It returns the activation
        # in the same natural deployment coordinate as parent_weight.
        a_hat = _dequantize_hif4(
            hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        ).to(torch.float32)
        if tuple(a_hat.shape) != tuple(a_fp.shape):
            return None
        if not bool(torch.isfinite(a_hat).all()):
            return None

        y_ref = a_fp.mm(weight.t())
        y_parent = a_hat.mm(parent_weight.t())
        if not bool(
            torch.isfinite(y_ref).all() and torch.isfinite(y_parent).all()
        ):
            return None

        if hessian is not None and cross is not None:
            hessian.add_(a_hat.t().mm(a_hat))
            cross.add_(a_hat.t().mm(y_ref))
        else:
            for group_index in range(groups):
                start = group_index * _V205_GROUP_SIZE
                end = start + _V205_GROUP_SIZE
                a_group = a_hat[:, start:end]
                group_hessian[group_index].add_(a_group.t().mm(a_group))
                group_cross[group_index].add_(a_group.t().mm(y_ref))

        records.append((a_hat, y_ref))
        yty += float(y_ref.square().sum())
        parent_loss += float((y_ref - y_parent).square().sum())
        fit_windows += 1
        fit_rows += int(a_fp.shape[0])

    diagnostics["v205_fit_windows"] = int(fit_windows)
    diagnostics["v205_fit_rows"] = int(fit_rows)
    diagnostics["v205_attempted"] = 1
    diagnostics["v205_blocks"] = int(blocks)
    diagnostics["v205_groups"] = int(groups)
    diagnostics["v205_loss_parent"] = float(parent_loss)
    if fit_windows == 0:
        return None

    try:
        weight_gram = parent_weight.t().mm(parent_weight)
        if hessian is not None and cross is not None:
            h8 = hessian.reshape(
                groups, _V205_GROUP_SIZE, groups, _V205_GROUP_SIZE
            )
            s8 = weight_gram.reshape(
                groups, _V205_GROUP_SIZE, groups, _V205_GROUP_SIZE
            )
            gram_g = torch.einsum("piqj,piqj->pq", h8, s8)
            rhs = (
                parent_weight.t().reshape(
                    groups, _V205_GROUP_SIZE, out_features
                )
                * cross.reshape(groups, _V205_GROUP_SIZE, out_features)
            ).sum(dim=(1, 2))
        else:
            s8 = weight_gram.reshape(
                groups, _V205_GROUP_SIZE, groups, _V205_GROUP_SIZE
            )
            gram_g = torch.zeros(
                groups, groups, dtype=torch.float32, device=device
            )
            rhs = torch.zeros(groups, dtype=torch.float32, device=device)
            for group_index in range(groups):
                local_s = s8[
                    group_index, :, group_index, :
                ]
                gram_g[group_index, group_index] = (
                    group_hessian[group_index] * local_s
                ).sum()
                start = group_index * _V205_GROUP_SIZE
                end = start + _V205_GROUP_SIZE
                local_weight = parent_weight[:, start:end]
                rhs[group_index] = (
                    local_weight.t() * group_cross[group_index]
                ).sum()

        gram64 = gram_g.to(torch.float64)
        rhs64 = rhs.to(torch.float64)
        ridge = _V205_RIDGE_RATIO * max(
            float(gram64.diagonal().mean()), 0.0
        ) + 1.0e-30
        gains = torch.linalg.solve(
            gram64 + ridge * torch.eye(
                groups, dtype=torch.float64, device=device
            ),
            rhs64,
        )
    except (RuntimeError, ValueError, TypeError):
        return None

    gains = torch.nan_to_num(
        gains,
        nan=1.0,
        posinf=_V205_GAIN_MAX,
        neginf=_V205_GAIN_MIN,
    ).clamp(min=_V205_GAIN_MIN, max=_V205_GAIN_MAX)
    if _V205_FORCE_IDENTITY:
        gains = torch.ones_like(gains)
    gains32 = gains.to(torch.float32)
    diagnostics["v205_gain_abs_dev_mean"] = float(
        (gains32 - 1.0).abs().mean()
    )
    diagnostics["v205_gains"] = gains32.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()

    def quadratic_loss(vector: torch.Tensor) -> float:
        if hessian is None:
            result = 0.0
            for group_index in range(groups):
                value = vector[group_index]
                result += float(
                    value * value * gram_g[group_index, group_index]
                    - 2.0 * value * rhs[group_index]
                )
            return yty + result
        return yty - 2.0 * float(vector @ rhs64) + float(
            vector @ (gram64 @ vector)
        )

    continuous_parent = quadratic_loss(torch.ones_like(rhs64))
    continuous_candidate = quadratic_loss(gains)
    diagnostics["v205_loss_continuous_parent"] = float(continuous_parent)
    diagnostics["v205_loss_continuous_candidate"] = float(
        continuous_candidate
    )
    if not (
        math.isfinite(continuous_parent)
        and math.isfinite(continuous_candidate)
        and continuous_candidate < continuous_parent
    ):
        return None

    scaled = parent_weight * gains32.repeat_interleave(
        _V205_GROUP_SIZE
    ).reshape(1, in_features)
    candidate_params = _dense_to_hif4(scaled)
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
    diagnostics["v205_loss_candidate"] = float(candidate_loss)
    changed_groups = int(
        (
            candidate_weight.reshape(
                out_features, groups, _V205_GROUP_SIZE
            )
            != parent_weight.reshape(out_features, groups, _V205_GROUP_SIZE)
        )
        .any(dim=(0, 2))
        .sum()
    )
    diagnostics["v205_changed_groups"] = changed_groups
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    if changed_groups == 0:
        return None
    diagnostics["v205_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v205_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v205_attempted": 0,
        "v205_accepted": 0,
        "v205_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v205_group_gain_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v205_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v205_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V205_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v205: v202 parent plus hierarchy-aligned hard-gated A@W."""

    result = _V205_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v205_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
