# ---------------------------------------------------------------------------
# v204 / L-AW1: deployment-coordinate 64-block scalar A@W fit.
#
# The parent returns weight codes in the final natural deployment coordinate.
# ``gptq_block_order`` is an activation execution order only; the dynamic
# activation wrapper inverse-maps its result before the product is formed.
# Therefore the gain vector below is indexed by the decoded weight carrier's
# natural 64-channel blocks and is never permuted by either state field.
# ---------------------------------------------------------------------------

_V204_GAIN_MIN = 0.5
_V204_GAIN_MAX = 2.0
_V204_RIDGE_RATIO = 1.0e-3
_V204_MAX_FULL_FIT_CHANNELS = 8192
_V204_FORCE_IDENTITY = False


@torch.no_grad()
def _v204_block_gain_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Fit one fixed scalar per deployed 64-channel weight block.

    The sufficient statistics are the exact output-product quadratic for
    ``Y = XW^T`` with the parent's hard activation ``Q(X)`` held fixed.  A
    candidate is accepted only after legal re-encoding and a second exact
    hard-output comparison on every calibration row.
    """

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        dtype=torch.float32
    )
    if weight.ndim != 2:
        return None
    out_features, in_features = map(int, weight.shape)
    if in_features <= 0 or in_features % _HIF4_BLOCK_SIZE != 0:
        return None
    blocks = in_features // _HIF4_BLOCK_SIZE

    parent_weight = _dequantize_hif4(parent_params).to(torch.float32)
    if tuple(parent_weight.shape) != (out_features, in_features):
        return None
    if not bool(torch.isfinite(parent_weight).all()):
        return None

    # The state is deliberately inspected, not reapplied: these fields
    # describe how the parent produced Q(X), while parent_weight is already
    # in the matching post-permutation/deployment coordinate.
    order = state.get("gptq_block_order")
    permutation = state.get("permutation")
    coordinate_ok = (
        order is None
        or (
            torch.is_tensor(order)
            and int(order.numel()) == blocks
        )
    ) and (
        permutation is None
        or (
            torch.is_tensor(permutation)
            and int(permutation.numel()) == in_features
        )
    )
    if not coordinate_ok:
        return None
    diagnostics["v204_coordinate"] = "deployment-natural-blocks"
    diagnostics["v204_gptq_order_used_for"] = "activation-only"

    device = weight.device
    hessian: Optional[torch.Tensor]
    cross: Optional[torch.Tensor]
    if in_features <= _V204_MAX_FULL_FIT_CHANNELS:
        hessian = torch.zeros(
            in_features, in_features, dtype=torch.float32, device=device
        )
        cross = torch.zeros(
            in_features, out_features, dtype=torch.float32, device=device
        )
        diagnostics["v204_solver"] = "full-block-quadratic"
    else:
        # Keep the same block-scalar A@W mechanism for very wide carriers
        # without creating an unbounded D-by-D temporary.  Each 64-block is
        # still fit from every calibration row and the final hard gate is the
        # same pooled output objective.
        hessian = None
        cross = None
        diagnostics["v204_solver"] = "independent-block-quadratic"

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

        # This is the parent's final dynamic path, including its compiled
        # block order and inverse mapping.  No candidate search is performed
        # here; this is an offline calibration product only.
        a_hat = _dequantize_hif4(
            hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        ).to(torch.float32)
        if tuple(a_hat.shape) != tuple(a_fp.shape):
            return None
        if not bool(torch.isfinite(a_hat).all()):
            return None

        y_ref = a_fp.mm(weight.t())
        y_parent = a_hat.mm(parent_weight.t())
        if not bool(torch.isfinite(y_ref).all() and torch.isfinite(y_parent).all()):
            return None

        if hessian is not None and cross is not None:
            hessian.add_(a_hat.t().mm(a_hat))
            cross.add_(a_hat.t().mm(y_ref))
        else:
            # Store only block-local statistics for the wide fallback.
            if fit_windows == 0:
                block_hessian = torch.zeros(
                    blocks, _HIF4_BLOCK_SIZE, _HIF4_BLOCK_SIZE,
                    dtype=torch.float32, device=device,
                )
                block_cross = torch.zeros(
                    blocks, _HIF4_BLOCK_SIZE, out_features,
                    dtype=torch.float32, device=device,
                )
            for block_index in range(blocks):
                start = block_index * _HIF4_BLOCK_SIZE
                end = start + _HIF4_BLOCK_SIZE
                a_block = a_hat[:, start:end]
                block_hessian[block_index].add_(a_block.t().mm(a_block))
                block_cross[block_index].add_(a_block.t().mm(y_ref))

        records.append((a_hat, y_ref))
        yty += float(y_ref.square().sum())
        parent_loss += float((y_ref - y_parent).square().sum())
        fit_windows += 1
        fit_rows += int(a_fp.shape[0])

    diagnostics["v204_fit_windows"] = int(fit_windows)
    diagnostics["v204_fit_rows"] = int(fit_rows)
    diagnostics["v204_attempted"] = 1
    diagnostics["v204_blocks"] = int(blocks)
    diagnostics["v204_loss_parent"] = float(parent_loss)
    if fit_windows == 0:
        return None

    try:
        w_gram = parent_weight.t().mm(parent_weight)
        if hessian is not None and cross is not None:
            h4 = hessian.reshape(
                blocks, _HIF4_BLOCK_SIZE, blocks, _HIF4_BLOCK_SIZE
            )
            s4 = w_gram.reshape(
                blocks, _HIF4_BLOCK_SIZE, blocks, _HIF4_BLOCK_SIZE
            )
            gram_g = torch.einsum("bicj,bicj->bc", h4, s4)
            rhs = (
                parent_weight.t().reshape(
                    blocks, _HIF4_BLOCK_SIZE, out_features
                )
                * cross.reshape(blocks, _HIF4_BLOCK_SIZE, out_features)
            ).sum(dim=(1, 2))
        else:
            s4 = w_gram.reshape(
                blocks, _HIF4_BLOCK_SIZE, blocks, _HIF4_BLOCK_SIZE
            )
            gram_g = torch.zeros(
                blocks, blocks, dtype=torch.float32, device=device
            )
            rhs = torch.zeros(blocks, dtype=torch.float32, device=device)
            for block_index in range(blocks):
                local_h = block_hessian[block_index]
                local_s = s4[block_index, :, block_index, :]
                gram_g[block_index, block_index] = (local_h * local_s).sum()
                local_cross = block_cross[block_index]
                local_weight = parent_weight[
                    :, block_index * _HIF4_BLOCK_SIZE:
                    (block_index + 1) * _HIF4_BLOCK_SIZE
                ]
                rhs[block_index] = (
                    local_weight.t() * local_cross
                ).sum()

        gram64 = gram_g.to(torch.float64)
        rhs64 = rhs.to(torch.float64)
        diagonal_mean = float(gram64.diagonal().mean())
        ridge = _V204_RIDGE_RATIO * max(diagonal_mean, 0.0) + 1.0e-30
        gains = torch.linalg.solve(
            gram64 + ridge * torch.eye(
                blocks, dtype=torch.float64, device=device
            ),
            rhs64,
        )
    except (RuntimeError, ValueError, TypeError):
        return None

    gains = torch.nan_to_num(
        gains,
        nan=1.0,
        posinf=_V204_GAIN_MAX,
        neginf=_V204_GAIN_MIN,
    ).clamp(min=_V204_GAIN_MIN, max=_V204_GAIN_MAX)
    if _V204_FORCE_IDENTITY:
        gains = torch.ones_like(gains)
    gains32 = gains.to(torch.float32)
    diagnostics["v204_gain_abs_dev_mean"] = float(
        (gains32 - 1.0).abs().mean()
    )
    diagnostics["v204_gains"] = gains32.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()

    def quadratic_loss(vector: torch.Tensor) -> float:
        if hessian is None:
            result = 0.0
            for block_index in range(blocks):
                value = vector[block_index]
                result += float(
                    value * value * gram_g[block_index, block_index]
                    - 2.0 * value * rhs[block_index]
                )
            return yty + result
        return yty - 2.0 * float(vector @ rhs64) + float(
            vector @ (gram64 @ vector)
        )

    continuous_parent = quadratic_loss(torch.ones_like(rhs64))
    continuous_candidate = quadratic_loss(gains)
    diagnostics["v204_loss_continuous_parent"] = float(continuous_parent)
    diagnostics["v204_loss_continuous_candidate"] = float(continuous_candidate)
    if not (
        math.isfinite(continuous_parent)
        and math.isfinite(continuous_candidate)
        and continuous_candidate < continuous_parent
    ):
        return None

    scaled = parent_weight * gains32.repeat_interleave(
        _HIF4_BLOCK_SIZE
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
    diagnostics["v204_loss_candidate"] = float(candidate_loss)
    changed = int((candidate_weight != parent_weight).any(dim=0).sum())
    diagnostics["v204_changed_blocks"] = int(
        (candidate_weight.reshape(out_features, blocks, _HIF4_BLOCK_SIZE)
         != parent_weight.reshape(out_features, blocks, _HIF4_BLOCK_SIZE))
        .any(dim=(0, 2)).sum()
    )
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    if changed == 0:
        return None
    diagnostics["v204_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v204_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v204_attempted": 0,
        "v204_accepted": 0,
        "v204_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v204_block_gain_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v204_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v204_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V204_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v204: v202 parent plus deployment-coordinate hard-gated A@W."""

    result = _V204_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v204_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
