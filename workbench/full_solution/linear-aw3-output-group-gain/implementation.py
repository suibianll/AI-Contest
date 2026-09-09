# ---------------------------------------------------------------------------
# v206 / L-AW3: output-group x 64-block A@W fit.
#
# The previous two cards shared each input-block gain across every output row.
# This card adds one fixed output-row group (64 rows) to the parameterization:
# each output group owns one gain for every deployed 64-channel input block.
# Groups are independent in the output objective, so the fit is a batch of
# small block systems rather than a high-dimensional weight search.
# ---------------------------------------------------------------------------

_V206_OUTPUT_GROUP_SIZE = 64
_V206_GAIN_MIN = 0.5
_V206_GAIN_MAX = 2.0
_V206_RIDGE_RATIO = 1.0e-3
_V206_FORCE_IDENTITY = False


@torch.no_grad()
def _v206_output_group_gain_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Fit one scalar per (64 output rows, 64 input channels) pair."""

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        dtype=torch.float32
    )
    if weight.ndim != 2:
        return None
    out_features, in_features = map(int, weight.shape)
    if (
        out_features <= 0
        or in_features <= 0
        or out_features % _V206_OUTPUT_GROUP_SIZE != 0
        or in_features % _HIF4_BLOCK_SIZE != 0
    ):
        return None
    blocks = in_features // _HIF4_BLOCK_SIZE
    output_groups = out_features // _V206_OUTPUT_GROUP_SIZE

    parent_weight = _dequantize_hif4(parent_params).to(torch.float32)
    if tuple(parent_weight.shape) != (out_features, in_features):
        return None
    if not bool(torch.isfinite(parent_weight).all()):
        return None

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
    diagnostics["v206_coordinate"] = "deployment-natural-output-groups"
    diagnostics["v206_gptq_order_used_for"] = "activation-only"
    diagnostics["v206_output_group_size"] = _V206_OUTPUT_GROUP_SIZE

    device = weight.device
    gram = torch.zeros(
        output_groups, blocks, blocks, dtype=torch.float32, device=device
    )
    rhs = torch.zeros(
        output_groups, blocks, dtype=torch.float32, device=device
    )
    target_energy = torch.zeros(
        output_groups, dtype=torch.float32, device=device
    )

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
        if not bool(
            torch.isfinite(y_ref).all() and torch.isfinite(y_parent).all()
        ):
            return None

        # Z[r,t,g,b] is the contribution of input block b to output row
        # group r, so each output group has its own exact A@W quadratic.
        a_blocks = a_hat.reshape(int(a_hat.shape[0]), blocks, _HIF4_BLOCK_SIZE)
        weight_groups = parent_weight.reshape(
            output_groups,
            _V206_OUTPUT_GROUP_SIZE,
            blocks,
            _HIF4_BLOCK_SIZE,
        )
        basis = torch.einsum("tbi,rgbi->rtgb", a_blocks, weight_groups)
        target_groups = y_ref.reshape(
            int(y_ref.shape[0]),
            output_groups,
            _V206_OUTPUT_GROUP_SIZE,
        ).permute(1, 0, 2)
        gram.add_(torch.einsum("rtgb,rtgc->rbc", basis, basis))
        rhs.add_(torch.einsum("rtgb,rtg->rb", basis, target_groups))
        target_energy.add_(target_groups.square().sum(dim=(1, 2)))

        records.append((a_hat, y_ref))
        parent_loss += float((y_ref - y_parent).square().sum())
        fit_windows += 1
        fit_rows += int(a_fp.shape[0])

    diagnostics["v206_fit_windows"] = int(fit_windows)
    diagnostics["v206_fit_rows"] = int(fit_rows)
    diagnostics["v206_attempted"] = 1
    diagnostics["v206_blocks"] = int(blocks)
    diagnostics["v206_output_groups"] = int(output_groups)
    diagnostics["v206_loss_parent"] = float(parent_loss)
    if fit_windows == 0:
        return None

    try:
        gram64 = gram.to(torch.float64)
        rhs64 = rhs.to(torch.float64)
        diagonal_mean = float(gram64.diagonal(dim1=-2, dim2=-1).mean())
        ridge = _V206_RIDGE_RATIO * max(diagonal_mean, 0.0) + 1.0e-30
        eye = torch.eye(blocks, dtype=torch.float64, device=device).expand(
            output_groups, blocks, blocks
        )
        gains = torch.linalg.solve(
            gram64 + ridge * eye,
            rhs64.unsqueeze(-1),
        ).squeeze(-1)
    except (RuntimeError, ValueError, TypeError):
        return None

    gains = torch.nan_to_num(
        gains,
        nan=1.0,
        posinf=_V206_GAIN_MAX,
        neginf=_V206_GAIN_MIN,
    ).clamp(min=_V206_GAIN_MIN, max=_V206_GAIN_MAX)
    if _V206_FORCE_IDENTITY:
        gains = torch.ones_like(gains)
    gains32 = gains.to(torch.float32)
    diagnostics["v206_gain_abs_dev_mean"] = float(
        (gains32 - 1.0).abs().mean()
    )
    diagnostics["v206_gains"] = gains32.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()

    def quadratic_loss(vector: torch.Tensor) -> float:
        quadratic = torch.einsum(
            "rb,rbc,rc->r", vector, gram64, vector
        )
        linear = (vector * rhs64).sum(dim=1)
        return float(
            (target_energy.to(torch.float64) - 2.0 * linear + quadratic).sum()
        )

    continuous_parent = quadratic_loss(torch.ones_like(gains))
    continuous_candidate = quadratic_loss(gains)
    diagnostics["v206_loss_continuous_parent"] = float(continuous_parent)
    diagnostics["v206_loss_continuous_candidate"] = float(
        continuous_candidate
    )
    if not (
        math.isfinite(continuous_parent)
        and math.isfinite(continuous_candidate)
        and continuous_candidate < continuous_parent
    ):
        return None

    scaled = parent_weight.reshape(
        output_groups,
        _V206_OUTPUT_GROUP_SIZE,
        blocks,
        _HIF4_BLOCK_SIZE,
    ) * gains32[:, None, :, None]
    candidate_params = _dense_to_hif4(
        scaled.reshape(out_features, in_features)
    )
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
    diagnostics["v206_loss_candidate"] = float(candidate_loss)
    changed_units = int(
        (
            candidate_weight.reshape(
                output_groups,
                _V206_OUTPUT_GROUP_SIZE,
                blocks,
                _HIF4_BLOCK_SIZE,
            )
            != parent_weight.reshape(
                output_groups,
                _V206_OUTPUT_GROUP_SIZE,
                blocks,
                _HIF4_BLOCK_SIZE,
            )
        )
        .any(dim=(1, 3))
        .sum()
    )
    diagnostics["v206_changed_output_group_blocks"] = changed_units
    if not math.isfinite(candidate_loss) or not candidate_loss < parent_loss:
        return None
    if changed_units == 0:
        return None
    diagnostics["v206_accepted"] = 1
    return candidate_params


@torch.no_grad()
def _v206_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "v206_attempted": 0,
        "v206_accepted": 0,
        "v206_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    candidate_params = None
    try:
        candidate_params = _v206_output_group_gain_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        candidate_params = None
    diagnostics["v206_arm"] = (
        "accepted"
        if candidate_params is not None
        else ("parent" if diagnostics.get("v206_attempted") else "unavailable")
    )
    state.update(diagnostics)
    if candidate_params is None:
        return result
    return dict(result, weight_params=candidate_params)


_V206_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v206: v202 parent plus output-group hard-gated A@W."""

    result = _V206_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _v206_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
