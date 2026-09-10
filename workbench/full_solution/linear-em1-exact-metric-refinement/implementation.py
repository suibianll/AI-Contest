# ---------------------------------------------------------------------------
# L-EM1: exact-metric activation mantissa code descent on the ideal target.
#
# Plan: docs/superpowers/plans/2026-09-10-linear-exact-metric-refinement-plan.md
#
# Let X_hat be the deployed Activation, W_hat the deployed Weight, and let
# X_ref, W be the reference pair the evaluator scores against, i.e. exactly
# ``_dequantize_nvfp4_float32`` of the two input pairs.  The player mse is
# ``L(X) = ||X W_hat^T - X_ref W^T||^2`` in that frame: the parent applies a
# transform to the weight and its inverse to the activation, so both products
# are evaluated in the model frame and the raw nvfp4 weight is the correct
# ``W``.  With
#
#     G = W_hat^T W_hat,  C = W^T W_hat,  H = G - C,
#     P = W_hat G^{-1} W_hat^T,  T = X_ref C G^{-1},
#
# the true output squared error L(X) = ||X W_hat^T - X_ref W^T||^2 splits as
#
#     L(X) = J(X) + ||X_ref W^T (I - P)||^2,   J(X) = tr((X-T) G (X-T)^T),
#
# and the second term is independent of X because (I-P) W_hat = 0 and
# (I-P) P = 0.  Any decrease of J is therefore *exactly* a decrease of the
# true output error: the mechanism cannot make a case worse.
#
# The half-gradient of J at the current point is
#
#     g = (X - T) G = (X - X_ref) G + X_ref H,
#
# and a group move delta changes J by the exact quadratic
#
#     DeltaJ = 2 <delta, g_cur> + delta^T G_gg delta.
#
# Calibration stores H (fp32) and the parent ridge scalar c.  The dynamic API
# rebuilds G = h_inv^{-1} - c I from the parent state, walks natural 64-channel
# blocks in ascending order and their 16 four-element groups in order 0..15,
# and accepts one mantissa code step (-1/+1 at fixed scale_factor/lv2/lv3) per
# group and row whenever the exact quadratic is strictly negative, refreshing
# g exactly after every accepted move.  One pass, no repeat, no candidate
# polling, no hierarchy/scale/permutation change.
#
# Fixed choices (pre-registered; no rank / damping / layer-list / threshold
# search): ideal target, pm1 candidate set (8 per group), ascending block and
# group order, exact sequential refresh, layers with in_features <= 4096.
#
# Both hooks are API-level post-processors appended to the retained root, so
# the parent weight encoder, activation encoder and every shared helper stay
# byte-identical, and the Attention dynamic APIs are untouched.
# ---------------------------------------------------------------------------

_EM1_CODE_STEP = 0.25
_EM1_CODE_MAX = 7.0
_EM1_MAX_CHANNELS = 4096
_EM1_VERSION = 1


@torch.no_grad()
def _em1_compile_metric(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Store H = (W_hat - W)^T W_hat and the parent ridge scalar for one layer."""

    state = result.get("activation_state") if isinstance(result, dict) else None
    params = result.get("weight_params") if isinstance(result, dict) else None
    if not isinstance(state, dict) or not isinstance(params, dict):
        diagnostics["em1_arm"] = "no-state"
        return
    in_features = int(state.get("in_features", -1))
    if in_features <= 0 or in_features > _EM1_MAX_CHANNELS:
        diagnostics["em1_arm"] = "out-of-scope"
        diagnostics["em1_channels"] = in_features
        return
    if in_features % _HIF4_BLOCK_SIZE != 0 or in_features % 4 != 0:
        diagnostics["em1_arm"] = "channels-not-grouped"
        return
    h_inv = state.get("h_inv")
    if not torch.is_tensor(h_inv):
        diagnostics["em1_arm"] = "no-h_inv"
        return

    dense = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(torch.float32)
    if dense.ndim != 2:
        diagnostics["em1_arm"] = "dense-unavailable"
        return
    deployed = _dequantize_hif4(params).to(device=dense.device, dtype=torch.float32)
    if deployed.ndim != 2 or tuple(deployed.shape) != tuple(dense.shape):
        diagnostics["em1_arm"] = "shape-mismatch"
        return
    channels = int(deployed.shape[1])
    if channels != in_features:
        diagnostics["em1_arm"] = "shape-mismatch"
        return

    device = dense.device
    gram = deployed.transpose(0, 1).mm(deployed)
    cross = dense.transpose(0, 1).mm(deployed)
    h_matrix = (gram - cross).to(torch.float32)

    h_inv_dev = h_inv.to(device=device, dtype=torch.float32)
    if tuple(h_inv_dev.shape) != (channels, channels):
        diagnostics["em1_arm"] = "h_inv-shape-mismatch"
        return
    try:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv_dev))
    except RuntimeError as error:
        diagnostics["em1_arm"] = "h_inv-not-pd"
        diagnostics["em1_error"] = f"{type(error).__name__}: {error}"
        return
    ridge = float((inverse - gram).diagonal().mean())
    residual = inverse - gram - ridge * torch.eye(
        channels, device=device, dtype=torch.float32
    )
    if not torch.isfinite(h_matrix).all():
        diagnostics["em1_arm"] = "nonfinite-metric"
        return

    state["em1"] = {
        "h": _cpu_state_tensor(h_matrix.contiguous()),
        "ridge": ridge,
        "channels": channels,
        "version": _EM1_VERSION,
    }
    diagnostics.update(
        {
            "em1_arm": "compiled",
            "em1_channels": channels,
            "em1_ridge": ridge,
            "em1_h_norm": float(h_matrix.norm()),
            "em1_ridge_residual_rel": float(
                residual.norm() / gram.norm().clamp_min(1.0e-30)
            ),
        }
    )


@torch.no_grad()
def _em1_metric(
    state: Any,
    channels: int,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Rebuild ``G`` from the parent ``h_inv`` and return ``(G, H)``."""

    if not isinstance(state, dict):
        return None
    payload = state.get("em1")
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != _EM1_VERSION:
        return None
    if int(payload.get("channels", -1)) != channels:
        return None
    h_inv = state.get("h_inv")
    if not torch.is_tensor(h_inv):
        return None
    h_inv = h_inv.to(device=device, dtype=torch.float32)
    if h_inv.ndim != 2 or tuple(h_inv.shape) != (channels, channels):
        return None
    try:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
    except RuntimeError:
        return None
    ridge = float(payload.get("ridge", 0.0))
    metric = inverse - ridge * torch.eye(channels, device=device, dtype=torch.float32)
    h_matrix = payload.get("h")
    if not torch.is_tensor(h_matrix):
        return None
    h_matrix = h_matrix.to(device=device, dtype=torch.float32)
    if tuple(h_matrix.shape) != (channels, channels):
        return None
    if not (torch.isfinite(metric).all() and torch.isfinite(h_matrix).all()):
        return None
    return metric, h_matrix


@torch.no_grad()
def _em1_dynamic_descent(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    state: Any,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """One exact sequential pass of mantissa code descent on the ideal target."""

    if not isinstance(state, dict) or not isinstance(result, dict):
        return result
    mant = result.get("mant")
    sign = result.get("sign")
    scale_factor = result.get("scale_factor")
    scale_lv2 = result.get("scale_lv2")
    scale_lv3 = result.get("scale_lv3")
    if not all(
        torch.is_tensor(value)
        for value in (mant, sign, scale_factor, scale_lv2, scale_lv3)
    ):
        return result
    if not all(
        torch.is_tensor(value)
        for value in (activation_quant, activation_scale)
    ):
        return result

    deployed = _dequantize_hif4(result).to(torch.float32)
    if deployed.ndim != 2:
        return result
    rows, channels = map(int, deployed.shape)
    if channels != int(state.get("in_features", -1)):
        return result
    if channels <= 0 or channels > _EM1_MAX_CHANNELS or channels % _HIF4_BLOCK_SIZE != 0:
        return result
    device = deployed.device
    blocks = channels // _HIF4_BLOCK_SIZE

    metric_pair = _em1_metric(state, channels, device)
    if metric_pair is None:
        diagnostics["em1_dynamic_arm"] = "no-metric"
        return result
    metric, h_matrix = metric_pair

    reference = _dequantize_nvfp4_float32(activation_quant, activation_scale).to(
        device=device, dtype=torch.float32
    )
    if reference.ndim != 2 or tuple(reference.shape) != (rows, channels):
        diagnostics["em1_dynamic_arm"] = "reference-shape-mismatch"
        return result

    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
    if not torch.isfinite(gradient).all():
        diagnostics["em1_dynamic_arm"] = "nonfinite-gradient"
        return result

    sign_v = sign.to(device=device, dtype=torch.float32).reshape(rows, blocks, 16, 4)
    scale_v = (
        scale_factor.to(device=device, dtype=torch.float32).reshape(rows, blocks, 1, 1, 1)
        * scale_lv2.to(device=device, dtype=torch.float32).reshape(rows, blocks, 8, 1, 1)
        * scale_lv3.to(device=device, dtype=torch.float32).reshape(rows, blocks, 8, 2, 1)
    ).reshape(rows, blocks, 16, 1)
    code_v = torch.round(
        mant.to(device=device, dtype=torch.float32).reshape(rows, blocks, 16, 4)
        / _EM1_CODE_STEP
    )

    element_index = torch.arange(4, device=device).reshape(4, 1).expand(4, 2).reshape(8)
    step_values = torch.tensor([-1.0, 1.0], device=device).reshape(1, 2).expand(4, 2).reshape(8)
    element_mask = torch.zeros(8, 4, dtype=torch.bool, device=device)
    element_mask[torch.arange(8, device=device), element_index] = True
    step_column = step_values.reshape(8, 1, 1)
    mask_row = element_mask.reshape(8, 1, 4)

    deployed_before = deployed
    deployed = deployed.clone()
    code_orig = code_v.clone()
    accepted_groups = 0
    accepted_rows = 0
    predicted_cost = 0.0
    for block in range(blocks):
        block_start = block * _HIF4_BLOCK_SIZE
        for group in range(16):
            start = block_start + group * 4
            sl = slice(start, start + 4)
            code_g = code_v[:, block, group, :]
            moved = (code_g.unsqueeze(0) + step_column).clamp_(0.0, _EM1_CODE_MAX)
            candidates = torch.where(mask_row, moved, code_g.unsqueeze(0))
            delta = (
                sign_v[:, block, group, :].unsqueeze(0)
                * (candidates - code_g.unsqueeze(0))
                * _EM1_CODE_STEP
                * scale_v[:, block, group, :].unsqueeze(0)
            )
            local_gram = metric[sl, sl]
            linear = (delta * gradient[:, sl].unsqueeze(0)).sum(dim=-1)
            quadratic = torch.einsum("kri,ij,krj->kr", delta, local_gram, delta)
            cost = 2.0 * linear + quadratic
            best_index = cost.argmin(dim=0)
            best_cost = cost.gather(0, best_index.unsqueeze(0)).squeeze(0)
            take = best_cost < 0.0
            if not bool(take.any()):
                continue
            chosen = delta.gather(
                0, best_index.reshape(1, rows, 1).expand(1, rows, 4)
            ).squeeze(0)
            chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
            deployed[:, sl] += chosen
            gradient += chosen.mm(metric[sl, :])
            predicted_cost += float(
                torch.where(take, best_cost, torch.zeros_like(best_cost)).sum()
            )
            code_step = torch.where(
                take.unsqueeze(-1),
                element_mask[best_index].to(torch.float32)
                * step_values[best_index].unsqueeze(-1),
                torch.zeros(rows, 4, device=device, dtype=torch.float32),
            )
            code_v[:, block, group, :] = (code_g + code_step).clamp_(
                0.0, _EM1_CODE_MAX
            )
            accepted_groups += 1
            accepted_rows += int(take.sum())

    if accepted_groups == 0:
        diagnostics.update(
            {
                "em1_dynamic_arm": "no-negative-group",
                "em1_groups": blocks * 16,
                "em1_changed_mantissa": 0,
                "em1_rows": rows,
            }
        )
        return result

    new_code = code_v.reshape(rows, blocks, 8, 2, 4)
    new_mant = (new_code * _EM1_CODE_STEP).to(device=mant.device, dtype=mant.dtype)
    sign_dev = sign.to(device=device, dtype=torch.float32)
    new_sign = torch.where(
        new_code == 0.0, torch.zeros_like(sign_dev), sign_dev
    ).to(device=sign.device, dtype=sign.dtype)
    diagnostics.update(
        {
            "em1_dynamic_arm": "applied",
            "em1_groups": blocks * 16,
            "em1_accepted_groups": accepted_groups,
            "em1_accepted_rows": accepted_rows,
            "em1_changed_mantissa": int((code_v != code_orig).sum()),
            "em1_changed_values": int((deployed != deployed_before).sum()),
            "em1_predicted_cost": predicted_cost,
            "em1_rows": rows,
            "em1_grad_abs_mean": float(gradient.abs().mean()),
        }
    )
    return dict(result, mant=new_mant.reshape_as(mant), sign=new_sign.reshape_as(sign))


_EM1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight
_EM1_PARENT_LINEAR_DYNAMIC = hif4_dynamic_quantize_activation


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v202 root plus the L-EM1 exact-metric compile (H and the ridge scalar)."""

    result = _EM1_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    diagnostics: dict[str, Any] = {"em1_arm": "unavailable"}
    try:
        _em1_compile_metric(weight_quant, weight_scale, result, diagnostics)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as error:
        diagnostics["em1_arm"] = "error"
        diagnostics["em1_error"] = f"{type(error).__name__}: {error}"
    state = result.get("activation_state") if isinstance(result, dict) else None
    if isinstance(state, dict):
        state.update(diagnostics)
        print(
            f"[L-EM1] compile arm={diagnostics.get('em1_arm')} "
            f"in={diagnostics.get('em1_channels')} "
            f"ridge={float(diagnostics.get('em1_ridge', 0.0)):.6e} "
            f"h_norm={float(diagnostics.get('em1_h_norm', 0.0)):.4e} "
            f"ridge_residual_rel={float(diagnostics.get('em1_ridge_residual_rel', 0.0)):.3e}",
            flush=True,
        )
    return result


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    """v202 root plus one exact sequential ideal-target code descent pass."""

    result = _EM1_PARENT_LINEAR_DYNAMIC(
        activation_quant, activation_scale, activation_state
    )
    diagnostics: dict[str, Any] = {"em1_dynamic_arm": "unavailable"}
    try:
        corrected = _em1_dynamic_descent(
            activation_quant,
            activation_scale,
            activation_state,
            result,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as error:
        diagnostics["em1_dynamic_arm"] = "error"
        diagnostics["em1_dynamic_error"] = f"{type(error).__name__}: {error}"
        corrected = result
    if isinstance(activation_state, dict):
        activation_state.update(diagnostics)
    if diagnostics.get("em1_dynamic_arm") != "unavailable":
        print(
            f"[L-EM1] dynamic arm={diagnostics.get('em1_dynamic_arm')} "
            f"groups={diagnostics.get('em1_groups')} "
            f"accepted_groups={diagnostics.get('em1_accepted_groups')} "
            f"changed={diagnostics.get('em1_changed_mantissa')} "
            f"rows={diagnostics.get('em1_rows')} "
            f"grad_mean={float(diagnostics.get('em1_grad_abs_mean', 0.0)):.3e} "
            f"error={diagnostics.get('em1_dynamic_error', '-')}",
            flush=True,
        )
    return corrected
