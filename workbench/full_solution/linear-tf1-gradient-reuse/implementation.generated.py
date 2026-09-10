# ---------------------------------------------------------------------------
# L-TF1 -- the first pass reuses the gradient the parent already computed.
#
# At _EM1_PASSES = 1 the parent evaluates
#
#     gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
#
# twice per call: once before the pass loop, where the result is checked for
# finiteness, and again as the first statement of the single pass, which rebinds
# the name and discards the earlier tensor.  The two evaluations are the same
# expression on the same tensors -- premise.py confirms statically that nothing
# between them changes `deployed` or `reference`, and verify.py confirms at
# runtime that the two results are bit-identical -- so the first one is reused
# and the second is not performed:
#
#   parent   for _pass in range(_EM1_PASSES):
#                gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
#                if not torch.isfinite(gradient).all():
#                    aborted = True
#                    break
#
#   L-TF1    for _pass in range(_EM1_PASSES):
#                if _pass:
#                    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
#                    if not torch.isfinite(gradient).all():
#                        aborted = True
#                        break
#
# The guard stops the reuse at pass 0 for a reason the code itself gives: the
# pass body mutates `deployed` and `gradient` in place (deployed.add_ and
# gradient.add_), so from the first pass onward the tensors the gradient was
# built from no longer exist and every later pass must still recompute, in the
# parent's own order.  At K = 1, which is the shipped root, the guarded branch
# never runs.
#
# Sole change.  The pre-loop finiteness check and its `nonfinite-gradient`
# diagnostic, the metric and its ridge, the candidate set, the 16-step
# group-major schedule, the coverage rule, the local cost and the whole-row
# joint acceptance are the parent's own code, byte for byte.  build.py reports
# both the byte delta and an AST comparison: the candidate's function
# parses to the parent's statement tree exactly, once the inserted `if _pass:`
# node is dissolved back into the pass body.
#
# The definition below shadows the parent's, which stays in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


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
    step_column = step_values.reshape(8, 1, 1, 1)
    mask_row = element_mask.reshape(8, 1, 1, 4)
    # (blocks, 16, 4, 4) exact within-group metric blocks, indexed by group id
    metric6 = metric.reshape(blocks, _EM1_GROUPS_PER_BLOCK, 4, blocks, _EM1_GROUPS_PER_BLOCK, 4)
    block_index = torch.arange(blocks, device=device).reshape(blocks, 1, 1, 1)
    group_index = torch.arange(_EM1_GROUPS_PER_BLOCK, device=device).reshape(
        1, _EM1_GROUPS_PER_BLOCK, 1, 1
    )
    group_gram = metric6[:, :, :, :, :, :][
        block_index, group_index, :, block_index, group_index, :
    ].reshape(blocks, _EM1_GROUPS_PER_BLOCK, 4, 4)

    deployed_before = deployed
    deployed = deployed.clone()
    code_orig = code_v.clone()
    accepted_moves = torch.zeros((), device=device, dtype=torch.float32)
    accepted_rows = torch.zeros((), device=device, dtype=torch.float32)
    accepted_steps = 0
    predicted_cost = torch.zeros((), device=device, dtype=torch.float32)
    aborted = False
    for _pass in range(_EM1_PASSES):
        if _pass:
            gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
            if not torch.isfinite(gradient).all():
                aborted = True
                break
        for step in range(_EM1_GROUPS_PER_BLOCK):
            # Step `step`: group `step` of every block proposes simultaneously.
            code_g = code_v[:, :, step, :]
            moved = (code_g.unsqueeze(0) + step_column).clamp_(0.0, _EM1_CODE_MAX)
            candidates = torch.where(mask_row, moved, code_g.unsqueeze(0))
            delta = (
                sign_v[:, :, step, :].unsqueeze(0)
                * (candidates - code_g.unsqueeze(0))
                * _EM1_CODE_STEP
                * scale_v[:, :, step, :].unsqueeze(0)
            )
            local_gram = group_gram[:, step]
            gradient4 = gradient.reshape(rows, blocks, _EM1_GROUPS_PER_BLOCK, 4)
            linear = (delta * gradient4[:, :, step, :].unsqueeze(0)).sum(dim=-1)
            quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)
            cost = 2.0 * linear + quadratic
            best_index = cost.argmin(dim=0)
            best_cost = cost.gather(0, best_index.unsqueeze(0)).squeeze(0)
            take = best_cost < 0.0
            chosen = delta.gather(
                0, best_index.reshape(1, rows, blocks, 1).expand(1, rows, blocks, 4)
            ).squeeze(0)
            chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
            row_delta = torch.zeros(rows, channels, device=device, dtype=torch.float32)
            row_delta.reshape(rows, blocks, _EM1_GROUPS_PER_BLOCK, 4)[
                :, :, step, :
            ].copy_(chosen)
            g_delta = row_delta.mm(metric)
            joint = 2.0 * (row_delta * gradient).sum(dim=1) + (row_delta * g_delta).sum(
                dim=1
            )
            keep_row = joint < 0.0
            keep_scale = keep_row.to(torch.float32).unsqueeze(1)
            deployed.add_(row_delta.mul_(keep_scale))
            gradient.add_(g_delta.mul_(keep_scale))
            keep = take & keep_row.unsqueeze(1)
            code_step = torch.where(
                keep.unsqueeze(-1),
                element_mask[best_index].to(torch.float32)
                * step_values[best_index].unsqueeze(-1),
                torch.zeros(rows, blocks, 4, device=device, dtype=torch.float32),
            )
            code_g.copy_((code_g + code_step).clamp_(0.0, _EM1_CODE_MAX))
            predicted_cost += (joint * keep_scale.reshape(rows)).sum()
            accepted_moves += keep.sum()
            accepted_rows += keep_row.sum()
            accepted_steps += 1

    accepted_moves_int = int(accepted_moves)
    accepted_rows_int = int(accepted_rows)
    if accepted_moves_int == 0:
        diagnostics.update(
            {
                "em1_dynamic_arm": "no-negative-group",
                "em1_groups": blocks * _EM1_GROUPS_PER_BLOCK,
                "em1_changed_mantissa": 0,
                "em1_rows": rows,
                "em1_aborted": aborted,
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
            "em1_groups": blocks * _EM1_GROUPS_PER_BLOCK,
            "em1_passes": _EM1_PASSES,
            "em1_accepted_steps": accepted_steps,
            "em1_accepted_groups": accepted_moves_int,
            "em1_accepted_rows": accepted_rows_int,
            "em1_changed_mantissa": int((code_v != code_orig).sum()),
            "em1_changed_values": int((deployed != deployed_before).sum()),
            "em1_predicted_cost": predicted_cost,
            "em1_rows": rows,
            "em1_grad_abs_mean": float(gradient.abs().mean()),
        }
    )
    return dict(result, mant=new_mant.reshape_as(mant), sign=new_sign.reshape_as(sign))
