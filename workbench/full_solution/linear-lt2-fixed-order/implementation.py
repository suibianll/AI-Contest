# ---------------------------------------------------------------------------
# v216 / L-T2: use the calibration-compiled activation GPTQ block order.
#
# v202 compiles a sample-energy order into activation_state, but its dynamic
# wrapper still recomputes a per-call activation-energy order.  This candidate
# uses the already compiled order directly, removing that runtime sort and
# preserving the same static calibration and six-API surface.  The resulting
# hard activation output is evaluated as a new runtime mechanism.
# ---------------------------------------------------------------------------


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(activation_state, dict):
        raise TypeError("activation_state must be a dict")
    order = activation_state.get("gptq_block_order")
    if order is None:
        return _COMBINED_LINEAR_BASE_DYNAMIC(
            activation_quant, activation_scale, activation_state
        )
    try:
        channels = int(activation_quant.shape[-1])
        if channels != int(activation_state.get("in_features", -1)):
            raise ValueError("Activation hidden size does not match calibration state")
        dense = _static_actorder_dense_from_state(
            activation_quant, activation_scale, activation_state
        )
        result = _combined_dynamic_fast_reordered_gptq(
            dense, activation_state, order
        )
        if result is not None:
            return result
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        pass
    return _COMBINED_LINEAR_BASE_DYNAMIC(
        activation_quant, activation_scale, activation_state
    )

