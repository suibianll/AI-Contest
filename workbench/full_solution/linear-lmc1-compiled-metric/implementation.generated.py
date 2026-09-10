# ---------------------------------------------------------------------------
# L-MC1 -- the fixed metric rebuild moves from the dynamic path into calibration.
#
# `_em1_metric` rebuilt G on every dynamic call:
#
#     inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
#     ridge   = mean(diag(inverse)) - gram_diag_mean
#     metric  = inverse; metric.diagonal().sub_(ridge)
#
# Nothing in that depends on the activation -- only on `h_inv` and the scalar
# `gram_diag_mean`, both fixed once calibration finishes -- so it is rebuilt
# work, and the card builds it once and stores it.
#
# The two shadows below carry the change.  `_em1_compile_metric` gains the
# inverse; `_em1_metric` loses it and loads the stored tensor instead.
#
# The stored tensor is computed on the run device and stored as a CPU tensor,
# because `validate_state` requires CPU.  That order matters: measured here, CPU
# and CUDA `cholesky_inverse` are not bitwise equal (relative difference ~1e-6),
# while a float32 CUDA->CPU->CUDA round trip is exact.  Computing on CPU would
# have changed the arithmetic; computing on the run device and round-tripping
# does not.
#
# The activation set is preserved rather than widened: the parent's em1 arm
# switched off when the Cholesky raised, so calibration stores a metric only when
# the same Cholesky succeeded, and the dynamic arm requires a stored tensor.
#
# Nothing downstream writes to the metric -- the descent reshapes and multiplies
# it -- so the ridge is applied exactly once even when one state dict serves many
# calls.  verify.py checks that by comparing the stored bytes across calls.
#
# The definitions below shadow the parent's, which stay in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


def _em1_compile_metric(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Store H = (W_hat - W)^T W_hat and mean(diag(W_hat^T W_hat)) for one layer."""

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
    if not torch.isfinite(h_matrix).all():
        diagnostics["em1_arm"] = "nonfinite-metric"
        return

    # The ridge c = mean(diag(h_inv^{-1} - gram)) is what makes h_inv^{-1} - cI
    # the deployed Gram.  It is split into its two means so that this hook never
    # needs h_inv^{-1}: `mean(diag(gram))` is a scalar here, and the dynamic API
    # already builds h_inv^{-1}, where it subtracts the second mean.  That
    # removes an n^3 Cholesky inverse from every calibration call -- measured at
    # 39.9 ms (in=4096) / 15.3 ms (in=2560), i.e. ~2.8 s over the official
    # 144 in-scope calibrations -- with no change to the stored state's size.
    gram_diag_mean = float(gram.diagonal().mean())

    # L-MC1: build G here, once, instead of on every dynamic call.  This is the
    # same expression on the same input the dynamic path used, run on the run
    # device so the stored value is what that path would have produced.
    metric = None
    try:
        inverse = torch.cholesky_inverse(
            torch.linalg.cholesky(h_inv.to(device=device, dtype=torch.float32))
        )
        ridge = float(inverse.diagonal().mean()) - gram_diag_mean
        inverse.diagonal().sub_(ridge)
        # Deliberately NOT _cpu_state_tensor: that helper calls .contiguous(),
        # and cholesky_inverse returns a transposed-stride tensor (stride (1, n),
        # the LAPACK layout).  The layout is part of the arithmetic -- a
        # contiguous copy carries the same values but makes the downstream .mm()
        # take a different cuBLAS path and round differently, which was measured
        # at the full five-field level before this was fixed.  nan_to_num and the
        # device round trip both preserve strides.
        metric = torch.nan_to_num(
            inverse.detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    except RuntimeError:
        # The parent's em1 arm switched itself off when the Cholesky failed;
        # storing nothing keeps the candidate's activation set identical.
        metric = None

    state["em1"] = {
        "h": _cpu_state_tensor(h_matrix.contiguous()),
        "gram_diag_mean": gram_diag_mean,
        "metric": metric,
        "channels": channels,
        "version": _EM1_VERSION,
    }
    diagnostics.update(
        {
            "em1_arm": "compiled",
            "em1_channels": channels,
            "em1_gram_diag_mean": gram_diag_mean,
            "em1_h_norm": float(h_matrix.norm()),
        }
    )


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
    # L-MC1: the calibration already built G -- same expression, same input,
    # run device, ridge applied -- so this path only loads it.  No Cholesky, no
    # inverse, and no write: the tensor is returned as the caller's operand, and
    # the descent only reshapes and multiplies it.
    metric = payload.get("metric")
    if not torch.is_tensor(metric):
        return None
    metric = metric.to(device=device, dtype=torch.float32)
    if metric.ndim != 2 or tuple(metric.shape) != (channels, channels):
        return None
    h_matrix = payload.get("h")
    if not torch.is_tensor(h_matrix):
        return None
    h_matrix = h_matrix.to(device=device, dtype=torch.float32)
    if tuple(h_matrix.shape) != (channels, channels):
        return None
    if not (torch.isfinite(metric).all() and torch.isfinite(h_matrix).all()):
        return None
    return metric, h_matrix
