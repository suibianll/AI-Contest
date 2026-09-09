# ---------------------------------------------------------------------------
# L-XR1: rank-4 dynamic activation output cross-residual correction.
#
# Plan: docs/superpowers/plans/2026-09-10-linear-cross-residual-correction-plan.md
#
# The parent Linear pair leaves the deployed output residual
#
#     R = X_hat W_hat^T - X W^T,
#
# and the exact change of the squared output error under an activation-only
# change DeltaX is
#
#     DeltaL = 2 <R W_hat, DeltaX> + tr(DeltaX G DeltaX^T),
#     R W_hat = (X_hat - X) G + X C,   G = W_hat^T W_hat,
#     C = (W_hat - W)^T W_hat.
#
# Calibration compiles a fixed rank-4 sketch of the off-block metric
# M = G - blockdiag4(G) and of the weight-error cross term C.  The dynamic
# API then evaluates
#
#     g = E G_local + (E U_g) diag(lambda_g) U_g^T + (X U_c) diag(S_c) V_c^T
#
# once, with E = X_hat - X, and walks natural 64-channel blocks in reverse
# parent order, flipping at most one 4-element group per row per 64-block by
# one mantissa code step (-1/+1) at fixed scale_factor/lv2/lv3.  A move is
# accepted only when the exact 4x4 local-Gram quadratic change is strictly
# negative.  One pass, no gradient refresh, no hierarchy change, weight five
# fields untouched, canonical zero sign preserved.
#
# Fixed choices (pre-registered; no rank / damping / layer-list / coverage /
# threshold search):
#   * rank 4, subspace iteration with a deterministic start (no RNG, no eigh
#     on the full matrix, no SVD of the full matrix);
#   * candidate set = one element of one natural 4-element group, mantissa
#     code +1 or -1, i.e. 8 candidates per group, 16 groups per 64-block;
#   * per row per 64-block at most one group is modified;
#   * the accept rule is the exact quadratic: for a single-element move the
#     4x4 Gram quadratic reduces to diag(G4)[i] * dx^2 + 2 g[i] dx.
#
# Both hooks are API-level post-processors appended to the retained root, so
# the parent weight encoder, activation encoder and every shared helper stay
# byte-identical, and the Attention dynamic APIs are untouched.
# ---------------------------------------------------------------------------

_XR1_RANK = 4
_XR1_ITERATIONS = 24
_XR1_CODE_STEP = 0.25
_XR1_CODE_MAX = 7.0


def _xr1_top_eigen(
    matrix: torch.Tensor,
    rank: int = _XR1_RANK,
    iterations: int = _XR1_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic truncated symmetric eigen-decomposition (subspace iteration).

    Returns ``(vectors, values)`` ordered by descending ``|value|``.  The start
    subspace is the ``rank`` columns of ``matrix`` with the largest 2-norm, so
    the routine needs no RNG and is reproducible bit for bit across runs.  It
    replaces a full ``eigh``/``svd`` only to bound calibration time.
    """

    size = int(matrix.shape[0])
    if size <= rank:
        values, vectors = torch.linalg.eigh(matrix)
        order = torch.argsort(values.abs(), descending=True)
        return vectors.index_select(1, order), values.index_select(0, order)
    column_norms = matrix.square().sum(dim=0)
    start = torch.argsort(column_norms, descending=True)[:rank]
    basis = matrix.index_select(1, start)
    basis, _ = torch.linalg.qr(basis)
    for _ in range(iterations):
        basis, _ = torch.linalg.qr(matrix.mm(basis))
    small = basis.transpose(0, 1).mm(matrix).mm(basis)
    values, vectors = torch.linalg.eigh(small)
    vectors = basis.mm(vectors)
    order = torch.argsort(values.abs(), descending=True)
    return vectors.index_select(1, order), values.index_select(0, order)


def _xr1_top_singular(
    matrix: torch.Tensor,
    rank: int = _XR1_RANK,
    iterations: int = _XR1_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic truncated rank-``rank`` SVD via ``matrix^T matrix``."""

    right, values = _xr1_top_eigen(
        matrix.transpose(0, 1).mm(matrix), rank=rank, iterations=iterations
    )
    singular = values.clamp_min(0.0).sqrt()
    left = matrix.mm(right) / singular.clamp_min(1.0e-30).unsqueeze(0)
    return left, singular, right


def _xr1_reconstruct_dense_weight(
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


def _xr1_local_block_matrix(
    flat_gram: torch.Tensor,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize the parent 4x4 block-local Gram as a dense ``channels^2`` matrix."""

    groups = channels // 4
    blocks4 = flat_gram.detach().to(device=device, dtype=dtype).reshape(groups, 4, 4)
    group_ids = torch.arange(channels, device=device, dtype=torch.int64) // 4
    mask = group_ids[:, None] == group_ids[None, :]
    matrix = torch.zeros(channels, channels, device=device, dtype=dtype)
    matrix[mask] = blocks4.reshape(-1)
    return matrix


@torch.no_grad()
def _xr1_compile_factors(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Compile the fixed rank-4 G/C factors for one Linear layer."""

    state = result.get("activation_state") if isinstance(result, dict) else None
    params = result.get("weight_params") if isinstance(result, dict) else None
    if not isinstance(state, dict) or not isinstance(params, dict):
        return
    flat_gram = state.get("gram")
    if flat_gram is None:
        diagnostics["xr1_arm"] = "wide-no-gram"
        return
    in_features = int(state.get("in_features", -1))
    if in_features <= 0 or in_features > _ACTIVATION_QUADRATIC_MAX_FEATURES:
        diagnostics["xr1_arm"] = "wide-no-gram"
        return
    if in_features % 4 != 0:
        diagnostics["xr1_arm"] = "channels-not-grouped"
        return

    dense = _xr1_reconstruct_dense_weight(weight_quant, weight_scale, state)
    if dense is None:
        diagnostics["xr1_arm"] = "dense-unavailable"
        return
    deployed = _dequantize_hif4(params).to(
        device=dense.device, dtype=torch.float32
    )
    if deployed.ndim != 2 or tuple(deployed.shape) != tuple(dense.shape):
        diagnostics["xr1_arm"] = "shape-mismatch"
        return
    channels = int(dense.shape[1])
    if channels != in_features or channels % _HIF4_BLOCK_SIZE != 0:
        diagnostics["xr1_arm"] = "shape-mismatch"
        return

    device = dense.device
    gram_output = deployed.transpose(0, 1).mm(deployed)
    error = deployed - dense
    cross = error.transpose(0, 1).mm(deployed)
    local = _xr1_local_block_matrix(flat_gram, channels, device, torch.float32)
    local_delta = (gram_output - local).abs().max().item()
    offblock = gram_output - local

    u_g, lam_g = _xr1_top_eigen(offblock)
    u_c, s_c, v_c = _xr1_top_singular(cross)

    offblock_energy = offblock.square().sum().clamp_min(1.0e-30)
    cross_energy = cross.square().sum().clamp_min(1.0e-30)
    capture_g = float(lam_g.square().sum() / offblock_energy)
    capture_c = float(s_c.square().sum() / cross_energy)

    state["xr1"] = {
        "u_g": _cpu_state_tensor(u_g.contiguous()),
        "lam_g": _cpu_state_tensor(lam_g.contiguous()),
        "u_c": _cpu_state_tensor(u_c.contiguous()),
        "s_c": _cpu_state_tensor(s_c.contiguous()),
        "v_c": _cpu_state_tensor(v_c.contiguous()),
        "rank": int(_XR1_RANK),
        "capture_g": capture_g,
        "capture_c": capture_c,
    }
    diagnostics.update(
        {
            "xr1_arm": "compiled",
            "xr1_rows": int(dense.shape[0]),
            "xr1_channels": channels,
            "xr1_local_gram_delta": float(local_delta),
            "xr1_capture_g": capture_g,
            "xr1_capture_c": capture_c,
            "xr1_offblock_rel": float(
                offblock_energy.sqrt() / gram_output.square().sum().clamp_min(1.0e-30).sqrt()
            ),
            "xr1_cross_rel": float(
                cross_energy.sqrt() / gram_output.square().sum().clamp_min(1.0e-30).sqrt()
            ),
        }
    )


@torch.no_grad()
def _xr1_dynamic_correct(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    state: Any,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """One frozen-gradient pass of at most one 4-element group per row per block."""

    if not isinstance(state, dict) or not isinstance(result, dict):
        return result
    factors = state.get("xr1")
    if not isinstance(factors, dict):
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
    flat_gram = state.get("gram")
    if flat_gram is None:
        return result

    dense = _static_actorder_dense_from_state(
        activation_quant, activation_scale, state
    )
    if dense.ndim != 2:
        return result
    dense = dense.to(torch.float32)
    rows, channels = map(int, dense.shape)
    if channels != int(state.get("in_features", -1)) or channels % _HIF4_BLOCK_SIZE != 0:
        return result
    blocks = channels // _HIF4_BLOCK_SIZE
    groups4 = channels // 4
    device = dense.device

    deployed = _dequantize_hif4(result).to(device=device, dtype=torch.float32)
    if tuple(deployed.shape) != (rows, channels):
        return result
    error = deployed - dense

    u_g = factors["u_g"].to(device=device, dtype=torch.float32).reshape(channels, -1)
    lam_g = factors["lam_g"].to(device=device, dtype=torch.float32).reshape(-1)
    u_c = factors["u_c"].to(device=device, dtype=torch.float32).reshape(channels, -1)
    s_c = factors["s_c"].to(device=device, dtype=torch.float32).reshape(-1)
    v_c = factors["v_c"].to(device=device, dtype=torch.float32).reshape(channels, -1)
    rank = int(u_g.shape[1])
    if int(lam_g.numel()) != rank or int(u_c.shape[1]) != rank or int(s_c.numel()) != rank:
        return result

    gram4 = flat_gram.detach().to(device=device, dtype=torch.float32).reshape(
        groups4, 4, 4
    )
    local_term = torch.einsum(
        "...gi,gij->...gj", error.reshape(rows, groups4, 4), gram4
    ).reshape(rows, channels)
    low_rank_term = (error.mm(u_g) * lam_g).mm(u_g.transpose(0, 1))
    cross_term = (dense.mm(u_c) * s_c).mm(v_c.transpose(0, 1))
    gradient = local_term + low_rank_term + cross_term
    if not bool(torch.isfinite(gradient).all()):
        diagnostics["xr1_dynamic_arm"] = "nonfinite-gradient"
        return result

    grad_view = gradient.reshape(rows, blocks, 8, 2, 4)
    mant_view = mant.to(device=device, dtype=torch.float32).reshape(
        rows, blocks, 8, 2, 4
    )
    sign_view = sign.to(device=device, dtype=torch.float32).reshape(
        rows, blocks, 8, 2, 4
    )
    scale_view = (
        scale_factor.to(device=device, dtype=torch.float32).reshape(rows, blocks, 1, 1, 1)
        * scale_lv2.to(device=device, dtype=torch.float32).reshape(rows, blocks, 8, 1, 1)
        * scale_lv3.to(device=device, dtype=torch.float32).reshape(rows, blocks, 8, 2, 1)
    ).expand_as(grad_view)
    diag_view = torch.diagonal(gram4.reshape(blocks, 8, 2, 4, 4), dim1=-2, dim2=-1)
    diag_view = diag_view.reshape(1, blocks, 8, 2, 4).expand_as(grad_view)

    code = mant_view / _XR1_CODE_STEP
    delta_up = sign_view * _XR1_CODE_STEP * scale_view
    delta_down = -delta_up
    cost_up = 2.0 * grad_view * delta_up + diag_view * delta_up.square()
    cost_down = 2.0 * grad_view * delta_down + diag_view * delta_down.square()
    cost_up = torch.where(
        code < _XR1_CODE_MAX, cost_up, torch.full_like(cost_up, float("inf"))
    )
    cost_down = torch.where(
        code > 0.0, cost_down, torch.full_like(cost_down, float("inf"))
    )
    best_cost = torch.minimum(cost_up, cost_down)
    best_dir = torch.where(cost_up <= cost_down, 1.0, -1.0)

    group_cost, group_elem = best_cost.min(dim=-1)  # (rows, blocks, 8, 2)
    block_cost, block_flat = group_cost.reshape(rows, blocks, 16).min(dim=-1)
    accept = block_cost < 0.0
    if not bool(accept.any()):
        diagnostics.update(
            {
                "xr1_dynamic_arm": "no-negative-group",
                "xr1_candidates": int(grad_view.numel()),
                "xr1_min_block_cost": float(block_cost.min()),
                "xr1_changed_mantissa": 0,
            }
        )
        return result

    index8 = torch.div(block_flat, 2, rounding_mode="floor")
    index2 = torch.remainder(block_flat, 2)
    index4 = group_elem.reshape(rows, blocks, 16).gather(
        2, block_flat.unsqueeze(2)
    ).squeeze(2)
    block_index = torch.arange(blocks, device=device).unsqueeze(0).expand(rows, blocks)
    flat_index = (
        block_index * 64 + index8 * 8 + index2 * 4 + index4
    )
    chosen = torch.zeros(rows, blocks * 64, device=device, dtype=torch.bool)
    chosen.scatter_(1, flat_index, accept)
    chosen = chosen.reshape(rows, blocks, 8, 2, 4)
    block_dir = best_dir.reshape(rows, blocks * 64).gather(1, flat_index)
    block_dir = torch.where(accept, block_dir, torch.zeros_like(block_dir))
    chosen_dir = torch.zeros(
        rows, blocks * 64, device=device, dtype=torch.float32
    )
    chosen_dir.scatter_(1, flat_index, block_dir)
    chosen_dir = chosen_dir.reshape(rows, blocks, 8, 2, 4)

    new_code = torch.where(chosen, code + chosen_dir, code).clamp_(
        0.0, _XR1_CODE_MAX
    )
    new_mant = (new_code * _XR1_CODE_STEP).reshape_as(mant_view)
    new_sign = torch.where(
        new_mant == 0.0, torch.zeros_like(sign_view), sign_view
    )
    changed = int(chosen.sum())
    diagnostics.update(
        {
            "xr1_dynamic_arm": "applied",
            "xr1_candidates": int(grad_view.numel()),
            "xr1_attempted_groups": int(accept.sum()),
            "xr1_changed_mantissa": changed,
            "xr1_min_block_cost": float(block_cost.min()),
            "xr1_predicted_cost": float(
                torch.where(accept, block_cost, torch.zeros_like(block_cost)).sum()
            ),
            "xr1_grad_abs_mean": float(gradient.abs().mean()),
            "xr1_cross_abs_mean": float(cross_term.abs().mean()),
            "xr1_local_abs_mean": float(local_term.abs().mean()),
            "xr1_lowrank_abs_mean": float(low_rank_term.abs().mean()),
        }
    )
    return dict(
        result,
        mant=new_mant.reshape_as(mant).to(mant.dtype),
        sign=new_sign.reshape_as(sign).to(sign.dtype),
    )


_XR1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight
_XR1_PARENT_LINEAR_DYNAMIC = hif4_dynamic_quantize_activation


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v202 root plus the L-XR1 rank-4 output cross-residual factor compile."""

    result = _XR1_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    diagnostics: dict[str, Any] = {"xr1_arm": "unavailable"}
    try:
        _xr1_compile_factors(weight_quant, weight_scale, result, diagnostics)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as error:
        diagnostics["xr1_arm"] = "error"
        diagnostics["xr1_error"] = f"{type(error).__name__}: {error}"
    state = result.get("activation_state") if isinstance(result, dict) else None
    if isinstance(state, dict):
        state.update(diagnostics)
        print(
            f"[L-XR1] compile arm={diagnostics.get('xr1_arm')} "
            f"in={diagnostics.get('xr1_channels')} "
            f"local_gram_delta={float(diagnostics.get('xr1_local_gram_delta', 0.0)):.3e} "
            f"capture_g={float(diagnostics.get('xr1_capture_g', 0.0)):.4f} "
            f"capture_c={float(diagnostics.get('xr1_capture_c', 0.0)):.4f} "
            f"offblock_rel={float(diagnostics.get('xr1_offblock_rel', 0.0)):.4f} "
            f"cross_rel={float(diagnostics.get('xr1_cross_rel', 0.0)):.4f}",
            flush=True,
        )
    return result


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    """v202 root plus one frozen-gradient rank-4 activation code correction."""

    result = _XR1_PARENT_LINEAR_DYNAMIC(
        activation_quant, activation_scale, activation_state
    )
    diagnostics: dict[str, Any] = {"xr1_dynamic_arm": "unavailable"}
    try:
        corrected = _xr1_dynamic_correct(
            activation_quant,
            activation_scale,
            activation_state,
            result,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as error:
        diagnostics["xr1_dynamic_arm"] = "error"
        diagnostics["xr1_dynamic_error"] = f"{type(error).__name__}: {error}"
        corrected = result
    if isinstance(activation_state, dict):
        activation_state.update(diagnostics)
    if diagnostics.get("xr1_dynamic_arm") not in ("unavailable",):
        print(
            f"[L-XR1] dynamic arm={diagnostics.get('xr1_dynamic_arm')} "
            f"groups={diagnostics.get('xr1_attempted_groups')} "
            f"changed={diagnostics.get('xr1_changed_mantissa')} "
            f"min_cost={float(diagnostics.get('xr1_min_block_cost', 0.0)):.6e} "
            f"pred_cost={float(diagnostics.get('xr1_predicted_cost', 0.0)):.6e} "
            f"grad_mean={float(diagnostics.get('xr1_grad_abs_mean', 0.0)):.3e} "
            f"local_mean={float(diagnostics.get('xr1_local_abs_mean', 0.0)):.3e} "
            f"lowrank_mean={float(diagnostics.get('xr1_lowrank_abs_mean', 0.0)):.3e} "
            f"cross_mean={float(diagnostics.get('xr1_cross_abs_mean', 0.0)):.3e}",
            flush=True,
        )
    return corrected
