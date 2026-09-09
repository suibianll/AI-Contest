# ---------------------------------------------------------------------------
# A-RB1 / v227: Q/K joint softmax-output shared rounding boundaries.
#
# The retained root encodes Q and K with a plain nearest-round mantissa
# decision inside every hierarchy candidate
#
#     c = clamp(round(4 * |x| / D), 0, 7),   D = scale_factor * lv2 * lv3.
#
# This card replaces that fixed 0.5 threshold with one learned boundary per
# lower-code class, shared across signs -- tau_q[m] and tau_k[m], m = 1..6,
# 12 shared scalars per layer:
#
#     c = m         if frac(u) < tau[role][m],
#     c = m + 1     otherwise,        m = floor(u).
#
# The boundaries are solved once from the frozen parent softmax-output
# residual: g = J^T(O_parent - O_ref) from one backward, a one-probe
# Hutchinson diagonal curvature h = mean_f (J^T r_f)^2 with a fixed
# Rademacher seed, 64 fixed fractional buckets and an analytic argmin with
# the parent boundary tau = 0.5 as the fallback.  The merged Q/K table is
# applied to the real deployed encoders once and accepted only when the
# case-equal mean causal attention output MSE strictly decreases.
#
# build.py threads an explicit ``boundaries`` argument through the two
# functions that generate every active Q/K mantissa by exact-text patch;
# ``boundaries = None`` is bit-identical to the parent encoder.
# ---------------------------------------------------------------------------

_ARB1_BOUNDARY_LENGTH = 7
_ARB1_BUCKETS = 64
_ARB1_CODE_STEP = 0.25
_ARB1_CODE_MAX = 7
_ARB1_SHARED_MIN = 1
_ARB1_SHARED_MAX = 6
_ARB1_PARENT_BOUNDARY = 0.5
_ARB1_RADEMACHER_SEED = 0


def _arb1_boundary_mantissa(
    x_abs: torch.Tensor,
    scale: torch.Tensor,
    boundaries: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mantissa under learned lower-code boundaries; None reproduces rounding."""

    u = x_abs * (4.0 / scale)
    if boundaries is None:
        return torch.round(u).clamp_(0.0, 7.0) * _ARB1_CODE_STEP
    tau = boundaries.detach().to(device=u.device, dtype=torch.float32).reshape(-1)
    if int(tau.numel()) != _ARB1_BOUNDARY_LENGTH:
        raise ValueError("boundary table must have 7 entries")
    floor_code = torch.floor(u)
    frac = u - floor_code
    index = floor_code.clamp(
        0.0, float(_ARB1_BOUNDARY_LENGTH - 1)
    ).to(torch.int64)
    tau_view = tau.index_select(0, index.reshape(-1)).reshape_as(u)
    # Exact halves keep torch.round's tie rule so that an all-0.5 table
    # restores the parent five fields bit for bit.
    code = torch.where(
        frac < tau_view,
        floor_code,
        torch.where(frac > tau_view, floor_code + 1.0, torch.round(u)),
    )
    return code.clamp_(0.0, 7.0) * _ARB1_CODE_STEP


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(q_state, q_num_heads, head_dim, "q")
    if int(q_quant.shape[-1]) != q_num_heads * head_dim:
        raise ValueError("Q width does not match q_num_heads * head_dim")
    return _nvfp4_to_hif4(
        q_quant,
        q_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        block_smooth_size=int(state.get("block_smooth_size", 0)),
        block_smooth_seed=int(state.get("block_smooth_seed", 0)),
        attention_rotation=state.get("rotation"),
        rotation_num_heads=int(q_num_heads),
        attention_rotation_block=state.get("rotation_block"),
        attention_block_signs=state.get("block_smooth_signs"),
        attention_pair_transform=state.get("pair_transform"),
        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(q_num_heads),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
        boundaries=state.get("boundaries"),
        capture_dense=state.get("_arb1_capture"),
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(k_state, kv_num_heads, head_dim, "k")
    if int(k_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("K width does not match kv_num_heads * head_dim")
    return _nvfp4_to_hif4(
        k_quant,
        k_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        block_smooth_size=int(state.get("block_smooth_size", 0)),
        block_smooth_seed=int(state.get("block_smooth_seed", 0)),
        attention_rotation=state.get("rotation"),
        rotation_num_heads=int(kv_num_heads),
        attention_rotation_block=state.get("rotation_block"),
        attention_block_signs=state.get("block_smooth_signs"),
        attention_pair_transform=state.get("pair_transform"),
        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(kv_num_heads),
        learned_center=state.get("learned_center"),
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        center_value=state.get("center_value"),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
        boundaries=state.get("boundaries"),
        capture_dense=state.get("_arb1_capture"),
    )


@torch.no_grad()
def _arb1_deployed_role(
    pair: tuple,
    state: dict,
    num_heads: int,
    head_dim: int,
    role: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Deployed parent five fields plus the pre-quantization dense operand."""

    capture: list[torch.Tensor] = []
    state["_arb1_capture"] = capture
    try:
        if role == "q":
            params = hif4_dynamic_quantize_q(
                pair[0], pair[1], num_heads, head_dim, state
            )
        else:
            params = hif4_dynamic_quantize_k(
                pair[0], pair[1], num_heads, head_dim, state
            )
    finally:
        state.pop("_arb1_capture", None)
    if len(capture) != 1:
        raise ValueError("deployed operand capture did not fire exactly once")
    return params, capture[0].to(torch.float32)


@torch.no_grad()
def _arb1_role_view(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-element parent denominator, sign, code, magnitude and nearest code."""

    rows, channels = map(int, dense.shape)
    blocks = channels // _HIF4_BLOCK_SIZE
    scale = params["scale_factor"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 1, 1, 1)
    lv2 = params["scale_lv2"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 1, 1)
    lv3 = params["scale_lv3"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, blocks, 8, 2, 1)
    denominator = (scale * lv2 * lv3).repeat_interleave(4, dim=-1).reshape(
        rows, channels
    )
    sign = params["sign"].to(
        device=dense.device, dtype=torch.float32
    ).reshape(rows, channels)
    code = torch.round(
        params["mant"].to(device=dense.device, dtype=torch.float32).reshape(
            rows, channels
        )
        / _ARB1_CODE_STEP
    ).clamp_(0.0, float(_ARB1_CODE_MAX))
    magnitude = 4.0 * dense.abs() / denominator.clamp_min(_EPS)
    magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    nearest = torch.round(magnitude).clamp(0.0, float(_ARB1_CODE_MAX))
    floor_code = torch.floor(magnitude)
    frac = (magnitude - floor_code).clamp(0.0, 1.0)
    eligible = (
        (code == nearest)
        & (floor_code >= float(_ARB1_SHARED_MIN))
        & (floor_code <= float(_ARB1_SHARED_MAX))
    )
    return {
        "denominator": denominator,
        "sign": sign,
        "code": code,
        "nearest": nearest,
        "floor": floor_code,
        "frac": frac,
        "eligible": eligible,
    }


def _arb1_pick_bucket(total_cost: torch.Tensor) -> int:
    """Argmin over 65 boundary positions, ties toward 0.5 then lower tau."""

    order = torch.argsort(total_cost, stable=True)
    best_cost = total_cost[order[0]]
    tied = order[total_cost[order] == best_cost]
    distance = (tied - _ARB1_BUCKETS // 2).abs()
    return int(tied[int(torch.argmin(distance))])


def _arb1_solve_role(
    stats: list[dict[str, torch.Tensor]],
    gradient: list[torch.Tensor],
    curvature: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """One boundary per lower-code class from the frozen residual statistics."""

    boundaries = torch.full(
        (_ARB1_BOUNDARY_LENGTH,),
        _ARB1_PARENT_BOUNDARY,
        device=device,
        dtype=torch.float32,
    )
    proposed = 0
    for lower in range(_ARB1_SHARED_MIN, _ARB1_SHARED_MAX + 1):
        bucket_cost = torch.zeros(_ARB1_BUCKETS, device=device, dtype=torch.float32)
        touched = 0
        for view, g_value, h_value in zip(stats, gradient, curvature):
            mask = view["eligible"] & (view["floor"] == float(lower))
            if not bool(mask.any()):
                continue
            touched += int(mask.sum())
            step = float(lower) + 1.0 - view["code"][mask]
            delta = (
                view["sign"][mask] * step * view["denominator"][mask]
                * _ARB1_CODE_STEP
            )
            cost = 2.0 * g_value[mask] * delta + h_value[mask] * delta.square()
            bucket_cost.scatter_add_(
                0,
                torch.clamp(
                    torch.floor(view["frac"][mask] * _ARB1_BUCKETS).to(torch.int64),
                    0,
                    _ARB1_BUCKETS - 1,
                ),
                cost,
            )
        if touched == 0:
            continue
        suffix = torch.cumsum(bucket_cost.flip(0), dim=0).flip(0)
        total = torch.cat(
            [suffix, torch.zeros(1, device=device, dtype=torch.float32)]
        )
        best_bucket = _arb1_pick_bucket(total)
        boundaries[lower] = float(best_bucket) / float(_ARB1_BUCKETS)
        if best_bucket != _ARB1_BUCKETS // 2:
            proposed += 1
    return boundaries, proposed, int(
        sum(int(view["eligible"].sum()) for view in stats)
    )


def _arb1_apply_table(
    view: dict[str, torch.Tensor],
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """New absolute codes and the code-changing mask for one window/role."""

    new_code = view["code"].clone()
    changed = torch.zeros_like(view["code"], dtype=torch.bool)
    for lower in range(_ARB1_SHARED_MIN, _ARB1_SHARED_MAX + 1):
        tau = float(boundaries[lower])
        mask = (view["floor"] == float(lower)) & (view["frac"] >= tau)
        if bool(mask.any()):
            new_code[mask] = float(lower) + 1.0
            changed |= mask
    changed &= new_code != view["code"]
    return new_code, changed


def _arb1_boundary_fit(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    states: dict,
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    """Learn tau_q and tau_k from the frozen parent softmax-output residual."""

    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        return None
    q_state = states.get("q_state")
    k_state = states.get("k_state")
    v_state = states.get("v_state")
    if not (
        isinstance(q_state, dict)
        and isinstance(k_state, dict)
        and isinstance(v_state, dict)
    ):
        return None
    q_state.pop("boundaries", None)
    k_state.pop("boundaries", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows: list[dict[str, Any]] = []
    for item in calib_qkv_list:
        if not isinstance(item, dict) or set(item) != {"q", "k", "v"}:
            return None
        q_pair, k_pair, v_pair = item["q"], item["k"], item["v"]
        q_ref = _dequantize_nvfp4_float32(*q_pair).to(device=device, dtype=torch.float32)
        k_ref = _dequantize_nvfp4_float32(*k_pair).to(device=device, dtype=torch.float32)
        v_ref = _dequantize_nvfp4_float32(*v_pair).to(device=device, dtype=torch.float32)
        q_params, q_dense = _arb1_deployed_role(
            q_pair, q_state, q_num_heads, head_dim, "q"
        )
        k_params, k_dense = _arb1_deployed_role(
            k_pair, k_state, kv_num_heads, head_dim, "k"
        )
        q_dense = q_dense.to(device=device, dtype=torch.float32)
        k_dense = k_dense.to(device=device, dtype=torch.float32)
        v_params = hif4_dynamic_quantize_v(
            v_pair[0], v_pair[1], kv_num_heads, head_dim, v_state
        )
        q_hat = _dequantize_hif4(q_params).to(device=device, dtype=torch.float32)
        k_hat = _dequantize_hif4(k_params).to(device=device, dtype=torch.float32)
        v_hat = _dequantize_hif4(v_params).to(device=device, dtype=torch.float32)
        if tuple(q_hat.shape) != tuple(q_ref.shape) or tuple(
            k_hat.shape
        ) != tuple(k_ref.shape) or tuple(v_hat.shape) != tuple(v_ref.shape):
            return None
        windows.append(
            {
                "q_pair": q_pair,
                "k_pair": k_pair,
                "q_ref": q_ref,
                "k_ref": k_ref,
                "v_ref": v_ref,
                "q_hat": q_hat,
                "k_hat": k_hat,
                "v_hat": v_hat,
                "q_params": q_params,
                "k_params": k_params,
                "q_view": _arb1_role_view(q_dense, q_params),
                "k_view": _arb1_role_view(k_dense, k_params),
            }
        )

    case_count = float(len(windows))
    output_width = float(q_num_heads * head_dim)
    generator = torch.Generator(device=device)
    generator.manual_seed(_ARB1_RADEMACHER_SEED)
    gradient: dict[str, list[torch.Tensor]] = {"q": [], "k": []}
    curvature: dict[str, list[torch.Tensor]] = {"q": [], "k": []}
    agreement = {"q": 0, "k": 0}
    elements = {"q": 0, "k": 0}
    parent_causal: list[float] = []
    parent_safety: list[float] = []

    for window in windows:
        q_hat = window["q_hat"]
        k_hat = window["k_hat"]
        v_hat = window["v_hat"]
        tokens = int(q_hat.shape[0])
        weight = 1.0 / (case_count * float(tokens) * output_width)
        out_parent = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
        )
        out_ref = _attention_forward(
            window["q_ref"], window["k_ref"], window["v_ref"],
            q_num_heads, kv_num_heads, head_dim, True,
        )
        out_parent_n = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
        )
        out_ref_n = _attention_forward(
            window["q_ref"], window["k_ref"], window["v_ref"],
            q_num_heads, kv_num_heads, head_dim, False,
        )
        if not bool(
            torch.isfinite(out_parent).all()
            and torch.isfinite(out_ref).all()
            and torch.isfinite(out_parent_n).all()
        ):
            return None
        window["out_parent"] = out_parent.detach()
        parent_causal.append(float((out_parent - out_ref).square().mean()))
        parent_safety.append(float((out_parent_n - out_ref_n).square().mean()))

        probe = (
            torch.randint(
                0,
                2,
                tuple(int(v) for v in out_parent.shape),
                generator=generator,
                device=device,
                dtype=torch.int64,
            ).to(torch.float32)
            * 2.0
            - 1.0
        )
        residual = (out_parent - out_ref).detach()
        with torch.enable_grad():
            q_leaf = q_hat.detach().clone().requires_grad_(True)
            k_leaf = k_hat.detach().clone().requires_grad_(True)
            out = _attention_forward(
                q_leaf, k_leaf, v_hat, q_num_heads, kv_num_heads, head_dim, True
            )
            grad_q, grad_k = torch.autograd.grad(
                out, (q_leaf, k_leaf), grad_outputs=weight * residual,
                retain_graph=True,
            )
            probe_q, probe_k = torch.autograd.grad(
                out, (q_leaf, k_leaf), grad_outputs=weight * probe,
            )
        for role, g_value, h_value in (
            ("q", grad_q, probe_q),
            ("k", grad_k, probe_k),
        ):
            if not bool(
                torch.isfinite(g_value).all() and torch.isfinite(h_value).all()
            ):
                return None
            gradient[role].append(g_value.detach())
            curvature[role].append(h_value.detach().square())
        for role in ("q", "k"):
            view = window[f"{role}_view"]
            agreement[role] += int((view["code"] == view["nearest"]).sum())
            elements[role] += int(view["code"].numel())

    diagnostics.update(
        {
            "arb1_attempted": 1,
            "arb1_coordinate": "qk-softmax-output-shared-rounding-boundary",
            "arb1_windows": len(windows),
            "arb1_elements_q": elements["q"],
            "arb1_elements_k": elements["k"],
            "arb1_rounding_agreement_q": agreement["q"],
            "arb1_rounding_agreement_k": agreement["k"],
        }
    )

    tables: dict[str, torch.Tensor] = {}
    proposals = 0
    eligible_counts = {"q": 0, "k": 0}
    fixed_changed = {"q": 0, "k": 0}
    for role in ("q", "k"):
        stats = [window[f"{role}_view"] for window in windows]
        boundaries, proposed, eligible_total = _arb1_solve_role(
            stats, gradient[role], curvature[role], device
        )
        tables[role] = boundaries
        proposals += proposed
        eligible_counts[role] = eligible_total
        changed_total = 0
        for view in stats:
            _, changed = _arb1_apply_table(view, boundaries)
            changed_total += int(changed.sum())
        fixed_changed[role] = changed_total

    diagnostics.update(
        {
            "arb1_proposals": proposals,
            "arb1_eligible_q": eligible_counts["q"],
            "arb1_eligible_k": eligible_counts["k"],
            "arb1_fixed_hierarchy_changed_q": fixed_changed["q"],
            "arb1_fixed_hierarchy_changed_k": fixed_changed["k"],
            "arb1_parent_causal_mse": float(sum(parent_causal) / len(parent_causal)),
            "arb1_parent_safety_mse": float(sum(parent_safety) / len(parent_safety)),
        }
    )
    if fixed_changed["q"] == 0 and fixed_changed["k"] == 0:
        diagnostics["arb1_arm"] = "parent"
        diagnostics["arb1_reason"] = "no-boundary-proposal"
        return None

    # Fixed-hierarchy reading: the table applied to the parent five fields.
    fixed_causal: list[float] = []
    fixed_output_delta = 0.0
    for window in windows:
        q_new = window["q_hat"].clone()
        k_new = window["k_hat"].clone()
        for role, target in (("q", q_new), ("k", k_new)):
            view = window[f"{role}_view"]
            new_code, changed = _arb1_apply_table(view, tables[role])
            delta = (
                view["sign"]
                * (new_code - view["code"])
                * view["denominator"]
                * _ARB1_CODE_STEP
            )
            target += torch.where(changed, delta, torch.zeros_like(delta))
        out_fixed = _attention_forward(
            q_new, k_new, window["v_hat"], q_num_heads, kv_num_heads, head_dim, True
        )
        fixed_causal.append(
            float(
                (
                    out_fixed
                    - _attention_forward(
                        window["q_ref"], window["k_ref"], window["v_ref"],
                        q_num_heads, kv_num_heads, head_dim, True,
                    )
                )
                .square()
                .mean()
            )
        )
        fixed_output_delta = max(
            fixed_output_delta,
            float((out_fixed - window["out_parent"]).abs().max()),
        )
    diagnostics["arb1_fixed_causal_mse"] = float(
        sum(fixed_causal) / len(fixed_causal)
    )
    diagnostics["arb1_fixed_output_delta"] = fixed_output_delta

    # Reselected-hierarchy reading: the real deployed encoders under the table.
    q_state["boundaries"] = tables["q"].detach().to(device="cpu")
    k_state["boundaries"] = tables["k"].detach().to(device="cpu")
    reselected_causal: list[float] = []
    reselected_safety: list[float] = []
    field_changed = {"q": 0, "k": 0}
    decoded_delta = {"q": 0.0, "k": 0.0}
    output_delta = 0.0
    for window in windows:
        q_params = hif4_dynamic_quantize_q(
            window["q_pair"][0], window["q_pair"][1],
            q_num_heads, head_dim, q_state,
        )
        k_params = hif4_dynamic_quantize_k(
            window["k_pair"][0], window["k_pair"][1],
            kv_num_heads, head_dim, k_state,
        )
        q_new = _dequantize_hif4(q_params).to(device=device, dtype=torch.float32)
        k_new = _dequantize_hif4(k_params).to(device=device, dtype=torch.float32)
        out_new = _attention_forward(
            q_new, k_new, window["v_hat"], q_num_heads, kv_num_heads, head_dim, True
        )
        out_new_n = _attention_forward(
            q_new, k_new, window["v_hat"], q_num_heads, kv_num_heads, head_dim, False
        )
        reselected_causal.append(
            float(
                (
                    out_new
                    - _attention_forward(
                        window["q_ref"], window["k_ref"], window["v_ref"],
                        q_num_heads, kv_num_heads, head_dim, True,
                    )
                )
                .square()
                .mean()
            )
        )
        reselected_safety.append(
            float(
                (
                    out_new_n
                    - _attention_forward(
                        window["q_ref"], window["k_ref"], window["v_ref"],
                        q_num_heads, kv_num_heads, head_dim, False,
                    )
                )
                .square()
                .mean()
            )
        )
        output_delta = max(
            output_delta, float((out_new - window["out_parent"]).abs().max())
        )
        for role, params in (("q", q_params), ("k", k_params)):
            parent_params = window[f"{role}_params"]
            for key in parent_params:
                new_field = params[key].to(device=device, dtype=torch.float32)
                old_field = parent_params[key].to(device=device, dtype=torch.float32)
                if not torch.equal(new_field, old_field):
                    field_changed[role] += int((new_field != old_field).sum())
            decoded_delta[role] = max(
                decoded_delta[role],
                float(
                    (
                        _dequantize_hif4(params).to(device=device, dtype=torch.float32)
                        - _dequantize_hif4(parent_params).to(
                            device=device, dtype=torch.float32
                        )
                    )
                    .abs()
                    .max()
                ),
            )

    parent_mean = float(sum(parent_causal) / len(parent_causal))
    candidate_mean = float(sum(reselected_causal) / len(reselected_causal))
    diagnostics.update(
        {
            "arb1_reselected_field_changed_q": field_changed["q"],
            "arb1_reselected_field_changed_k": field_changed["k"],
            "arb1_reselected_decoded_delta_q": decoded_delta["q"],
            "arb1_reselected_decoded_delta_k": decoded_delta["k"],
            "arb1_reselected_output_delta": output_delta,
            "arb1_candidate_causal_mse": candidate_mean,
            "arb1_candidate_safety_mse": float(
                sum(reselected_safety) / len(reselected_safety)
            ),
            "arb1_causal_delta": candidate_mean - parent_mean,
        }
    )
    if field_changed["q"] == 0 and field_changed["k"] == 0:
        q_state.pop("boundaries", None)
        k_state.pop("boundaries", None)
        diagnostics["arb1_arm"] = "parent"
        diagnostics["arb1_reason"] = "ABSORBED_BY_HIERARCHY"
        return None
    if not math.isfinite(candidate_mean) or not candidate_mean < parent_mean:
        q_state.pop("boundaries", None)
        k_state.pop("boundaries", None)
        diagnostics["arb1_arm"] = "parent"
        diagnostics["arb1_reason"] = "no-causal-improvement"
        return None
    diagnostics["arb1_arm"] = "accepted"
    return {
        "q_boundaries": tables["q"].detach().to(device="cpu"),
        "k_boundaries": tables["k"].detach().to(device="cpu"),
    }


@torch.no_grad()
def _arb1_postprocess(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    states: dict,
) -> dict:
    diagnostics: dict[str, Any] = {
        "arb1_attempted": 0,
        "arb1_arm": "unavailable",
    }
    if not isinstance(states, dict):
        return states
    q_state = states.get("q_state")
    k_state = states.get("k_state")
    if not isinstance(q_state, dict) or not isinstance(k_state, dict):
        return states
    learned = None
    try:
        learned = _arb1_boundary_fit(
            calib_qkv_list, q_num_heads, kv_num_heads, head_dim, states, diagnostics
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        learned = None
    if learned is None:
        q_state.pop("boundaries", None)
        k_state.pop("boundaries", None)
    q_state.update(diagnostics)
    k_state.update(diagnostics)
    if diagnostics.get("arb1_attempted"):
        def _text(table: Any) -> str:
            if not torch.is_tensor(table):
                return "n/a"
            return ",".join(f"{float(v):.3f}" for v in table.reshape(-1).tolist())

        print(
            f"[A-RB1] arm={diagnostics['arb1_arm']} "
            f"windows={diagnostics.get('arb1_windows')} "
            f"proposals={diagnostics.get('arb1_proposals')} "
            f"eligible={diagnostics.get('arb1_eligible_q')}/"
            f"{diagnostics.get('arb1_eligible_k')} "
            f"fixed_changed={diagnostics.get('arb1_fixed_hierarchy_changed_q')}/"
            f"{diagnostics.get('arb1_fixed_hierarchy_changed_k')} "
            f"reselected_changed={diagnostics.get('arb1_reselected_field_changed_q')}/"
            f"{diagnostics.get('arb1_reselected_field_changed_k')} "
            f"causal_delta={float(diagnostics.get('arb1_causal_delta', 0.0)):.6e} "
            f"tau_q=[{_text(q_state.get('boundaries'))}] "
            f"tau_k=[{_text(k_state.get('boundaries'))}]",
            flush=True,
        )
    return states


_ARB1_PARENT_ATTENTION_CALIBRATION = hif4_calibration_attention


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """A-RB1: v195 attention root plus shared Q/K output rounding boundaries."""

    states = _ARB1_PARENT_ATTENTION_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    return _arb1_postprocess(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim, states
    )
