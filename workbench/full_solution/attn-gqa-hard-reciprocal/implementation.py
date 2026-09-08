# ---------------------------------------------------------------------------
# v199: GQA x 64-block hard reciprocal coordinate search.
#
# The parent R3/v195 Q/K path is left intact.  Calibration searches one
# scalar reciprocal coordinate per (KV group, 64-channel head block).  For
# each coordinate only the first positive and negative hard-code boundary is
# proposed.  The proposals are evaluated through the real HiF4 encoder and
# causal attention output; at most one block is retained per KV group.  The
# final combined coordinate is checked on the held-out calibration window.
# V is never changed.
# ---------------------------------------------------------------------------

_V199_U_BOUND = math.log(2.0)
_V199_BOUNDARY_STEPS = 14
_V199_BOUNDARY_MARGIN = 2.0e-4
_V199_OUTPUT_TOKENS = 256
_V199_BLOCK = 64


def _v199_clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in state.items()
    }


def _v199_annotate(
    states: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    result = {
        "q_state": _v199_clone_state(states["q_state"]),
        "k_state": _v199_clone_state(states["k_state"]),
        "v_state": states["v_state"],
    }
    result["q_state"].update(audit)
    result["k_state"].update(audit)
    return result


def _v199_parent_dense(
    dense: torch.Tensor,
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
    *,
    is_k: bool,
) -> torch.Tensor:
    """Return the continuous tensor immediately before the new diagonal."""

    transformed = _attention_state_transform_dense(
        dense, state, num_heads, head_dim, is_k=is_k
    )
    learned_rotation = state.get("learned_rotation")
    if learned_rotation is not None:
        transformed = _a2_apply_group_rotation(
            transformed, num_heads, learned_rotation
        )
    learned_center = state.get("learned_center")
    if learned_center is not None:
        lead = transformed.shape[:-1]
        center = learned_center.to(
            device=transformed.device, dtype=torch.float32
        ).reshape(*([1] * len(lead)), int(num_heads), int(head_dim))
        transformed = (
            transformed.reshape(*lead, int(num_heads), int(head_dim)) + center
        ).reshape_as(transformed)
    return transformed


def _v199_encode_dense(
    dense: torch.Tensor, state: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Mirror the parent dynamic Q/K encoder after its continuous prefix."""

    importance = state.get("importance")
    if _ACTIVATION_SAMPLE_IMPORTANCE and dense.ndim == 2:
        importance = torch.sqrt(
            dense.square().mean(dim=0).clamp_min(_EPS)
        )
    return _dense_to_hif4(
        dense,
        importance=importance,
        search_offsets=state.get("offsets"),
        error_threshold=float(state.get("error_threshold", 0.0)),
        accept_margin=float(state.get("accept_margin", 0.0)),
        max_refine_ratio=float(state.get("max_refine_ratio", 0.0)),
        max_refine_blocks=int(state.get("max_refine_blocks", 0)),
    )


def _v199_encode_selected(
    dense: torch.Tensor,
    state: dict[str, Any],
    channel_indices: torch.Tensor,
    *,
    importance_indices: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Encode only independent 64-blocks used by a boundary test."""

    selected = dense.index_select(-1, channel_indices)
    importance = state.get("importance")
    if torch.is_tensor(importance):
        flat = importance.detach().reshape(-1)
        if importance_indices is not None and int(importance_indices.numel()) == int(selected.shape[-1]):
            importance = flat.index_select(
                0,
                importance_indices.to(device=flat.device),
            )
        elif int(flat.numel()) == int(selected.shape[-1]):
            importance = flat
    return _dense_to_hif4(
        selected,
        importance=importance,
        search_offsets=state.get("offsets"),
        error_threshold=float(state.get("error_threshold", 0.0)),
        accept_margin=float(state.get("accept_margin", 0.0)),
        max_refine_ratio=float(state.get("max_refine_ratio", 0.0)),
        max_refine_blocks=int(state.get("max_refine_blocks", 0)),
    )


def _v199_block_indices(
    group_index: int,
    block_index: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = int(q_num_heads) // int(kv_num_heads)
    blocks_per_head = int(head_dim) // _V199_BLOCK
    q_heads = range(
        int(group_index) * group_size,
        (int(group_index) + 1) * group_size,
    )
    q_blocks = [
        int(head) * blocks_per_head + int(block_index) for head in q_heads
    ]
    k_blocks = [
        int(group_index) * blocks_per_head + int(block_index)
    ]
    q_channels = []
    for block in q_blocks:
        start = int(block) * _V199_BLOCK
        q_channels.extend(range(start, start + _V199_BLOCK))
    k_block = k_blocks[0]
    k_start = int(k_block) * _V199_BLOCK
    q_indices = torch.tensor(q_channels, dtype=torch.int64, device=device)
    k_indices = torch.arange(
        k_start, k_start + _V199_BLOCK, dtype=torch.int64, device=device
    )
    q_block_indices = torch.tensor(
        q_blocks, dtype=torch.int64, device=device
    )
    k_block_indices = torch.tensor(
        k_blocks, dtype=torch.int64, device=device
    )
    return q_indices, k_indices, q_block_indices, k_block_indices


def _v199_changed(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    parent_block_indices: torch.Tensor,
) -> bool:
    for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
        reference = parent[key].index_select(1, parent_block_indices)
        if not torch.equal(reference, candidate[key]):
            return True
    return False


def _v199_boundary_changed(
    item: dict[str, Any],
    group_index: int,
    block_index: int,
    signed_u: float,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> bool:
    q_indices, k_indices, q_blocks, k_blocks = _v199_block_indices(
        group_index,
        block_index,
        q_num_heads,
        kv_num_heads,
        head_dim,
        item["q_boundary"].device,
    )
    factor_q = math.exp(float(signed_u))
    factor_k = math.exp(-float(signed_u))
    q_candidate = _v199_encode_selected(
        item["q_boundary"].index_select(-1, q_indices) * factor_q,
        item["q_state"],
        torch.arange(
            int(q_indices.numel()),
            dtype=torch.int64,
            device=q_indices.device,
        ),
        importance_indices=q_indices,
    )
    k_candidate = _v199_encode_selected(
        item["k_boundary"].index_select(-1, k_indices) * factor_k,
        item["k_state"],
        torch.arange(
            int(k_indices.numel()),
            dtype=torch.int64,
            device=k_indices.device,
        ),
        importance_indices=k_indices,
    )
    return _v199_changed(
        item["q_boundary_parent"], q_candidate, q_blocks
    ) or _v199_changed(
        item["k_boundary_parent"], k_candidate, k_blocks
    )


def _v199_first_boundary(
    items: list[dict[str, Any]],
    group_index: int,
    block_index: int,
    direction: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Optional[float]:
    """Locate the first hard-code change in one reciprocal direction."""

    hi = _V199_U_BOUND
    for item in items:
        if _v199_boundary_changed(
            item,
            group_index,
            block_index,
            int(direction) * hi,
            q_num_heads,
            kv_num_heads,
            head_dim,
        ):
            break
    else:
        return None

    lo = 0.0
    for _ in range(_V199_BOUNDARY_STEPS):
        mid = 0.5 * (lo + hi)
        changed = False
        for item in items:
            if _v199_boundary_changed(
                item,
                group_index,
                block_index,
                int(direction) * mid,
                q_num_heads,
                kv_num_heads,
                head_dim,
            ):
                changed = True
                break
        if changed:
            hi = mid
        else:
            lo = mid

    step = min(
        _V199_U_BOUND,
        hi + max(_V199_BOUNDARY_MARGIN, hi * 1.0e-3),
    )
    # A boundary can fall on a round-to-even tie.  The fixed margin is only
    # to move the proposal onto the changed side, not a second scale search.
    for item in items:
        if _v199_boundary_changed(
            item,
            group_index,
            block_index,
            int(direction) * step,
            q_num_heads,
            kv_num_heads,
            head_dim,
        ):
            return float(direction) * step
    if step < _V199_U_BOUND:
        step = min(_V199_U_BOUND, step + _V199_BOUNDARY_MARGIN)
        for item in items:
            if _v199_boundary_changed(
                item,
                group_index,
                block_index,
                int(direction) * step,
                q_num_heads,
                kv_num_heads,
                head_dim,
            ):
                return float(direction) * step
    return None


def _v199_factor_vectors(
    u: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks_per_head = int(head_dim) // _V199_BLOCK
    group_size = int(q_num_heads) // int(kv_num_heads)
    q_factors = torch.ones(
        int(q_num_heads) * int(head_dim), dtype=torch.float32, device=device
    )
    k_factors = torch.ones(
        int(kv_num_heads) * int(head_dim), dtype=torch.float32, device=device
    )
    for group_index in range(int(kv_num_heads)):
        for block_index in range(blocks_per_head):
            value = float(u[group_index, block_index])
            if value == 0.0:
                continue
            q_value = math.exp(value)
            k_value = math.exp(-value)
            for q_head in range(
                group_index * group_size, (group_index + 1) * group_size
            ):
                start = q_head * int(head_dim) + block_index * _V199_BLOCK
                q_factors[start : start + _V199_BLOCK] = q_value
            start = group_index * int(head_dim) + block_index * _V199_BLOCK
            k_factors[start : start + _V199_BLOCK] = k_value
    return q_factors, k_factors


def _v199_attention_mse(
    item: dict[str, Any],
    u: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> float:
    q_factors, k_factors = _v199_factor_vectors(
        u,
        q_num_heads,
        kv_num_heads,
        head_dim,
        item["q_eval"].device,
    )
    q_params = _v199_encode_dense(
        item["q_eval_parent"] * q_factors.reshape(1, -1),
        item["q_state"],
    )
    k_params = _v199_encode_dense(
        item["k_eval_parent"] * k_factors.reshape(1, -1),
        item["k_state"],
    )
    q_hat = _dequantize_hif4(q_params).to(torch.float32)
    k_hat = _dequantize_hif4(k_params).to(torch.float32)
    output = _attention_forward(
        q_hat,
        k_hat,
        item["v_hat"],
        q_num_heads,
        kv_num_heads,
        head_dim,
        True,
    )
    return float((output - item["reference"]).square().mean())


def _v199_prepare_items(
    calib_qkv_list: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    q_state = states["q_state"]
    k_state = states["k_state"]
    v_state = states["v_state"]
    for sample in calib_qkv_list:
        q_raw = _dequantize_nvfp4_float32(*sample["q"]).to(torch.float32)
        k_raw = _dequantize_nvfp4_float32(*sample["k"]).to(torch.float32)
        v_raw = _dequantize_nvfp4_float32(*sample["v"]).to(torch.float32)
        q_parent = _v199_parent_dense(
            q_raw, q_state, q_num_heads, head_dim, is_k=False
        )
        k_parent = _v199_parent_dense(
            k_raw, k_state, kv_num_heads, head_dim, is_k=True
        )
        prefix = min(int(q_raw.shape[0]), _V199_OUTPUT_TOKENS)
        q_eval_parent = q_parent[:prefix]
        k_eval_parent = k_parent[:prefix]
        v_quant, v_scale = sample["v"]
        v_params = _V199_PARENT_DYNAMIC_V(
            v_quant[:prefix],
            v_scale[:prefix],
            kv_num_heads,
            head_dim,
            v_state,
        )
        v_hat = _dequantize_hif4(v_params).to(torch.float32)
        reference = _attention_forward(
            q_raw[:prefix],
            k_raw[:prefix],
            v_raw[:prefix],
            q_num_heads,
            kv_num_heads,
            head_dim,
            True,
        )
        q_boundary = q_eval_parent
        k_boundary = k_eval_parent
        q_boundary_parent = _v199_encode_dense(q_boundary, q_state)
        k_boundary_parent = _v199_encode_dense(k_boundary, k_state)
        items.append(
            {
                "q_state": q_state,
                "k_state": k_state,
                "q_eval": q_raw[:prefix],
                "k_eval": k_raw[:prefix],
                "q_eval_parent": q_eval_parent,
                "k_eval_parent": k_eval_parent,
                "q_boundary": q_boundary,
                "k_boundary": k_boundary,
                "q_boundary_parent": q_boundary_parent,
                "k_boundary_parent": k_boundary_parent,
                "v_hat": v_hat,
                "reference": reference,
            }
        )
    return items


def _v199_candidate_u(
    items: list[dict[str, Any]],
    group_index: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[list[tuple[float, int]], int]:
    blocks_per_head = int(head_dim) // _V199_BLOCK
    candidates: list[tuple[float, int]] = []
    attempted = 0
    for block_index in range(blocks_per_head):
        for direction in (-1, 1):
            step = _v199_first_boundary(
                items,
                group_index,
                block_index,
                direction,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            if step is not None:
                candidates.append((float(step), int(block_index)))
                attempted += 1
    return candidates, attempted


_V199_PARENT_CALIBRATION = hif4_calibration_attention
_V199_PARENT_DYNAMIC_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Current parent plus one hard reciprocal coordinate per KV group."""

    states = _V199_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    audit: dict[str, Any] = {
        "v199_arm": "fallback",
        "v199_attempted": 0,
        "v199_accepted": 0,
        "v199_fit_windows": 0,
        "v199_selected_groups": 0,
        "v199_selected_blocks": 0,
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < 2
        or int(q_num_heads) % int(kv_num_heads) != 0
        or int(head_dim) % _V199_BLOCK != 0
    ):
        audit["v199_arm"] = "ineligible"
        return _v199_annotate(states, audit)

    try:
        items = _v199_prepare_items(
            calib_qkv_list,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        if not items:
            audit["v199_arm"] = "ineligible"
            return _v199_annotate(states, audit)
        fit_items = items[:-1] or items
        holdout_item = items[-1]
        audit["v199_fit_windows"] = int(len(fit_items))
        device = fit_items[0]["q_eval_parent"].device
        zero_u = torch.zeros(
            int(kv_num_heads),
            int(head_dim) // _V199_BLOCK,
            dtype=torch.float32,
            device=device,
        )
        fit_parent_losses = [
            _v199_attention_mse(
                item, zero_u, q_num_heads, kv_num_heads, head_dim
            )
            for item in fit_items
        ]
        fit_parent = sum(fit_parent_losses) / float(len(fit_parent_losses))
        selected_u = zero_u.clone()
        selected_blocks = 0
        selected_fit_losses: list[float] = []
        for group_index in range(int(kv_num_heads)):
            proposals, attempted = _v199_candidate_u(
                fit_items,
                group_index,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            audit["v199_attempted"] += int(attempted)
            best_loss = fit_parent
            best_u = 0.0
            best_block = -1
            for proposal, block_index in proposals:
                trial_u = zero_u.clone()
                trial_u[group_index, block_index] = float(proposal)
                losses = [
                    _v199_attention_mse(
                        item, trial_u, q_num_heads, kv_num_heads, head_dim
                    )
                    for item in fit_items
                ]
                trial_loss = sum(losses) / float(len(losses))
                if math.isfinite(trial_loss) and trial_loss < best_loss:
                    best_loss = trial_loss
                    best_u = float(proposal)
                    best_block = int(block_index)
            if best_block >= 0:
                selected_u[group_index, best_block] = best_u
                selected_blocks += 1
                selected_fit_losses.append(float(best_loss))

        fit_candidate = (
            sum(selected_fit_losses) / float(len(selected_fit_losses))
            if selected_fit_losses
            else fit_parent
        )
        holdout_parent = _v199_attention_mse(
            holdout_item, zero_u, q_num_heads, kv_num_heads, head_dim
        )
        holdout_candidate = _v199_attention_mse(
            holdout_item, selected_u, q_num_heads, kv_num_heads, head_dim
        )
        audit.update(
            {
                "v199_fit_parent_mse": float(fit_parent),
                "v199_fit_candidate_mse": float(fit_candidate),
                "v199_holdout_parent_mse": float(holdout_parent),
                "v199_holdout_candidate_mse": float(holdout_candidate),
                "v199_selected_blocks": int(selected_blocks),
                "v199_u_max_abs": float(selected_u.abs().max()),
                "v199_u_nonzero": int((selected_u != 0).sum()),
            }
        )
        if (
            selected_blocks > 0
            and math.isfinite(holdout_candidate)
            and math.isfinite(holdout_parent)
            and holdout_candidate < holdout_parent
        ):
            q_factors, k_factors = _v199_factor_vectors(
                selected_u,
                q_num_heads,
                kv_num_heads,
                head_dim,
                device,
            )
            q_state = _v199_clone_state(states["q_state"])
            k_state = _v199_clone_state(states["k_state"])
            q_state["diag_scale"] = _cpu_state_tensor(q_factors)
            k_state["diag_scale"] = _cpu_state_tensor(k_factors)
            q_state["v199_u"] = _cpu_state_tensor(selected_u)
            k_state["v199_u"] = _cpu_state_tensor(selected_u.clone())
            audit["v199_arm"] = "accepted"
            audit["v199_accepted"] = 1
            audit["v199_selected_groups"] = int(selected_blocks)
            return _v199_annotate(
                {
                    "q_state": q_state,
                    "k_state": k_state,
                    "v_state": states["v_state"],
                },
                audit,
            )
        audit["v199_arm"] = (
            "parent" if int(audit["v199_attempted"]) > 0 else "no_boundary"
        )
        return _v199_annotate(states, audit)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        audit["v199_arm"] = "fallback"
        audit["v199_error"] = f"{type(exc).__name__}: {exc}"
        return _v199_annotate(states, audit)


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(q_state, q_num_heads, head_dim, "q")
    if int(q_quant.shape[-1]) != int(q_num_heads) * int(head_dim):
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
        attention_diag_scale=state.get("diag_scale"),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
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
    if int(k_quant.shape[-1]) != int(kv_num_heads) * int(head_dim):
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
        attention_diag_scale=state.get("diag_scale"),
        center_mode=int(state["center_mode"]),
        center_num_heads=int(kv_num_heads),
        center_head_dim=int(head_dim),
        center_value=state.get("center_value"),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )
