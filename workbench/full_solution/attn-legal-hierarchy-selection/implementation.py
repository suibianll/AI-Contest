# ---------------------------------------------------------------------------
# A4 / v203: joint legal Q/K hierarchy-neighbor selection.
#
# The parent Q/K continuous transform is frozen.  Calibration proposes one
# adjacent E6M2 scale-factor code in each direction for each GQA group/block,
# recomputes the legal lv2/lv3 hierarchy and mantissa once, and scores the
# resulting hard Q/K pair through the real causal Attention output.  At most
# one proposal is retained per KV group.  Deployment applies the same fixed
# per-64-block code offset to the current input's parent scale code; no
# continuous reciprocal coordinate, STE, or dynamic search is introduced.
# ---------------------------------------------------------------------------

_A4_BLOCK = 64
_A4_OUTPUT_TOKENS = 256
_A4_ROLES = ("q", "k", "qk")
_A4_DIRECTIONS = (-1, 1)


def _a4_clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in state.items()
    }


def _a4_annotate(
    states: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    result = {
        "q_state": _a4_clone_state(states["q_state"]),
        "k_state": _a4_clone_state(states["k_state"]),
        "v_state": states["v_state"],
    }
    result["q_state"].update(audit)
    result["k_state"].update(audit)
    return result


@torch.no_grad()
def _a4_parent_dense(
    dense: torch.Tensor,
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
    *,
    is_k: bool,
) -> torch.Tensor:
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


@torch.no_grad()
def _a4_encode_dense(
    dense: torch.Tensor, state: dict[str, Any]
) -> dict[str, torch.Tensor]:
    return _dense_to_hif4(
        dense,
        importance=state.get("importance"),
        search_offsets=state.get("offsets"),
        error_threshold=float(state.get("error_threshold", 0.0)),
        accept_margin=float(state.get("accept_margin", 0.0)),
        max_refine_ratio=float(state.get("max_refine_ratio", 0.0)),
        max_refine_blocks=int(state.get("max_refine_blocks", 0)),
    )


@torch.no_grad()
def _a4_apply_hierarchy_offsets(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    offsets: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-encode selected 64-blocks at an adjacent legal scale-factor code."""

    channels = int(dense.shape[-1])
    if channels % _A4_BLOCK != 0:
        raise ValueError("A4 width must be divisible by 64")
    blocks = channels // _A4_BLOCK
    offset = offsets.detach().to(device=dense.device, dtype=torch.int64).reshape(-1)
    if int(offset.numel()) != blocks:
        raise ValueError("A4 hierarchy offset width does not match tensor")
    selected = torch.nonzero(offset != 0, as_tuple=False).reshape(-1)
    if int(selected.numel()) == 0:
        return params

    prefix = tuple(int(value) for value in dense.shape[:-1])
    rows = 1
    for value in prefix:
        rows *= int(value)
    dense_flat = dense.reshape(rows, blocks, 8, 2, 4)
    parent_scale = params["scale_factor"].reshape(rows, blocks)
    parent_code = _e6m2_encode_nearest(parent_scale)
    candidate_code = (
        parent_code.index_select(1, selected)
        + offset.index_select(0, selected).reshape(1, -1)
    ).clamp(min=0, max=254)
    candidate_scale = _e6m2_decode(candidate_code)
    x_selected = dense_flat.index_select(1, selected)
    x_abs = x_selected.abs()
    sign = torch.sign(x_selected)

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is None:
        importance_selected = None
    else:
        channel_importance = channel_importance.to(
            device=dense.device, dtype=torch.float32
        )
        importance_selected = (
            channel_importance.reshape(blocks, 8, 2, 4)
            .index_select(0, selected)
            .unsqueeze(0)
            .expand(rows, -1, 8, 2, 4)
        )
    _, scale_lv2, scale_lv3, mantissa = _solve_exact_hierarchy(
        x_abs,
        candidate_scale,
        importance_selected,
        sign,
        None,
    )
    sign = torch.where(mantissa == 0.0, torch.zeros_like(sign), sign)
    replacements = {
        "scale_factor": candidate_scale.reshape(rows, -1, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(rows, -1, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(rows, -1, 8, 2, 1),
        "sign": sign,
        "mant": mantissa,
    }
    result = {key: value.clone() for key, value in params.items()}
    for key, replacement in replacements.items():
        flat = result[key].reshape(rows, blocks, *result[key].shape[2:])
        flat.index_copy_(1, selected, replacement)
        result[key] = flat.reshape_as(result[key])
    return result


def _a4_prepare_items(
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
        q_parent = _a4_parent_dense(
            q_raw, q_state, q_num_heads, head_dim, is_k=False
        )
        k_parent = _a4_parent_dense(
            k_raw, k_state, kv_num_heads, head_dim, is_k=True
        )
        prefix = min(int(q_raw.shape[0]), _A4_OUTPUT_TOKENS)
        q_parent = q_parent[:prefix]
        k_parent = k_parent[:prefix]
        q_params = _a4_encode_dense(q_parent, q_state)
        k_params = _a4_encode_dense(k_parent, k_state)
        v_quant, v_scale = sample["v"]
        v_params = _A4_PARENT_DYNAMIC_V(
            v_quant[:prefix],
            v_scale[:prefix],
            kv_num_heads,
            head_dim,
            v_state,
        )
        items.append(
            {
                "q_dense": q_parent,
                "k_dense": k_parent,
                "q_params": q_params,
                "k_params": k_params,
                "q_hat": _dequantize_hif4(q_params).to(torch.float32),
                "k_hat": _dequantize_hif4(k_params).to(torch.float32),
                "v_hat": _dequantize_hif4(v_params).to(torch.float32),
                "reference": _attention_forward(
                    q_raw[:prefix],
                    k_raw[:prefix],
                    v_raw[:prefix],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    True,
                ),
            }
        )
    return items


@torch.no_grad()
def _a4_attention_loss(
    item: dict[str, Any],
    q_state: dict[str, Any],
    k_state: dict[str, Any],
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> float:
    q_params = _a4_apply_hierarchy_offsets(
        item["q_dense"], item["q_params"], q_offsets, q_state.get("importance")
    )
    k_params = _a4_apply_hierarchy_offsets(
        item["k_dense"], item["k_params"], k_offsets, k_state.get("importance")
    )
    output = _attention_forward(
        _dequantize_hif4(q_params).to(torch.float32),
        _dequantize_hif4(k_params).to(torch.float32),
        item["v_hat"],
        q_num_heads,
        kv_num_heads,
        head_dim,
        True,
    )
    return float((output - item["reference"]).square().mean())


def _a4_proposals(
    group_index: int,
    block_index: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> list[tuple[str, int, torch.Tensor, torch.Tensor]]:
    group_size = int(q_num_heads) // int(kv_num_heads)
    blocks_per_head = int(head_dim) // _A4_BLOCK
    proposals = []
    for role in _A4_ROLES:
        for direction in _A4_DIRECTIONS:
            q_offsets = torch.zeros(
                int(q_num_heads) * blocks_per_head,
                dtype=torch.int8,
                device=device,
            )
            k_offsets = torch.zeros(
                int(kv_num_heads) * blocks_per_head,
                dtype=torch.int8,
                device=device,
            )
            if role in ("q", "qk"):
                for head in range(
                    int(group_index) * group_size,
                    (int(group_index) + 1) * group_size,
                ):
                    q_offsets[head * blocks_per_head + int(block_index)] = int(
                        direction
                    )
            if role in ("k", "qk"):
                k_offsets[
                    int(group_index) * blocks_per_head + int(block_index)
                ] = int(direction)
            proposals.append((role, direction, q_offsets, k_offsets))
    return proposals


_A4_PARENT_CALIBRATION = hif4_calibration_attention
_A4_PARENT_DYNAMIC_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    states = _A4_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    audit: dict[str, Any] = {
        "a4_arm": "fallback",
        "a4_attempted": 0,
        "a4_ranked": 0,
        "a4_accepted": 0,
        "a4_fit_windows": 0,
        "a4_selected_groups": 0,
        "a4_selected_blocks": 0,
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
        or int(head_dim) % _A4_BLOCK != 0
    ):
        audit["a4_arm"] = "ineligible"
        return _a4_annotate(states, audit)

    try:
        items = _a4_prepare_items(
            calib_qkv_list,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        if not items:
            audit["a4_arm"] = "ineligible"
            return _a4_annotate(states, audit)
        fit_items = items[:-1] or items
        holdout_item = items[-1]
        audit["a4_fit_windows"] = int(len(fit_items))
        device = fit_items[0]["q_dense"].device
        q_blocks = int(q_num_heads) * int(head_dim) // _A4_BLOCK
        k_blocks = int(kv_num_heads) * int(head_dim) // _A4_BLOCK
        selected_q = torch.zeros(q_blocks, dtype=torch.int8, device=device)
        selected_k = torch.zeros(k_blocks, dtype=torch.int8, device=device)
        zero_q = selected_q.clone()
        zero_k = selected_k.clone()
        fit_parent = sum(
            _a4_attention_loss(
                item,
                states["q_state"],
                states["k_state"],
                zero_q,
                zero_k,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            for item in fit_items
        ) / float(len(fit_items))
        selected_fit = fit_parent
        selected_groups = 0
        ranked_count = 0
        blocks_per_head = int(head_dim) // _A4_BLOCK
        for group_index in range(int(kv_num_heads)):
            best_loss = selected_fit
            best_q = None
            best_k = None
            for block_index in range(blocks_per_head):
                for role, direction, proposal_q, proposal_k in _a4_proposals(
                    group_index,
                    block_index,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    device,
                ):
                    del role, direction
                    trial_q = selected_q.clone()
                    trial_k = selected_k.clone()
                    trial_q = torch.where(
                        proposal_q != 0, proposal_q, trial_q
                    )
                    trial_k = torch.where(
                        proposal_k != 0, proposal_k, trial_k
                    )
                    losses = [
                        _a4_attention_loss(
                            item,
                            states["q_state"],
                            states["k_state"],
                            trial_q,
                            trial_k,
                            q_num_heads,
                            kv_num_heads,
                            head_dim,
                        )
                        for item in fit_items
                    ]
                    trial_loss = sum(losses) / float(len(losses))
                    audit["a4_attempted"] += 1
                    ranked_count += 1
                    if math.isfinite(trial_loss) and trial_loss < best_loss:
                        best_loss = trial_loss
                        best_q = trial_q
                        best_k = trial_k
            if best_q is not None and best_k is not None:
                selected_q = best_q
                selected_k = best_k
                selected_fit = best_loss
                selected_groups += 1

        holdout_parent = _a4_attention_loss(
            holdout_item,
            states["q_state"],
            states["k_state"],
            zero_q,
            zero_k,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        holdout_candidate = _a4_attention_loss(
            holdout_item,
            states["q_state"],
            states["k_state"],
            selected_q,
            selected_k,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        selected_blocks = int(torch.count_nonzero(selected_q).item()) + int(
            torch.count_nonzero(selected_k).item()
        )
        audit.update(
            {
                "a4_ranked": int(ranked_count),
                "a4_fit_parent_mse": float(fit_parent),
                "a4_fit_candidate_mse": float(selected_fit),
                "a4_holdout_parent_mse": float(holdout_parent),
                "a4_holdout_candidate_mse": float(holdout_candidate),
                "a4_selected_groups": int(selected_groups),
                "a4_selected_blocks": int(selected_blocks),
                "a4_q_nonzero": int(torch.count_nonzero(selected_q).item()),
                "a4_k_nonzero": int(torch.count_nonzero(selected_k).item()),
            }
        )
        if (
            selected_groups > 0
            and math.isfinite(holdout_candidate)
            and math.isfinite(holdout_parent)
            and holdout_candidate < holdout_parent
        ):
            q_state = _a4_clone_state(states["q_state"])
            k_state = _a4_clone_state(states["k_state"])
            q_state["a4_scale_offsets"] = _cpu_state_tensor(selected_q)
            k_state["a4_scale_offsets"] = _cpu_state_tensor(selected_k)
            audit["a4_arm"] = "accepted"
            audit["a4_accepted"] = 1
            return _a4_annotate(
                {
                    "q_state": q_state,
                    "k_state": k_state,
                    "v_state": states["v_state"],
                },
                audit,
            )
        audit["a4_arm"] = (
            "parent" if int(audit["a4_attempted"]) > 0 else "no_boundary"
        )
        return _a4_annotate(states, audit)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        audit["a4_arm"] = "fallback"
        audit["a4_error"] = f"{type(exc).__name__}: {exc}"
        return _a4_annotate(states, audit)


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
        a4_hierarchy_offsets=state.get("a4_scale_offsets"),
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
        a4_hierarchy_offsets=state.get("a4_scale_offsets"),
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
