# ---------------------------------------------------------------------------
# v201: hard-logit residual weighted reciprocal coordinate selection.
#
# The v199 boundary generator remains fixed: one first +/- hard HiF4
# boundary per GQA group/64-channel block.  This card changes only the
# candidate ranking.  A hard Q/K pair is scored with the residual
#     E = Q_hat K_hat^T - Q K^T
# weighted by the local softmax Jacobian and frozen V deviation.  The two
# lowest residual-weighted proposals per GQA group then go through the real
# causal Attention output scorer, and the combined winner is validated on the
# final calibration window.  No STE, optimizer, or dynamic search is added.
# ---------------------------------------------------------------------------

_V201_TOP_CANDIDATES = 2


@torch.no_grad()
def _v201_logit_residual_score(
    item: dict[str, Any],
    u: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> float:
    """First-order output-sensitivity score for one hard reciprocal state."""

    device = item["q_eval_parent"].device
    q_factor, k_factor = _v199_factor_vectors(
        u, q_num_heads, kv_num_heads, head_dim, device
    )
    q_params = _v199_encode_dense(
        item["q_eval_parent"] * q_factor.reshape(1, -1), item["q_state"]
    )
    k_params = _v199_encode_dense(
        item["k_eval_parent"] * k_factor.reshape(1, -1), item["k_state"]
    )
    q_hat = _dequantize_hif4(q_params).to(torch.float32)
    k_hat = _dequantize_hif4(k_params).to(torch.float32)
    tokens = int(item["q_eval_parent"].shape[0])
    group = int(q_num_heads) // int(kv_num_heads)
    q_cont = item["q_eval_parent"].reshape(tokens, q_num_heads, head_dim)
    k_cont = item["k_eval_parent"].reshape(tokens, kv_num_heads, head_dim)
    q_hard = q_hat.reshape(tokens, q_num_heads, head_dim)
    k_hard = k_hat.reshape(tokens, kv_num_heads, head_dim)
    q_cont = q_cont.transpose(0, 1)
    k_cont = k_cont.transpose(0, 1).repeat_interleave(group, dim=0)
    q_hard = q_hard.transpose(0, 1)
    k_hard = k_hard.transpose(0, 1).repeat_interleave(group, dim=0)
    values = item["v_hat"].reshape(tokens, kv_num_heads, head_dim)
    values = values.transpose(0, 1).repeat_interleave(group, dim=0)

    scale = 1.0 / math.sqrt(float(head_dim))
    logits = torch.matmul(q_cont, k_cont.transpose(-1, -2)) * scale
    logits = logits + torch.triu(
        torch.full(
            (tokens, tokens), float("-inf"), device=device, dtype=logits.dtype
        ),
        diagonal=1,
    )
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, values)
    value_deviation = values[:, None, :, :] - output[:, :, None, :]
    sensitivity = probabilities * value_deviation.square().sum(dim=-1)
    residual = (
        torch.matmul(q_hard, k_hard.transpose(-1, -2))
        - torch.matmul(q_cont, k_cont.transpose(-1, -2))
    ) * scale
    score = (sensitivity * residual.square()).mean()
    return float(score) if bool(torch.isfinite(score)) else float("inf")


def _v201_ranked_proposals(
    items: list[dict[str, Any]],
    group_index: int,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    zero_u: torch.Tensor,
) -> tuple[list[tuple[float, int]], int, int]:
    proposals, attempted = _v199_candidate_u(
        items,
        group_index,
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    ranked: list[tuple[float, float, int]] = []
    for proposal, block_index in proposals:
        trial_u = zero_u.clone()
        trial_u[group_index, block_index] = float(proposal)
        scores = [
            _v201_logit_residual_score(
                item, trial_u, q_num_heads, kv_num_heads, head_dim
            )
            for item in items
        ]
        ranked.append(
            (
                sum(scores) / float(len(scores)),
                float(proposal),
                int(block_index),
            )
        )
    ranked.sort(key=lambda value: (value[0], abs(value[1]), value[2]))
    chosen = [
        (proposal, block_index)
        for _, proposal, block_index in ranked[:_V201_TOP_CANDIDATES]
    ]
    return chosen, int(attempted), int(len(ranked))


_V201_PARENT_CALIBRATION = _V199_PARENT_CALIBRATION


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Current v195 parent with residual-ranked hard reciprocal moves."""

    states = _V201_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    audit: dict[str, Any] = {
        "v201_arm": "fallback",
        "v201_attempted": 0,
        "v201_ranked": 0,
        "v201_accepted": 0,
        "v201_fit_windows": 0,
        "v201_selected_groups": 0,
        "v201_selected_blocks": 0,
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
        audit["v201_arm"] = "ineligible"
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
            audit["v201_arm"] = "ineligible"
            return _v199_annotate(states, audit)
        fit_items = items[:-1] or items
        holdout_item = items[-1]
        audit["v201_fit_windows"] = int(len(fit_items))
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
        selected_fit_losses: list[float] = []
        residual_ranked = 0
        for group_index in range(int(kv_num_heads)):
            proposals, attempted, ranked_count = _v201_ranked_proposals(
                fit_items,
                group_index,
                q_num_heads,
                kv_num_heads,
                head_dim,
                zero_u,
            )
            audit["v201_attempted"] += int(attempted)
            residual_ranked += int(ranked_count)
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
                selected_fit_losses.append(float(best_loss))

        selected_blocks = len(selected_fit_losses)
        fit_candidate = (
            sum(selected_fit_losses) / float(selected_blocks)
            if selected_blocks
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
                "v201_ranked": int(residual_ranked),
                "v201_selected_blocks": int(selected_blocks),
                "v201_fit_parent_mse": float(fit_parent),
                "v201_fit_candidate_mse": float(fit_candidate),
                "v201_holdout_parent_mse": float(holdout_parent),
                "v201_holdout_candidate_mse": float(holdout_candidate),
                "v201_u_max_abs": float(selected_u.abs().max()),
                "v201_u_nonzero": int((selected_u != 0).sum()),
            }
        )
        if (
            selected_blocks > 0
            and math.isfinite(holdout_candidate)
            and math.isfinite(holdout_parent)
            and holdout_candidate < holdout_parent
        ):
            q_factor, k_factor = _v199_factor_vectors(
                selected_u,
                q_num_heads,
                kv_num_heads,
                head_dim,
                device,
            )
            q_state = _v199_clone_state(states["q_state"])
            k_state = _v199_clone_state(states["k_state"])
            q_state["diag_scale"] = _cpu_state_tensor(q_factor)
            k_state["diag_scale"] = _cpu_state_tensor(k_factor)
            q_state["v201_u"] = _cpu_state_tensor(selected_u)
            k_state["v201_u"] = _cpu_state_tensor(selected_u.clone())
            audit["v201_arm"] = "accepted"
            audit["v201_accepted"] = 1
            audit["v201_selected_groups"] = int(selected_blocks)
            return _v199_annotate(
                {
                    "q_state": q_state,
                    "k_state": k_state,
                    "v_state": states["v_state"],
                },
                audit,
            )
        audit["v201_arm"] = (
            "parent" if int(audit["v201_attempted"]) > 0 else "no_boundary"
        )
        return _v199_annotate(states, audit)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        audit["v201_arm"] = "fallback"
        audit["v201_error"] = f"{type(exc).__name__}: {exc}"
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
