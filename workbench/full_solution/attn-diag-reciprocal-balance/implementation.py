# ---------------------------------------------------------------------------
# Attention direction 1: calibration-only diagonal reciprocal Q/K balance.
# ---------------------------------------------------------------------------

_ATTN_DIAG_EPS = 1.0e-12
_ATTN_DIAG_BOUND = 0.5 * math.log(2.0)
_ATTN_DIAG_FIT_WINDOWS = 3
_ATTN_DIAG_GATE_WINDOWS = (3, 4)
_ATTN_DIAG_MAX_TOKENS = 256
_ATTN_DIAG_CHUNK_TOKENS = 8


def _attn_diag_clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in state.items()
    }


def _attn_diag_parent_dense(
    dense: torch.Tensor,
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
    *,
    is_k: bool,
) -> torch.Tensor:
    """Return the continuous parent tensor consumed by the HiF4 encoder."""

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
    diag_scale = state.get("diag_scale")
    if diag_scale is not None:
        scale = _safe_positive_vector(
            diag_scale, int(num_heads) * int(head_dim)
        ).to(device=transformed.device)
        transformed = transformed * scale.reshape(
            *([1] * (transformed.ndim - 1)), -1
        )
    return transformed


def _attn_diag_prefix_window(sample: dict, limit: int) -> dict:
    q_quant, q_scale = sample["q"]
    k_quant, k_scale = sample["k"]
    v_quant, v_scale = sample["v"]
    tokens = min(
        int(q_quant.shape[0]), int(k_quant.shape[0]), int(v_quant.shape[0]),
        int(limit),
    )
    return {
        "q": (q_quant[:tokens], q_scale[:tokens]),
        "k": (k_quant[:tokens], k_scale[:tokens]),
        "v": (v_quant[:tokens], v_scale[:tokens]),
    }


@torch.no_grad()
def _attn_diag_output_sensitivity(
    q_parent: torch.Tensor,
    k_parent: torch.Tensor,
    v_parent: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum squared attention-output Jacobian energy per Q/K coordinate.

    The Q derivative is evaluated by a small query-token chunk.  For K, the
    derivative factorizes into the squared value deviation and q-coordinate
    square, avoiding a full [query,key,head_dim,head_dim] temporary.
    """

    tokens = int(q_parent.shape[0])
    if tokens <= 0 or int(k_parent.shape[0]) != tokens:
        raise ValueError("Q/K sensitivity inputs must have equal nonzero length")
    if int(v_parent.shape[0]) != tokens:
        raise ValueError("V sensitivity input must match Q/K length")
    if int(q_num_heads) % int(kv_num_heads) != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    group = int(q_num_heads) // int(kv_num_heads)
    device = q_parent.device
    qh = q_parent.to(torch.float32).reshape(tokens, q_num_heads, head_dim).transpose(0, 1)
    kh = k_parent.to(torch.float32).reshape(tokens, kv_num_heads, head_dim).transpose(0, 1)
    kh = kh.repeat_interleave(group, dim=0)
    vh = v_parent.to(torch.float32).reshape(tokens, kv_num_heads, head_dim).transpose(0, 1)
    vh = vh.repeat_interleave(group, dim=0)
    scale = 1.0 / math.sqrt(float(head_dim))
    logits = qh @ kh.transpose(-1, -2) * scale
    causal_mask = torch.triu(
        torch.ones(tokens, tokens, dtype=torch.bool, device=device), diagonal=1
    )
    q_energy = torch.zeros(q_num_heads, tokens, head_dim, device=device)
    k_energy = torch.zeros(q_num_heads, tokens, head_dim, device=device)
    for causal in (True, False):
        current_logits = (
            logits.masked_fill(causal_mask, float("-inf"))
            if causal
            else logits
        )
        probabilities = torch.softmax(current_logits, dim=-1)
        output = probabilities @ vh
        q_energy_mask = torch.zeros_like(q_energy)
        k_energy_mask = torch.zeros_like(k_energy)
        chunk = max(1, int(_ATTN_DIAG_CHUNK_TOKENS))
        for start in range(0, tokens, chunk):
            stop = min(tokens, start + chunk)
            p = probabilities[:, start:stop, :]
            q_chunk = qh[:, start:stop, :]
            delta_v = vh[:, None, :, :] - output[:, start:stop, None, :]
            dq = torch.einsum(
                "hcs,hcsl,hsj->hclj", p, delta_v, kh
            ) * scale
            q_energy_mask[:, start:stop, :] = dq.square().sum(dim=2)
            value_energy = delta_v.square().sum(dim=-1)
            dk_energy = torch.einsum(
                "hcs,hcj->hsj", p.square() * value_energy, q_chunk.square()
            ) * (scale * scale)
            k_energy_mask += dk_energy
        q_energy += q_energy_mask * 0.5
        k_energy += k_energy_mask * 0.5
    return q_energy, k_energy.reshape(
        kv_num_heads, group, tokens, head_dim
    ).sum(dim=1)


@torch.no_grad()
def _attn_diag_error_energy(
    calib_qkv_list: list,
    parent_states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    """Accumulate output-propagated Q/K quantization error over three folds."""

    if len(calib_qkv_list) < _ATTN_DIAG_FIT_WINDOWS:
        return None
    q_state = parent_states["q_state"]
    k_state = parent_states["k_state"]
    v_state = parent_states["v_state"]
    a_total = None
    b_total = None
    used = 0
    for sample in calib_qkv_list[:_ATTN_DIAG_FIT_WINDOWS]:
        window = _attn_diag_prefix_window(sample, _ATTN_DIAG_MAX_TOKENS)
        q_raw = _dequantize_nvfp4_float32(*window["q"]).to(torch.float32)
        k_raw = _dequantize_nvfp4_float32(*window["k"]).to(torch.float32)
        v_params = _ATTN_DIAG_PARENT_V(
            window["v"][0], window["v"][1], kv_num_heads, head_dim, v_state
        )
        v_parent = _dequantize_hif4(v_params).to(torch.float32)
        q_parent = _attn_diag_parent_dense(
            q_raw, q_state, q_num_heads, head_dim, is_k=False
        )
        k_parent = _attn_diag_parent_dense(
            k_raw, k_state, kv_num_heads, head_dim, is_k=True
        )
        q_params = _ATTN_DIAG_PARENT_Q(
            window["q"][0], window["q"][1], q_num_heads, head_dim, q_state
        )
        k_params = _ATTN_DIAG_PARENT_K(
            window["k"][0], window["k"][1], kv_num_heads, head_dim, k_state
        )
        q_hat = _dequantize_hif4(q_params).to(torch.float32)
        k_hat = _dequantize_hif4(k_params).to(torch.float32)
        q_sensitivity, k_sensitivity = _attn_diag_output_sensitivity(
            q_parent, k_parent, v_parent,
            q_num_heads, kv_num_heads, head_dim,
        )
        q_error = q_hat - q_parent
        k_error = k_hat - k_parent
        group = int(q_num_heads) // int(kv_num_heads)
        q_energy = (
            q_sensitivity.permute(1, 0, 2) * q_error.reshape(
                q_error.shape[0], q_num_heads, head_dim
            ).square()
        ).sum(dim=0).reshape(kv_num_heads, group, head_dim).sum(dim=1)
        k_energy = (
            k_sensitivity * k_error.reshape(
                k_error.shape[0], kv_num_heads, head_dim
            ).permute(1, 0, 2).square()
        ).sum(dim=1)
        if a_total is None:
            a_total = q_energy
            b_total = k_energy
        else:
            a_total = a_total + q_energy
            b_total = b_total + k_energy
        used += 1
    if used == 0 or a_total is None or b_total is None:
        return None
    a_total = torch.nan_to_num(a_total / float(used), nan=0.0, posinf=0.0, neginf=0.0)
    b_total = torch.nan_to_num(b_total / float(used), nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(torch.isfinite(a_total).all() and torch.isfinite(b_total).all()):
        return None
    return a_total, b_total, {
        "fit_windows": int(used),
        "a_mean": float(a_total.mean()),
        "b_mean": float(b_total.mean()),
    }


def _attn_diag_reciprocal_balance(
    parent_states: dict[str, Any],
    a_energy: torch.Tensor,
    b_energy: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile the unique groupwise zero-mean bounded diagonal solution."""

    expected = (int(kv_num_heads), int(head_dim))
    if tuple(a_energy.shape) != expected or tuple(b_energy.shape) != expected:
        raise ValueError("Diagonal energy shape does not match GQA metadata")
    raw_d = 0.25 * torch.log(
        (b_energy.to(torch.float32) + _ATTN_DIAG_EPS)
        / (a_energy.to(torch.float32) + _ATTN_DIAG_EPS)
    )
    d = torch.nan_to_num(raw_d, nan=0.0, posinf=0.0, neginf=0.0)
    d = d - d.mean(dim=1, keepdim=True)
    d = d.clamp(min=-_ATTN_DIAG_BOUND, max=_ATTN_DIAG_BOUND)
    if not bool(torch.isfinite(d).all()):
        raise ValueError("Diagonal reciprocal balance is non-finite")
    q_factor = d.exp()
    k_factor = d.neg().exp()
    q_state = _attn_diag_clone_state(parent_states["q_state"])
    k_state = _attn_diag_clone_state(parent_states["k_state"])
    folded = False
    q_rotation = q_state.get("learned_rotation")
    k_rotation = k_state.get("learned_rotation")
    if (
        torch.is_tensor(q_rotation)
        and torch.is_tensor(k_rotation)
        and tuple(q_rotation.shape) == (int(kv_num_heads), int(head_dim), int(head_dim))
        and tuple(k_rotation.shape) == (int(kv_num_heads), int(head_dim), int(head_dim))
    ):
        q_matrix = q_rotation.to(
            device=q_factor.device, dtype=torch.float32
        ) @ torch.diag_embed(q_factor)
        k_matrix = k_rotation.to(
            device=k_factor.device, dtype=torch.float32
        ) @ torch.diag_embed(k_factor)
        q_state["learned_rotation"] = _cpu_state_tensor(q_matrix)
        k_state["learned_rotation"] = _cpu_state_tensor(k_matrix)
        learned_center = k_state.get("learned_center")
        if torch.is_tensor(learned_center) and tuple(learned_center.shape) == expected:
            k_state["learned_center"] = _cpu_state_tensor(
                learned_center.to(
                    device=k_factor.device, dtype=torch.float32
                ) * k_factor
            )
        folded = True
    else:
        q_state["diag_scale"] = _cpu_state_tensor(
            q_factor.repeat_interleave(int(q_num_heads) // int(kv_num_heads), dim=0).reshape(-1)
        )
        k_state["diag_scale"] = _cpu_state_tensor(k_factor.reshape(-1))
    info = {
        "d_norm": float(d.norm()),
        "d_max_abs": float(d.abs().max()),
        "d_mean_abs": float(d.abs().mean()),
        "d_zero_fraction": float((d == 0).to(torch.float32).mean()),
        "fold_mode": "learned_rotation_and_center" if folded else "post_transform_vector",
        "d": _cpu_state_tensor(d),
    }
    return q_state, k_state, info


@torch.no_grad()
def _attn_true_output_loss(
    sample: dict,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Real causal/non-causal Attention MSE through the deployed HiF4 path."""

    window = _attn_diag_prefix_window(sample, _ATTN_DIAG_MAX_TOKENS)
    q_quant, q_scale = window["q"]
    k_quant, k_scale = window["k"]
    v_quant, v_scale = window["v"]
    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)
    q_params = hif4_dynamic_quantize_q(
        q_quant, q_scale, q_num_heads, head_dim, states["q_state"]
    )
    k_params = hif4_dynamic_quantize_k(
        k_quant, k_scale, kv_num_heads, head_dim, states["k_state"]
    )
    v_params = hif4_dynamic_quantize_v(
        v_quant, v_scale, kv_num_heads, head_dim, states["v_state"]
    )
    q_hat = _dequantize_hif4(q_params).to(torch.float32)
    k_hat = _dequantize_hif4(k_params).to(torch.float32)
    v_hat = _dequantize_hif4(v_params).to(torch.float32)
    causal = []
    noncausal = []
    for is_causal in (True, False):
        reference = _attention_forward(
            q_ref, k_ref, v_ref, q_num_heads, kv_num_heads, head_dim, is_causal
        )
        player = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, is_causal
        )
        value = float((player - reference).square().mean())
        (causal if is_causal else noncausal).append(value)
    causal_value = causal[0]
    noncausal_value = noncausal[0]
    return {
        "causal": causal_value,
        "noncausal": noncausal_value,
        "mean": 0.5 * (causal_value + noncausal_value),
        "q_params": q_params,
        "k_params": k_params,
    }


def _attn_diag_gate_passes(parent_loss: dict, candidate_loss: dict) -> bool:
    if not all(
        math.isfinite(float(candidate_loss[key]))
        and math.isfinite(float(parent_loss[key]))
        for key in ("causal", "noncausal", "mean")
    ):
        return False
    if float(candidate_loss["mean"]) >= float(parent_loss["mean"]):
        return False
    return all(
        float(candidate_loss[key]) <= float(parent_loss[key]) + 1.0e-12
        for key in ("causal", "noncausal")
    )


def _attn_diag_changed_codes(parent_params: dict, candidate_params: dict) -> int:
    changed = 0
    for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
        changed += int((parent_params[key] != candidate_params[key]).sum())
    return changed


def _attn_diag_annotated_states(
    states: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "q_state": _attn_diag_clone_state(states["q_state"]),
        "k_state": _attn_diag_clone_state(states["k_state"]),
        "v_state": states["v_state"],
    }
    result["q_state"].update(audit)
    result["k_state"].update(audit)
    return result


_ATTN_DIAG_PARENT_CALIBRATION = hif4_calibration_attention
_ATTN_DIAG_PARENT_Q = hif4_dynamic_quantize_q
_ATTN_DIAG_PARENT_K = hif4_dynamic_quantize_k
_ATTN_DIAG_PARENT_V = hif4_dynamic_quantize_v


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Current complete parent plus one closed-form reciprocal balance arm."""

    states = _ATTN_DIAG_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    if (
        not isinstance(states, dict)
        or not all(role in states and isinstance(states[role], dict) for role in ("q_state", "k_state", "v_state"))
        or len(calib_qkv_list) < max(_ATTN_DIAG_GATE_WINDOWS) + 1
    ):
        return states
    audit = {
        "diag_arm": "fallback",
        "diag_attempted": 0,
        "diag_accepted": 0,
        "diag_fit_windows": 0,
        "diag_gate_windows": len(_ATTN_DIAG_GATE_WINDOWS),
    }
    try:
        energy = _attn_diag_error_energy(
            calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim
        )
        if energy is None:
            audit["diag_arm"] = "ineligible"
            return _attn_diag_annotated_states(states, audit)
        a_energy, b_energy, energy_info = energy
        q_candidate, k_candidate, balance_info = _attn_diag_reciprocal_balance(
            states, a_energy, b_energy, q_num_heads, kv_num_heads, head_dim
        )
        candidate_states = {
            "q_state": q_candidate,
            "k_state": k_candidate,
            "v_state": states["v_state"],
        }
        audit.update(energy_info)
        audit["diag_fit_windows"] = int(energy_info["fit_windows"])
        audit.update({key: value for key, value in balance_info.items() if key != "d"})
        audit["diag_attempted"] = 1
        accepted = True
        gate_records = []
        for index in _ATTN_DIAG_GATE_WINDOWS:
            parent_loss = _attn_true_output_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _attn_true_output_loss(
                calib_qkv_list[index], candidate_states,
                q_num_heads, kv_num_heads, head_dim,
            )
            current_pass = _attn_diag_gate_passes(parent_loss, candidate_loss)
            accepted = accepted and current_pass
            gate_records.append({
                "window": int(index),
                "pass": bool(current_pass),
                "parent_mean": float(parent_loss["mean"]),
                "candidate_mean": float(candidate_loss["mean"]),
                "parent_causal": float(parent_loss["causal"]),
                "candidate_causal": float(candidate_loss["causal"]),
                "parent_noncausal": float(parent_loss["noncausal"]),
                "candidate_noncausal": float(candidate_loss["noncausal"]),
                "q_changed_codes": _attn_diag_changed_codes(
                    parent_loss["q_params"], candidate_loss["q_params"]
                ),
                "k_changed_codes": _attn_diag_changed_codes(
                    parent_loss["k_params"], candidate_loss["k_params"]
                ),
            })
        audit["diag_gate"] = gate_records
        audit["diag_q_changed_codes"] = int(sum(item["q_changed_codes"] for item in gate_records))
        audit["diag_k_changed_codes"] = int(sum(item["k_changed_codes"] for item in gate_records))
        if accepted:
            audit["diag_arm"] = "accepted"
            audit["diag_accepted"] = 1
            return _attn_diag_annotated_states(candidate_states, audit)
        audit["diag_arm"] = "parent"
        return _attn_diag_annotated_states(states, audit)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        audit["diag_arm"] = "fallback"
        audit["diag_error"] = f"{type(exc).__name__}: {exc}"
        return _attn_diag_annotated_states(states, audit)


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
        attention_diag_scale=state.get("diag_scale"),
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
    )
