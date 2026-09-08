"""One fixed 64-dimensional block-triangular Q/K transport arm."""

import math
from typing import Any, Optional

import torch


_TRI_BLOCK = 64
_TRI_BLOCKS = 4
_TRI_PAIRS = ((0, 1), (2, 3))
_TRI_FIT_WINDOWS = 3
_TRI_GATE_WINDOWS = (3, 4)
_TRI_FIT_TOKENS = 256
_TRI_BOUND_INITIAL = 1.0e-5
_TRI_BOUND_GROW_STEPS = 16
_TRI_BOUND_BINARY_STEPS = 10
_TRI_EPS = 1.0e-12


def _tri_clone_state(state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().to(device="cpu").clone()
    return out


def _tri_annotated_states(
    states: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    out = {
        "q_state": _tri_clone_state(states["q_state"]),
        "k_state": _tri_clone_state(states["k_state"]),
        "v_state": _tri_clone_state(states["v_state"]),
    }
    out["q_state"].update(audit)
    out["k_state"].update(audit)
    return out


def _tri_prefix_sample(sample: dict[str, Any], limit: Optional[int]) -> dict[str, Any]:
    if limit is None:
        return sample
    q_quant, q_scale = sample["q"]
    k_quant, k_scale = sample["k"]
    v_quant, v_scale = sample["v"]
    rows = min(
        int(limit),
        int(q_quant.shape[0]),
        int(k_quant.shape[0]),
        int(v_quant.shape[0]),
    )
    if rows <= 0:
        raise ValueError("Attention sample has no rows")
    return {
        "q": (q_quant[:rows], q_scale[:rows]),
        "k": (k_quant[:rows], k_scale[:rows]),
        "v": (v_quant[:rows], v_scale[:rows]),
    }


def _tri_reference_dense(sample: dict[str, Any], limit: Optional[int]) -> tuple:
    current = _tri_prefix_sample(sample, limit)
    q = _dequantize_nvfp4_float32(*current["q"]).to(torch.float32)
    k = _dequantize_nvfp4_float32(*current["k"]).to(torch.float32)
    v = _dequantize_nvfp4_float32(*current["v"]).to(torch.float32)
    return current, q, k, v


def _tri_parent_outputs(
    sample: dict[str, Any],
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    limit: Optional[int],
) -> tuple:
    current, q_ref, k_ref, v_ref = _tri_reference_dense(sample, limit)
    q_params = _TRI_PARENT_Q(
        current["q"][0], current["q"][1], q_num_heads, head_dim, states["q_state"]
    )
    k_params = _TRI_PARENT_K(
        current["k"][0], current["k"][1], kv_num_heads, head_dim, states["k_state"]
    )
    v_params = _TRI_PARENT_V(
        current["v"][0], current["v"][1], kv_num_heads, head_dim, states["v_state"]
    )
    return (
        current,
        q_ref,
        k_ref,
        v_ref,
        _dequantize_hif4(q_params).to(torch.float32),
        _dequantize_hif4(k_params).to(torch.float32),
        _dequantize_hif4(v_params).to(torch.float32),
        q_params,
        k_params,
    )


def _tri_attention_backward(
    d_output: torch.Tensor,
    q_dense: torch.Tensor,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = int(q_dense.shape[0])
    group = int(q_num_heads) // int(kv_num_heads)
    q_heads = q_dense.reshape(tokens, q_num_heads, head_dim).transpose(0, 1)
    k_heads = k_dense.reshape(tokens, kv_num_heads, head_dim).transpose(0, 1)
    v_heads = v_dense.reshape(tokens, kv_num_heads, head_dim).transpose(0, 1)
    k_for_q = k_heads.repeat_interleave(group, dim=0)
    v_for_q = v_heads.repeat_interleave(group, dim=0)
    logits = q_heads @ k_for_q.transpose(-1, -2) / math.sqrt(float(head_dim))
    if causal:
        logits = logits + torch.triu(
            torch.full(
                (tokens, tokens), float("-inf"), device=logits.device, dtype=logits.dtype
            ),
            1,
        )
    probabilities = torch.softmax(logits, dim=-1)
    d_out_heads = d_output.reshape(tokens, q_num_heads, head_dim).transpose(0, 1)
    d_probability = d_out_heads @ v_for_q.transpose(-1, -2)
    centered = d_probability - (
        d_probability * probabilities
    ).sum(dim=-1, keepdim=True)
    d_logits = centered * probabilities / math.sqrt(float(head_dim))
    d_q = torch.einsum("hts,hsd->htd", d_logits, k_for_q)
    d_k_for_q = torch.einsum("hts,htd->hsd", d_logits, q_heads)
    d_k = d_k_for_q.reshape(kv_num_heads, group, tokens, head_dim).sum(dim=1)
    return (
        d_q.transpose(0, 1).reshape(tokens, q_num_heads * head_dim),
        d_k.transpose(0, 1).reshape(tokens, kv_num_heads * head_dim),
    )


def _tri_block_gradient(
    q_dense: torch.Tensor,
    k_dense: torch.Tensor,
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if int(head_dim) != _TRI_BLOCK * _TRI_BLOCKS:
        raise ValueError("The triangular transport requires four 64D blocks")
    group = int(q_num_heads) // int(kv_num_heads)
    q = q_dense.reshape(-1, q_num_heads, _TRI_BLOCKS, _TRI_BLOCK)
    dq = d_q.reshape(-1, q_num_heads, _TRI_BLOCKS, _TRI_BLOCK)
    k = k_dense.reshape(-1, kv_num_heads, _TRI_BLOCKS, _TRI_BLOCK)
    dk = d_k.reshape(-1, kv_num_heads, _TRI_BLOCKS, _TRI_BLOCK)
    gradients = torch.zeros(
        kv_num_heads, len(_TRI_PAIRS), _TRI_BLOCK, _TRI_BLOCK,
        device=q_dense.device, dtype=torch.float32,
    )
    for group_index in range(int(kv_num_heads)):
        q_slice = slice(group_index * group, (group_index + 1) * group)
        for pair_index, (source, target) in enumerate(_TRI_PAIRS):
            q_source = q[:, q_slice, source, :].reshape(-1, _TRI_BLOCK)
            dq_target = dq[:, q_slice, target, :].reshape(-1, _TRI_BLOCK)
            q_term = q_source.transpose(0, 1) @ dq_target
            k_source = k[:, group_index, target, :]
            dk_target = dk[:, group_index, source, :]
            k_term = -(dk_target.transpose(0, 1) @ k_source)
            gradients[group_index, pair_index].add_(q_term + k_term)
    return gradients


@torch.no_grad()
def _attn_block_output_gradient(
    calib_qkv_list: list,
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(calib_qkv_list) < _TRI_FIT_WINDOWS:
        raise ValueError("Not enough calibration windows for triangular transport")
    total = None
    used = 0
    for sample in calib_qkv_list[:_TRI_FIT_WINDOWS]:
        (
            _current,
            q_ref,
            k_ref,
            v_ref,
            q_parent,
            k_parent,
            v_parent,
            _q_params,
            _k_params,
        ) = _tri_parent_outputs(
            sample, states, q_num_heads, kv_num_heads, head_dim, _TRI_FIT_TOKENS
        )
        sample_gradient = torch.zeros(
            kv_num_heads, len(_TRI_PAIRS), _TRI_BLOCK, _TRI_BLOCK,
            device=q_parent.device, dtype=torch.float32,
        )
        for causal in (True, False):
            parent_output = _attention_forward(
                q_parent, k_parent, v_parent,
                q_num_heads, kv_num_heads, head_dim, causal,
            )
            reference_output = _attention_forward(
                q_ref, k_ref, v_ref,
                q_num_heads, kv_num_heads, head_dim, causal,
            )
            d_output = 2.0 * (parent_output - reference_output) / float(
                max(int(parent_output.numel()), 1)
            )
            d_q, d_k = _tri_attention_backward(
                d_output, q_parent, k_parent, v_parent,
                q_num_heads, kv_num_heads, head_dim, causal,
            )
            sample_gradient.add_(_tri_block_gradient(
                q_parent, k_parent, d_q, d_k,
                q_num_heads, kv_num_heads, head_dim,
            ))
        if total is None:
            total = sample_gradient
        else:
            total.add_(sample_gradient)
        used += 2
    if total is None or used <= 0:
        raise ValueError("No usable calibration windows")
    total.div_(float(used))
    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    singular_values = []
    for group_index in range(int(kv_num_heads)):
        singular_values.append([
            float(torch.linalg.svdvals(total[group_index, pair_index])[0].item())
            for pair_index in range(len(_TRI_PAIRS))
        ])
    return total, {
        "tri_fit_windows": _TRI_FIT_WINDOWS,
        "tri_fit_masks": used,
        "tri_singular_values": singular_values,
    }


def _tri_probe_states(
    q_state: dict[str, Any],
    k_state: dict[str, Any],
    blocks: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    q_probe = _tri_clone_state(q_state)
    k_probe = _tri_clone_state(k_state)
    q_probe["triangular_blocks"] = blocks.detach().to(device="cpu").clone()
    k_probe["triangular_blocks"] = blocks.detach().to(device="cpu").clone()
    q_probe["triangular_inverse"] = False
    k_probe["triangular_inverse"] = True
    return q_probe, k_probe


def _tri_params_changed(
    parent: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> int:
    changed = 0
    for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
        left = parent[key]
        right = candidate[key]
        if tuple(left.shape) != tuple(right.shape):
            return 1
        changed += int((left != right).sum().item())
    return changed


@torch.no_grad()
def _attn_first_code_boundary_step(
    sample: dict[str, Any],
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    group_index: int,
    pair_index: int,
    direction: torch.Tensor,
) -> tuple[float, int, int, int]:
    current, _q_ref, _k_ref, _v_ref = _tri_reference_dense(
        sample, _TRI_FIT_TOKENS
    )
    q_base = _TRI_PARENT_Q(
        current["q"][0], current["q"][1], q_num_heads, head_dim, states["q_state"]
    )
    k_base = _TRI_PARENT_K(
        current["k"][0], current["k"][1], kv_num_heads, head_dim, states["k_state"]
    )
    kv = int(kv_num_heads)

    def probe(alpha: float) -> tuple[int, int]:
        blocks = torch.zeros(
            kv, len(_TRI_PAIRS), _TRI_BLOCK, _TRI_BLOCK,
            dtype=torch.float32,
        )
        blocks[group_index, pair_index] = direction * float(alpha)
        q_probe, k_probe = _tri_probe_states(
            states["q_state"], states["k_state"], blocks
        )
        q_candidate = hif4_dynamic_quantize_q(
            current["q"][0], current["q"][1],
            q_num_heads, head_dim, q_probe,
        )
        k_candidate = hif4_dynamic_quantize_k(
            current["k"][0], current["k"][1],
            kv_num_heads, head_dim, k_probe,
        )
        return (
            _tri_params_changed(q_base, q_candidate),
            _tri_params_changed(k_base, k_candidate),
        )

    high = _TRI_BOUND_INITIAL
    q_changed = 0
    k_changed = 0
    for _ in range(_TRI_BOUND_GROW_STEPS):
        q_changed, k_changed = probe(high)
        if q_changed or k_changed:
            break
        high *= 2.0
    else:
        return 0.0, 0, 0, 0

    low = 0.0
    for _ in range(_TRI_BOUND_BINARY_STEPS):
        middle = 0.5 * (low + high)
        q_changed, k_changed = probe(middle)
        if q_changed or k_changed:
            high = middle
        else:
            low = middle
    q_changed, k_changed = probe(high)
    return float(high), 1, int(q_changed), int(k_changed)


@torch.no_grad()
def _attn_compile_pair_transform(
    states: dict[str, Any],
    gradients: torch.Tensor,
    calib_sample: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if int(head_dim) != _TRI_BLOCK * _TRI_BLOCKS:
        raise ValueError("The triangular transport requires head_dim=256")
    blocks = torch.zeros(
        kv_num_heads, len(_TRI_PAIRS), _TRI_BLOCK, _TRI_BLOCK,
        dtype=torch.float32,
    )
    step_records = []
    total_q_changed = 0
    total_k_changed = 0
    reachable = 0
    for group_index in range(int(kv_num_heads)):
        for pair_index in range(len(_TRI_PAIRS)):
            gradient = gradients[group_index, pair_index]
            if float(gradient.abs().max().item()) <= _TRI_EPS:
                step_records.append({
                    "group": group_index,
                    "pair": pair_index,
                    "singular_value": 0.0,
                    "step": 0.0,
                    "reachable": 0,
                    "q_changed_codes": 0,
                    "k_changed_codes": 0,
                })
                continue
            u, singular, vh = torch.linalg.svd(gradient, full_matrices=False)
            direction = -u[:, :1] @ vh[:1, :]
            step, is_reachable, q_changed, k_changed = _attn_first_code_boundary_step(
                calib_sample, states, q_num_heads, kv_num_heads, head_dim,
                group_index, pair_index, direction,
            )
            if is_reachable:
                blocks[group_index, pair_index] = direction.cpu() * step
                reachable += 1
            total_q_changed += q_changed
            total_k_changed += k_changed
            step_records.append({
                "group": group_index,
                "pair": pair_index,
                "singular_value": float(singular[0].item()),
                "step": float(step),
                "reachable": int(is_reachable),
                "q_changed_codes": int(q_changed),
                "k_changed_codes": int(k_changed),
            })
    if reachable <= 0:
        return _tri_clone_state(states["q_state"]), _tri_clone_state(states["k_state"]), {
            "tri_steps": step_records,
            "tri_reachable_blocks": 0,
            "tri_q_changed_codes": total_q_changed,
            "tri_k_changed_codes": total_k_changed,
        }
    q_candidate, k_candidate = _tri_probe_states(
        states["q_state"], states["k_state"], blocks
    )
    return q_candidate, k_candidate, {
        "tri_steps": step_records,
        "tri_reachable_blocks": reachable,
        "tri_q_changed_codes": total_q_changed,
        "tri_k_changed_codes": total_k_changed,
        "tri_block_norm": float(blocks.norm().item()),
        "tri_block_max_abs": float(blocks.abs().max().item()),
    }


@torch.no_grad()
def _tri_true_output_loss(
    sample: dict[str, Any],
    states: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    (
        _current,
        q_ref,
        k_ref,
        v_ref,
        q_hat,
        k_hat,
        v_hat,
        _q_params,
        _k_params,
    ) = _tri_parent_outputs(
        sample, states, q_num_heads, kv_num_heads, head_dim, None
    )
    causal = _attention_forward(
        q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
    )
    causal_ref = _attention_forward(
        q_ref, k_ref, v_ref, q_num_heads, kv_num_heads, head_dim, True
    )
    noncausal = _attention_forward(
        q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
    )
    noncausal_ref = _attention_forward(
        q_ref, k_ref, v_ref, q_num_heads, kv_num_heads, head_dim, False
    )
    causal_loss = float((causal - causal_ref).square().mean().item())
    noncausal_loss = float((noncausal - noncausal_ref).square().mean().item())
    return {
        "causal": causal_loss,
        "noncausal": noncausal_loss,
        "mean": 0.5 * (causal_loss + noncausal_loss),
    }


def _tri_gate_passes(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    for key in ("mean", "causal", "noncausal"):
        if not math.isfinite(float(candidate[key])):
            return False
        if float(candidate[key]) > float(parent[key]) + 1.0e-12:
            return False
    return True


_TRI_PARENT_CALIBRATION = hif4_calibration_attention
_TRI_PARENT_Q = hif4_dynamic_quantize_q
_TRI_PARENT_K = hif4_dynamic_quantize_k
_TRI_PARENT_V = hif4_dynamic_quantize_v


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    states = _TRI_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    audit = {
        "tri_arm": "fallback",
        "tri_attempted": 0,
        "tri_accepted": 0,
        "tri_fit_windows": 0,
        "tri_gate_windows": len(_TRI_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(role in states and isinstance(states[role], dict) for role in ("q_state", "k_state", "v_state"))
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) <= max(_TRI_GATE_WINDOWS)
        or int(head_dim) != _TRI_BLOCK * _TRI_BLOCKS
        or int(q_num_heads) % int(kv_num_heads) != 0
    ):
        audit["tri_arm"] = "ineligible"
        return _tri_annotated_states(states, audit)
    try:
        gradients, gradient_info = _attn_block_output_gradient(
            calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim
        )
        q_candidate, k_candidate, transform_info = _attn_compile_pair_transform(
            states, gradients, calib_qkv_list[0],
            q_num_heads, kv_num_heads, head_dim,
        )
        candidate_states = {
            "q_state": q_candidate,
            "k_state": k_candidate,
            "v_state": states["v_state"],
        }
        audit.update(gradient_info)
        audit.update(transform_info)
        audit["tri_attempted"] = 1
        accepted = True
        gate_records = []
        for index in _TRI_GATE_WINDOWS:
            parent_loss = _tri_true_output_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _tri_true_output_loss(
                calib_qkv_list[index], candidate_states,
                q_num_heads, kv_num_heads, head_dim,
            )
            current_pass = _tri_gate_passes(parent_loss, candidate_loss)
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
            })
        audit["tri_gate"] = gate_records
        if accepted and int(transform_info.get("tri_reachable_blocks", 0)) > 0:
            audit["tri_arm"] = "accepted"
            audit["tri_accepted"] = 1
            return _tri_annotated_states(candidate_states, audit)
        audit["tri_arm"] = "parent"
        return _tri_annotated_states(states, audit)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        audit["tri_arm"] = "fallback"
        audit["tri_error"] = f"{type(exc).__name__}: {exc}"
        return _tri_annotated_states(states, audit)


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
        attention_triangular_blocks=state.get("triangular_blocks"),
        attention_triangular_inverse=bool(state.get("triangular_inverse", False)),
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
        attention_triangular_blocks=state.get("triangular_blocks"),
        attention_triangular_inverse=bool(state.get("triangular_inverse", False)),
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


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(v_state, kv_num_heads, head_dim, "v")
    if int(v_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("V width does not match kv_num_heads * head_dim")
    return _TRI_PARENT_V(
        v_quant, v_scale, kv_num_heads, head_dim, state
    )
