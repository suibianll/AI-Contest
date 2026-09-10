# ---------------------------------------------------------------------------
# A-CT1 -- the arm-independent work inside the A-GR1 gate is done once.
#
# A-GR1's gate calls _agr1_gate_loss twice for each gate window, once with the
# parent states and once with the candidate states.  Each call re-derives the
# dense reference Q/K/V from the window, recomputes the reference attention
# output `target`, and re-quantises V.  None of that depends on the arm: the two
# arms differ only in the Q/K rotation and the K center they carry, and the V
# state is equal-valued in both because the candidate is a copy of the parent
# with only q_state/k_state touched.
#
# A-CT1 collapses the pair into one call:
#
#   A-GR1    for index in _AGR1_GATE_WINDOWS:
#                parent_loss = _agr1_gate_loss(
#                    calib_qkv_list[index], states,
#                    q_num_heads, kv_num_heads, head_dim,
#                )
#                candidate_loss = _agr1_gate_loss(
#                    calib_qkv_list[index], candidate,
#                    q_num_heads, kv_num_heads, head_dim,
#                )
#
#   A-CT1    for index in _AGR1_GATE_WINDOWS:
#                parent_loss, candidate_loss = _act1_gate_pair(
#                    calib_qkv_list[index], states, candidate,
#                    q_num_heads, kv_num_heads, head_dim,
#                )
#
# Per gate window the new call does one reference decode instead of two, one
# reference attention forward instead of two, and one V quantisation instead of
# two.  The Q/K quantisation is still done once per arm, because that is the
# part the arms disagree about, and each arm still gets its own attention
# forward against the shared target.  Over the two shipped gate windows that is
# two fewer reference forwards, two fewer V quantisations and two fewer sets of
# reference decodes.
#
# Nothing else moves.  Every window's two losses, the strict
# `candidate_loss < parent_loss` per window, the AND across windows, the info
# fields and the final state compilation are the A-GR1 code, byte for byte.
# The shared work is held in a local for the duration of the call -- no global
# cache, no write into any deployment state, and nothing that moves with M is
# cached.
#
# The definitions below shadow the A-GR1 ones, which stay in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _act1_gate_pair(
    item: dict,
    parent_states: dict,
    candidate_states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
):
    """Both A-GR1 gate losses for one window, sharing the arm-independent work.

    Returns ``(parent_loss, candidate_loss)``.  Each arm's Q/K are still
    quantised and decoded from its own state, and each arm still gets its own
    attention forward; only the reference decode, the reference `target` and the
    parent-side V are computed once.  The V five fields are arm-independent
    because the two arms hold equal-valued ``v_state`` -- the candidate is a
    copy of the parent with only ``q_state``/``k_state`` modified -- and because
    the V API reads its state without writing it.
    """

    reference = [
        _dequantize_nvfp4_float32(*item[role]).to(torch.float32)[None]
        for role in ("q", "k", "v")
    ]
    target = _a2_attention_forward(
        reference[0], reference[1], reference[2], q_heads, kv_heads, head_dim
    )
    v_hat = _dequantize_hif4(
        _AGR1_PARENT_V(*item["v"], kv_heads, head_dim, parent_states["v_state"])
    ).to(torch.float32)[None]
    losses = []
    for states in (parent_states, candidate_states):
        q_hat = _dequantize_hif4(
            _AGR1_PARENT_Q(*item["q"], q_heads, head_dim, states["q_state"])
        ).to(torch.float32)[None]
        k_hat = _dequantize_hif4(
            _AGR1_PARENT_K(*item["k"], kv_heads, head_dim, states["k_state"])
        ).to(torch.float32)[None]
        actual = _a2_attention_forward(
            q_hat, k_hat, v_hat, q_heads, kv_heads, head_dim
        )
        losses.append(float((actual - target).square().mean().item()))
    return losses[0], losses[1]


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict:
    states = _AGR1_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    info = {
        "agr1_arm": "fallback",
        "agr1_attempted": 0,
        "agr1_accepted": 0,
        "agr1_fit_windows": 0,
        "agr1_gate_windows": len(_AGR1_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < max(_AGR1_GATE_WINDOWS) + 1
        or int(q_num_heads) % int(kv_num_heads) != 0
    ):
        info["agr1_arm"] = "ineligible"
        return _agr1_annotated_states(states, info)
    try:
        fit_windows = calib_qkv_list[:len(calib_qkv_list) - len(_AGR1_GATE_WINDOWS)]
        train_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tq, tk, center, train_info = _agr1_train(
            fit_windows,
            states,
            q_num_heads,
            kv_num_heads,
            head_dim,
            train_device,
        )
        candidate = _agr1_parent_copy(states)
        candidate["q_state"]["learned_rotation"] = tq
        candidate["k_state"]["learned_rotation"] = tk
        if center is not None:
            candidate["k_state"]["learned_center"] = center
        gate_records = []
        accepted = True
        for index in _AGR1_GATE_WINDOWS:
            parent_loss, candidate_loss = _act1_gate_pair(
                calib_qkv_list[index], states, candidate,
                q_num_heads, kv_num_heads, head_dim,
            )
            current_pass = candidate_loss < parent_loss
            accepted = accepted and current_pass
            gate_records.append({
                "window": int(index),
                "pass": bool(current_pass),
                "parent_mse": float(parent_loss),
                "candidate_mse": float(candidate_loss),
            })
        info.update(train_info)
        info["agr1_attempted"] = 1
        info["agr1_fit_windows"] = len(fit_windows)
        info["agr1_gate_parent_mse"] = float(
            sum(item["parent_mse"] for item in gate_records) / len(gate_records)
        )
        info["agr1_gate_candidate_mse"] = float(
            sum(item["candidate_mse"] for item in gate_records) / len(gate_records)
        )
        if accepted:
            info["agr1_arm"] = "accepted"
            info["agr1_accepted"] = 1
            return _agr1_annotated_states(candidate, info)
        info["agr1_arm"] = "parent"
        return _agr1_annotated_states(states, info)
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
        info["agr1_arm"] = "fallback"
        info["agr1_error"] = f"{type(exc).__name__}: {exc}"
        return _agr1_annotated_states(states, info)
