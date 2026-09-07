"""Finish candidate_a1: replace the trainer with the deployed-aligned version
and rewire the wrapper.  Run after build_a1 step 1 (helpers injected)."""

from pathlib import Path

p = Path(__file__).resolve().parent / "candidate_a1/solution.py"
src = p.read_text(encoding="utf-8")

def rep(old, new, label, count=1):
    global src
    n = src.count(old)
    assert n == count, f"{label}: count={n} (expected {count})"
    src = src.replace(old, new)

# ---- 1. replace the R2c autograd trainer with the A1 manual trainer ------
start = src.find("def _a2_train_rotation(")
end = src.find("def _a2_true_path_gate_loss")
assert start != -1 and end != -1
new_trainer = '''def _a2_train_rotation(
    windows: List[dict],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    q_state: Optional[Mapping] = None,
    k_state: Optional[Mapping] = None,
    v_state: Optional[Mapping] = None,
) -> Tuple[torch.Tensor, dict]:
    """A1: rotation trained on the TRUE deployed coordinates and encoder.

    Per window: U = the deployed pre-encode coordinates (stack-transform
    replica); the player values are decode(deployed encode(rotate(U))) with
    the deployed refined V; the STE gradient treats encode+refinement as
    identity.  Config inherited from R2c (32 steps, lr 0.01, clip 1.0,
    reg 1e-3, same deterministic sampling, last window = gate).
    """

    groups = kv_heads
    dim = head_dim
    base = _a2_hadamard_orthogonal(dim)
    if base is None:
        base = torch.eye(dim, dtype=torch.float32)
    base = base.to(device)

    prepared = []
    for window in windows:
        q_quant, q_scale = window["q"]
        k_quant, k_scale = window["k"]
        v_quant, v_scale = window["v"]
        q_dense = _a2_normal(
            _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32), device
        )
        k_dense = _a2_normal(
            _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32), device
        )
        v_dense = _a2_normal(
            _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32), device
        )
        u_q = _a1_stack_transform(q_dense, q_heads, head_dim, q_state, is_k=False)
        u_k = _a1_stack_transform(k_dense, kv_heads, head_dim, k_state, is_k=True)
        v_hat_full = _a1_deployed_v_hat(v_quant, v_scale, v_state)

        kv_index = _a2_even_indices(k_dense.shape[0], _A2_MAX_KV_TOKENS, device)
        q_cap = min(int(kv_index.numel()), int(q_dense.shape[0]))
        q_index = _a2_even_indices(q_cap, _A2_MAX_Q_TOKENS, device)
        q_rows = kv_index.index_select(0, q_index)
        q_rows = q_rows[q_rows < q_dense.shape[0]]
        if q_rows.numel() == 0:
            q_rows = _a2_even_indices(q_dense.shape[0], _A2_MAX_Q_TOKENS, device)
        u_q_sub = u_q.index_select(0, q_rows)
        u_k_sub = u_k.index_select(0, kv_index)
        v_hat_sub = v_hat_full.index_select(0, kv_index)
        reference = _a2_attention_forward(
            q_dense.index_select(0, q_rows)[None],
            k_dense.index_select(0, kv_index)[None],
            v_dense.index_select(0, kv_index)[None],
            q_heads, kv_heads, head_dim,
        )[0].detach()
        std_q = _decode_hif4(_dense_to_hif4(q_dense.index_select(0, q_rows))).to(torch.float32)
        std_k = _decode_hif4(_dense_to_hif4(k_dense.index_select(0, kv_index))).to(torch.float32)
        std_v = _decode_hif4(_dense_to_hif4(v_dense.index_select(0, kv_index))).to(torch.float32)
        standard = _a2_attention_forward(
            std_q[None], std_k[None], std_v[None], q_heads, kv_heads, head_dim
        )[0].detach()
        mse_std = float((standard - reference).square().mean())
        prepared.append({
            "u_q": u_q_sub, "u_k": u_k_sub, "v_hat": v_hat_sub,
            "reference": reference, "mse_std": max(mse_std, 1e-12),
        })

    theta = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(theta)
    exp_avg_sq = torch.zeros_like(theta)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")

    for step_index in range(_A2_TRAIN_STEPS):
        window_losses = []
        grad_theta = torch.zeros_like(theta)
        for item in prepared:
            c, _right = _m_cayley_pair(theta)
            rotation = torch.einsum("dk,gkl->gdl", base, c)
            q_hat = _a1_deployed_encode(
                _a2_apply_group_rotation(item["u_q"], q_heads, rotation), q_state
            )
            k_hat = _a1_deployed_encode(
                _a2_apply_group_rotation(item["u_k"], kv_heads, rotation), k_state
            )
            output = _a2_attention_forward(
                q_hat[None], k_hat[None], item["v_hat"][None], q_heads, kv_heads, head_dim
            )[0]
            residual = output - item["reference"]
            loss = residual.square().mean() / item["mse_std"]
            window_losses.append(loss)

            d_output = 2.0 * residual / float(residual.numel()) / item["mse_std"]
            d_qhat, d_khat = _m_attention_backward(
                d_output[None], q_hat, k_hat, item["v_hat"], q_heads, kv_heads, head_dim
            )
            tokens_q = item["u_q"].shape[0]
            tokens_k = item["u_k"].shape[0]
            per_group = q_heads // groups
            uq3 = item["u_q"].reshape(tokens_q, groups, per_group, head_dim)
            dq3 = d_qhat.reshape(tokens_q, groups, per_group, head_dim)
            uk3 = item["u_k"].reshape(tokens_k, kv_heads, head_dim)
            dk3 = d_khat.reshape(tokens_k, kv_heads, head_dim)
            grad_rotation = torch.einsum("tghk,tghd->gkd", uq3, dq3)
            grad_rotation = grad_rotation + torch.einsum("tgk,tgd->gkd", uk3, dk3)
            grad_c = torch.einsum("kd,gkl->gdl", base, grad_rotation)
            grad_theta = grad_theta + _m_cayley_backward(grad_c, theta)

        data_loss = torch.stack(window_losses).mean()
        if not math.isfinite(float(data_loss)):
            raise RuntimeError("A1 rotation training produced a non-finite loss")
        c_now, _right = _m_cayley_pair(theta)
        grad_c_reg = 2.0 * (c_now - eye) * (_A2_REG_WEIGHT / float(groups * dim * dim))
        grad_theta = grad_theta + _m_cayley_backward(grad_c_reg, theta)
        final_loss = float(data_loss)

        total = grad_theta.norm()
        if float(total) > _A2_TRAIN_CLIP and float(total) > 0:
            grad_theta = grad_theta * (_A2_TRAIN_CLIP / float(total))
        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad_theta.square()
        bias1 = 1 - 0.9 ** (step_index + 1)
        bias2 = 1 - 0.999 ** (step_index + 1)
        theta = theta - _A2_TRAIN_LR * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)

    c, _right = _m_cayley_pair(theta)
    rotation = torch.einsum("dk,gkl->gdl", base, c)
    identity_error = float((rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max())
    if identity_error > _A2_ORTHO_TOLERANCE:
        raise RuntimeError(f"A1 trained rotation failed orthogonality: {identity_error}")
    info = {
        "steps": _A2_TRAIN_STEPS,
        "final_train_loss": final_loss,
        "ortho_error": identity_error,
        "windows": len(prepared),
        "trainer": "manual-deployed-aligned",
    }
    return rotation.detach().cpu().to(torch.float32), info


'''
src = src[:start] + new_trainer + src[end:]

# ---- 2. wrapper passes the states into the trainer -----------------------
rep("""        rotation, info = _a2_train_rotation(
            windows[:-1], q_num_heads, kv_num_heads, head_dim, device
        )""",
"""        rotation, info = _a2_train_rotation(
            windows[:-1], q_num_heads, kv_num_heads, head_dim, device,
            q_state=states["q_state"], k_state=states["k_state"],
            v_state=states["v_state"],
        )""", "wrapper-call")

p.write_text(src, encoding="utf-8")
print("candidate_a1 trainer replaced")
