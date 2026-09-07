
# A24: residual inverse transform on the complete parent coordinates trained
# DIRECTLY on the true quantized attention output error (STE manual
# gradients, A2-style). Only the residual loss differs from A22-2; the base
# flow, parent construction, gate, optimizer budget and compile rules are
# identical. No scale-regularization term: the error target is the objective.
def _a21_exp(s, sign=1.0):
    values, vectors = torch.linalg.eigh(s)
    ev = (sign * values).exp()
    result = (vectors * ev.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return result, (values, vectors, ev, sign)


def _a21_exp_backward(grad, cache):
    values, vectors, ev, sign = cache
    diff = values.unsqueeze(-1) - values.unsqueeze(-2)
    near = diff.abs() < 1e-6
    ratio = (ev.unsqueeze(-1) - ev.unsqueeze(-2)) / torch.where(near, torch.ones_like(diff), diff)
    midpoint = (sign * (values.unsqueeze(-1) + values.unsqueeze(-2)) * 0.5).exp() * sign
    divided = torch.where(near, midpoint, ratio)
    local = vectors.transpose(-1, -2) @ ((grad + grad.transpose(-1, -2)) * 0.5) @ vectors
    return vectors @ (local * divided) @ vectors.transpose(-1, -2)


def _a21_project(s):
    values, vectors = torch.linalg.eigh((s + s.transpose(-1, -2)) * 0.5)
    bound = math.log(2.0) / 2.0
    lo = values.amin(-1, keepdim=True) - bound
    hi = values.amax(-1, keepdim=True) + bound
    # Projection onto both the spectral box and zero-trace plane, not successive
    # clamping/centering (which would violate one of the constraints).
    for _ in range(32):
        mid = (lo + hi) * 0.5
        positive = (values - mid).clamp(-bound, bound).sum(-1, keepdim=True) > 0
        lo = torch.where(positive, mid, lo)
        hi = torch.where(positive, hi, mid)
    values = (values - (lo + hi) * 0.5).clamp(-bound, bound)
    return (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _a21_matrix_grad(x, grad, heads, groups):
    dim = x.shape[-1] // heads
    return torch.einsum("tghi,tghj->gij", x.reshape(-1, groups, heads // groups, dim),
                        grad.reshape(-1, groups, heads // groups, dim))


def _a21_gate_loss(item, states, qh, kh, dim):
    decoded, reference = [], []
    for role, heads in (("q", qh), ("k", kh), ("v", kh)):
        api = {"q": hif4_dynamic_quantize_q, "k": hif4_dynamic_quantize_k, "v": hif4_dynamic_quantize_v}[role]
        params = api(*item[role], heads, dim, states[role + "_state"])
        decoded.append(_dequantize_hif4(params).float()[None])
        reference.append(_dequantize_nvfp4_float32(*item[role]).float()[None])
    actual = _a2_attention_forward(*decoded, qh, kh, dim)
    target = _a2_attention_forward(*reference, qh, kh, dim)
    return float((actual - target).square().mean())


def _a24_train(windows, states, qh, kh, dim, device, force_zero=False):
    """Residual scale training with the TRUE quantized-output error objective.

    Same parent coordinates and compile rules as A22-2; the loss is the real
    deployed-path attention output MSE against the NVFP4 reference (A2-style
    mse_std normalization, straight-through manual gradients). Gradient is a
    per-window SUM then clip-norm, exactly like the R3 rotation trainer.
    """
    parent_q = states["q_state"].get("learned_rotation")
    parent_k = states["k_state"].get("learned_rotation")
    parent_c = states["k_state"].get("learned_center")
    rq = parent_q.to(device=device, dtype=torch.float32) if parent_q is not None else torch.eye(dim, device=device).expand(kh, dim, dim)
    rk = parent_k.to(device=device, dtype=torch.float32) if parent_k is not None else torch.eye(dim, device=device).expand(kh, dim, dim)
    has_center = parent_c is not None
    c = parent_c.to(device=device, dtype=torch.float32) if has_center else torch.zeros(kh, dim, device=device)
    s = torch.zeros(kh, dim, dim, device=device)
    prepared = []
    for window in windows:
        q_full = _a2_normal(window["q"], device)
        k_full = _a2_normal(window["k"], device)
        v_full = _a2_normal(window["v"], device)
        kv_index = _a2_even_indices(k_full.shape[0], _A2_MAX_KV_TOKENS, device)
        q_cap = min(int(kv_index.numel()), int(q_full.shape[0]))
        q_index = _a2_even_indices(q_cap, _A2_MAX_Q_TOKENS, device)
        q_rows = kv_index.index_select(0, q_index)
        q_rows = q_rows[q_rows < q_full.shape[0]]
        if q_rows.numel() == 0:
            q_rows = _a2_even_indices(q_full.shape[0], _A2_MAX_Q_TOKENS, device)
        q_sub = q_full.index_select(0, q_rows)
        k_sub = k_full.index_select(0, kv_index)
        v_sub = v_full.index_select(0, kv_index)
        reference = _a2_attention_forward(
            q_sub[None], k_sub[None], v_sub[None], qh, kh, dim
        )[0].detach()
        std_q = _dequantize_hif4(_dense_to_hif4(q_sub)).to(torch.float32)
        std_k = _dequantize_hif4(_dense_to_hif4(k_sub)).to(torch.float32)
        std_v = _dequantize_hif4(_dense_to_hif4(v_sub)).to(torch.float32)
        standard = _a2_attention_forward(
            std_q[None], std_k[None], std_v[None], qh, kh, dim
        )[0].detach()
        mse_std = max(float((standard - reference).square().mean()), 1e-12)
        v_hat = _dequantize_hif4(_dense_to_hif4(v_sub)).to(torch.float32)
        # Complete parent coordinates (rotation applied; K includes center).
        x_q = _a2_apply_group_rotation(
            _a1_stack_transform(q_sub, qh, dim, states["q_state"], False), qh, rq)
        x_k = _a2_apply_group_rotation(
            _a1_stack_transform(k_sub, kh, dim, states["k_state"], True), kh, rk)
        if has_center:
            lead = x_k.shape[:-1]
            x_k = (x_k.reshape(*lead, kh, x_k.shape[-1] // kh)
                   + c.reshape(*([1] * len(lead)), kh, x_k.shape[-1] // kh)).reshape(x_k.shape)
        prepared.append({"x_q": x_q.detach(), "x_k": x_k.detach(),
                         "v_hat": v_hat, "reference": reference, "mse_std": mse_std})
    m, v = torch.zeros_like(s), torch.zeros_like(s)
    initial_loss = final_loss = 0.0
    for step in range(1, _A2_TRAIN_STEPS + 1):
        ep, cp = _a21_exp(s)
        em, cm = _a21_exp(s, -1.0)
        gp, gm = torch.zeros_like(s), torch.zeros_like(s)
        window_losses = []
        for it in prepared:
            y_q = _a2_apply_group_rotation(it["x_q"], qh, ep)
            y_k = _a2_apply_group_rotation(it["x_k"], kh, em)
            q_hat = _dequantize_hif4(_dense_to_hif4(y_q)).to(torch.float32)
            k_hat = _dequantize_hif4(_dense_to_hif4(y_k)).to(torch.float32)
            output = _a2_attention_forward(q_hat[None], k_hat[None], it["v_hat"][None], qh, kh, dim)[0]
            residual = output - it["reference"]
            loss_item = residual.square().mean() / it["mse_std"]
            window_losses.append(loss_item)
            if force_zero:
                continue
            d_output = 2.0 * residual / float(residual.numel()) / it["mse_std"]
            d_qhat, d_khat = _m_attention_backward(
                d_output[None], q_hat, k_hat, it["v_hat"], qh, kh, dim)
            # Straight-through: quantized decode/encode is treated as identity,
            # so the loss gradient flows to the pre-encode coordinates x@exp(±S).
            gp.add_(_a21_matrix_grad(it["x_q"], d_qhat, qh, kh))
            gm.add_(_a21_matrix_grad(it["x_k"], d_khat, kh, kh))
        data_loss = torch.stack(window_losses).mean()
        if step == 1:
            initial_loss = float(data_loss)
        if not math.isfinite(float(data_loss)):
            raise RuntimeError("A24 output-error training produced a non-finite loss")
        if force_zero:
            continue
        grad = _a21_exp_backward(gp, cp) + _a21_exp_backward(gm, cm)
        grad = (grad + grad.transpose(-1, -2)) * 0.5
        grad = grad - torch.diag_embed(grad.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).expand(kh, dim))
        total = grad.norm()
        if float(total) > _A2_TRAIN_CLIP and float(total) > 0:
            grad = grad * (_A2_TRAIN_CLIP / float(total))
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        bias1 = 1 - 0.9 ** step
        bias2 = 1 - 0.999 ** step
        s = _a21_project(s - _A2_TRAIN_LR * (m / bias1) / ((v / bias2).sqrt() + 1e-8))
    if not force_zero:
        ep, _ = _a21_exp(s)
        em, _ = _a21_exp(s, -1.0)
        final_acc = []
        for it in prepared:
            y_q = _a2_apply_group_rotation(it["x_q"], qh, ep)
            y_k = _a2_apply_group_rotation(it["x_k"], kh, em)
            q_hat = _dequantize_hif4(_dense_to_hif4(y_q)).to(torch.float32)
            k_hat = _dequantize_hif4(_dense_to_hif4(y_k)).to(torch.float32)
            output = _a2_attention_forward(q_hat[None], k_hat[None], it["v_hat"][None], qh, kh, dim)[0]
            final_acc.append(float(((output - it["reference"]).square().mean()) / it["mse_std"]))
        final_loss = sum(final_acc) / len(final_acc)
    else:
        ep = em = torch.eye(dim, device=device).expand(kh, dim, dim)
    # Deployment compile: parent rotation carries the residual; the additive
    # K-center must transform with exp(-S) or the parent QK continuity breaks.
    tq, tk = rq @ ep, rk @ em
    c_new = (c.unsqueeze(-2) @ em).squeeze(-2) if has_center else None
    inverse_error = float((tq @ tk.transpose(-1, -2) - (rq @ rk.transpose(-1, -2))).abs().max())
    info = {"a24_steps": _A2_TRAIN_STEPS, "a24_attempted_groups": kh,
            "a24_initial_loss": initial_loss, "a24_final_loss": final_loss,
            "a24_inverse_error": inverse_error, "a24_s_norm": float(s.norm()),
            "a24_parent_arm": "rotation" if parent_q is not None else "identity",
            "a24_center_compiled": bool(has_center),
            "a24_objective": "true-quantized-output-mse-STE"}
    return tq.cpu(), tk.cpu(), (c_new.cpu() if has_center else None), info


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    """A24 output-error residual proposal gated against the COMPLETE parent
    state (identical flow to A22-2; only the residual training objective
    differs: true quantized attention output error instead of scale proxies)."""
    # B: the base stack, computed exactly once.
    states = _V189_CALIBRATION_ATTENTION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states
    # P: the complete parent - ORIGINAL R3 training and selection logic.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = [
        {
            "q": _dequantize_nvfp4_float32(*item["q"]).to(torch.float32),
            "k": _dequantize_nvfp4_float32(*item["k"]).to(torch.float32),
            "v": _dequantize_nvfp4_float32(*item["v"]).to(torch.float32),
        }
        for item in calib_qkv_list
    ]
    rotation, a2_info, center = _a2_train_rotation(
        windows[:-1], q_num_heads, kv_num_heads, head_dim, device
    )
    gate_window = calib_qkv_list[-1]
    loss_identity = _a2_true_path_gate_loss(
        gate_window, q_num_heads, kv_num_heads, head_dim, states, None, device
    )
    loss_rotation = _a2_true_path_gate_loss(
        gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device, center
    )
    parent = {key: dict(value) for key, value in states.items()}
    parent_arm = "identity"
    if loss_rotation < loss_identity:
        cpu_rotation = rotation.detach().cpu().to(torch.float32).clone()
        parent["q_state"]["learned_rotation"] = cpu_rotation
        parent["k_state"]["learned_rotation"] = cpu_rotation.clone()
        parent["k_state"]["learned_center"] = center
        parent_arm = "rotation"
    # C': the output-error residual proposal in the COMPLETE parent
    # coordinates (same optimizer budget and compile rules as A22-2).
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tq, tk, c_new, info = _a24_train(
        windows[:-1], parent, q_num_heads, kv_num_heads, head_dim, train_device
    )
    candidate = {key: dict(value) for key, value in parent.items()}
    candidate["q_state"]["learned_rotation"] = tq
    candidate["k_state"]["learned_rotation"] = tk
    if c_new is not None:
        candidate["k_state"]["learned_center"] = c_new
    # Gate: candidate C' versus the COMPLETE parent P on the original last
    # calibration window through the true deployed readout path; strictly
    # smaller accepts, ties (S=0 reproduces P exactly) retain the parent.
    parent_loss = _a21_gate_loss(gate_window, parent, q_num_heads, kv_num_heads, head_dim)
    candidate_loss = _a21_gate_loss(gate_window, candidate, q_num_heads, kv_num_heads, head_dim)
    accepted = candidate_loss < parent_loss
    result = candidate if accepted else parent
    info.update(
        a24_gate_parent_mse=float(parent_loss),
        a24_gate_candidate_mse=float(candidate_loss),
        a24_accepted=int(bool(accepted)),
        a24_fallback_states="complete_parent",
        a2_gate_loss_identity=float(loss_identity),
        a2_gate_loss_rotation=float(loss_rotation),
        a2_steps=int(a2_info["steps"]),
        a2_train_loss=float(a2_info["final_train_loss"]),
        a2_ortho_error=float(a2_info["ortho_error"]),
    )
    result["q_state"].update(info)
    result["k_state"].update(info)
    return result
