
# A22-2: residual inverse transform on top of the COMPLETE R3 parent
# coordinates. S=0 restores the parent bit-for-bit; the deployment compiles
# Rq_new = Rq @ exp(S), Rk_new = Rk @ exp(-S) and c_new = c @ exp(-S) so the
# additive K-center transforms together with the rotation.
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


def _a21_scale_loss_grad(x, denominator):
    # Use actual contiguous 64-wide codec blocks, including head boundaries.
    blocks = x.reshape(-1, x.shape[-1] // 64, 64)
    amax = blocks.abs().amax(-1, keepdim=True)
    denom = denominator.clamp_min(1e-12)
    loss = (amax / denom).square().mean()
    ties = blocks.abs() == amax
    grad = 2 * amax / denom.square() / amax.numel()
    grad = grad * blocks.sign() * ties / ties.sum(-1, keepdim=True)
    return loss, grad.reshape_as(x)


def _a21_matrix_grad(x, grad, heads, groups):
    dim = x.shape[-1] // heads
    return torch.einsum("tghi,tghj->gij", x.reshape(-1, groups, heads // groups, dim),
                        grad.reshape(-1, groups, heads // groups, dim))


def _a22b_train(windows, states, qh, kh, dim, device, force_zero=False):
    """Residual scale training in the COMPLETE parent coordinates.

    windows carry raw calibration items; ``states`` is the COMPLETE parent
    state dict P (rotation arm carries learned_rotation/learned_center, the
    identity arm carries none). Denominators are the S=0 parent-coordinate
    amaxes. With ``force_zero`` the residual is pinned to S=0, which must
    reproduce the parent exactly.
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
    for item in windows:
        fold = []
        for role, heads in (("q", qh), ("k", kh)):
            dense = _dequantize_nvfp4_float32(*item[role]).to(device=device, dtype=torch.float32)
            u = _a1_stack_transform(dense, heads, dim, states[role + "_state"], role == "k")
            rotation = rq if role == "q" else rk
            x = _a2_apply_group_rotation(u, heads, rotation)
            if role == "k" and has_center:
                lead = x.shape[:-1]
                x = (x.reshape(*lead, heads, x.shape[-1] // heads)
                     + c.reshape(*([1] * len(lead)), heads, x.shape[-1] // heads)).reshape(x.shape)
            denominator = x.reshape(-1, x.shape[-1] // 64, 64).abs().amax(-1, keepdim=True)
            fold.append((x, denominator, heads))
        prepared.append(fold)
    m, v = torch.zeros_like(s), torch.zeros_like(s)
    initial_loss = final_loss = 0.0
    for step in range(1, 33):
        ep, cp = _a21_exp(s)
        em, cm = _a21_exp(s, -1.0)
        gp, gm = torch.zeros_like(s), torch.zeros_like(s)
        loss = s.square().mean() * 1e-3
        for fold in prepared:
            for (x, denominator, heads), e, g in zip(fold, (ep, em), (gp, gm)):
                transformed = _a2_apply_group_rotation(x, heads, e)
                value, dx = _a21_scale_loss_grad(transformed, denominator)
                loss = loss + value / len(prepared)
                g.add_(_a21_matrix_grad(x, dx, heads, kh) / len(prepared))
        if step == 1:
            initial_loss = float(loss)
        if force_zero:
            continue
        grad = _a21_exp_backward(gp, cp) + _a21_exp_backward(gm, cm) + 2e-3 * s / s.numel()
        grad = (grad + grad.transpose(-1, -2)) * 0.5
        grad = grad - torch.diag_embed(grad.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).expand(kh, dim))
        grad = grad * (1.0 / grad.norm().clamp_min(1.0))
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        s = _a21_project(s - 0.01 * (m / (1 - 0.9 ** step)) / ((v / (1 - 0.999 ** step)).sqrt() + 1e-8))
    if not force_zero:
        ep, _ = _a21_exp(s)
        em, _ = _a21_exp(s, -1.0)
        ratios = [[], []]
        for fold in prepared:
            for role, ((x, denominator, heads), e) in enumerate(zip(fold, (ep, em))):
                value, _ = _a21_scale_loss_grad(_a2_apply_group_rotation(x, heads, e), denominator)
                ratios[role].append(float(value))
        final_loss = sum(sum(r) / len(r) for r in ratios) + 1e-3 * float(s.square().mean())
    else:
        ep = em = torch.eye(dim, device=device).expand(kh, dim, dim)
    # Deployment compile: parent rotation carries the residual; the additive
    # K-center must transform with exp(-S) or the parent QK continuity breaks.
    tq, tk = rq @ ep, rk @ em
    c_new = (c.unsqueeze(-2) @ em).squeeze(-2) if has_center else None
    if has_center:
        # QK continuity of the compiled pair: UQ tq (UK tk)^T is preserved by
        # construction; the center term is additive and checked in verify.
        pass
    inverse_error = float((tq @ tk.transpose(-1, -2) - (rq @ rk.transpose(-1, -2))).abs().max())
    info = {"a22b_steps": 32, "a22b_attempted_groups": kh,
            "a22b_initial_loss": initial_loss, "a22b_final_loss": final_loss,
            "a22b_q_scale_ratio2": sum(ratios[0]) / len(ratios[0]) if not force_zero else 1.0,
            "a22b_k_scale_ratio2": sum(ratios[1]) / len(ratios[1]) if not force_zero else 1.0,
            "a22b_inverse_error": inverse_error, "a22b_s_norm": float(s.norm()),
            "a22b_parent_arm": "rotation" if parent_q is not None else "identity",
            "a22b_center_compiled": bool(has_center)}
    return tq.cpu(), tk.cpu(), (c_new.cpu() if has_center else None), info


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


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    """Residual scale proposal gated against the COMPLETE R3 parent state."""
    # B: the base stack, computed exactly once.
    states = _V189_CALIBRATION_ATTENTION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states
    # P: the complete R3 parent - ORIGINAL training and selection logic.
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
    # C': the residual proposal trained in the COMPLETE parent coordinates.
    train_device = calib_qkv_list[0]["q"][0].device
    tq, tk, c_new, info = _a22b_train(
        calib_qkv_list[:-1], parent, q_num_heads, kv_num_heads, head_dim, train_device
    )
    candidate = {key: dict(value) for key, value in parent.items()}
    candidate["q_state"]["learned_rotation"] = tq
    candidate["k_state"]["learned_rotation"] = tk
    if c_new is not None:
        candidate["k_state"]["learned_center"] = c_new
    # A22 gate: candidate C' versus the COMPLETE parent P on the original last
    # calibration window through the true deployed readout path; strictly
    # smaller accepts, ties (S=0 reproduces P exactly) retain the parent.
    parent_loss = _a21_gate_loss(gate_window, parent, q_num_heads, kv_num_heads, head_dim)
    candidate_loss = _a21_gate_loss(gate_window, candidate, q_num_heads, kv_num_heads, head_dim)
    accepted = candidate_loss < parent_loss
    result = candidate if accepted else parent
    info.update(
        a22_gate_parent_mse=float(parent_loss),
        a22_gate_candidate_mse=float(candidate_loss),
        a22_accepted=int(bool(accepted)),
        a22_fallback_states="complete_r3",
        a2_gate_loss_identity=float(loss_identity),
        a2_gate_loss_rotation=float(loss_rotation),
        a2_steps=int(a2_info["steps"]),
        a2_train_loss=float(a2_info["final_train_loss"]),
        a2_ortho_error=float(a2_info["ortho_error"]),
    )
    result["q_state"].update(info)
    result["k_state"].update(info)
    return result
