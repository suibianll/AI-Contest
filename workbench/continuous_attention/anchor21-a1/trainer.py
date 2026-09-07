
# A21-1: only continuous block scales are optimized; actual output is gated once.
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


def _a21_train(windows, states, qh, kh, dim, device):
    r0 = _a2_hadamard_orthogonal(dim)
    if r0 is None:
        r0 = torch.eye(dim)
    r0 = r0.to(device).expand(kh, dim, dim).contiguous()
    s = torch.zeros(kh, dim, dim, device=device)
    prepared = []
    for item in windows:
        fold = []
        for role, heads in (("q", qh), ("k", kh)):
            dense = _dequantize_nvfp4_float32(*item[role]).to(device=device, dtype=torch.float32)
            u = _a1_stack_transform(dense, heads, dim, states[role + "_state"], role == "k")
            x = _a2_apply_group_rotation(u, heads, r0)
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
        grad = _a21_exp_backward(gp, cp) + _a21_exp_backward(gm, cm) + 2e-3 * s / s.numel()
        grad = (grad + grad.transpose(-1, -2)) * 0.5
        grad = grad - torch.diag_embed(grad.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).expand(kh, dim))
        grad = grad * (1.0 / grad.norm().clamp_min(1.0))
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        s = _a21_project(s - 0.01 * (m / (1 - 0.9 ** step)) / ((v / (1 - 0.999 ** step)).sqrt() + 1e-8))
    ep, _ = _a21_exp(s)
    em, _ = _a21_exp(s, -1.0)
    ratios = [[], []]
    for fold in prepared:
        for role, ((x, denominator, heads), e) in enumerate(zip(fold, (ep, em))):
            value, _ = _a21_scale_loss_grad(_a2_apply_group_rotation(x, heads, e), denominator)
            ratios[role].append(float(value))
    final_loss = sum(sum(r) / len(r) for r in ratios) + 1e-3 * float(s.square().mean())
    tq, tk = r0 @ ep, r0 @ em
    inverse_error = float((tq @ tk.transpose(-1, -2) - torch.eye(dim, device=device)).abs().max())
    info = {"a21_steps": 32, "a21_attempted_groups": kh,
            "a21_initial_loss": initial_loss, "a21_final_loss": final_loss,
            "a21_q_scale_ratio2": sum(ratios[0]) / len(ratios[0]),
            "a21_k_scale_ratio2": sum(ratios[1]) / len(ratios[1]),
            "a21_inverse_error": inverse_error, "a21_s_norm": float(s.norm())}
    return tq.cpu(), tk.cpu(), info


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
    states = _V189_CALIBRATION_ATTENTION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if len(calib_qkv_list) < 2:
        return states
    device = calib_qkv_list[0]["q"][0].device
    tq, tk, info = _a21_train(calib_qkv_list[:-1], states, q_num_heads, kv_num_heads, head_dim, device)
    candidate = {key: dict(value) for key, value in states.items()}
    candidate["q_state"]["learned_rotation"] = tq
    candidate["k_state"]["learned_rotation"] = tk
    baseline_loss = _a21_gate_loss(calib_qkv_list[-1], states, q_num_heads, kv_num_heads, head_dim)
    candidate_loss = _a21_gate_loss(calib_qkv_list[-1], candidate, q_num_heads, kv_num_heads, head_dim)
    accepted = candidate_loss < baseline_loss
    result = candidate if accepted else states
    info.update(a21_accepted_groups=kv_num_heads if accepted else 0,
                a21_gate_base_mse=baseline_loss, a21_gate_candidate_mse=candidate_loss)
    result["q_state"].update(info)
    result["k_state"].update(info)
    return result
