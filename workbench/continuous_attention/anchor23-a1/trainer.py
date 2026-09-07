
# A23: residual inverse transform on the complete parent coordinates with the
# JOINT Q/K block-scale PRODUCT objective. Everything else matches A22-2
# byte-for-byte in structure (same base flow, same parent construction, same
# gate, same optimizer); only the training loss changes. The product sees
# through constant Q*c / K/c rescalings that the additive objective cannot.
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


def _a23_block_amax(x, nb):
    """Per-token amax of each contiguous 64-wide codec block."""
    return x.reshape(x.shape[0], x.shape[-1] // 64, 64).abs().amax(-1)


def _a23_moments(xq, xk, qh, kh, dim):
    """Joint Q/K block moments with the layout-aware aggregation.

    Head-aligned layouts (dim % 64 == 0, every evaluation panel): returns
    (aQ, aK) of shape (kh, nb) where aQ(g,b) averages amax^2 over the
    group's Q heads and tokens and aK(g,b) over tokens of K head g.
    Other layouts (contract fuzz only): group-level aggregation, (kh,).
    """
    if dim % 64 == 0:
        nb = dim // 64
        amax_q = xq.reshape(xq.shape[0], qh, nb, 64).abs().amax(-1)
        amax_k = xk.reshape(xk.shape[0], kh, nb, 64).abs().amax(-1)
        hpg = qh // kh
        a_q = amax_q.square().reshape(xq.shape[0], kh, hpg, nb).mean(dim=(0, 2))
        a_k = amax_k.square().mean(0)
        return a_q, a_k
    amax_q = _a23_block_amax(xq, dim // 64)
    amax_k = _a23_block_amax(xk, dim // 64)
    hpg = max(qh // kh, 1)
    group_q = ((torch.arange(amax_q.shape[1], device=xq.device) * 64) // dim // hpg).clamp_max(kh - 1)
    group_k = ((torch.arange(amax_k.shape[1], device=xk.device) * 64) // dim).clamp_max(kh - 1)
    count_q = torch.bincount(group_q, minlength=kh).clamp_min(1).to(amax_q.dtype)
    count_k = torch.bincount(group_k, minlength=kh).clamp_min(1).to(amax_k.dtype)
    a_q = torch.zeros(kh, device=xq.device, dtype=amax_q.dtype).index_add_(0, group_q, amax_q.square().mean(0)) / count_q
    a_k = torch.zeros(kh, device=xk.device, dtype=amax_k.dtype).index_add_(0, group_k, amax_k.square().mean(0)) / count_k
    return a_q, a_k


def _a23_product_loss_grad(xq, xk, qh, kh, dim, denom_q, denom_k):
    """Joint product loss and manual element gradients on head-aligned blocks.

    The denominators are the S=0 parent aggregates (precomputed constants).
    Ties share the subgradient equally; zero blocks get zero gradient.
    """
    nb = dim // 64
    tokens_q = xq.shape[0]
    tokens_k = xk.shape[0]
    hpg = qh // kh
    bq = xq.reshape(tokens_q, qh, nb, 64)
    bk = xk.reshape(tokens_k, kh, nb, 64)
    amax_q = bq.abs().amax(-1)
    amax_k = bk.abs().amax(-1)
    a_q = amax_q.square().reshape(tokens_q, kh, hpg, nb).mean(dim=(0, 2))
    a_k = amax_k.square().mean(0)
    dq = denom_q.clamp_min(1e-12)
    dk = denom_k.clamp_min(1e-12)
    prod = a_q * a_k / (dq * dk)
    loss = prod.mean()
    ga_q = a_k / (dq * dk) / prod.numel()
    ga_k = a_q / (dq * dk) / prod.numel()
    # d loss / d amax via the aggregate means; zero blocks vanish (2*amax = 0).
    g_amax_q = 2.0 * amax_q * ga_q.repeat_interleave(hpg, dim=0)[None] / (tokens_q * hpg)
    g_amax_k = 2.0 * amax_k * ga_k[None] / tokens_k
    ties_q = bq.abs() == amax_q.unsqueeze(-1)
    ties_k = bk.abs() == amax_k.unsqueeze(-1)
    n_ties_q = ties_q.sum(-1, keepdim=True).clamp_min(1)
    n_ties_k = ties_k.sum(-1, keepdim=True).clamp_min(1)
    grad_q = torch.where(ties_q, g_amax_q.unsqueeze(-1) * bq.sign() / n_ties_q,
                         torch.zeros_like(bq)).reshape(tokens_q, qh * dim)
    grad_k = torch.where(ties_k, g_amax_k.unsqueeze(-1) * bk.sign() / n_ties_k,
                         torch.zeros_like(bk)).reshape(tokens_k, kh * dim)
    return loss, grad_q, grad_k


def _a23_product_loss_grad_groupfallback(xq, xk, qh, kh, dim, denom_q, denom_k):
    """Layout-constrained fallback for dim % 64 != 0 (contract fuzz only).

    Blocks cannot align inside heads, so the product aggregates at group
    level: blocks map to the group of their starting head and merge within
    it. Not used by any evaluation panel.
    """
    tokens_q, tokens_k = xq.shape[0], xk.shape[0]
    bq = xq.reshape(tokens_q, -1, 64)
    bk = xk.reshape(tokens_k, -1, 64)
    amax_q = bq.abs().amax(-1)
    amax_k = bk.abs().amax(-1)
    hpg = max(qh // kh, 1)
    group_q = ((torch.arange(amax_q.shape[1], device=xq.device) * 64) // dim // hpg).clamp_max(kh - 1)
    group_k = ((torch.arange(amax_k.shape[1], device=xk.device) * 64) // dim).clamp_max(kh - 1)
    count_q = torch.bincount(group_q, minlength=kh).clamp_min(1).to(amax_q.dtype)
    count_k = torch.bincount(group_k, minlength=kh).clamp_min(1).to(amax_k.dtype)
    a_q = torch.zeros(kh, device=xq.device, dtype=amax_q.dtype).index_add_(0, group_q, amax_q.square().mean(0)) / count_q
    a_k = torch.zeros(kh, device=xk.device, dtype=amax_k.dtype).index_add_(0, group_k, amax_k.square().mean(0)) / count_k
    dq = denom_q.clamp_min(1e-12)
    dk = denom_k.clamp_min(1e-12)
    prod = a_q * a_k / (dq * dk)
    loss = prod.mean()
    ga_q = a_k / (dq * dk) / prod.numel()
    ga_k = a_q / (dq * dk) / prod.numel()
    per_block_q = (tokens_q * count_q[group_q])[None]
    per_block_k = (tokens_k * count_k[group_k])[None]
    g_amax_q = 2.0 * amax_q * ga_q[group_q][None] / per_block_q
    g_amax_k = 2.0 * amax_k * ga_k[group_k][None] / per_block_k
    ties_q = bq.abs() == amax_q.unsqueeze(-1)
    ties_k = bk.abs() == amax_k.unsqueeze(-1)
    n_ties_q = ties_q.sum(-1, keepdim=True).clamp_min(1)
    n_ties_k = ties_k.sum(-1, keepdim=True).clamp_min(1)
    grad_q = torch.where(ties_q, g_amax_q.unsqueeze(-1) * bq.sign() / n_ties_q,
                         torch.zeros_like(bq)).reshape(tokens_q, -1)
    grad_k = torch.where(ties_k, g_amax_k.unsqueeze(-1) * bk.sign() / n_ties_k,
                         torch.zeros_like(bk)).reshape(tokens_k, -1)
    return loss, grad_q, grad_k


def _a23_train(windows, states, qh, kh, dim, device, force_zero=False):
    """Residual scale training with the JOINT Q/K product objective.

    Identical to A22-2's ``_a22b_train`` except the loss: the S=0 parent
    aggregate product replaces the per-token additive amax ratios.
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
        dense_q = _dequantize_nvfp4_float32(*item["q"]).to(device=device, dtype=torch.float32)
        u_q = _a1_stack_transform(dense_q, qh, dim, states["q_state"], False)
        x_q = _a2_apply_group_rotation(u_q, qh, rq)
        dense_k = _dequantize_nvfp4_float32(*item["k"]).to(device=device, dtype=torch.float32)
        u_k = _a1_stack_transform(dense_k, kh, dim, states["k_state"], True)
        x_k = _a2_apply_group_rotation(u_k, kh, rk)
        if has_center:
            lead = x_k.shape[:-1]
            x_k = (x_k.reshape(*lead, kh, x_k.shape[-1] // kh)
                   + c.reshape(*([1] * len(lead)), kh, x_k.shape[-1] // kh)).reshape(x_k.shape)
        prepared.append((x_q, x_k))
    # S=0 parent aggregates are the loss denominators (precomputed constants).
    denom_q_list, denom_k_list = [], []
    for x_q, x_k in prepared:
        d_q, d_k = _a23_moments(x_q, x_k, qh, kh, dim)
        denom_q_list.append(d_q)
        denom_k_list.append(d_k)
    m, v = torch.zeros_like(s), torch.zeros_like(s)
    initial_loss = final_loss = 0.0
    loss_fn = _a23_product_loss_grad if dim % 64 == 0 else _a23_product_loss_grad_groupfallback
    for step in range(1, 33):
        ep, cp = _a21_exp(s)
        em, cm = _a21_exp(s, -1.0)
        gp, gm = torch.zeros_like(s), torch.zeros_like(s)
        loss = s.square().mean() * 1e-3
        for (x_q, x_k), d_q, d_k in zip(prepared, denom_q_list, denom_k_list):
            t_q = _a2_apply_group_rotation(x_q, qh, ep)
            t_k = _a2_apply_group_rotation(x_k, kh, em)
            value, dx_q, dx_k = loss_fn(t_q, t_k, qh, kh, dim, d_q, d_k)
            loss = loss + value / len(prepared)
            gp.add_(_a21_matrix_grad(x_q, dx_q, qh, kh) / len(prepared))
            gm.add_(_a21_matrix_grad(x_k, dx_k, kh, kh) / len(prepared))
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
        final_acc = []
        for (x_q, x_k), d_q, d_k in zip(prepared, denom_q_list, denom_k_list):
            t_q = _a2_apply_group_rotation(x_q, qh, ep)
            t_k = _a2_apply_group_rotation(x_k, kh, em)
            value, _, _ = loss_fn(t_q, t_k, qh, kh, dim, d_q, d_k)
            final_acc.append(float(value))
        final_loss = sum(final_acc) / len(final_acc) + 1e-3 * float(s.square().mean())
    else:
        ep = em = torch.eye(dim, device=device).expand(kh, dim, dim)
    # Deployment compile: parent rotation carries the residual; the additive
    # K-center must transform with exp(-S) or the parent QK continuity breaks.
    tq, tk = rq @ ep, rk @ em
    c_new = (c.unsqueeze(-2) @ em).squeeze(-2) if has_center else None
    inverse_error = float((tq @ tk.transpose(-1, -2) - (rq @ rk.transpose(-1, -2))).abs().max())
    info = {"a23_steps": 32, "a23_attempted_groups": kh,
            "a23_initial_loss": initial_loss, "a23_final_loss": final_loss,
            "a23_product_ratio_final": sum(float(vv) for vv in final_acc) / len(final_acc) if not force_zero else 1.0,
            "a23_inverse_error": inverse_error, "a23_s_norm": float(s.norm()),
            "a23_parent_arm": "rotation" if parent_q is not None else "identity",
            "a23_center_compiled": bool(has_center),
            "a23_layout": "head-aligned" if dim % 64 == 0 else "group-fallback"}
    return tq.cpu(), tk.cpu(), (c_new.cpu() if has_center else None), info


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    """A23 product-objective residual proposal gated against the COMPLETE
    parent state (identical flow to A22-2; only the residual loss differs)."""
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
    # C': the product-objective residual proposal in the COMPLETE parent
    # coordinates (same optimizer, same 32 steps, same compile rules).
    train_device = calib_qkv_list[0]["q"][0].device
    tq, tk, c_new, info = _a23_train(
        calib_qkv_list[:-1], parent, q_num_heads, kv_num_heads, head_dim, train_device
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
        a23_gate_parent_mse=float(parent_loss),
        a23_gate_candidate_mse=float(candidate_loss),
        a23_accepted=int(bool(accepted)),
        a23_fallback_states="complete_parent",
        a2_gate_loss_identity=float(loss_identity),
        a2_gate_loss_rotation=float(loss_rotation),
        a2_steps=int(a2_info["steps"]),
        a2_train_loss=float(a2_info["final_train_loss"]),
        a2_ortho_error=float(a2_info["ortho_error"]),
    )
    result["q_state"].update(info)
    result["k_state"].update(info)
    return result
