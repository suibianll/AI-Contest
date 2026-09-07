"""Generate anchor26-a1/solution.py from anchor25-a1/solution.py.

Keeps: full base stack, _a21_exp/_a21_exp_backward/_a21_project (eigh +
analytic backward + spectral projection), _a21_gate_loss (true readout gate).
Replaces: the amax scale-proxy trainer (_a25_train) with the A26-A
sin grid-alignment trainer; entry hif4_calibration_attention updated in place.
Removes: dead amax helpers (_a21_scale_loss_grad, _a21_matrix_grad).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "anchor25-a1/solution.py"
DST = HERE / "solution.py"

text = SRC.read_text(encoding="utf-8")

A26_HEADER = '''# A26-A: phase grid-alignment inverse transform (sawtooth-sign triangular-
# amplitude grid-residual output error model, ledger grid F2). Single trunk
# on the base stack; S=0 starts from identity (no extra Hadamard
# pre-rotation). The training loss is the smooth first-order model of the
# F2 output effect with the residual eps = (delta/2)*|sin(pi*y)|*sgn(frac(y)
# -0.5), y = x/delta (delta = 0.25*base fixed at S=0, detached, base = 2^floor(log2(amax)) = the real codec scale_factor structure):
# zero at code points, maximum at half-grid, rounding-sawtooth sign.
# Unlike the amax-family proxies (A23 product / A25 scale) it uses amplitude
# AND phase; unlike A24 STE the residual model is an explicit nonzero
# analytic function, so the inverse float main path cancels only the
# Q_b*K_b^T term and the residual cross terms carry nonzero gradient.
# Gate is the true five-field readout MSE vs B.
'''

NEW_TRAINER = '''def _a26_grid_residual(x, delta):
    """Sawtooth-sign x triangular-amplitude grid-residual model.

    Per 64-element block: y = x/delta (delta fixed per block, detached);
    eps = (delta/2) * |sin(pi*y)| * sgn(frac(y) - 0.5).  Code points give
    eps = 0 with the largest gradient magnitude; half-grid gives the maximum
    |eps| = delta/2 (matching true nearest-neighbor rounding, which the pure
    first-term sine misses - it vanishes at half-grid too); the sign matches
    the rounding sawtooth on every unit interval, preserving the symbol
    correlation in the Q*eps_k^T cross terms.
    """
    blocks = x.reshape(x.shape[0], -1, 64)
    y = blocks / delta.clamp_min(1e-12)
    frac = y - torch.floor(y)
    sign_saw = torch.where(frac < 0.5, 1.0, -1.0)
    eps = (delta * 0.5) * torch.sin(math.pi * y).abs() * sign_saw
    return eps.reshape_as(x)


class _A26Exp(torch.autograd.Function):
    """exp(+-S) with the A24-fixed analytic eigh backward (float32 stable)."""

    @staticmethod
    def forward(ctx, s, sign):
        result, (values, vectors, ev, sign_value) = _a21_exp(s, sign)
        ctx.save_for_backward(values, vectors, ev)
        ctx.a26_sign = sign_value
        return result

    @staticmethod
    def backward(ctx, grad):
        values, vectors, ev = ctx.saved_tensors
        return _a21_exp_backward(grad, (values, vectors, ev, ctx.a26_sign)), None


def _a26_output_loss(u_q, u_k, e_q, e_k, delta_q, delta_k, qh, kh, dim):
    """Smooth first-order model of the F2 output effect, per GQA group.

    L = || Q_b eps_k^T + eps_q K_b^T - eps_q eps_k^T ||_F^2 over the full
    token cross product; Q_b/K_b are the rotated floats, eps the sin grid
    residuals. With eps == 0 this is identically zero (A24 check).
    """
    tokens = u_q.shape[0]
    q_rot = _a2_apply_group_rotation(u_q, qh, e_q)
    k_rot = _a2_apply_group_rotation(u_k, kh, e_k)
    eps_q = _a26_grid_residual(q_rot, delta_q).reshape(tokens, qh, dim)
    eps_k = _a26_grid_residual(k_rot, delta_k).reshape(tokens, kh, dim)
    qg = q_rot.reshape(tokens, kh, qh // kh, dim)
    kg = k_rot.reshape(tokens, kh, 1, dim)
    # Q terms head-batched: (T, kh, per_group, d) x (T, kh, 1, d) -> (T, kh, per_group, T)
    e1 = torch.einsum("tghd,tgjd->tghj", qg, eps_k.reshape(tokens, kh, 1, dim))
    e2 = torch.einsum("tghd,tgjd->tghj", eps_q.reshape(tokens, kh, qh // kh, dim),
                      kg.reshape(tokens, kh, 1, dim))
    e3 = torch.einsum("tghd,tgjd->tghj", eps_q.reshape(tokens, kh, qh // kh, dim),
                      eps_k.reshape(tokens, kh, 1, dim))
    err = e1 + e2 - e3
    return err.square().mean()


def _a26_train(windows, states, qh, kh, dim, device, force_zero=False):
    """A26-A training: sin grid-alignment first-order output error objective.

    Single trunk on the base stack coordinates; S=0 identity start. Fixed
    pre-registered configuration (see mechanism.md): Adam 32 steps, lr 0.01,
    clip 1.0, reg 1e-3*mean(S^2), symmetric zero-trace spectral projection
    clamp +-log2/2, delta = 0.25*block-amax frozen at S=0.
    """
    s = torch.zeros(kh, dim, dim, device=device)
    prepared = []
    for item in windows:
        fold = {}
        for role, heads in (("q", qh), ("k", kh)):
            dense = _dequantize_nvfp4_float32(*item[role]).to(device=device, dtype=torch.float32)
            u = _a1_stack_transform(dense, heads, dim, states[role + "_state"], role == "k")
            # Real codec grid: scale_factor is a power of two (2^floor(log2(amax)));
            # HiF4 code points sit on multiples of 0.25*base, NOT 0.25*amax.
            amax = u.reshape(u.shape[0], -1, 64).abs().amax(-1, keepdim=True).clamp_min(1e-12)
            base = torch.exp2(torch.floor(torch.log2(amax)))
            delta = (base * 0.25).detach()
            fold[role] = (u, delta, heads)
        prepared.append(fold)
    m = torch.zeros_like(s)
    v = torch.zeros_like(s)
    initial_loss = final_loss = 0.0
    for step in range(1, 33):
        s_leaf = s.detach().requires_grad_(True)
        e_q = _A26Exp.apply(s_leaf, 1.0)
        e_k = _A26Exp.apply(s_leaf, -1.0)
        loss = s_leaf.square().mean() * 1e-3
        for fold in prepared:
            u_q, delta_q, _ = fold["q"]
            u_k, delta_k, _ = fold["k"]
            loss = loss + _a26_output_loss(u_q, u_k, e_q, e_k, delta_q, delta_k, qh, kh, dim) / len(prepared)
        if step == 1:
            initial_loss = float(loss.detach())
        if force_zero:
            continue
        loss.backward()
        grad = s_leaf.grad.detach()
        grad = (grad + grad.transpose(-1, -2)) * 0.5
        grad = grad - torch.diag_embed(grad.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).expand(kh, dim))
        grad = grad * (1.0 / grad.norm().clamp_min(1.0))
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        s = _a21_project(s - 0.01 * (m / (1 - 0.9 ** step)) / ((v / (1 - 0.999 ** step)).sqrt() + 1e-8))
    if not force_zero:
        e_q = _A26Exp.apply(s.detach(), 1.0)
        e_k = _A26Exp.apply(s.detach(), -1.0)
        with torch.no_grad():
            total = 0.0
            for fold in prepared:
                u_q, delta_q, _ = fold["q"]
                u_k, delta_k, _ = fold["k"]
                total += float(_a26_output_loss(u_q, u_k, e_q, e_k, delta_q, delta_k, qh, kh, dim)) / len(prepared)
            final_loss = total + 1e-3 * float(s.square().mean())
    else:
        e_q = torch.eye(dim, device=device).expand(kh, dim, dim)
        e_k = e_q
    tq, tk = e_q.detach().cpu(), e_k.detach().cpu()
    inverse_error = float((tq.to(torch.float64) @ tk.to(torch.float64).transpose(-1, -2)
                           - torch.eye(dim, dtype=torch.float64)).abs().max())
    info = {"a26_steps": 32, "a26_attempted_groups": kh,
            "a26_initial_loss": initial_loss, "a26_final_loss": final_loss,
            "a26_inverse_error": inverse_error, "a26_s_norm": float(s.norm()),
            "a26_trunk": "single (no old training, S=0 start, sin grid-alignment objective)"}
    return tq, tk, info


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    """A26-A sin grid-alignment training gated against the BASE stack B."""
    # B: the base stack - sole parent, no old training retained.
    states = _V189_CALIBRATION_ATTENTION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states
    # C: the sin grid-alignment proposal trained on B coordinates.
    train_device = calib_qkv_list[0]["q"][0].device
    tq, tk, info = _a26_train(
        calib_qkv_list[:-1], states, q_num_heads, kv_num_heads, head_dim, train_device
    )
    candidate = {key: dict(value) for key, value in states.items()}
    candidate["q_state"]["learned_rotation"] = tq
    candidate["k_state"]["learned_rotation"] = tk
    # Gate: candidate C versus the BASE stack B on the original last calibration
    # window through the true deployed readout path; strictly smaller accepts,
    # ties (S=0 reproduces B exactly) retain B.
    parent_loss = _a21_gate_loss(calib_qkv_list[-1], states, q_num_heads, kv_num_heads, head_dim)
    candidate_loss = _a21_gate_loss(calib_qkv_list[-1], candidate, q_num_heads, kv_num_heads, head_dim)
    accepted = candidate_loss < parent_loss
    result = candidate if accepted else states
    info.update(
        a26_gate_base_mse=float(parent_loss),
        a26_gate_candidate_mse=float(candidate_loss),
        a26_accepted=int(bool(accepted)),
        a26_fallback_states="base_stack",
    )
    result["q_state"].update(info)
    result["k_state"].update(info)
    return result
'''

# 1) replace the A25 header comment (exact 3 lines)
old_header = '''# A25: single-trunk inverse transform training. No R3 old training retained;
# the new exp(+-S) scale-proxy training is the sole optimization. S=0 starts
# from identity (no Hadamard pre-rotation). Gate is the true readout MSE vs B.
'''
assert text.count(old_header) == 1, "A25 header not found"
text = text.replace(old_header, A26_HEADER)

# 2) drop dead amax helpers (between _a21_project end and _a21_gate_loss)
start = text.index("def _a21_scale_loss_grad")
end = text.index("def _a21_gate_loss")
text = text[:start] + text[end:]

# 3) replace trainer + entry (from _a25_train to EOF)
start = text.index("def _a25_train")
text = text[:start] + NEW_TRAINER

DST.write_text(text, encoding="utf-8")
print(f"written {DST} ({len(text.splitlines())} lines)")
