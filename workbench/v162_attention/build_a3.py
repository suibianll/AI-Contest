"""Build candidate_a3 (A3): bounded-condition symmetric mixing T=exp(S)
trained jointly with the rotation, on top of the A1 deployed-aligned trainer.

Per the A3 card: T = V exp(clamp(lambda)) V^T from the symmetric zero-trace
S (init 0), eigenvalues clamped to [-log2/2, +log2/2] => cond(T) <= 2.
Q multiplies T, K multiplies T^-1 (continuous QK invariant).  Manual
gradients: Daleckii-Krein for the matrix exponential; K-side chain through
T^-1.  Everything else (deployed-aligned forward, guards, config) inherited
from candidate_a1.
"""

from pathlib import Path

p = Path(__file__).resolve().parent / "candidate_a3/solution.py"
src = Path(__file__).resolve().parent / "candidate_a1/solution.py".__str__()
src = Path("workbench/v162_attention/candidate_a1/solution.py").read_text(encoding="utf-8")

def rep(old, new, label, count=1):
    global src
    n = src.count(old)
    assert n == count, f"{label}: count={n} (expected {count})"
    src = src.replace(old, new)

# ---- 1. matrix-exponential helpers (calibration-only) --------------------
helpers = '''

_A3_LOG2_HALF = 0.34657359027997264  # log(2)/2


def _m_exp_sym(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """T = exp(clamp-eig(S)) for symmetric S; returns (T, T_inv, V, lam_clamped)."""

    lam, vec = torch.linalg.eigh(s)
    lam_c = lam.clamp(-_A3_LOG2_HALF, _A3_LOG2_HALF)
    exp_lam = torch.exp(lam_c)
    t = vec @ torch.diag_embed(exp_lam) @ vec.transpose(-1, -2)
    t_inv = vec @ torch.diag_embed(torch.exp(-lam_c)) @ vec.transpose(-1, -2)
    return t, t_inv, vec, lam_c


def _m_exp_sym_backward(
    grad_t: torch.Tensor,
    vec: torch.Tensor,
    lam: torch.Tensor,
    lam_c: torch.Tensor,
) -> torch.Tensor:
    """dL/dS via the Daleckii-Krein formula with clamped eigenvalues.

    f(l) = exp(clamp(l)); D_ij = (f(li)-f(lj))/(li-lj) (limit f'(li));
    dL/dS = V (G_s {hadamard} D) V^T with the diagonal using f'(li) and the
    clamp zeroing f' outside the bounds.
    """

    g_s = vec.transpose(-1, -2) @ grad_t @ vec
    lam_lo = lam_c <= -_A3_LOG2_HALF + 1e-12
    lam_hi = lam_c >= _A3_LOG2_HALF - 1e-12
    in_bounds = ~(lam_lo | lam_hi)
    f = torch.exp(lam_c)
    f_prime = torch.where(in_bounds, f, torch.zeros_like(f))
    d = f[..., :, None] - f[..., None, :]
    lam_diff = lam[..., :, None] - lam[..., None, :]
    same = lam_diff.abs() < 1e-12
    d = torch.where(same, f_prime[..., :, None].expand_as(d), d / lam_diff.clamp_min(1e-30) if False else torch.where(same, f_prime[..., :, None].expand_as(d), d / torch.where(same, torch.ones_like(lam_diff), lam_diff)))
    n = g_s * d
    n = n - torch.diag_embed(torch.diagonal(n, dim1=-2, dim2=-1)) + torch.diag_embed(g_s.diagonal(dim1=-2, dim2=-1) * f_prime)
    return vec @ n @ vec.transpose(-1, -2)
'''
anchor = "def _a2_train_rotation("
assert src.count(anchor) == 1
src = src.replace(anchor, helpers.strip() + "\n\n\n" + anchor)

# ---- 2. trainer: joint (theta, S) parameters ------------------------------
rep("""    theta = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(theta)
    exp_avg_sq = torch.zeros_like(theta)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")""",
"""    theta = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    theta_s = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(theta)
    exp_avg_sq = torch.zeros_like(theta)
    exp_avg_s = torch.zeros_like(theta_s)
    exp_avg_sq_s = torch.zeros_like(theta_s)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")""", "params")

# ---- 3. forward: rotation then symmetric mixing ---------------------------
rep("""            q_hat = _a1_deployed_encode(
                _a2_apply_group_rotation(item["u_q"], q_heads, rotation), q_state
            )
            k_hat = _a1_deployed_encode(
                _a2_apply_group_rotation(item["u_k"], kv_heads, rotation), k_state
            )""",
"""            t_sym, t_inv, evec, lam_c = _m_exp_sym(theta_s)
            q_pre = _a2_apply_group_rotation(item["u_q"], q_heads, rotation)
            k_pre = _a2_apply_group_rotation(item["u_k"], kv_heads, rotation)
            q_pre = torch.einsum("tghk,gkd->tghd", q_pre.reshape(tokens_q := q_pre.shape[0], groups, -1, dim), t_sym).reshape(tokens_q, -1)
            k_pre = torch.einsum("tgk,gkd->tgd", k_pre.reshape(k_pre.shape[0], groups, dim), t_inv).reshape(k_pre.shape[0], -1)
            q_hat = _a1_deployed_encode(q_pre, q_state)
            k_hat = _a1_deployed_encode(k_pre, k_state)""", "fwd-mix")

# ---- 4. backward: T chains on both sides ----------------------------------
rep("""            d_qhat, d_khat = _m_attention_backward(
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
            grad_theta = grad_theta + _m_cayley_backward(grad_c, theta)""",
"""            d_qhat, d_khat = _m_attention_backward(
                d_output[None], q_hat, k_hat, item["v_hat"], q_heads, kv_heads, head_dim
            )
            tokens_q = item["u_q"].shape[0]
            tokens_k = item["u_k"].shape[0]
            per_group = q_heads // groups
            uq3 = item["u_q"].reshape(tokens_q, groups, per_group, head_dim)
            dq_pre = torch.einsum("tghd,gde->tghk", d_qhat.reshape(tokens_q, groups, per_group, head_dim), t_sym).reshape(tokens_q, -1)
            dk_pre = torch.einsum("tgd,gde->tgk", d_khat.reshape(tokens_k, groups, head_dim), t_inv).reshape(tokens_k, -1)
            dq3 = dq_pre.reshape(tokens_q, groups, per_group, head_dim)
            uk3 = item["u_k"].reshape(tokens_k, kv_heads, head_dim)
            dk3 = dk_pre.reshape(tokens_k, kv_heads, head_dim)
            grad_rotation = torch.einsum("tghk,tghd->gkd", uq3, dq3)
            grad_rotation = grad_rotation + torch.einsum("tgk,tgd->gkd", uk3, dk3)
            grad_c = torch.einsum("kd,gkl->gdl", base, grad_rotation)
            grad_theta = grad_theta + _m_cayley_backward(grad_c, theta)
            ur3 = torch.einsum("tghk,gkh->tghd" if False else "tghk,gkd->tghd",
                               uq3, torch.einsum("gkd,gde->gke", rotation, t_sym))
            grad_t_q = torch.einsum("tghk,tghd->gkd", ur3, d_qhat.reshape(tokens_q, groups, per_group, head_dim))
            ukr = torch.einsum("tgk,gke->tge", uk3, torch.einsum("gkd,gde->gke", rotation, t_inv))
            grad_t_inv = torch.einsum("tge,tgd->gde", ukr, d_khat.reshape(tokens_k, groups, head_dim))
            grad_t = -torch.einsum("gde,gdf,gef->gdf" if False else "gef,gde,gdf->gdf",
                                   t_inv, grad_t_inv, t_inv) + grad_t_q
            grad_theta_s = _m_exp_sym_backward(grad_t, evec, lam, lam_c)""", "bwd-mix")

# ---- 5. regularizer: mean(S^2) ---------------------------------------------
rep("""        c_now, _right = _m_cayley_pair(theta)
        grad_c_reg = 2.0 * (c_now - eye) * (_A2_REG_WEIGHT / float(groups * dim * dim))
        grad_theta = grad_theta + _m_cayley_backward(grad_c_reg, theta)
        final_loss = float(data_loss)""",
"""        c_now, _right = _m_cayley_pair(theta)
        grad_c_reg = 2.0 * (c_now - eye) * (_A2_REG_WEIGHT / float(groups * dim * dim))
        grad_theta = grad_theta + _m_cayley_backward(grad_c_reg, theta)
        grad_theta_s = grad_theta_s + 2.0 * theta_s * (_A2_REG_WEIGHT / float(groups * dim * dim))
        final_loss = float(data_loss)""", "reg")

# ---- 6. Adam covers S ------------------------------------------------------
rep("""        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad_theta.square()
        bias1 = 1 - 0.9 ** (step_index + 1)
        bias2 = 1 - 0.999 ** (step_index + 1)
        theta = theta - _A2_TRAIN_LR * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)""",
"""        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad_theta.square()
        exp_avg_s = 0.9 * exp_avg_s + 0.1 * grad_theta_s
        exp_avg_sq_s = 0.999 * exp_avg_sq_s + 0.001 * grad_theta_s.square()
        bias1 = 1 - 0.9 ** (step_index + 1)
        bias2 = 1 - 0.999 ** (step_index + 1)
        theta = theta - _A2_TRAIN_LR * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)
        theta_s = theta_s - _A2_TRAIN_LR * (exp_avg_s / bias1) / ((exp_avg_sq_s / bias2).sqrt() + 1e-8)""", "adam-s")

# ---- 7. final extraction: R and T ship to the states ----------------------
rep("""    c, _right = _m_cayley_pair(theta)
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
    return rotation.detach().cpu().to(torch.float32), info""",
"""    c, _right = _m_cayley_pair(theta)
    rotation = torch.einsum("dk,gkl->gdl", base, c)
    identity_error = float((rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max())
    if identity_error > _A2_ORTHO_TOLERANCE:
        raise RuntimeError(f"A1 trained rotation failed orthogonality: {identity_error}")
    t_sym, t_inv, _vec, _lam = _m_exp_sym(theta_s)
    info = {
        "steps": _A2_TRAIN_STEPS,
        "final_train_loss": final_loss,
        "ortho_error": identity_error,
        "windows": len(prepared),
        "trainer": "manual-deployed-aligned+symmix",
    }
    return (
        rotation.detach().cpu().to(torch.float32),
        info,
        t_sym.detach().cpu().to(torch.float32),
        t_inv.detach().cpu().to(torch.float32),
    )""", "return-rt")

# ---- 8. wrapper: ship T/T_inv into the states ------------------------------
rep("""        rotation, info = _a2_train_rotation(
            calib_qkv_list[:-1], q_num_heads, kv_num_heads, head_dim, device,
            q_state=states["q_state"], k_state=states["k_state"],
            v_state=states["v_state"],
        )""",
"""        rotation, info, sym_t, sym_t_inv = _a2_train_rotation(
            calib_qkv_list[:-1], q_num_heads, kv_num_heads, head_dim, device,
            q_state=states["q_state"], k_state=states["k_state"],
            v_state=states["v_state"],
        )""", "unwrap")
rep("""            states["q_state"]["learned_rotation"] = cpu_rotation
            states["k_state"]["learned_rotation"] = cpu_rotation.clone()""",
"""            states["q_state"]["learned_rotation"] = cpu_rotation
            states["k_state"]["learned_rotation"] = cpu_rotation.clone()
            states["q_state"]["learned_sym_t"] = sym_t
            states["k_state"]["learned_sym_t_inv"] = sym_t_inv""", "state-inject")

# ---- 9. dynamic Q/K pass the symmetric mixes; injection applies them ------
rep("""        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(q_num_heads),""",
"""        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(q_num_heads),
        learned_sym_t=state.get("learned_sym_t"),""", "q-pass")

rep("""        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(kv_num_heads),""",
"""        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(kv_num_heads),
        learned_sym_t_inv=state.get("learned_sym_t_inv"),""", "k-pass")

rep("""    learned_rotation: Optional[torch.Tensor] = None,
    learned_rotation_num_heads: Optional[int] = None,
) -> dict[str, torch.Tensor]:""",
"""    learned_rotation: Optional[torch.Tensor] = None,
    learned_rotation_num_heads: Optional[int] = None,
    learned_sym_t: Optional[torch.Tensor] = None,
    learned_sym_t_inv: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:""", "sig")

rep("""    if learned_rotation is not None and learned_rotation_num_heads is not None:
        try:
            dense = _a2_apply_group_rotation(
                dense, int(learned_rotation_num_heads), learned_rotation
            )
        except Exception:  # noqa: BLE001 - degrade to the unrotated legal path
            pass""",
"""    if learned_rotation is not None and learned_rotation_num_heads is not None:
        try:
            dense = _a2_apply_group_rotation(
                dense, int(learned_rotation_num_heads), learned_rotation
            )
            if learned_sym_t is not None:
                dense = _a2_apply_sym_mix(dense, int(learned_rotation_num_heads), learned_sym_t)
        except Exception:  # noqa: BLE001 - degrade to the unrotated legal path
            pass
    if learned_sym_t_inv is not None and learned_rotation_num_heads is not None:
        try:
            dense = _a2_apply_sym_mix(dense, int(learned_rotation_num_heads), learned_sym_t_inv)
        except Exception:  # noqa: BLE001 - degrade to the unscaled legal path
            pass""", "inject")

# ---- 10. the symmetric-mix applier -----------------------------------------
mix_fn = '''

def _a2_apply_sym_mix(
    dense: torch.Tensor,
    num_heads: int,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply the per-group symmetric matrix to the head dimension (right mult)."""

    tokens = dense.shape[0]
    head_dim = dense.shape[-1] // num_heads
    grouped = dense.reshape(tokens, num_heads, head_dim)
    transformed = torch.einsum(
        "thk,gkd->thd" if False else "thk,gkd->thd",
        grouped.reshape(tokens, num_heads, head_dim),
        matrix.to(device=grouped.device, dtype=torch.float32),
    )
    return transformed.reshape(tokens, -1)
'''
anchor2 = "def _a2_train_rotation("
assert src.count(anchor2) == 1
src = src.replace(anchor2, mix_fn.strip() + "\n\n\n" + anchor2)

p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(src, encoding="utf-8")
print("candidate_a3 written")
