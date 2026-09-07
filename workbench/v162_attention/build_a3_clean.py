"""Clean builder: candidate_a3 = A1 + bounded-condition symmetric mixing (A3 card).

Replaces the autograd-dependent R2c trainer wholesale with the manual
deployed-aligned trainer extended with the joint T=exp(S) symmetric mixing
(Daleckii-Krein analytic gradients, eigenvalue clamping to cond<=2).
Single comprehensive rebuild from candidate_a1 - no incremental patches.
"""

from pathlib import Path

SRC = Path("workbench/v162_attention/candidate_a1/solution.py")
OUT = Path("workbench/v162_attention/candidate_a3/solution.py")

src = SRC.read_text(encoding="utf-8")

def rep(old, new, label, count=1):
    global src
    n = src.count(old)
    assert n == count, f"{label}: count={n} (expected {count})"
    src = src.replace(old, new)

# ---- 1. helpers: symmetric exp + Daleckii-Krein backward + applier --------
helpers = '''

_A3_LOG2_HALF = 0.34657359027997264  # log(2)/2


def _m_exp_sym(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """T = exp(clamp-eig(S)) for symmetric S; returns (T, T_inv, V, lam_c)."""

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
    sign: float = 1.0,
) -> torch.Tensor:
    """dL/dS for the branch T_sign = V exp(sign*clamp(lam)) V^T.

    Daleckii-Krein with f(l) = exp(sign*clamp(l)):
    D_ij = (f(li)-f(lj))/(li-lj) (limit f'(li)); dL/dS = V (G_s {hadamard} D) V^T.
    The result is symmetrized (S is symmetric; non-symmetric W makes the raw
    adjoint asymmetric).
    """

    g_s = vec.transpose(-1, -2) @ grad_t @ vec
    in_bounds = (lam_c > -_A3_LOG2_HALF + 1e-12) & (lam_c < _A3_LOG2_HALF - 1e-12)
    f = torch.exp(lam_c)
    f_prime = torch.where(in_bounds, sign * f, torch.zeros_like(f))
    d = f[..., :, None] - f[..., None, :]
    lam_diff = lam[..., :, None] - lam[..., None, :]
    same = lam_diff.abs() < 1e-12
    safe_diff = torch.where(same, torch.ones_like(lam_diff), lam_diff)
    d = torch.where(same, f_prime[..., :, None].expand_as(d), d / safe_diff)
    n = g_s * d
    n = n - torch.diag_embed(torch.diagonal(n, dim1=-2, dim2=-1)) + torch.diag_embed(
        g_s.diagonal(dim1=-2, dim2=-1) * f_prime
    )
    n = 0.5 * (n + n.transpose(-1, -2))
    return vec @ n @ vec.transpose(-1, -2)


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
        "thk,gkd->thd", grouped, matrix.to(device=grouped.device, dtype=torch.float32)
    )
    return transformed.reshape(tokens, -1)


def _a1_state_on_device(state: Mapping, device: torch.device) -> dict:
    """Copy the state dict with all tensors moved to ``device``."""

    moved = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            moved[key] = (
                value.detach().to(device, torch.float32)
                if value.is_floating_point()
                else value.detach().to(device)
            )
        else:
            moved[key] = value
    return moved


def _a1_stack_transform(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: Mapping,
    is_k: bool,
) -> torch.Tensor:
    """Replicate the deployed pre-encode chain for training (A0-verified)."""

    state = _a1_state_on_device(state, dense.device)
    channels = int(dense.shape[-1])
    out = dense.to(torch.float32)
    if is_k and int(state.get("center_mode", 0) or 0) != 0:
        out = _center_attention_k(
            out,
            int(state.get("num_heads", num_heads)),
            int(head_dim),
            int(state["center_mode"]),
            state.get("center_value"),
        )
    multiplier = state.get("multiplier")
    if multiplier is not None:
        scale = _safe_positive_vector(multiplier, channels).to(out.device)
        out = out * scale.reshape(*([1] * (out.ndim - 1)), channels)
    signs = state.get("rotation")
    if signs is not None:
        out = _apply_attention_rotation(
            out, int(num_heads), int(head_dim), signs, state.get("rotation_block")
        )
    if int(state.get("block_smooth_size", 0) or 0) != 0:
        block_signs = state.get("block_smooth_signs")
        if block_signs is not None:
            out = _apply_attention_rotation(
                out,
                int(num_heads),
                int(block_signs.shape[1]),
                block_signs,
                int(state["block_smooth_size"]),
            )
        else:
            out = _block_hadamard_transform(
                out, int(state["block_smooth_size"]), int(state.get("block_smooth_seed", 0))
            )
    pair = state.get("pair_transform")
    if pair is not None:
        out = _apply_attention_pair_transform(out, int(num_heads), int(head_dim), pair)
    return out


def _a1_deployed_encode(
    dense: torch.Tensor,
    state: Mapping,
) -> torch.Tensor:
    """Replicate the deployed encode+refinement and decode."""

    state = _a1_state_on_device(state, dense.device)
    if _ACTIVATION_SAMPLE_IMPORTANCE and dense.ndim == 2:
        refine_importance = torch.sqrt(
            dense.square().mean(dim=0).clamp_min(_EPS)
        )
    else:
        refine_importance = state.get("importance")
    params = _dense_to_hif4(
        dense,
        importance=refine_importance,
        search_offsets=state.get("offsets"),
        error_threshold=float(state.get("error_threshold", 0.0)),
        accept_margin=float(state.get("accept_margin", 0.0)),
        max_refine_ratio=float(state.get("max_refine_ratio", 0.0)),
        max_refine_blocks=(
            int(state["max_refine_blocks"])
            if state.get("max_refine_blocks") is not None
            else None
        ),
    )
    return _dequantize_hif4(params).to(torch.float32)


def _a1_deployed_v_hat(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    v_state: Mapping,
) -> torch.Tensor:
    """Deployed dynamic V output (decode of the refined params)."""

    params = _nvfp4_to_hif4(
        v_quant,
        v_scale,
        importance=v_state["importance"],
        search_offsets=v_state["offsets"],
        error_threshold=float(v_state["error_threshold"]),
        accept_margin=float(v_state["accept_margin"]),
        max_refine_ratio=float(v_state["max_refine_ratio"]),
        max_refine_blocks=int(v_state["max_refine_blocks"]),
    )
    return _dequantize_hif4(params).to(torch.float32)
'''
anchor = "def _a2_train_rotation("
assert src.count(anchor) == 1
src = src.replace(anchor, helpers.strip() + "\n\n\n" + anchor)

# ---- 2. replace the trainer wholesale --------------------------------------
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
) -> Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """A3: rotation + bounded symmetric mixing T=exp(S), deployed-aligned.

    Training forward: U (deployed pre-encode coordinates, A0-bitwise-verified)
    -> rotate(R) -> sym-mix(T / T^-1) -> deployed refined encode -> decode.
    STE gradients through the encode; analytic attention/Cayley/Daleckii-Krein
    backward.  Config inherited from R2c via A1.
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
        q_quant = q_quant.to(device)
        q_scale = q_scale.to(device)
        k_quant = k_quant.to(device)
        k_scale = k_scale.to(device)
        v_quant = v_quant.to(device)
        v_scale = v_scale.to(device)
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
        reference = _a2_attention_forward(
            q_dense.index_select(0, q_rows)[None],
            k_dense.index_select(0, kv_index)[None],
            v_dense.index_select(0, kv_index)[None],
            q_heads, kv_heads, head_dim,
        )[0].detach()
        std_q = _dequantize_hif4(_dense_to_hif4(q_dense.index_select(0, q_rows))).to(torch.float32)
        std_k = _dequantize_hif4(_dense_to_hif4(k_dense.index_select(0, kv_index))).to(torch.float32)
        std_v = _dequantize_hif4(_dense_to_hif4(v_dense.index_select(0, kv_index))).to(torch.float32)
        standard = _a2_attention_forward(
            std_q[None], std_k[None], std_v[None], q_heads, kv_heads, head_dim
        )[0].detach()
        mse_std = float((standard - reference).square().mean())
        prepared.append({
            "u_q": u_q.index_select(0, q_rows),
            "u_k": u_k.index_select(0, kv_index),
            "v_hat": v_hat_full.index_select(0, kv_index),
            "reference": reference, "mse_std": max(mse_std, 1e-12),
        })

    theta = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    theta_s = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(theta)
    exp_avg_sq = torch.zeros_like(theta)
    exp_avg_s = torch.zeros_like(theta_s)
    exp_avg_sq_s = torch.zeros_like(theta_s)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")

    for step_index in range(_A2_TRAIN_STEPS):
        window_losses = []
        grad_theta = torch.zeros_like(theta)
        grad_theta_s = torch.zeros_like(theta_s)
        for item in prepared:
            c, _right = _m_cayley_pair(theta)
            rotation = torch.einsum("dk,gkl->gdl", base, c)
            t_sym, t_inv, evec, lam_c = _m_exp_sym(theta_s)
            q_pre = _a2_apply_group_rotation(item["u_q"], q_heads, rotation)
            k_pre = _a2_apply_group_rotation(item["u_k"], kv_heads, rotation)
            tokens_q = q_pre.shape[0]
            tokens_k = k_pre.shape[0]
            q_pre = torch.einsum(
                "tghk,gkd->tghd",
                q_pre.reshape(tokens_q, groups, -1, dim), t_sym,
            ).reshape(tokens_q, -1)
            k_pre = torch.einsum(
                "tgk,gkd->tgd", k_pre.reshape(tokens_k, groups, dim), t_inv,
            ).reshape(tokens_k, -1)
            q_hat = _a1_deployed_encode(q_pre, q_state)
            k_hat = _a1_deployed_encode(k_pre, k_state)
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
            per_group = q_heads // groups
            dq_pre = torch.einsum(
                "tghd,gde->tghk",
                d_qhat.reshape(tokens_q, groups, per_group, head_dim), t_sym,
            ).reshape(tokens_q, -1)
            dk_pre = torch.einsum(
                "tgd,gde->tgk", d_khat.reshape(tokens_k, groups, head_dim), t_inv,
            ).reshape(tokens_k, -1)
            uq3 = item["u_q"].reshape(tokens_q, groups, per_group, head_dim)
            dq3 = dq_pre.reshape(tokens_q, groups, per_group, head_dim)
            uk3 = item["u_k"].reshape(tokens_k, kv_heads, head_dim)
            dk3 = dk_pre.reshape(tokens_k, kv_heads, head_dim)
            grad_rotation = torch.einsum("tghk,tghd->gkd", uq3, dq3)
            grad_rotation = grad_rotation + torch.einsum("tgk,tgd->gkd", uk3, dk3)
            grad_c = torch.einsum("kd,gkl->gdl", base, grad_rotation)
            grad_theta = grad_theta + _m_cayley_backward(grad_c, theta)
            ur3 = torch.einsum(
                "tghk,gkd->tghd", uq3,
                torch.einsum("gkd,gde->gke", rotation, t_sym),
            )
            grad_t_q = torch.einsum(
                "tghk,tghd->gkd", ur3, d_qhat.reshape(tokens_q, groups, per_group, head_dim)
            )
            ukr = torch.einsum(
                "tgk,gke->tge", uk3,
                torch.einsum("gkd,gde->gke", rotation, t_inv),
            )
            grad_t_inv = torch.einsum(
                "tge,tgd->gde", ukr, d_khat.reshape(tokens_k, groups, head_dim)
            )
            grad_theta_s = grad_theta_s + _m_exp_sym_backward(
                grad_t_q, evec, lam_c, lam_c, sign=1.0
            )
            grad_theta_s = grad_theta_s + _m_exp_sym_backward(
                grad_t_inv, evec, lam_c, lam_c, sign=-1.0
            )

        data_loss = torch.stack(window_losses).mean()
        if not math.isfinite(float(data_loss)):
            raise RuntimeError("A3 rotation training produced a non-finite loss")
        c_now, _right = _m_cayley_pair(theta)
        grad_c_reg = 2.0 * (c_now - eye) * (_A2_REG_WEIGHT / float(groups * dim * dim))
        grad_theta = grad_theta + _m_cayley_backward(grad_c_reg, theta)
        grad_theta_s = grad_theta_s + 2.0 * theta_s * (
            _A2_REG_WEIGHT / float(groups * dim * dim)
        )
        final_loss = float(data_loss)

        total = grad_theta.norm()
        if float(total) > _A2_TRAIN_CLIP and float(total) > 0:
            grad_theta = grad_theta * (_A2_TRAIN_CLIP / float(total))
        total_s = grad_theta_s.norm()
        if float(total_s) > _A2_TRAIN_CLIP and float(total_s) > 0:
            grad_theta_s = grad_theta_s * (_A2_TRAIN_CLIP / float(total_s))
        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad_theta.square()
        exp_avg_s = 0.9 * exp_avg_s + 0.1 * grad_theta_s
        exp_avg_sq_s = 0.999 * exp_avg_sq_s + 0.001 * grad_theta_s.square()
        bias1 = 1 - 0.9 ** (step_index + 1)
        bias2 = 1 - 0.999 ** (step_index + 1)
        theta = theta - _A2_TRAIN_LR * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)
        theta_s = theta_s - _A2_TRAIN_LR * (exp_avg_s / bias1) / ((exp_avg_sq_s / bias2).sqrt() + 1e-8)

    c, _right = _m_cayley_pair(theta)
    rotation = torch.einsum("dk,gkl->gdl", base, c)
    identity_error = float((rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max())
    if identity_error > _A2_ORTHO_TOLERANCE:
        raise RuntimeError(f"A3 trained rotation failed orthogonality: {identity_error}")
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
    )


'''
src = src[:start] + new_trainer + src[end:]

# ---- 3. wrapper: pass states; ship T/T_inv ---------------------------------
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

# ---- 4. dynamic Q/K pass the mixes; signature; injection -------------------
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

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(src, encoding="utf-8")
print("candidate_a3 (clean rebuild) written")
