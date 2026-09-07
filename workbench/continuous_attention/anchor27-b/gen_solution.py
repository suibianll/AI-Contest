"""Generate anchor27-b/solution.py from anchor26-a1/solution.py.

Keeps: full base stack, _a21_exp/_a21_exp_backward/_a21_project (generic
spectral utilities), _a21_gate_loss (true readout gate), the
_V189_CALIBRATION_ATTENTION alias.
Replaces: the A26-A analytic grid-residual trainer (dead: LOCAL_REJECTED,
proxy anti-correlated with the adaptive encoder) with the A27-B true-readout
central-difference sign trainer on low-dim spectral S; entry
hif4_calibration_attention updated in place (same gate structure).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "anchor26-a1/solution.py"
DST = HERE / "solution.py"

text = SRC.read_text(encoding="utf-8")

A27_HEADER = '''# A27-B: low-dim spectral residual transform trained by TRUE-readout finite
# differences (ledger grid F2). Per KV group, S = U diag(d) U^T with U the
# normalized Sylvester (Kronecker-order) Hadamard and d constant on 8
# contiguous 32-wide bands; exp(+-S) is closed-form in U (no eigh). The
# training signal is the real deployed readout MSE (base stack -> learned
# group rotation -> _dense_to_hif4 with the state refine params -> decode ->
# per-group attention vs NVFP4 floats) evaluated by central differences
# eps=0.05 with sign steps eta=0.05, 8-step coordinate rotation; group
# losses are separable under GQA. Unlike A26-A (analytic independent-
# residual proxy, anti-correlated with the adaptive hierarchical encoder)
# and A24 (full-dim STE, zero gradient) this objective cannot disagree with
# the encoder it is evaluated on. Gate is the true five-field readout MSE
# vs B (unchanged from A26-A).
'''

A27_TAIL = '''def _a27_hadamard(dim, device):
    """Normalized Sylvester (Kronecker-order) Hadamard; orthonormal for powers of two."""

    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError("A27 Hadamard basis requires a power-of-two head_dim")
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / math.sqrt(dim)).to(torch.float32)


def _a27_band_expand(c, dim, bands):
    """(groups, bands) coefficients -> (groups, dim) diagonal values (contiguous bands)."""

    idx = torch.arange(dim, device=c.device) * bands // dim
    return c.gather(1, idx.unsqueeze(0).expand(c.shape[0], dim))


def _a27_exp_pair(c, u, bands):
    """exp(+S) / exp(-S) per group in the fixed basis U (closed form, no eigh)."""

    d = _a27_band_expand(c, u.shape[0], bands)
    pos = u @ torch.diag_embed(d.exp()) @ u.t()
    neg = u @ torch.diag_embed((-d).exp()) @ u.t()
    return pos, neg


def _a27_project(c, bound):
    """Zero-trace + box projection on band coefficients.

    Shift bisection (32 iterations) so the clamped coefficient sum is zero -
    the exact _a21_project eigenvalue semantics restricted to the 8-dim
    spectral coefficient space.
    """

    lo = c.amin(-1, keepdim=True) - bound
    hi = c.amax(-1, keepdim=True) + bound
    for _ in range(32):
        mid = (lo + hi) * 0.5
        positive = (c - mid).clamp(-bound, bound).sum(-1, keepdim=True) > 0
        lo = torch.where(positive, mid, lo)
        hi = torch.where(positive, hi, mid)
    return (c - (lo + hi) * 0.5).clamp(-bound, bound)


def _a27_group_attn(q, k, v, dim):
    """Per-GQA-group attention on token-major slices.

    q (T, per_group, dim); k, v (T, 1, dim) -> output (T, per_group, dim).
    Transposes to (heads, tokens, dim) first so the logits are token-token
    (per_group, T, T), matching _a2_attention_forward semantics.
    """

    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    logits = q_t @ k_t.transpose(-1, -2) / math.sqrt(dim)
    return (torch.softmax(logits, dim=-1) @ v_t).transpose(0, 1)


def _a27_train(windows, states, qh, kh, dim, device, force_zero=False):
    """A27-B training: true-readout central-difference sign gradient on a
    low-dim spectral S (8 Hadamard band coefficients per KV group).

    Fixed pre-registered configuration (mechanism.md): bands=8 contiguous
    32-wide Kronecker-order Hadamard bands; eps=0.05 central difference;
    eta=0.05 sign gradient (zero FD keeps the coordinate); 8-step coordinate
    rotation (step j updates coefficient j of every group; group losses are
    separable under GQA); zero-trace + +-log2/2 projection; per-fold
    evenly-spaced token cap 256 with identical Q/K indices; folds [:-1]
    train, last fold gates. The training readout replicates the deployed
    chain exactly: base stack -> learned group rotation -> _dense_to_hif4
    (state refine params) -> _dequantize_hif4 -> per-group attention MSE vs
    the NVFP4-float reference (v fixed from the parent v_state).
    """

    bands = 8
    bound = math.log(2.0) / 2.0
    eps = 0.05
    eta = 0.05
    cap = 256
    per_group = qh // kh
    q_width = per_group * dim
    u_mat = _a27_hadamard(dim, device)
    qs, ks = states["q_state"], states["k_state"]
    imp_q_all = qs.get("importance")
    if imp_q_all is not None:
        imp_q_all = imp_q_all.detach().to(device=device, dtype=torch.float32).reshape(-1)
    imp_k_all = ks.get("importance")
    if imp_k_all is not None:
        imp_k_all = imp_k_all.detach().to(device=device, dtype=torch.float32).reshape(-1)
    refine_q = dict(
        search_offsets=qs["offsets"],
        error_threshold=float(qs["error_threshold"]),
        accept_margin=float(qs["accept_margin"]),
        max_refine_ratio=float(qs["max_refine_ratio"]),
        max_refine_blocks=qs.get("max_refine_blocks"),
    )
    refine_k = dict(
        search_offsets=ks["offsets"],
        error_threshold=float(ks["error_threshold"]),
        accept_margin=float(ks["accept_margin"]),
        max_refine_ratio=float(ks["max_refine_ratio"]),
        max_refine_blocks=ks.get("max_refine_blocks"),
    )

    with torch.no_grad():
        prepared = []
        for item in windows:
            dense_q = _dequantize_nvfp4_float32(*item["q"]).to(device=device, dtype=torch.float32)
            dense_k = _dequantize_nvfp4_float32(*item["k"]).to(device=device, dtype=torch.float32)
            u_q = _a1_stack_transform(dense_q, qh, dim, qs, False)
            u_k = _a1_stack_transform(dense_k, kh, dim, ks, True)
            idx = _a2_even_indices(int(u_k.shape[0]), cap, device)
            keep = int(idx.numel())
            u_q = u_q.index_select(0, idx)
            u_k = u_k.index_select(0, idx)
            dense_q = dense_q.index_select(0, idx)
            dense_k = dense_k.index_select(0, idx)
            v_params = hif4_dynamic_quantize_v(*item["v"], kh, dim, states["v_state"])
            v_hat = _dequantize_hif4(v_params).to(torch.float32).index_select(0, idx)
            v_ref = _dequantize_nvfp4_float32(*item["v"]).to(torch.float32).index_select(0, idx)
            prepared.append({
                "u_q": u_q.reshape(keep, kh, q_width),
                "u_k": u_k.reshape(keep, kh, dim),
                "q_ref": dense_q.reshape(keep, kh, per_group, dim),
                "k_ref": dense_k.reshape(keep, kh, 1, dim),
                "v_hat": v_hat.reshape(keep, kh, dim),
                "v_ref": v_ref.reshape(keep, kh, dim),
            })

        def group_loss(g, c_g):
            d_g = _a27_band_expand(c_g.unsqueeze(0), dim, bands)[0]
            e_q = (u_mat @ torch.diag_embed(d_g.exp()) @ u_mat.t()).unsqueeze(0)
            e_k = (u_mat @ torch.diag_embed((-d_g).exp()) @ u_mat.t()).unsqueeze(0)
            total = 0.0
            for fold in prepared:
                q_rot = _a2_apply_group_rotation(fold["u_q"][:, g], per_group, e_q)
                k_rot = _a2_apply_group_rotation(fold["u_k"][:, g], 1, e_k)
                pq = _dense_to_hif4(
                    q_rot,
                    importance=None if imp_q_all is None else imp_q_all[g * q_width:(g + 1) * q_width],
                    **refine_q,
                )
                pk = _dense_to_hif4(
                    k_rot,
                    importance=None if imp_k_all is None else imp_k_all[g * dim:(g + 1) * dim],
                    **refine_k,
                )
                q_hat = _dequantize_hif4(pq).to(torch.float32).reshape(-1, per_group, dim)
                k_hat = _dequantize_hif4(pk).to(torch.float32).reshape(-1, 1, dim)
                actual = _a27_group_attn(q_hat, k_hat, fold["v_hat"][:, g:g + 1], dim)
                target = _a27_group_attn(fold["q_ref"][:, g], fold["k_ref"][:, g], fold["v_ref"][:, g:g + 1], dim)
                total = total + float((actual - target).square().mean())
            return total / len(prepared)

        c = torch.zeros(kh, bands, device=device)
        initial_loss = sum(group_loss(g, c[g]) for g in range(kh)) / kh
        moved = 0
        fd_nonzero = 0
        final_loss = float(initial_loss)
        if not force_zero:
            for step in range(bands):
                for g in range(kh):
                    cp = c.clone()
                    cp[g, step] += eps
                    cm = c.clone()
                    cm[g, step] -= eps
                    fd = group_loss(g, cp[g]) - group_loss(g, cm[g])
                    if fd != 0.0:
                        fd_nonzero += 1
                        cn = c.clone()
                        cn[g, step] -= eta * (1.0 if fd > 0.0 else -1.0)
                        c = _a27_project(cn, bound)
                        moved += 1
            final_loss = sum(group_loss(g, c[g]) for g in range(kh)) / kh
        e_q, e_k = _a27_exp_pair(c, u_mat, bands)
        tq, tk = e_q.detach().cpu(), e_k.detach().cpu()
        inverse_error = float((tq.to(torch.float64) @ tk.to(torch.float64).transpose(-1, -2)
                               - torch.eye(dim, dtype=torch.float64)).abs().max())
    info = {
        "a27_steps": bands, "a27_attempted_coords": bands * kh,
        "a27_moved_coords": int(moved), "a27_fd_nonzero": int(fd_nonzero),
        "a27_initial_loss": float(initial_loss), "a27_final_loss": float(final_loss),
        "a27_inverse_error": inverse_error, "a27_c_norm": float(c.norm()),
        "a27_epsilon": eps, "a27_eta": eta, "a27_token_cap": cap,
        "a27_trunk": "single (no old training, S=0 start, true-readout FD sign gradient)",
    }
    return tq, tk, info


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    """A27-B true-readout FD training gated against the BASE stack B."""
    # B: the base stack - sole parent, no old training retained.
    states = _V189_CALIBRATION_ATTENTION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states
    # C: the low-dim spectral proposal trained on B coordinates by true-readout FD.
    train_device = calib_qkv_list[0]["q"][0].device
    tq, tk, info = _a27_train(
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
        a27_gate_base_mse=float(parent_loss),
        a27_gate_candidate_mse=float(candidate_loss),
        a27_accepted=int(bool(accepted)),
        a27_fallback_states="base_stack",
    )
    result["q_state"].update(info)
    result["k_state"].update(info)
    return result
'''

# 1) replace the A26 header comment block (up to the first kept utility)
start = text.index("# A26-A: phase grid-alignment inverse transform")
end = text.index("def _a21_exp(s, sign=1.0):")
text = text[:start] + A27_HEADER + "\n" + text[end:]

# 2) replace the A26 trainer + entry (from _a26_grid_residual to EOF)
start = text.index("def _a26_grid_residual")
text = text[:start] + A27_TAIL

DST.write_text(text, encoding="utf-8")
print(f"written {DST} ({len(text.splitlines())} lines)")
