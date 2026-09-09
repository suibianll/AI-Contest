# ---------------------------------------------------------------------------
# A-C76.5: residual-directed C76.4 orthogonal candidate.
#
# v205 subtraction pricing shows removing C76.4 costs 84 official points, so
# C76.4 stays untouched.  A-C76.5 only adds one deterministic calibration
# candidate per block size B in {16, 32}, derived from the final-output
# residual at the C76.4 search's parent state:
#   G_g = sum over cases ( Q_g^T dQ_g + K_g^T dK_g ),
# where Q_g/K_g are the pre-C76.4 dense values and dQ_g/dK_g come from the
# existing Attention backward.  For each contiguous B x B diagonal block of
# G_g, C = (G_block + G_block^T)/2; the eigenvector with the largest
# |eigenvalue| (ties: smaller index) gives the block signs s_j = +1 if u_j>=0
# else -1, with the first nonzero element fixed positive.  The candidate then
# uses the existing dynamic path x -> (x * signs) @ H_B with no permutation,
# no seed/threshold/eigenvector search.  It is dropped into the same complete
# deployed-MSE selection as the existing C76.4 candidates.
# ---------------------------------------------------------------------------

_C765_BLOCKS = (16, 32)


def _c765_pre_c764_dense(
    dense: torch.Tensor,
    state: dict,
    is_k: bool,
    num_heads: int,
) -> torch.Tensor:
    """Deployed dense value at the C76.4 sign insertion point."""

    out = dense
    channels = int(out.shape[-1])
    if is_k and int(state.get("center_mode", 0)) != 0:
        out = _center_attention_k(
            out,
            int(num_heads),
            channels // int(num_heads),
            int(state.get("center_mode", 0)),
            state.get("center_value"),
        )
    multiplier = state.get("multiplier")
    if multiplier is not None:
        scale = _safe_positive_vector(multiplier, channels).to(out.device)
        out = out * scale.reshape(*([1] * (out.ndim - 1)), channels)
    permutation = state.get("permutation")
    if permutation is not None:
        order = permutation.detach().to(
            device=out.device, dtype=torch.int64
        ).reshape(-1)
        out = out.index_select(-1, order)
    return out


def _c765_qk_coupling(
    q_pairs: list,
    k_pairs: list,
    v_hats: list,
    refs: list,
    q_state: dict,
    k_state: dict,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> Optional[torch.Tensor]:
    """Case-equal Q/K right-transform coupling at the C76.4 parent state."""

    groups = int(kv_num_heads)
    per_group = q_num_heads // groups
    coupling = None
    for (q_quant, q_scale), (k_quant, k_scale), v_hat, (ref_c, _ref_n) in zip(
        q_pairs, k_pairs, v_hats, refs
    ):
        q_hat = _dequantize_hif4(
            hif4_dynamic_quantize_q(
                q_quant, q_scale, q_num_heads, head_dim, q_state
            )
        ).to(torch.float32)
        k_hat = _dequantize_hif4(
            hif4_dynamic_quantize_k(
                k_quant, k_scale, kv_num_heads, head_dim, k_state
            )
        ).to(torch.float32)
        out_c = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
        )
        residual = out_c - ref_c
        d_output = 2.0 * residual / float(residual.numel())
        d_qhat, d_khat = _m_attention_backward(
            d_output[None], q_hat, k_hat, v_hat,
            q_num_heads, kv_num_heads, head_dim,
        )
        u_q = _c765_pre_c764_dense(
            _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32),
            q_state, False, q_num_heads,
        )
        u_k = _c765_pre_c764_dense(
            _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32),
            k_state, True, kv_num_heads,
        )
        tokens_q = int(u_q.shape[0])
        tokens_k = int(u_k.shape[0])
        u_q3g = u_q.reshape(tokens_q, groups, per_group, head_dim)
        dq3g = d_qhat.reshape(tokens_q, groups, per_group, head_dim)
        u_k3 = u_k.reshape(tokens_k, kv_num_heads, head_dim)
        dk3 = d_khat.reshape(tokens_k, kv_num_heads, head_dim)
        contribution = torch.einsum("tghk,tghd->gkd", u_q3g, dq3g)
        contribution = contribution + torch.einsum("tgk,tgd->gkd", u_k3, dk3)
        coupling = contribution if coupling is None else coupling + contribution
    if coupling is None:
        return None
    return coupling / float(len(q_pairs))


def _c765_signs_from_coupling(
    coupling: torch.Tensor,
    block: int,
) -> torch.Tensor:
    """Dominant-eigenvector sign pattern per contiguous block of each group."""

    groups, head_dim, _ = coupling.shape
    signs = torch.ones(groups, head_dim, dtype=torch.float32)
    for group in range(groups):
        for start in range(0, head_dim, block):
            sub = coupling[group, start:start + block, start:start + block]
            sym = 0.5 * (sub + sub.transpose(-1, -2))
            evals, evecs = torch.linalg.eigh(sym)
            vec = evecs[:, int(evals.abs().argmax())]
            nonzero = vec != 0
            if bool(nonzero.any()):
                first = int(nonzero.nonzero()[0])
                if float(vec[first]) < 0:
                    vec = -vec
            signs[group, start:start + block] = torch.where(
                vec >= 0, torch.ones_like(vec), -torch.ones_like(vec)
            )
    return signs
