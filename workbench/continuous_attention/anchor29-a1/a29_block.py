# ============================ A29 (anchor29-a1) ============================
# Final-output-residual-driven QK inverse-pair compensation.
# Frozen card: workbench/continuous_attention/anchor29-a1/mechanism.md +
# config.json. Per-GQA-group symmetric traceless S appended on the
# learned_rotation output side (row-vector right-multiply) with compiled
# center c' = c @ expm(-S); continuous QK logits are strictly invariant.
# Closed-form direction from the softmax-Jacobian minimum-norm logit residual
# of the deployed parent (rcond=1e-6, RMS-clamped to the parent logit-error
# RMS), one projected-CG solve per (driver fold, GQA group) on the frozen
# normal equations G_q S G_ek - G_e S G_k = Q~^T dL E_k - E_q^T dL K~
# (tol 1e-8, max 500 iters, no damping, no sweeps; twelve instances executed
# as one batched CG with tensor-masked freeze semantics identical to the
# per-instance procedure), top-4 |eigenvalue| rank, Frobenius-normalized
# equal-weight fold0-2 average, single code-boundary alpha
# (output-sensitivity weighted 1/64 quantile of first positive
# code-crossing distances, ||alpha*S||_F <= log(2)/2), fold3 strict
# selection between full parent and the unique proposal, fold4 fully
# independent hard-output holdout. A layer deploys only when BOTH gates
# strictly decrease the causal deployed hard-output MSE; otherwise the
# parent state is kept bitwise (S=0 restores R3 exactly because A29 never
# writes when any stage fails).
# Driver Q rows are evenly-spaced sampled (A2 even-indices precedent) to fit
# the card's cost model; fold3/4 gates remain full-window. Recorded in the
# A29 report as a cost-model correction.

_A29_MANT_LADDER = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_A29_CG_TOL = 1e-8
_A29_CG_MAXITER = 500
_A29_RANK = 4
_A29_QUANTILE = 1.0 / 64.0
_A29_FNORM_CAP = float(math.log(2.0) / 2.0)
_A29_RCOND = 1e-6
_A29_MIN_FOLDS = 5
_A29_FOLDS_DRIVER = 3
_A29_FOLD_SELECT = 3
_A29_FOLD_HOLDOUT = 4
_A29_ZERO_EPS = 1e-12
_A29_MANT_TOL = 1e-7
_A29_Q_ROWS = 32


def _a29_sym_traceless(S: torch.Tensor) -> torch.Tensor:
    """Project onto symmetric traceless matrices (Frobenius-orthogonal)."""

    S = 0.5 * (S + S.transpose(-1, -2))
    eye = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)
    tr = torch.diagonal(S, dim1=-2, dim2=-1).mean(dim=-1)
    return S - tr[..., None, None] * eye


def _a29_batched_sym_pinv(A: torch.Tensor, rcond: float) -> torch.Tensor:
    """Symmetric rcond-truncated pseudo-inverse via eigh (batched)."""

    values, vectors = torch.linalg.eigh(A)
    limit = rcond * values[..., -1:].clamp_min(_EPS)
    keep = values > limit
    safe = torch.where(keep, values, torch.ones_like(values))
    inv = torch.where(keep, 1.0 / safe, torch.zeros_like(safe))
    return (vectors * inv.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _a29_cg_solve(apply_op, rhs: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Single projected CG (reference implementation, kept for tests)."""

    b = _a29_sym_traceless(rhs)
    S = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    b_norm = float(b.norm()) + _A29_ZERO_EPS
    rs_old = float((r * r).sum())
    iters = 0
    for iters in range(1, _A29_CG_MAXITER + 1):
        Ap = _a29_sym_traceless(apply_op(p))
        pAp = float((p * Ap).sum())
        if not math.isfinite(pAp) or pAp <= 0.0:
            break
        alpha = rs_old / pAp
        S = _a29_sym_traceless(S + alpha * p)
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if math.sqrt(rs_new) / b_norm < _A29_CG_TOL:
            rs_old = rs_new
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    residual = math.sqrt(max(rs_old, 0.0)) / b_norm
    return S, residual, iters


def _a29_cg_solve_batched(apply_batched, rhs: torch.Tensor):
    """Batched projected CG over the twelve (driver fold, GQA group)
    instances. Per-instance procedure identical to _a29_cg_solve: a
    non-finite/non-positive curvature step freezes that instance (registered
    breakdown) and convergence below tol freezes it; all decisions are
    tensor-masked with a single sync at the end."""

    b = _a29_sym_traceless(rhs)
    n = b.shape[0]
    S = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    b_norm = b.norm(dim=(-2, -1)).clamp_min(_A29_ZERO_EPS)
    rs_old = (r * r).sum(dim=(-2, -1))
    active = torch.ones(n, device=b.device, dtype=torch.bool)
    iters_used = torch.zeros(n, device=b.device, dtype=torch.int32)
    for _it in range(1, _A29_CG_MAXITER + 1):
        if not bool(active.any()):
            break
        Ap = _a29_sym_traceless(apply_batched(p))
        pAp = (p * Ap).sum(dim=(-2, -1))
        step = active & (pAp > 0.0)
        alpha = torch.where(step, rs_old / pAp.clamp_min(_A29_ZERO_EPS), torch.zeros_like(pAp))
        S = _a29_sym_traceless(S + alpha[..., None, None] * p)
        r = r - alpha[..., None, None] * Ap
        rs_new = (r * r).sum(dim=(-2, -1))
        res = rs_new.clamp_min(0.0).sqrt() / b_norm
        newly_done = active & (~step | (res < _A29_CG_TOL))
        iters_used = torch.where(newly_done, torch.full_like(iters_used, _it), iters_used)
        active = active & ~newly_done
        beta = torch.where(step, rs_new / rs_old.clamp_min(_A29_ZERO_EPS), torch.zeros_like(rs_new))
        p = torch.where(
            (active & step)[..., None, None],
            r + beta[..., None, None] * p,
            torch.where(active[..., None, None], r, p),
        )
        rs_old = torch.where(active, rs_new, rs_old)
    residual = rs_old.clamp_min(0.0).sqrt() / b_norm
    return S, residual, iters_used


def _a29_exp_sym(S: torch.Tensor) -> torch.Tensor:
    """Matrix exponential of symmetric matrices via batched eigh."""

    values, vectors = torch.linalg.eigh(S)
    return (vectors * torch.exp(values).unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _a29_lift(coords: dict, device: torch.device) -> dict:
    """Move the float tensors of one coordinate bundle to the compute device
    (params dicts are lifted inside _a29_code_mid_and_block)."""

    lifted = {}
    for key, val in coords.items():
        if isinstance(val, torch.Tensor):
            lifted[key] = val.to(device)
        else:
            lifted[key] = val
    return lifted


def _a29_cont_coords(
    item: dict,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict:
    """Continuous R3 coordinates (learned_rotation output side, K center
    added) plus the parent hard decode (five-field params retained for the
    code-boundary step). Follows the input device for the deployed path."""

    q_quant, q_scale = item["q"]
    k_quant, k_scale = item["k"]
    v_quant, v_scale = item["v"]
    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)
    q_params = hif4_dynamic_quantize_q(q_quant, q_scale, q_heads, head_dim, states["q_state"])
    k_params = hif4_dynamic_quantize_k(k_quant, k_scale, kv_heads, head_dim, states["k_state"])
    v_params = hif4_dynamic_quantize_v(v_quant, v_scale, kv_heads, head_dim, states["v_state"])
    q_hat = _dequantize_hif4(q_params).to(torch.float32)
    k_hat = _dequantize_hif4(k_params).to(torch.float32)
    v_hat = _dequantize_hif4(v_params).to(torch.float32)

    q_cont = _attention_state_transform_dense(q_ref, states["q_state"], q_heads, head_dim, is_k=False)
    k_cont = _attention_state_transform_dense(k_ref, states["k_state"], kv_heads, head_dim, is_k=True)
    rot = states["q_state"].get("learned_rotation")
    if rot is not None:
        rot_d = rot.detach().to(device=q_cont.device, dtype=torch.float32)
        q_cont = _a2_apply_group_rotation(q_cont, q_heads, rot_d)
        k_cont = _a2_apply_group_rotation(k_cont, kv_heads, rot_d)
    center = states["k_state"].get("learned_center")
    if center is not None:
        c = center.detach().to(device=k_cont.device, dtype=torch.float32)
        T_k = int(k_cont.shape[0])
        k_cont = (k_cont.reshape(T_k, kv_heads, head_dim) + c.reshape(1, kv_heads, head_dim)).reshape(k_cont.shape)
    return {
        "q_ref": q_ref, "k_ref": k_ref, "v_ref": v_ref,
        "q_hat": q_hat, "k_hat": k_hat, "v_hat": v_hat,
        "q_params": q_params, "k_params": k_params,
        "q_cont": q_cont, "k_cont": k_cont,
    }


def _a29_fold_driver(
    coords: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict:
    """One driver fold on evenly-spaced Q rows (full rows when the fold is
    small): per-group Gram matrices and normal-equation RHS.

    The minimum-norm delta-l rows come from one stacked batched eigh over
    all (head, row) instances of A_i = V^T J_i^2 V. All four Q heads of a
    GQA group share the group K Gram matrices; the RHS sums per-head
    contributions (registered aggregation).
    """

    coords = _a29_lift(coords, device)
    T_q = int(coords["q_ref"].shape[0])
    T_k = int(coords["k_ref"].shape[0])
    group = q_heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    dev = device
    assert T_q == T_k, "A29 driver expects square calibration folds"

    if T_q > _A29_Q_ROWS:
        rows = _a2_even_indices(T_q, _A29_Q_ROWS, torch.device("cpu")).to(dev)
    else:
        rows = torch.arange(T_q, device=dev)
    Rn = int(rows.numel())

    q_hat4 = coords["q_hat"].reshape(T_q, kv_heads, group, head_dim)
    k_hatg = coords["k_hat"].reshape(T_k, kv_heads, head_dim)
    v_hatg = coords["v_hat"].reshape(T_k, kv_heads, head_dim)
    q_ref4 = coords["q_ref"].reshape(T_q, kv_heads, group, head_dim)
    k_refg = coords["k_ref"].reshape(T_k, kv_heads, head_dim)
    q_cont4 = coords["q_cont"].reshape(T_q, kv_heads, group, head_dim)
    k_contg = coords["k_cont"].reshape(T_k, kv_heads, head_dim)

    y_ref = _attention_forward(
        coords["q_ref"], coords["k_ref"], coords["v_ref"],
        q_heads, kv_heads, head_dim, True,
    ).reshape(T_q, kv_heads, group, head_dim)

    neg = torch.finfo(torch.float32).min

    # pass 1: row-sampled probs/residuals + parent logit-error RMS (one scalar per fold)
    probs_cache = torch.empty(kv_heads * group, Rn, T_k, device=dev, dtype=torch.float32)
    r_out_cache = torch.empty(kv_heads * group, Rn, head_dim, device=dev, dtype=torch.float32)
    err_sq = 0.0
    err_cnt = 0
    for g in range(kv_heads):
        Kcg = k_contg[:, g, :]
        Krg = k_refg[:, g, :]
        Vg = v_hatg[:, g, :]
        for h in range(group):
            gh = g * group + h
            Qh = q_hat4[rows, g, h, :]
            Qch = q_cont4[rows, g, h, :]
            # sampled query rows attend over the FULL key sequence (Rn, T_k)
            causal = rows[:, None] >= torch.arange(T_k, device=dev)[None, :]
            logits = torch.where(
                causal,
                (Qh @ Kcg.transpose(0, 1)) * scale,
                torch.full((Rn, T_k), neg, device=dev),
            )
            probs_cache[gh] = torch.softmax(logits, dim=-1)
            r_out_cache[gh] = y_ref[rows, g, h, :] - probs_cache[gh] @ Vg
            l_err = ((Qch @ Kcg.transpose(0, 1)) - (q_ref4[rows, g, h, :] @ Krg.transpose(0, 1))) * scale
            diff = torch.where(causal, l_err, torch.zeros_like(l_err))
            err_sq += float(diff.square().sum())
            err_cnt += int(causal.sum())
    l_rms = err_sq / max(float(err_cnt), 1.0)

    # pass 2: stacked minimum-norm delta-l over all (head, row) instances
    n_head = kv_heads * group
    A_stack = torch.empty(n_head * Rn, head_dim, head_dim, device=dev, dtype=torch.float32)
    for gh in range(n_head):
        g = gh // group
        Vg = v_hatg[:, g, :]
        probs = probs_cache[gh]
        r_out = r_out_cache[gh]  # placeholder, filled below
        p2 = probs.square()
        U = p2 @ Vg
        Sg = probs @ Vg
        P2 = p2.sum(dim=1)
        A = torch.einsum("it,td,te->ide", p2, Vg, Vg)
        A = A - U[:, :, None] * Sg[:, None, :] - Sg[:, :, None] * U[:, None, :]
        A = A + P2[:, None, None] * (Sg[:, :, None] * Sg[:, None, :])
        A_stack[gh * Rn:(gh + 1) * Rn] = A
    pinv_stack = _a29_batched_sym_pinv(A_stack, _A29_RCOND)

    G_q = [None] * kv_heads
    G_e = [None] * kv_heads
    G_k = [None] * kv_heads
    G_ek = [None] * kv_heads
    B = [None] * kv_heads
    keep = rows[:, None] >= torch.arange(T_k, device=dev)[None, :]
    keep_cnt = max(float(keep.float().sum()), 1.0)
    for gh in range(n_head):
        g = gh // group
        h = gh % group
        Kg = k_hatg[:, g, :]
        Kcg = k_contg[:, g, :]
        Vg = v_hatg[:, g, :]
        Qch = q_cont4[rows, g, h, :]
        Eqh = q_hat4[rows, g, h, :] - Qch
        Ekg = Kg - Kcg
        probs = probs_cache[gh]
        r_out = r_out_cache[gh]
        pinvA = pinv_stack[gh * Rn:(gh + 1) * Rn]

        tmp = torch.einsum("bde,bd->be", pinvA, r_out)
        w = torch.einsum("be,te->bt", tmp, Vg)
        dL = probs * w - probs * (probs * w).sum(dim=1, keepdim=True)
        d_ms = float(dL.square().sum()) / keep_cnt
        if d_ms > l_rms and d_ms > 0.0:
            dL = dL * math.sqrt(l_rms / d_ms)

        B_h = Qch.transpose(0, 1) @ dL @ Ekg - Eqh.transpose(0, 1) @ dL @ Kcg
        B[g] = B_h if B[g] is None else B[g] + B_h
        G_q[g] = (Qch.transpose(0, 1) @ Qch) if G_q[g] is None else G_q[g] + (Qch.transpose(0, 1) @ Qch)
        G_e[g] = (Eqh.transpose(0, 1) @ Eqh) if G_e[g] is None else G_e[g] + (Eqh.transpose(0, 1) @ Eqh)
        if G_k[g] is None:
            G_k[g] = Kcg.transpose(0, 1) @ Kcg
            G_ek[g] = Ekg.transpose(0, 1) @ Ekg

    return {"G_q": G_q, "G_e": G_e, "G_k": G_k, "G_ek": G_ek, "B": B, "logit_rms": math.sqrt(l_rms)}


def _a29_solve_group_S(S: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Top-4 |eigenvalue| rank reconstruction of a solved S, Frobenius-
    normalized to unit norm (the CG solve itself is batched upstream)."""

    values, vectors = torch.linalg.eigh(S)
    top = torch.argsort(values.abs(), descending=True)[:_A29_RANK]
    S4 = (vectors[:, top] * values[top].unsqueeze(0)) @ vectors[:, top].transpose(0, 1)
    nrm = float(S4.norm())
    S_hat = S4 / nrm if nrm > _A29_ZERO_EPS else torch.zeros_like(S4)
    return S_hat, {"norm": nrm}


def _a29_code_mid_and_block(
    params: dict,
    T: int,
    channels: int,
    device: torch.device,
) -> dict:
    """Per-element next-positive-code midpoint and validity mask.

    mant ladder (0, .5, 1, 1.5, 2, 3, 4, 6); positive crossing = dequant
    value increases: mant=0 crosses to +0.5 (midpoint 0.25*s), sign>0 climbs
    one rung (mant=6 saturated -> excluded), sign<0 moves toward zero.
    """

    mant = params["mant"].detach().to(device=device, dtype=torch.float32)
    sign = params["sign"].detach().to(device=device, dtype=torch.float32)
    s = (
        params["scale_factor"].detach().to(device=device, dtype=torch.float32)
        * params["scale_lv2"].detach().to(device=device, dtype=torch.float32)
        * params["scale_lv3"].detach().to(device=device, dtype=torch.float32)
    )
    ladder = torch.tensor(_A29_MANT_LADDER, device=device, dtype=torch.float32).reshape(8, 1, 1, 1, 1, 1)
    near = (mant.unsqueeze(0) - ladder).abs() < _A29_MANT_TOL
    idx = torch.argmax(near.to(torch.int32), dim=0).clamp(0, 7)
    ladder_flat = ladder.reshape(-1)
    mant_next = ladder_flat[(idx + 1).clamp(max=7)]
    mant_prev = ladder_flat[(idx - 1).clamp(min=0)]
    v_next = torch.where(
        sign > 0, mant_next * s,
        torch.where(sign < 0, mant_prev * s, 0.5 * s),
    )
    v = sign * mant * s
    mid = 0.5 * (v + v_next)
    valid = (sign != 0) | (mant == 0.0)
    valid = valid & ~((sign > 0) & (idx >= 7))
    return {
        "mid": mid.reshape(T, channels),
        "valid": valid.reshape(T, channels),
        "blocks": channels // 64,
    }


def _a29_step_alpha(
    driver_coords: list,
    S_dir: list,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[float, dict]:
    """Unique alpha: output-sensitivity weighted 1/64 quantile of the first
    positive code-crossing distances over driver folds, Q and K merged.

    Sensitivity weight (fixed, parameter-free): |dQ|*||K̂_g column|| per Q
    element (its own head's logits) and |dK|*(sum_h ||Q̂_gh column||) per K
    element (it feeds every head of its group). First-order logit
    sensitivity bound; no tuning parameters.
    """

    group = q_heads // kv_heads
    q_cols = torch.zeros(kv_heads, group, head_dim, device=device, dtype=torch.float64)
    k_cols = torch.zeros(kv_heads, head_dim, device=device, dtype=torch.float64)
    for c in driver_coords:
        T_q = int(c["q_ref"].shape[0])
        q4 = c["q_hat"].to(device).reshape(T_q, kv_heads, group, head_dim)
        k4 = c["k_hat"].to(device).reshape(T_q, kv_heads, head_dim)
        q_cols += q4.square().sum(dim=0).to(torch.float64)
        k_cols += k4.square().sum(dim=0).to(torch.float64)
    q_coln = q_cols.clamp_min(1.0).sqrt().to(torch.float32)
    k_coln = k_cols.clamp_min(1.0).sqrt().to(torch.float32)
    q_coln_sum = q_coln.sum(dim=1)  # (kv_heads, head_dim)

    S_stack = torch.stack(S_dir).to(device=device, dtype=torch.float32)
    q_info = [
        _a29_code_mid_and_block(c["q_params"], int(c["q_ref"].shape[0]), q_heads * head_dim, device)
        for c in driver_coords
    ]
    k_info = [
        _a29_code_mid_and_block(c["k_params"], int(c["k_ref"].shape[0]), kv_heads * head_dim, device)
        for c in driver_coords
    ]

    dists = []
    weights = []
    for fi, c in enumerate(driver_coords):
        T_q = int(c["q_ref"].shape[0])
        q_cont4 = c["q_cont"].to(device).reshape(T_q, kv_heads, group, head_dim)
        k_cont4 = c["k_cont"].to(device).reshape(T_q, kv_heads, head_dim)
        # Q side: dQ = Q~ @ S_g per (g,h)
        dQ = torch.einsum("tghd,gde->tghe", q_cont4, S_stack)
        sens = dQ.abs() * k_coln[None, :, None, :]
        mid = q_info[fi]["mid"].reshape(T_q, kv_heads, group, head_dim)
        valid = q_info[fi]["valid"].reshape(T_q, kv_heads, group, head_dim)
        pos = valid & (dQ > 0) & (mid > q_cont4)
        d_elem = torch.where(pos, (mid - q_cont4) / dQ.clamp_min(_EPS), torch.full_like(mid, float("inf")))
        w_elem = torch.where(pos, sens, torch.zeros_like(sens))
        nb = q_info[fi]["blocks"]
        dists.append(d_elem.reshape(T_q, nb, 64).amin(dim=2).reshape(-1))
        weights.append(w_elem.reshape(T_q, nb, 64).sum(dim=2).reshape(-1))
        # K side: dK = -K~ @ S_g
        dK = -torch.einsum("tgd,gde->tge", k_cont4, S_stack)
        sens_k = dK.abs() * q_coln_sum[None]
        mid_k = k_info[fi]["mid"].reshape(T_q, kv_heads, head_dim)
        valid_k = k_info[fi]["valid"].reshape(T_q, kv_heads, head_dim)
        pos_k = valid_k & (dK > 0) & (mid_k > k_cont4)
        d_elem_k = torch.where(pos_k, (mid_k - k_cont4) / dK.clamp_min(_EPS), torch.full_like(mid_k, float("inf")))
        w_elem_k = torch.where(pos_k, sens_k, torch.zeros_like(sens_k))
        nbk = k_info[fi]["blocks"]
        dists.append(d_elem_k.reshape(T_q, nbk, 64).amin(dim=2).reshape(-1))
        weights.append(w_elem_k.reshape(T_q, nbk, 64).sum(dim=2).reshape(-1))

    d_all = torch.cat(dists)
    w_all = torch.cat(weights)
    finite = torch.isfinite(d_all) & (d_all > 0)
    d_all = d_all[finite]
    w_all = w_all[finite]
    if d_all.numel() == 0 or float(w_all.sum()) <= 0.0:
        return -1.0, {"valid_blocks": int(d_all.numel())}
    order = torch.argsort(d_all)
    d_sorted = d_all[order]
    cum = torch.cumsum(w_all[order], dim=0)
    target = float(w_all.sum()) * _A29_QUANTILE
    hit = torch.nonzero(cum >= target, as_tuple=False)
    alpha = float(d_sorted[hit[0]]) if hit.numel() > 0 else float(d_sorted[-1])
    return alpha, {"valid_blocks": int(d_all.numel()), "total_weight": float(w_all.sum())}


def _a29_proposal_states(
    states: dict,
    S_dir: list,
    alpha: float,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict:
    """Build proposal states: learned_rotation -> old @ expm(+/- alpha S_g)
    (row-vector right-multiply) and compiled center c' = c @ expm(-alpha S).
    Zero new deployed operators."""

    S_stack = torch.stack(S_dir).to(device=device, dtype=torch.float32) * alpha
    exp_p = _a29_exp_sym(S_stack)
    exp_m = _a29_exp_sym(-S_stack)
    new_q = dict(states["q_state"])
    new_k = dict(states["k_state"])
    rot = states["q_state"].get("learned_rotation")
    if rot is not None:
        rot_d = rot.detach().to(device=device, dtype=torch.float32)
        new_q["learned_rotation"] = (rot_d @ exp_p).cpu().to(torch.float32)
        new_k["learned_rotation"] = (rot_d @ exp_m).cpu().to(torch.float32)
    else:
        new_q["learned_rotation"] = exp_p.cpu().to(torch.float32)
        new_k["learned_rotation"] = exp_m.cpu().to(torch.float32)
    center = states["k_state"].get("learned_center")
    if center is not None:
        c = center.detach().to(device=device, dtype=torch.float32)
        new_k["learned_center"] = (c.unsqueeze(-2) @ exp_m).squeeze(-2).cpu().to(torch.float32)
    return {"q_state": new_q, "k_state": new_k, "v_state": states["v_state"]}


def _a29_causal_mse(
    item: dict,
    q_state: dict,
    k_state: dict,
    v_state: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> float:
    """Causal deployed hard-output MSE against the NVFP4 reference output."""

    q_quant, q_scale = item["q"]
    k_quant, k_scale = item["k"]
    v_quant, v_scale = item["v"]
    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)
    q_hat = _dequantize_hif4(hif4_dynamic_quantize_q(q_quant, q_scale, q_heads, head_dim, q_state)).to(torch.float32)
    k_hat = _dequantize_hif4(hif4_dynamic_quantize_k(k_quant, k_scale, kv_heads, head_dim, k_state)).to(torch.float32)
    v_hat = _dequantize_hif4(hif4_dynamic_quantize_v(v_quant, v_scale, kv_heads, head_dim, v_state)).to(torch.float32)
    out = _attention_forward(q_hat, k_hat, v_hat, q_heads, kv_heads, head_dim, True)
    ref = _attention_forward(q_ref, k_ref, v_ref, q_heads, kv_heads, head_dim, True)
    return float((out - ref).square().mean())


def _a29_changed_codes(params_a: dict, params_b: dict) -> int:
    """Count elements whose (sign, mant) codes differ between two params."""

    diff = (params_a["mant"] != params_b["mant"]) | (params_a["sign"] != params_b["sign"])
    return int(diff.sum())


def _a29_apply(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    states: dict,
) -> dict:
    """A29 main routine: closed-form direction, single proposal, dual gate."""

    audit = {"a29_arm": "skip"}
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < _A29_MIN_FOLDS:
        states["q_state"].update(audit)
        states["k_state"].update(audit)
        return states
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        coords = [
            _a29_cont_coords(item, states, q_num_heads, kv_num_heads, head_dim, device)
            for item in calib_qkv_list[:_A29_FOLDS_DRIVER]
        ]
        # twelve (fold, group) instances solved as one batched CG
        G_q, G_e, G_k, G_ek, B = [], [], [], [], []
        for c in coords:
            d = _a29_fold_driver(c, q_num_heads, kv_num_heads, head_dim, device)
            for g in range(kv_num_heads):
                G_q.append(d["G_q"][g])
                G_e.append(d["G_e"][g])
                G_k.append(d["G_k"][g])
                G_ek.append(d["G_ek"][g])
                B.append(d["B"][g])
        B_stack = torch.stack(B)

        def apply_batched(X):
            # X: (12, D, D); per-instance L(X) = G_q X G_ek - G_e X G_k
            return G_q_stack @ X @ G_ek_stack - G_e_stack @ X @ G_k_stack

        G_q_stack = torch.stack(G_q)
        G_e_stack = torch.stack(G_e)
        G_k_stack = torch.stack(G_k)
        G_ek_stack = torch.stack(G_ek)
        S_stack, residuals, iters_used = _a29_cg_solve_batched(apply_batched, B_stack)
        res_list = [float(x) for x in residuals.detach().cpu()]
        it_list = [int(x) for x in iters_used.detach().cpu()]
        audit["a29_cg"] = repr([(i, round(res_list[i], 6), it_list[i]) for i in range(len(res_list))])

        # top-4 rank per instance, Frobenius-normalize, fold-average per group
        S_dir = [None] * kv_num_heads
        fnorms = []
        for g in range(kv_num_heads):
            acc = torch.zeros(head_dim, head_dim, device=device, dtype=torch.float32)
            for fi in range(_A29_FOLDS_DRIVER):
                inst = fi * kv_num_heads + g
                S_hat, info = _a29_solve_group_S(S_stack[inst])
                acc = acc + S_hat
            S_dir[g] = acc / float(_A29_FOLDS_DRIVER)
            fnorms.append(float(S_dir[g].norm()))
        fnorm = max(fnorms)
        audit["a29_s_fnorm"] = fnorm
        if fnorm < _A29_ZERO_EPS:
            audit["a29_arm"] = "zero_dir"
            states["q_state"].update(audit)
            states["k_state"].update(audit)
            return states

        alpha, alpha_audit = _a29_step_alpha(coords, S_dir, q_num_heads, kv_num_heads, head_dim, device)
        audit["a29_alpha"] = alpha
        audit["a29_alpha_blocks"] = alpha_audit.get("valid_blocks", 0)
        if alpha <= 0.0:
            audit["a29_arm"] = "invalid_alpha"
            states["q_state"].update(audit)
            states["k_state"].update(audit)
            return states
        cap = _A29_FNORM_CAP / max(fnorm, _A29_ZERO_EPS)
        if alpha > cap:
            alpha = cap
            audit["a29_alpha_clamped"] = True
        proposal = _a29_proposal_states(states, S_dir, alpha, kv_num_heads, head_dim, device)

        f3 = calib_qkv_list[_A29_FOLD_SELECT]
        f4 = calib_qkv_list[_A29_FOLD_HOLDOUT]
        mse_p3 = _a29_causal_mse(f3, states["q_state"], states["k_state"], states["v_state"], q_num_heads, kv_num_heads, head_dim)
        mse_q3 = _a29_causal_mse(f3, proposal["q_state"], proposal["k_state"], states["v_state"], q_num_heads, kv_num_heads, head_dim)
        audit["a29_lhard_parent_f3"] = mse_p3
        audit["a29_lhard_prop_f3"] = mse_q3
        audit["a29_arm"] = "parent_fold3"
        if not (mse_q3 < mse_p3):
            states["q_state"].update(audit)
            states["k_state"].update(audit)
            return states

        q_par = hif4_dynamic_quantize_q(f3["q"][0], f3["q"][1], q_num_heads, head_dim, states["q_state"])
        q_prp = hif4_dynamic_quantize_q(f3["q"][0], f3["q"][1], q_num_heads, head_dim, proposal["q_state"])
        k_par = hif4_dynamic_quantize_k(f3["k"][0], f3["k"][1], kv_num_heads, head_dim, states["k_state"])
        k_prp = hif4_dynamic_quantize_k(f3["k"][0], f3["k"][1], kv_num_heads, head_dim, proposal["k_state"])
        audit["a29_changed_codes_q"] = _a29_changed_codes(q_prp, q_par)
        audit["a29_changed_codes_k"] = _a29_changed_codes(k_prp, k_par)

        mse_p4 = _a29_causal_mse(f4, states["q_state"], states["k_state"], states["v_state"], q_num_heads, kv_num_heads, head_dim)
        mse_q4 = _a29_causal_mse(f4, proposal["q_state"], proposal["k_state"], states["v_state"], q_num_heads, kv_num_heads, head_dim)
        audit["a29_lhard_parent_f4"] = mse_p4
        audit["a29_lhard_prop_f4"] = mse_q4
        audit["a29_arm"] = "parent_fold4"
        if not (mse_q4 < mse_p4):
            states["q_state"].update(audit)
            states["k_state"].update(audit)
            return states

        # Both gates passed strictly: fold the proposal into the state
        states["q_state"]["learned_rotation"] = proposal["q_state"]["learned_rotation"]
        states["k_state"]["learned_rotation"] = proposal["k_state"]["learned_rotation"]
        if "learned_center" in proposal["k_state"]:
            states["k_state"]["learned_center"] = proposal["k_state"]["learned_center"]
        audit["a29_arm"] = "deployed"
        states["q_state"].update(audit)
        states["k_state"].update(audit)
        return states
    except Exception as exc:  # noqa: BLE001 - any A29 failure degrades to exact R3
        ln = exc.__traceback__.tb_lineno if exc.__traceback__ is not None else -1
        msg = f"{type(exc).__name__}@L{ln}: {exc}"[:200]
        states["q_state"]["a29_arm"] = "fallback"
        states["k_state"]["a29_arm"] = "fallback"
        states["q_state"]["a29_error"] = msg
        states["k_state"]["a29_error"] = msg
        return states


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """R3 full stack calibration (kept bitwise) + A29 closed-form QK
    inverse-pair compensation with dual hard-output gates (anchor29-a1)."""

    states = _r3_a2_calibration_attention(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    try:
        states = _a29_apply(
            calib_qkv_list, q_num_heads, kv_num_heads, head_dim, states
        )
    except Exception:  # noqa: BLE001 - unreachable; kept for safety
        states.setdefault("q_state", {})["a29_arm"] = "fallback"
        states.setdefault("k_state", {})["a29_arm"] = "fallback"
    return states
