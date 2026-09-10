"""C-FR1 mechanism block, appended to the v237 root copy by build.py."""

CFR_BLOCK = r'''

# ---------------------------------------------------------------------------
# C-FR1 (2026-09-10): closed-form full-matrix reciprocal transform.
#
# Official scores in this family rise with the *expressiveness* of the transform:
#
#   diagonal, closed form   (v190)           official 0
#   triangular, searched    (v191)           official 0
#   symmetric full, 32-step (v192)           official +22, full package TIMEOUT
#   general full, 32-step   (A-GR1 / v234)   official +29, full package TIMEOUT
#
# The diagonal rungs move no codes at all -- a block-scaled quantizer absorbs a
# diagonal reparameterisation into the block scale -- while the full-matrix rungs
# do move codes and do score.  Both good numbers, however, were bought with a
# 32-step Adam training run, and every member of the family that added
# calibration-time work timed out.  The external 21071/283s anchor is *faster*
# than the current root's 289s while scoring 2553 more, so a mechanism that
# reaches it cannot be adding calibration compute.  This card takes the
# full-matrix rung and removes the optimizer.
#
# The transform.  Per KV group, in the parent coordinate,
#
#   Q @ M    and    K @ M^{-T}
#
# leaves the continuous QK product exactly unchanged: (qM)(kM^{-T})^T = q k^T.
# M is chosen so the two encoded operands carry the same channel second moments.
# With A and B the per-group channel second moments of the parent-coordinate Q
# and K, and A = R R^T the Cholesky factorisation (R lower triangular),
#
#   R^T B R = U diag(lam) U^T          (symmetric eigendecomposition)
#   M       = R^{-T} U diag(lam)^{1/4}
#   M^{-1}  = diag(lam)^{-1/4} U^T R^T
#   M^{-T}  = R U diag(lam)^{-1/4}
#
# which gives, by construction,
#
#   M^T A M = diag(lam)^{1/2}   and   M^{-1} B M^{-T} = diag(lam)^{1/2},
#
# i.e. the two sides carry the same channel energy profile.  That is the closed
# form of "reduce each side's scale" without moving magnitude out of the product.
# Nothing is searched, nothing is trained, and no gradient exists.
#
# Note the factors: it is R^{-T} and R^T B R, not R^{-1} and R B R^T.  The wrong
# pairing is self-consistent-looking and still yields an invertible M, so it
# produces no error -- it just silently fails to balance anything.  verify.py
# carries the invariant M^T A M = M^{-1} B M^{-T} as a check precisely because a
# gate rejection cannot distinguish "wrong mechanism" from "wrong algebra".
#
# Cost.  Collecting the two moments is one dequantise plus one parent-coordinate
# pass over the fit windows -- no encode, no attention forward -- then one
# Cholesky and one symmetric eigendecomposition of a 256x256 matrix per KV group.
# The only added forward passes are the gate's, which every member of this family
# pays.  Compare A-GR1: 32 steps of Adam, 33 inversions and 33 SVDs per layer.
#
# Deployment.  M is folded at calibration time into fields the root already
# compiles -- learned_rotation and learned_center -- so the dynamic path stays a
# single matmul plus a shift and never inverts a matrix:
#
#   q_state.learned_rotation <- R_parent @ M
#   k_state.learned_rotation <- R_parent @ M^{-T}
#   k_state.learned_center   <- center    @ M^{-T}
#
# _nvfp4_to_hif4, both dynamic Q/K APIs, the six-API surface, the state format,
# the five-field encoding and every other mechanism in the root are untouched.
#
# The root's own calibration (the v189 stack plus the A2 rotation/center trainer
# and its gate) runs first and is frozen; this composes on whatever arm the root
# selected.  Acceptance is the family's own rule: strictly better true
# deployed-path output MSE on every gate window, otherwise the candidate falls
# back to the parent state bit for bit.
# ---------------------------------------------------------------------------

_CFR_RIDGE = 1e-3
_CFR_GATE_WINDOWS = (3, 4)

_CFR_PARENT_Q = hif4_dynamic_quantize_q
_CFR_PARENT_K = hif4_dynamic_quantize_k
_CFR_PARENT_V = hif4_dynamic_quantize_v


def _cfr_parent_coordinate(
    dense: torch.Tensor,
    state: dict,
    heads: int,
    head_dim: int,
    is_k: bool,
) -> torch.Tensor:
    """The root's complete pre-reciprocal coordinate stack, dense float32."""

    out = _attention_state_transform_dense(
        dense, state, int(heads), int(head_dim), is_k=bool(is_k)
    )
    rotation = state.get("learned_rotation")
    if rotation is not None:
        out = _a2_apply_group_rotation(out, int(heads), rotation)
    if is_k:
        center = state.get("learned_center")
        if center is not None:
            lead = out.shape[:-1]
            out = (
                out.reshape(*lead, int(heads), int(head_dim))
                + center.to(device=out.device, dtype=torch.float32).reshape(
                    *([1] * len(lead)), int(heads), int(head_dim)
                )
            ).reshape_as(out)
    return out


def _cfr_parent_copy(states: dict) -> dict:
    out = dict(states)
    for role in ("q_state", "k_state", "v_state"):
        state = dict(states[role])
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().to(device="cpu").clone()
        out[role] = state
    return out


@torch.no_grad()
def _cfr_moments(
    windows: list,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
):
    """Per-group channel second moments of the parent-coordinate Q and K.

    No encode and no attention forward happen here: this is a dequantise plus
    the root's own continuous prefix, which the parent calibration already
    performs once per window.
    """

    groups = int(kv_heads)
    per_group = int(q_heads) // groups
    dim = int(head_dim)
    aq = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    ak = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    for item in windows:
        q_dense = _dequantize_nvfp4_float32(*item["q"]).to(torch.float32).to(device)
        k_dense = _dequantize_nvfp4_float32(*item["k"]).to(torch.float32).to(device)
        q_c = _cfr_parent_coordinate(
            q_dense, states["q_state"], int(q_heads), dim, False
        )
        k_c = _cfr_parent_coordinate(
            k_dense, states["k_state"], groups, dim, True
        )
        q4 = q_c.reshape(-1, groups, per_group, dim)
        aq += torch.einsum("tghd,tghe->gde", q4, q4) / float(q4.shape[0] * per_group)
        k3 = k_c.reshape(-1, groups, dim)
        ak += torch.einsum("tgd,tge->gde", k3, k3) / float(k3.shape[0])
        del q_dense, k_dense, q_c, k_c, q4, k3
    count = float(max(len(windows), 1))
    return aq / count, ak / count


@torch.no_grad()
def _cfr_matrix(aq: torch.Tensor, ak: torch.Tensor, ridge: float):
    """Returns (M, M^{-T}) for M = R^{-T} U diag(lam)^{1/4}; see the block header.

    M^{-T} comes from the factorisation, not from inverting M, so no general
    inverse is ever formed.  That is both the cheaper route and the only one
    available here: torch.linalg.inv and torch.linalg.solve dispatch to lu_solve
    on this batched input and raise "Pivots given to lu_solve must all be
    greater or equal to 1", even though the matrix is finite and well
    conditioned (min eigenvalue 1.1e-2 against a diagonal mean of 0.043).
    """

    dim = int(aq.shape[-1])
    eye = torch.eye(dim, device=aq.device, dtype=aq.dtype)
    rq = aq.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
    rk = ak.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
    aq = aq + (float(ridge) * rq) * eye
    ak = ak + (float(ridge) * rk) * eye
    chol = torch.linalg.cholesky(aq)                 # lower, aq = chol @ chol^T
    cross = chol.transpose(-1, -2) @ ak @ chol       # R^T B R
    cross = 0.5 * (cross + cross.transpose(-1, -2))
    lam, u = torch.linalg.eigh(cross)
    lam = lam.clamp_min(1e-12)
    scaled = (u * lam.pow(0.25).unsqueeze(-2)).contiguous()
    # M = R^{-T} U diag(lam)^{1/4}; R^T is upper triangular.
    m = torch.linalg.solve_triangular(
        chol.transpose(-1, -2).contiguous(), scaled, upper=True
    )
    # M^{-T} = R U diag(lam)^{-1/4}
    m_inv_t = chol @ (u * lam.pow(-0.25).unsqueeze(-2))
    return m.contiguous(), m_inv_t.contiguous()


@torch.no_grad()
def _cfr_compile(
    states: dict,
    m: torch.Tensor,
    m_inv_t: torch.Tensor,
    kv_heads: int,
    head_dim: int,
):
    """Fold M into the root's compiled fields; the dynamic path is unchanged."""

    groups = int(kv_heads)
    dim = int(head_dim)
    eye = torch.eye(dim, device=m.device, dtype=m.dtype)[None].expand(groups, dim, dim)

    def _parent(role: str) -> torch.Tensor:
        value = states[role + "_state"].get("learned_rotation")
        if value is None:
            return eye
        return value.to(device=m.device, dtype=torch.float32)

    rotation_q = _parent("q") @ m
    rotation_k = _parent("k") @ m_inv_t
    center = states["k_state"].get("learned_center")
    center_k = None
    if center is not None:
        center_k = (
            center.to(device=m.device, dtype=torch.float32).reshape(groups, dim) @ m_inv_t
        )
    return rotation_q, rotation_k, center_k


@torch.no_grad()
def _cfr_gate_loss(
    item: dict,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> float:
    """True deployed-path output MSE for one calibration window, one arm."""

    decoded = []
    reference = []
    for role, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
        api = {
            "q": _CFR_PARENT_Q,
            "k": _CFR_PARENT_K,
            "v": _CFR_PARENT_V,
        }[role]
        params = api(*item[role], heads, head_dim, states[role + "_state"])
        decoded.append(_dequantize_hif4(params).to(torch.float32)[None])
        reference.append(
            _dequantize_nvfp4_float32(*item[role]).to(torch.float32)[None]
        )
    actual = _a2_attention_forward(
        decoded[0], decoded[1], decoded[2], q_heads, kv_heads, head_dim
    )
    target = _a2_attention_forward(
        reference[0], reference[1], reference[2], q_heads, kv_heads, head_dim
    )
    return float((actual - target).square().mean().item())


def _cfr_annotate(states: dict, info: dict) -> dict:
    out = _cfr_parent_copy(states)
    for role in ("q_state", "k_state"):
        out[role].update(info)
    return out


_CFR_PARENT_CALIBRATION = hif4_calibration_attention


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict:
    """The root's calibration, then one closed-form reciprocal transform."""

    states = _CFR_PARENT_CALIBRATION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    info = {
        "cfr_arm": "ineligible",
        "cfr_attempted": 0,
        "cfr_accepted": 0,
        "cfr_ridge": float(_CFR_RIDGE),
        "cfr_gate_windows": list(_CFR_GATE_WINDOWS),
    }
    if (
        not isinstance(states, dict)
        or not all(
            role in states and isinstance(states[role], dict)
            for role in ("q_state", "k_state", "v_state")
        )
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) < max(_CFR_GATE_WINDOWS) + 1
        or int(kv_num_heads) <= 0
        or int(q_num_heads) % int(kv_num_heads) != 0
        or int(head_dim) <= 0
    ):
        info["cfr_arm"] = "ineligible"
        return _cfr_annotate(states, info)

    info["cfr_attempted"] = 1
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fit_windows = calib_qkv_list[
            : len(calib_qkv_list) - len(_CFR_GATE_WINDOWS)
        ]
        aq, ak = _cfr_moments(
            fit_windows, states, q_num_heads, kv_num_heads, head_dim, device
        )
        m, m_inv_t = _cfr_matrix(aq, ak, _CFR_RIDGE)
        rotation_q, rotation_k, center_k = _cfr_compile(
            states, m, m_inv_t, kv_num_heads, head_dim
        )
        singular = torch.linalg.svdvals(m.to(torch.float32))
        info["cfr_singular_min"] = float(singular.min())
        info["cfr_singular_max"] = float(singular.max())
        info["cfr_identity_distance"] = float(
            (m - torch.eye(int(head_dim), device=m.device, dtype=m.dtype)[None])
            .abs()
            .max()
        )

        candidate = _cfr_parent_copy(states)
        candidate["q_state"]["learned_rotation"] = rotation_q
        candidate["k_state"]["learned_rotation"] = rotation_k
        if center_k is not None:
            candidate["k_state"]["learned_center"] = center_k

        accepted = True
        parent_losses = []
        candidate_losses = []
        for index in _CFR_GATE_WINDOWS:
            parent_loss = _cfr_gate_loss(
                calib_qkv_list[index], states, q_num_heads, kv_num_heads, head_dim
            )
            candidate_loss = _cfr_gate_loss(
                calib_qkv_list[index], candidate, q_num_heads, kv_num_heads, head_dim
            )
            parent_losses.append(parent_loss)
            candidate_losses.append(candidate_loss)
            if not candidate_loss < parent_loss:
                accepted = False
        info["cfr_gate_parent"] = [float(x) for x in parent_losses]
        info["cfr_gate_candidate"] = [float(x) for x in candidate_losses]

        if accepted:
            info["cfr_arm"] = "closed-form"
            info["cfr_accepted"] = 1
            return _cfr_annotate(candidate, info)

        info["cfr_arm"] = "parent"
        return _cfr_annotate(states, info)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the parent
        info["cfr_arm"] = "fallback"
        info["cfr_error"] = f"{type(exc).__name__}: {exc}"
        return _cfr_annotate(states, info)


'''
