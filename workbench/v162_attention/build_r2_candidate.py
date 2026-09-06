"""Patch candidate_v2 into candidate_v3 (R2: learned rotation inside v189 stack).

Injection points:
1. _nvfp4_to_hif4: new keywords learned_rotation/learned_rotation_num_heads,
   applied right before the encode (after every stack transform), so the
   continuous QK product stays exactly invariant versus R1.
2. v189 dynamic Q/K: pass state["learned_rotation"] through.
3. A wrapper hif4_calibration_attention (appended, shadows the v189 one):
   run the untouched v189 calibration, train the per-KV-group rotation with
   the frozen A2 config, gate identity-vs-rotation on the true deployed path
   over the held-out last calibration window, and inject the rotation into
   the states only on a strict win.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parent / "candidate_v3/solution.py"

SIG_OLD = """    attention_rotation: Optional[torch.Tensor] = None,
    rotation_num_heads: Optional[int] = None,
    attention_rotation_block: Optional[int] = None,
    attention_block_signs: Optional[torch.Tensor] = None,
    attention_pair_transform: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:"""
SIG_NEW = """    attention_rotation: Optional[torch.Tensor] = None,
    rotation_num_heads: Optional[int] = None,
    attention_rotation_block: Optional[int] = None,
    attention_block_signs: Optional[torch.Tensor] = None,
    attention_pair_transform: Optional[torch.Tensor] = None,
    learned_rotation: Optional[torch.Tensor] = None,
    learned_rotation_num_heads: Optional[int] = None,
) -> dict[str, torch.Tensor]:"""

INJECT_OLD = """    refine_importance = importance
    if _ACTIVATION_SAMPLE_IMPORTANCE and dense.ndim == 2:"""
INJECT_NEW = """    if learned_rotation is not None:
        if learned_rotation_num_heads is None:
            raise ValueError("learned rotation requires head count")
        dense = _a2_apply_group_rotation(
            dense, int(learned_rotation_num_heads), learned_rotation
        )
    refine_importance = importance
    if _ACTIVATION_SAMPLE_IMPORTANCE and dense.ndim == 2:"""

Q_OLD = """        attention_pair_transform=state.get("pair_transform"),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_k("""
Q_NEW = """        attention_pair_transform=state.get("pair_transform"),
        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(q_num_heads),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_k("""
K_OLD = """        attention_pair_transform=state.get("pair_transform"),
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        center_value=state.get("center_value"),
        importance=state["importance"],"""
K_NEW = """        attention_pair_transform=state.get("pair_transform"),
        learned_rotation=state.get("learned_rotation"),
        learned_rotation_num_heads=int(kv_num_heads),
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        center_value=state.get("center_value"),
        importance=state["importance"],"""

APPENDIX = '''

# ---------------------------------------------------------------------------
# v162-independent Attention branch, R2 step (A agent): per-KV-group learned
# orthogonal rotation deployed inside the v186/v189 attention stack.  The
# rotation composes after every stack transform and before the HiF4 encode,
# so the continuous QK product is exactly the R1 one; training uses the
# frozen A2 config and deployment is gated per layer on the true path.
# ---------------------------------------------------------------------------

_A2_TRAIN_STEPS = 32
_A2_TRAIN_LR = 0.01
_A2_TRAIN_CLIP = 1.0
_A2_REG_WEIGHT = 1e-3
_A2_MAX_KV_TOKENS = 128
_A2_MAX_Q_TOKENS = 32
_A2_ORTHO_TOLERANCE = 1e-3


def _a2_hadamard_orthogonal(dim: int) -> Optional[torch.Tensor]:
    if dim < 1 or (dim & (dim - 1)) != 0 or dim > 4096:
        return None
    matrix = torch.ones(1, 1, dtype=torch.float64)
    size = 1
    while size < dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
        size *= 2
    return (matrix / math.sqrt(dim)).to(torch.float32)


def _a2_cayley_orthogonal(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    skew = theta - theta.transpose(-1, -2)
    eye = torch.eye(theta.shape[-1], device=theta.device, dtype=theta.dtype)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(
        right.transpose(-1, -2), left.transpose(-1, -2)
    ).transpose(-1, -2)
    reg = (c - eye).square().mean()
    return c, reg


def _a2_even_indices(total: int, limit: int, device: torch.device) -> torch.Tensor:
    if total <= limit:
        return torch.arange(total, device=device)
    positions = torch.linspace(0, total - 1, limit, device=device).round()
    return torch.unique(positions.to(torch.int64))


def _a2_apply_group_rotation(
    dense: torch.Tensor,
    num_heads: int,
    rotation: torch.Tensor,
) -> torch.Tensor:
    head_dim = int(dense.shape[-1]) // int(num_heads)
    groups = int(rotation.shape[0])
    per_group = int(num_heads) // groups
    lead = dense.shape[:-1]
    grouped = dense.reshape(*lead, groups, per_group, head_dim)
    rotated = torch.einsum(
        "tghk,gkd->tghd",
        grouped.reshape(-1, groups, per_group, head_dim),
        rotation.to(device=grouped.device, dtype=torch.float32),
    )
    return rotated.reshape(*lead, int(num_heads) * head_dim)


def _a2_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, tokens, _ = q.shape
    qh = q.reshape(batch, tokens, q_heads, head_dim).transpose(1, 2)
    group = q_heads // kv_heads
    kh = k.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    vh = v.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    probabilities = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (probabilities @ vh).transpose(1, 2).reshape(batch, tokens, q_heads * head_dim)


def _a2_train_rotation(
    windows: List[dict],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    groups = kv_heads
    dim = head_dim
    base = _a2_hadamard_orthogonal(dim)
    if base is None:
        base = torch.eye(dim, dtype=torch.float32)
    base = base.to(device)

    prepared = []
    for window in windows:
        q_full = window["q"].to(device)
        k_full = window["k"].to(device)
        v_full = window["v"].to(device)
        kv_index = _a2_even_indices(k_full.shape[0], _A2_MAX_KV_TOKENS, device)
        q_index = _a2_even_indices(kv_index.numel(), _A2_MAX_Q_TOKENS, device)
        q_rows = kv_index.index_select(0, q_index)
        k_sub = k_full.index_select(0, kv_index)
        v_sub = v_full.index_select(0, kv_index)
        q_sub = q_full.index_select(0, q_rows)
        reference = _a2_attention_forward(
            q_sub[None], k_sub[None], v_sub[None], q_heads, kv_heads, head_dim
        )[0].detach()
        std_q = _dequantize_hif4(_dense_to_hif4(q_sub)).to(torch.float32)
        std_k = _dequantize_hif4(_dense_to_hif4(k_sub)).to(torch.float32)
        std_v = _dequantize_hif4(_dense_to_hif4(v_sub)).to(torch.float32)
        standard = _a2_attention_forward(
            std_q[None], std_k[None], std_v[None], q_heads, kv_heads, head_dim
        )[0].detach()
        mse_std = float((standard - reference).square().mean())
        prepared.append({
            "q": q_sub, "k": k_sub, "v": v_sub,
            "reference": reference, "mse_std": max(mse_std, 1e-12),
        })

    theta = torch.zeros(
        groups, dim, dim, dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=_A2_TRAIN_LR)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")
    for _step in range(_A2_TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        window_losses = []
        reg_total = theta.new_zeros(())
        for item in prepared:
            c, reg = _a2_cayley_orthogonal(theta)
            rotation = torch.einsum("dk,gkl->gdl", base, c)
            reg_total = reg_total + reg
            q_rot = _a2_apply_group_rotation(item["q"], q_heads, rotation)
            k_rot = _a2_apply_group_rotation(item["k"], kv_heads, rotation)
            q_hat = q_rot + (_dequantize_hif4(_dense_to_hif4(q_rot)) - q_rot).detach()
            k_hat = k_rot + (_dequantize_hif4(_dense_to_hif4(k_rot)) - k_rot).detach()
            v_hat = item["v"] + (_dequantize_hif4(_dense_to_hif4(item["v"])) - item["v"]).detach()
            output = _a2_attention_forward(
                q_hat[None], k_hat[None], v_hat[None], q_heads, kv_heads, head_dim
            )[0]
            loss = (output - item["reference"]).square().mean() / item["mse_std"]
            window_losses.append(loss)
        data_loss = torch.stack(window_losses).mean()
        objective = data_loss + _A2_REG_WEIGHT * reg_total / len(prepared)
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("A2 rotation training produced a non-finite loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_([theta], _A2_TRAIN_CLIP)
        optimizer.step()
        final_loss = float(data_loss.detach())
    with torch.no_grad():
        c, _reg = _a2_cayley_orthogonal(theta)
        rotation = torch.einsum("dk,gkl->gdl", base, c)
        identity_error = float(
            (rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max()
        )
    if identity_error > _A2_ORTHO_TOLERANCE:
        raise RuntimeError(
            f"A2 trained rotation failed orthogonality: {identity_error}"
        )
    info = {
        "steps": _A2_TRAIN_STEPS,
        "final_train_loss": final_loss,
        "ortho_error": identity_error,
        "windows": len(prepared),
    }
    return rotation.detach().cpu().to(torch.float32), info


def _a2_true_path_gate_loss(
    gate_window: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    states: dict,
    rotation: Optional[torch.Tensor],
    device: torch.device,
) -> float:
    """Normalized full-window output loss through the true deployed path."""

    def _run(state_q: dict, state_k: dict) -> float:
        q_quant, q_scale = gate_window["q"]
        k_quant, k_scale = gate_window["k"]
        v_quant, v_scale = gate_window["v"]
        q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
        k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
        v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)
        q_params = hif4_dynamic_quantize_q(
            q_quant, q_scale, q_heads, head_dim, state_q
        )
        k_params = hif4_dynamic_quantize_k(
            k_quant, k_scale, kv_heads, head_dim, state_k
        )
        v_params = hif4_dynamic_quantize_v(
            v_quant, v_scale, kv_heads, head_dim, states["v_state"]
        )
        q_hat = _dequantize_hif4(q_params).to(torch.float32)
        k_hat = _dequantize_hif4(k_params).to(torch.float32)
        v_hat = _dequantize_hif4(v_params).to(torch.float32)
        reference = _a2_attention_forward(
            q_ref[None], k_ref[None], v_ref[None], q_heads, kv_heads, head_dim
        )[0]
        player = _a2_attention_forward(
            q_hat[None], k_hat[None], v_hat[None], q_heads, kv_heads, head_dim
        )[0]
        return float((player - reference).square().mean())

    player_mse = _run(
        dict(states["q_state"], learned_rotation=rotation),
        dict(states["k_state"], learned_rotation=rotation),
    )
    standard_mse = _run(states["q_state"], states["k_state"])
    return player_mse / max(standard_mse, 1e-12)


_V189_CALIBRATION_ATTENTION = hif4_calibration_attention


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """v189 stack calibration + gated learned rotation (A2 frozen config)."""

    states = _V189_CALIBRATION_ATTENTION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        windows = [
            {
                "q": _dequantize_nvfp4_float32(*item["q"]).to(torch.float32),
                "k": _dequantize_nvfp4_float32(*item["k"]).to(torch.float32),
                "v": _dequantize_nvfp4_float32(*item["v"]).to(torch.float32),
            }
            for item in calib_qkv_list
        ]
        rotation, info = _a2_train_rotation(
            windows[:-1], q_num_heads, kv_num_heads, head_dim, device
        )
        gate_window = calib_qkv_list[-1]
        loss_identity = _a2_true_path_gate_loss(
            gate_window, q_num_heads, kv_num_heads, head_dim, states, None, device
        )
        loss_rotation = _a2_true_path_gate_loss(
            gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device
        )
        if loss_rotation < loss_identity:
            cpu_rotation = rotation.detach().cpu().to(torch.float32).clone()
            states["q_state"]["learned_rotation"] = cpu_rotation
            states["k_state"]["learned_rotation"] = cpu_rotation.clone()
            audit = {
                "a2_arm": "rotation",
                "a2_gate_loss_identity": float(loss_identity),
                "a2_gate_loss_rotation": float(loss_rotation),
                "a2_steps": int(info["steps"]),
                "a2_train_loss": float(info["final_train_loss"]),
                "a2_ortho_error": float(info["ortho_error"]),
            }
        else:
            audit = {
                "a2_arm": "identity",
                "a2_gate_loss_identity": float(loss_identity),
                "a2_gate_loss_rotation": float(loss_rotation),
                "a2_steps": int(info["steps"]),
                "a2_train_loss": float(info["final_train_loss"]),
                "a2_ortho_error": float(info["ortho_error"]),
            }
        states["q_state"].update(audit)
        states["k_state"].update(audit)
    except (RuntimeError, ValueError) as error:
        # A rotation failure must not corrupt the recovered R1 behavior;
        # record the fallback and ship the plain v189 stack states.
        states["q_state"]["a2_arm"] = f"fallback:{type(error).__name__}"
        states["k_state"]["a2_arm"] = states["q_state"]["a2_arm"]
    return states
'''

src = SRC.read_text(encoding="utf-8")
for old, new, label in (
    (SIG_OLD, SIG_NEW, "signature"),
    (INJECT_OLD, INJECT_NEW, "inject"),
    (Q_OLD, Q_NEW, "dynamic Q"),
    (K_OLD, K_NEW, "dynamic K"),
):
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"patch {label}: expected 1 occurrence, found {count}")
    src = src.replace(old, new)
src = src + APPENDIX
SRC.write_text(src, encoding="utf-8")
print("candidate_v3 patched")
