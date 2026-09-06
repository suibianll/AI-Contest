"""v162-independent Attention candidate: per-KV-group learnable orthogonal rotation.

Linear APIs, dynamic V and the shared standard codec are frozen at v162.
The attention side learns, per KV group, one orthogonal matrix R_g that is
applied to every query head of the group and to the group key head, so the
continuous Q'K'^T product is invariant while HiF4 re-encoding sees rotated
distributions.  Training runs only inside ``hif4_calibration_attention`` on
calibration windows with the full attention output loss (hard v162 HiF4
forward, STE for backward); the dynamic Q/K APIs execute one fixed matrix
multiplication per head before the standard encode and add no search, no
candidate loops and no cross-call caching.  V stays standard v162.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


_HIF4_BLOCK_SIZE = 64
_NVFP4_BLOCK_SIZE = 16
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_HIF4_MAX_INNER = 7.0
_BF16_ONE_SEVENTH = 0.142578125

# --- frozen attention-side research configuration (workpackage A2 table) ---
_ROT_ARM = "gate"  # "identity" | "h" | "learned" | "gate"; deployed candidate keeps "gate"
_TRAIN_STEPS = 32
_TRAIN_LR = 0.01
_TRAIN_CLIP = 1.0
_REG_WEIGHT = 1e-3
_MAX_KV_TOKENS = 128
_MAX_Q_TOKENS = 32
_GATE_CHUNK = 512
_ORTHO_TOLERANCE = 1e-3


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Official NVFP4 dequantization with the BF16 rounding point."""

    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by NVFP4 block size {blk_size}"
        )
    expected_scale_shape = tuple(quant_float.shape[:-1]) + (channels // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 scale shape {tuple(scale_float.shape)} != {expected_scale_shape}"
        )
    grouped = quant_float.detach().to(torch.float32).unflatten(-1, (-1, blk_size))
    result = grouped * scale_float.detach().to(torch.float32).unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    """Encode non-negative FP32 values into finite unsigned E6M2 codes."""

    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_E6M2_MIN,
        posinf=_E6M2_MAX,
        neginf=_E6M2_MIN,
    ).clamp(min=_E6M2_MIN, max=_E6M2_MAX)

    exponent = torch.floor(torch.log2(x))
    base = torch.pow(2.0, exponent)
    mantissa_field = torch.round((x / base - 1.0) * 4.0).to(torch.int64)

    carry = mantissa_field >= 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa_field = torch.where(
        carry, torch.zeros_like(mantissa_field), mantissa_field
    ).clamp(min=0, max=3)

    exponent_field = (exponent.to(torch.int64) + 48).clamp(min=0, max=63)
    code = exponent_field * 4 + mantissa_field
    return code.clamp(min=0, max=254).to(torch.int16)


def _e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(min=0, max=254)
    exponent_field = torch.bitwise_right_shift(c, 2)
    mantissa_field = torch.bitwise_and(c, 3)
    exponent = exponent_field.to(torch.float32) - 48.0
    return torch.pow(2.0, exponent) * (
        1.0 + mantissa_field.to(torch.float32) * 0.25
    )


def _standard_e6m2_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official amax/7 base scale with a BF16 intermediate."""

    high_precision_scale = (
        amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = _e6m2_encode_nearest(high_precision_scale)
    return code, _e6m2_decode(code)


def _solve_standard_hierarchy(
    absolute: torch.Tensor,
    scale_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the minimum-MSE legal hierarchy for each eight-value group."""

    losses: list[torch.Tensor] = []
    for total_exponent in (0, 1, 2):
        local_scale = scale_factor[..., None, None, None] * float(
            1 << total_exponent
        )
        mantissa = (
            torch.round(absolute * (4.0 / local_scale)).clamp_(0.0, 7.0) * 0.25
        )
        losses.append((absolute - mantissa * local_scale).square().sum(dim=-1))

    loss_0, loss_1, loss_2 = losses
    choose_01 = loss_1 < loss_0
    choose_12 = loss_2 < loss_1
    cost_lv2_1 = torch.minimum(loss_0, loss_1).sum(dim=-1)
    cost_lv2_2 = torch.minimum(loss_1, loss_2).sum(dim=-1)
    use_lv2_2 = cost_lv2_2 < cost_lv2_1
    use_lv3_2 = torch.where(use_lv2_2[..., None], choose_12, choose_01)

    scale_lv2 = 1.0 + use_lv2_2.to(torch.float32)
    scale_lv3 = 1.0 + use_lv3_2.to(torch.float32)
    denominator = (
        scale_factor[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(absolute * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )
    return scale_lv2, scale_lv3, mantissa


def _encode_standard_hif4(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor with the standard HiF4 codec."""

    if dense.ndim < 1:
        raise ValueError("dense must have at least one dimension")
    prefix = tuple(int(v) for v in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by HiF4 block size 64"
        )
    blocks = channels // _HIF4_BLOCK_SIZE

    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    sign = torch.sign(x_grouped)

    amax = x_abs.amax(dim=(-1, -2, -3))
    _, standard_scale = _standard_e6m2_scale(amax)
    scale_lv2, scale_lv3, mantissa = _solve_standard_hierarchy(
        x_abs, standard_scale
    )

    sign_out = sign.reshape(*prefix, blocks, 8, 2, 4)
    mantissa_out = mantissa.reshape(*prefix, blocks, 8, 2, 4)
    sign_out = torch.where(
        mantissa_out == 0.0, torch.zeros_like(sign_out), sign_out
    )
    return {
        "scale_factor": standard_scale.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign_out,
        "mant": mantissa_out,
    }


def _decode_hif5_fields(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Decode standard HiF4 fields back to a dense tensor (candidate side)."""

    return (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    ).flatten(start_dim=-4, end_dim=-1)


def _standard_params(quant_float: torch.Tensor, scale_float: torch.Tensor):
    dense = dequantize_nvfp4(quant_float, scale_float).to(torch.float32)
    return _encode_standard_hif4(dense)


# ---------------------------------------------------------------------------
# Attention-side rotation helpers (frozen Linear/V paths never call these).
# ---------------------------------------------------------------------------


def _hadamard_orthogonal(dim: int) -> torch.Tensor | None:
    """Normalized Sylvester Hadamard matrix for power-of-two dims."""

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


def _cayley_orthogonal(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cayley transform C=(I-A)(I+A)^{-1} for skew A=Theta-Theta^T, plus reg."""

    skew = theta - theta.transpose(-1, -2)
    eye = torch.eye(theta.shape[-1], device=theta.device, dtype=theta.dtype)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(
        right.transpose(-1, -2), left.transpose(-1, -2)
    ).transpose(-1, -2)
    reg = (c - eye).square().mean()
    return c, reg


def _even_indices(total: int, limit: int, device: torch.device) -> torch.Tensor:
    """Deterministic evenly spaced subset of at most ``limit`` indices."""

    if total <= limit:
        return torch.arange(total, device=device)
    positions = torch.linspace(0, total - 1, limit, device=device).round()
    return torch.unique(positions.to(torch.int64))


def _decode_pair_normal(
    pair: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    decoded = dequantize_nvfp4(*pair).to(torch.float32)
    return _normal_tensor(decoded, device)


def _decode_window(
    item: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Decode one NVFP4 QKV window to float32 (official BF16 rounding point)."""

    return {
        name: dequantize_nvfp4(*item[name]).to(torch.float32)
        for name in ("q", "k", "v")
    }


def _attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Non-causal GQA forward mirroring the evaluator ``_attention``."""

    batch, tokens, _ = q.shape
    qh = q.reshape(batch, tokens, q_heads, head_dim).transpose(1, 2)
    group = q_heads // kv_heads
    kh = k.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    vh = v.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    probabilities = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (probabilities @ vh).transpose(1, 2).reshape(batch, tokens, q_heads * head_dim)


def _ste_encode(dense: torch.Tensor) -> torch.Tensor:
    """Hard v162 HiF4 forward with a straight-through gradient path."""

    params = _encode_standard_hif4(dense)
    decoded = _decode_hif5_fields(params).to(dense.dtype)
    return dense + (decoded - dense).detach()


def _rotate_rows(
    dense: torch.Tensor,
    num_heads: int,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply per-group right multiplication ``head_row @ R_g`` to (T, C) rows."""

    tokens = dense.shape[0]
    head_dim = dense.shape[-1] // num_heads
    groups = rotation.shape[0]
    per_group = num_heads // groups
    grouped = dense.reshape(tokens, groups, per_group, head_dim)
    rotated = torch.einsum(
        "tghk,gkd->tghd", grouped, rotation.to(device=grouped.device, dtype=torch.float32)
    )
    return rotated.reshape(tokens, dense.shape[-1])


def _normal_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rebuild a tensor as a normal (non-inference) tensor on ``device``."""

    with torch.inference_mode(False):
        normal = t.detach().to(device=device, dtype=torch.float32).clone()
    if normal.is_inference():
        plain = torch.empty(
            tuple(normal.shape), device=device, dtype=torch.float32
        )
        plain.copy_(normal)
        normal = plain
    return normal


def _train_rotation(
    windows: list[dict[str, torch.Tensor]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Train one ``[G, D, D]`` rotation on the sampled calibration windows.

    The whole loop runs inside an inference-mode-off/grad-on bubble so the
    official harness may call calibration under any context.
    """

    groups = kv_heads
    dim = head_dim
    base = _hadamard_orthogonal(dim)
    if base is None:
        base = torch.eye(dim, dtype=torch.float32)
    base = base.to(device)

    prepared: list[dict[str, torch.Tensor]] = []
    for window in windows:
        q_full = _normal_tensor(window["q"], device)
        k_full = _normal_tensor(window["k"], device)
        v_full = _normal_tensor(window["v"], device)
        kv_index = _even_indices(k_full.shape[0], _MAX_KV_TOKENS, device)
        q_index = _even_indices(min(kv_index.numel(), q_full.shape[0]), _MAX_Q_TOKENS, device)
        q_rows = kv_index.index_select(0, q_index)
        q_rows = q_rows[q_rows < q_full.shape[0]]
        if q_rows.numel() == 0:
            q_rows = _even_indices(q_full.shape[0], _MAX_Q_TOKENS, device)
        k_sub = k_full.index_select(0, kv_index)
        v_sub = v_full.index_select(0, kv_index)
        q_sub = q_full.index_select(0, q_rows)
        reference = _attention_forward(
            q_sub[None], k_sub[None], v_sub[None], q_heads, kv_heads, head_dim
        )[0].detach()
        std_q = _decode_hif5_fields(_encode_standard_hif4(q_sub)).to(torch.float32)
        std_k = _decode_hif5_fields(_encode_standard_hif4(k_sub)).to(torch.float32)
        std_v = _decode_hif5_fields(_encode_standard_hif4(v_sub)).to(torch.float32)
        standard = _attention_forward(
            std_q[None], std_k[None], std_v[None], q_heads, kv_heads, head_dim
        )[0].detach()
        mse_std = float((standard - reference).square().mean())
        v_hat = _ste_encode(v_sub).detach()
        prepared.append({
            "q": q_sub, "k": k_sub, "v": v_sub, "v_hat": v_hat,
            "reference": reference, "mse_std": max(mse_std, 1e-12),
        })

    with torch.inference_mode(False), torch.enable_grad():
        return _train_rotation_loop(
            prepared, base, groups, dim, q_heads, kv_heads, head_dim, device
        )


def _train_rotation_loop(
    prepared: list[dict[str, torch.Tensor]],
    base: torch.Tensor,
    groups: int,
    dim: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    theta = torch.zeros(
        groups, dim, dim, dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=_TRAIN_LR)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")
    for _step in range(_TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        window_losses = []
        reg_total = theta.new_zeros(())
        for item in prepared:
            c, reg = _cayley_orthogonal(theta)
            rotation = base[None] @ c if base.shape[0] == 1 else torch.einsum(
                "kd,gkl->gdl", base, c
            )
            reg_total = reg_total + reg
            q_rot = _rotate_rows(item["q"], q_heads, rotation)
            k_rot = _rotate_rows(item["k"], kv_heads, rotation)
            q_hat = _ste_encode(q_rot)
            k_hat = _ste_encode(k_rot)
            output = _attention_forward(
                q_hat[None], k_hat[None], item["v_hat"][None],
                q_heads, kv_heads, head_dim,
            )[0]
            loss = (output - item["reference"]).square().mean() / item["mse_std"]
            window_losses.append(loss)
        data_loss = torch.stack(window_losses).mean()
        objective = data_loss + _REG_WEIGHT * reg_total / len(prepared)
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("rotation training produced a non-finite loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_([theta], _TRAIN_CLIP)
        optimizer.step()
        final_loss = float(data_loss.detach())
    with torch.no_grad():
        c, _reg = _cayley_orthogonal(theta)
        rotation = torch.einsum("dk,gkl->gdl", base, c)
        identity_error = float(
            (rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max()
        )
    if identity_error > _ORTHO_TOLERANCE:
        raise RuntimeError(
            f"trained rotation failed the FP orthogonality check: {identity_error}"
        )
    info = {
        "steps": _TRAIN_STEPS,
        "final_train_loss": final_loss,
        "ortho_error": identity_error,
        "windows": len(prepared),
    }
    return rotation.detach().cpu().to(torch.float32), info


def _gate_loss_for_rotation(
    window: dict[str, torch.Tensor],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rotation: torch.Tensor | None,
    device: torch.device,
) -> float:
    """Normalized full-window output loss for one rotation arm.

    Non-causal attention: Q rows are chunked for memory but every row attends
    to the complete K/V, so K and V are never chunked.
    """

    q = window["q"].to(device)
    k = window["k"].to(device)
    v = window["v"].to(device)
    rotation_dev = rotation.to(device) if rotation is not None else None
    k_std_dense = _decode_hif5_fields(_encode_standard_hif4(k)).to(torch.float32)
    v_std_dense = _decode_hif5_fields(_encode_standard_hif4(v)).to(torch.float32)
    if rotation_dev is None:
        k_player_dense = k_std_dense
    else:
        k_rot = _rotate_rows(k, kv_heads, rotation_dev)
        k_player_dense = _decode_hif5_fields(_encode_standard_hif4(k_rot)).to(torch.float32)
    player_sum = 0.0
    standard_sum = 0.0
    count = 0
    for start in range(0, q.shape[0], _GATE_CHUNK):
        end = min(start + _GATE_CHUNK, q.shape[0])
        q_chunk = q[start:end]
        reference = _attention_forward(
            q_chunk[None], k[None], v[None], q_heads, kv_heads, head_dim
        )[0]
        standard_q = _decode_hif5_fields(_encode_standard_hif4(q_chunk)).to(torch.float32)
        standard = _attention_forward(
            standard_q[None], k_std_dense[None], v_std_dense[None],
            q_heads, kv_heads, head_dim,
        )[0]
        if rotation_dev is None:
            player = standard
        else:
            player_q = _decode_hif5_fields(
                _encode_standard_hif4(_rotate_rows(q_chunk, q_heads, rotation_dev))
            ).to(torch.float32)
            player = _attention_forward(
                player_q[None], k_player_dense[None], v_std_dense[None],
                q_heads, kv_heads, head_dim,
            )[0]
        player_sum += float((player - reference).square().sum())
        standard_sum += float((standard - reference).square().sum())
        count += reference.numel()
    mse_std = standard_sum / max(1, count)
    mse_player = player_sum / max(1, count)
    return mse_player / max(mse_std, 1e-12)

def _finite_or_sentinel(value: Any) -> float:
    """Finite float for state storage; -1.0 marks an unmeasured arm."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0
    return number if math.isfinite(number) else -1.0


def _build_state(
    rotation: torch.Tensor | None,
    arm: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    audit: Mapping[str, float],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "format": "group-rotation-v1",
        "arm": str(arm),
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "mode": "rotation" if rotation is not None else "identity",
        "gate_loss_identity": _finite_or_sentinel(audit.get("identity")),
        "gate_loss_h": _finite_or_sentinel(audit.get("h")),
        "gate_loss_learned": _finite_or_sentinel(audit.get("learned")),
        "trained_steps": int(audit.get("steps", 0)),
    }
    if rotation is not None:
        state["r"] = rotation.detach().cpu().to(torch.float32).clone()
    return state


def _state_rotation(
    state: Any,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor | None:
    """Return the usable rotation from a state, or None for identity/mismatch."""

    if not isinstance(state, Mapping) or state.get("mode") != "rotation":
        return None
    rotation = state.get("r")
    if not torch.is_tensor(rotation):
        return None
    if int(state.get("head_dim", -1)) != int(head_dim):
        return None
    groups = int(rotation.shape[0]) if rotation.ndim == 3 else 0
    if groups <= 0 or num_heads % groups != 0:
        return None
    return rotation


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Standard v162 baseline: encode the weight, ignore calibration samples."""

    return {
        "weight_params": _standard_params(weight_quant, weight_scale),
        "activation_state": {},
    }


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    return _standard_params(activation_quant, activation_scale)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Learn the per-KV-group rotation on calibration windows only."""

    if q_num_heads % kv_num_heads != 0 or head_dim < 1:
        return {"q_state": {}, "k_state": {}, "v_state": {}}
    if not calib_qkv_list:
        return {"q_state": {}, "k_state": {}, "v_state": {}}
    try:
        return _calibration_attention_impl(
            calib_qkv_list, q_num_heads, kv_num_heads, head_dim
        )
    except Exception:  # noqa: BLE001 - any failure must degrade to v162
        return {"q_state": {}, "k_state": {}, "v_state": {}}


def _calibration_attention_impl(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = [
        {
            name: (
                dequantize_nvfp4(*pair)
                if False
                else _decode_pair_normal(pair, device)
            )
            for name, pair in item.items()
        }
        for item in calib_qkv_list
    ]

    audit: dict[str, Any] = {"arm": _ROT_ARM, "steps": 0, "windows": len(windows)}
    learned: torch.Tensor | None = None
    if _ROT_ARM in ("learned", "gate") and len(windows) >= 2:
        with torch.enable_grad():
            learned, info = _train_rotation(
                windows[:-1], q_num_heads, kv_num_heads, head_dim, device
            )
        audit.update(info)

    gate_window = windows[-1]
    loss_identity = _gate_loss_for_rotation(
        gate_window, q_num_heads, kv_num_heads, head_dim, None, device
    )
    audit["identity"] = loss_identity

    base = _hadamard_orthogonal(head_dim)
    fixed_h: torch.Tensor | None = None
    if base is not None and kv_num_heads >= 1:
        fixed_h = base[None].repeat(kv_num_heads, 1, 1)
    loss_h: float | None = None
    if fixed_h is not None:
        loss_h = _gate_loss_for_rotation(
            gate_window, q_num_heads, kv_num_heads, head_dim, fixed_h, device
        )
        audit["h"] = loss_h
    loss_learned: float | None = None
    if learned is not None:
        loss_learned = _gate_loss_for_rotation(
            gate_window, q_num_heads, kv_num_heads, head_dim, learned, device
        )
        audit["learned"] = loss_learned

    if _ROT_ARM == "identity":
        deployed: torch.Tensor | None = None
        deployed_arm = "identity"
    elif _ROT_ARM == "learned" and learned is not None:
        deployed = learned
        deployed_arm = "learned"
    else:
        # "h" and "gate": pick between H and learned on the gate first.
        picked = fixed_h
        picked_arm = "h"
        if (
            _ROT_ARM == "gate"
            and loss_learned is not None
            and loss_h is not None
            and loss_learned < loss_h
        ):
            picked = learned
            picked_arm = "learned"
        # Then deploy only on a strict improvement over identity.
        picked_loss = loss_learned if picked_arm == "learned" else loss_h
        if picked is not None and picked_loss is not None and picked_loss < loss_identity:
            deployed = picked
            deployed_arm = picked_arm
        else:
            deployed = None
            deployed_arm = "identity"
    audit["deployed"] = deployed_arm

    rotation_cpu = deployed.detach().cpu().to(torch.float32) if deployed is not None else None
    q_state = _build_state(
        rotation_cpu, deployed_arm, q_num_heads, kv_num_heads, head_dim, audit
    )
    k_state = _build_state(
        rotation_cpu, deployed_arm, kv_num_heads, kv_num_heads, head_dim, audit
    )
    return {"q_state": q_state, "k_state": k_state, "v_state": {}}


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    rotation = _state_rotation(q_state, q_num_heads, head_dim)
    if rotation is None:
        return _standard_params(q_quant, q_scale)
    try:
        dense = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
        rows = dense.reshape(-1, dense.shape[-1])
        rotated = _rotate_rows(rows, q_num_heads, rotation)
        return _encode_standard_hif4(rotated.reshape(dense.shape))
    except Exception:  # noqa: BLE001 - degrade to the legal standard path
        return _standard_params(q_quant, q_scale)


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    rotation = _state_rotation(k_state, kv_num_heads, head_dim)
    if rotation is None:
        return _standard_params(k_quant, k_scale)
    try:
        dense = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
        rows = dense.reshape(-1, dense.shape[-1])
        rotated = _rotate_rows(rows, kv_num_heads, rotation)
        return _encode_standard_hif4(rotated.reshape(dense.shape))
    except Exception:  # noqa: BLE001 - degrade to the legal standard path
        return _standard_params(k_quant, k_scale)


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _standard_params(v_quant, v_scale)
