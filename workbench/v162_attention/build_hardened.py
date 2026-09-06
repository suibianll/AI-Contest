"""Build hardened A2b/R2b candidates after the official wrong-answer root cause.

Confirmed locally: the official harness runs APIs under an inference context;
the rotation training `backward()` raises there (A2 WA), and unguarded custom
paths anywhere produce submission-fatal exceptions (v107 precedent).  A2b/R2b
therefore:
  1. run training inside `torch.inference_mode(False) + torch.enable_grad()`
     with inputs rebuilt via `.detach().clone()` (normal tensors);
  2. guard EVERY custom path with `except Exception` falling back to the
     exact parent behavior (v162 standard for A2b, R1 for R2b);
  3. fix the L_q < L_kv index landmine in training sampling;
  4. guard the dynamic rotation application (A2b) and the `_nvfp4_to_hif4`
     injection (R2b) individually.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATT = ROOT / "workbench/v162_attention"


def patch_a2b() -> None:
    src = (ATT / "candidate/solution.py").read_text(encoding="utf-8")

    # 1. normal-tensor rebuild helper + training bubble -----------------------
    old_train_head = """def _train_rotation(
    windows: list[dict[str, torch.Tensor]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    \"\"\"Train one ``[G, D, D]`` rotation on the sampled calibration windows.\"\"\"

    groups = kv_heads
    dim = head_dim"""
    new_train_head = """def _normal_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    \"\"\"Rebuild a tensor as a normal (non-inference) tensor on ``device``.\"\"\"

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
    \"\"\"Train one ``[G, D, D]`` rotation on the sampled calibration windows.

    The whole loop runs inside an inference-mode-off/grad-on bubble so the
    official harness may call calibration under any context.
    \"\"\"

    groups = kv_heads
    dim = head_dim"""
    assert src.count(old_train_head) == 1
    src = src.replace(old_train_head, new_train_head)

    old_window_prep = """    prepared: list[dict[str, torch.Tensor]] = []
    for window in windows:
        q_full = window["q"].to(device)
        k_full = window["k"].to(device)
        v_full = window["v"].to(device)"""
    new_window_prep = """    prepared: list[dict[str, torch.Tensor]] = []
    for window in windows:
        q_full = _normal_tensor(window["q"], device)
        k_full = _normal_tensor(window["k"], device)
        v_full = _normal_tensor(window["v"], device)"""
    assert src.count(old_window_prep) == 1
    src = src.replace(old_window_prep, new_window_prep)

    # 2. mixed-q landmine: keep Q subset within the actual Q rows -------------
    old_q_rows = """        kv_index = _even_indices(k_full.shape[0], _MAX_KV_TOKENS, device)
        q_index = _even_indices(kv_index.numel(), _MAX_Q_TOKENS, device)
        q_rows = kv_index.index_select(0, q_index)"""
    new_q_rows = """        kv_index = _even_indices(k_full.shape[0], _MAX_KV_TOKENS, device)
        q_index = _even_indices(min(kv_index.numel(), q_full.shape[0]), _MAX_Q_TOKENS, device)
        q_rows = kv_index.index_select(0, q_index)
        q_rows = q_rows[q_rows < q_full.shape[0]]
        if q_rows.numel() == 0:
            q_rows = _even_indices(q_full.shape[0], _MAX_Q_TOKENS, device)"""
    assert src.count(old_q_rows) == 1
    src = src.replace(old_q_rows, new_q_rows)

    # 3. wrap the optimization loop in the inference-off bubble --------------
    old_loop_head = """    theta = torch.zeros(
        groups, dim, dim, dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=_TRAIN_LR)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")
    for _step in range(_TRAIN_STEPS):"""
    new_loop_head = """    with torch.inference_mode(False), torch.enable_grad():
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
    for _step in range(_TRAIN_STEPS):"""
    assert src.count(old_loop_head) == 1
    src = src.replace(old_loop_head, new_loop_head)

    # 4. calibration total guard ---------------------------------------------
    old_calib_body = """    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = [_decode_window(item) for item in calib_qkv_list]"""
    new_calib_body = """    try:
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
    ]"""
    assert src.count(old_calib_body) == 1
    src = src.replace(old_calib_body, new_calib_body)

    # _decode_window used by impl: normal-tensor decode
    old_decode_window = """def _decode_window(
    item: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    \"\"\"Decode one NVFP4 QKV window to float32 (official BF16 rounding point).\"\"\"

    return {
        name: dequantize_nvfp4(*item[name]).to(torch.float32)
        for name in ("q", "k", "v")
    }"""
    new_decode_window = """def _decode_pair_normal(
    pair: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    decoded = dequantize_nvfp4(*pair).to(torch.float32)
    return _normal_tensor(decoded, device)


def _decode_window(
    item: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    \"\"\"Decode one NVFP4 QKV window to float32 (official BF16 rounding point).\"\"\"

    return {
        name: dequantize_nvfp4(*item[name]).to(torch.float32)
        for name in ("q", "k", "v")
    }"""
    assert src.count(old_decode_window) == 1
    src = src.replace(old_decode_window, new_decode_window)

    # 5. dynamic Q/K guards ---------------------------------------------------
    old_q = """    rotation = _state_rotation(q_state, q_num_heads, head_dim)
    if rotation is None:
        return _standard_params(q_quant, q_scale)
    dense = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    rows = dense.reshape(-1, dense.shape[-1])
    rotated = _rotate_rows(rows, q_num_heads, rotation)
    return _encode_standard_hif4(rotated.reshape(dense.shape))"""
    new_q = """    rotation = _state_rotation(q_state, q_num_heads, head_dim)
    if rotation is None:
        return _standard_params(q_quant, q_scale)
    try:
        dense = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
        rows = dense.reshape(-1, dense.shape[-1])
        rotated = _rotate_rows(rows, q_num_heads, rotation)
        return _encode_standard_hif4(rotated.reshape(dense.shape))
    except Exception:  # noqa: BLE001 - degrade to the legal standard path
        return _standard_params(q_quant, q_scale)"""
    assert src.count(old_q) == 1
    src = src.replace(old_q, new_q)

    old_k = """    rotation = _state_rotation(k_state, kv_num_heads, head_dim)
    if rotation is None:
        return _standard_params(k_quant, k_scale)
    dense = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    rows = dense.reshape(-1, dense.shape[-1])
    rotated = _rotate_rows(rows, kv_num_heads, rotation)
    return _encode_standard_hif4(rotated.reshape(dense.shape))"""
    new_k = """    rotation = _state_rotation(k_state, kv_num_heads, head_dim)
    if rotation is None:
        return _standard_params(k_quant, k_scale)
    try:
        dense = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
        rows = dense.reshape(-1, dense.shape[-1])
        rotated = _rotate_rows(rows, kv_num_heads, rotation)
        return _encode_standard_hif4(rotated.reshape(dense.shape))
    except Exception:  # noqa: BLE001 - degrade to the legal standard path
        return _standard_params(k_quant, k_scale)"""
    assert src.count(old_k) == 1
    src = src.replace(old_k, new_k)

    out = ATT / "candidate_b/solution.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8")
    print("candidate_b (A2b) written")


def patch_r2b() -> None:
    src = (ATT / "candidate_v3/solution.py").read_text(encoding="utf-8")

    # 1. move the v189 calibration call inside the total guard ----------------
    old_wrapper_head = """    states = _V189_CALIBRATION_ATTENTION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""
    new_wrapper_head = """    try:
        states = _V189_CALIBRATION_ATTENTION(
            calib_qkv_list, q_num_heads, kv_num_heads, head_dim
        )
        if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
            return states
    except Exception:  # noqa: BLE001 - unreachable here; kept for safety
        return {"q_state": {}, "k_state": {}, "v_state": {}}
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""
    assert src.count(old_wrapper_head) == 1
    src = src.replace(old_wrapper_head, new_wrapper_head)

    # 2. broaden the fallback catch + fallback target = R1 states (not empty)
    old_catch = """    except (RuntimeError, ValueError) as error:
        # A rotation failure must not corrupt the recovered R1 behavior;
        # record the fallback and ship the plain v189 stack states.
        states["q_state"]["a2_arm"] = f"fallback:{type(error).__name__}"
        states["k_state"]["a2_arm"] = states["q_state"]["a2_arm"]
    return states"""
    new_catch = """    except Exception:  # noqa: BLE001 - any failure degrades to exact R1
        states["q_state"].pop("learned_rotation", None)
        states["k_state"].pop("learned_rotation", None)
        states["q_state"]["a2_arm"] = "fallback"
        states["k_state"]["a2_arm"] = "fallback"
    return states"""
    assert src.count(old_catch) == 1
    src = src.replace(old_catch, new_catch)

    # 3. training bubble: wrap _a2_train_rotation's loop ----------------------
    old_loop = """    theta = torch.zeros(
        groups, dim, dim, dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=_A2_TRAIN_LR)"""
    new_loop = """    with torch.inference_mode(False), torch.enable_grad():
        theta = torch.zeros(
            groups, dim, dim, dtype=torch.float32, device=device, requires_grad=True
        )
        optimizer = torch.optim.Adam([theta], lr=_A2_TRAIN_LR)"""
    assert src.count(old_loop) == 1
    src = src.replace(old_loop, new_loop)
    # indent the remainder of the loop body one level inside the bubble
    lines = src.split("\n")
    start = next(i for i, l in enumerate(lines) if "optimizer = torch.optim.Adam([theta], lr=_A2_TRAIN_LR)" in l)
    end = next(i for i, l in enumerate(lines) if "return rotation.detach().cpu().to(torch.float32), info" in l)
    indent_zone = False
    for i in range(start + 1, end + 1):
        line = lines[i]
        if line.strip().startswith("with torch.no_grad():"):
            indent_zone = True
        if indent_zone and line.strip():
            lines[i] = "    " + line
    src = "\n".join(lines)

    # 4. rebuild training inputs as normal tensors ----------------------------
    old_prep = """    prepared = []
    for window in windows:
        q_full = window["q"].to(device)
        k_full = window["k"].to(device)
        v_full = window["v"].to(device)"""
    new_prep = """    prepared = []
    for window in windows:
        q_full = _a2_normal(window["q"], device)
        k_full = _a2_normal(window["k"], device)
        v_full = _a2_normal(window["v"], device)"""
    assert src.count(old_prep) == 1
    src = src.replace(old_prep, new_prep)

    helper = '''

def _a2_normal(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rebuild a tensor as a normal (non-inference) tensor on ``device``."""

    with torch.inference_mode(False):
        normal = t.detach().to(device=device, dtype=torch.float32).clone()
    if normal.is_inference():
        plain = torch.empty(tuple(normal.shape), device=device, dtype=torch.float32)
        plain.copy_(normal)
        normal = plain
    return normal
'''
    anchor = "_V189_CALIBRATION_ATTENTION = hif4_calibration_attention"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + helper)

    # 5. guard the _nvfp4_to_hif4 injection -----------------------------------
    old_inject = """    if learned_rotation is not None:
        if learned_rotation_num_heads is None:
            raise ValueError("learned rotation requires head count")
        dense = _a2_apply_group_rotation(
            dense, int(learned_rotation_num_heads), learned_rotation
        )"""
    new_inject = """    if learned_rotation is not None and learned_rotation_num_heads is not None:
        try:
            dense = _a2_apply_group_rotation(
                dense, int(learned_rotation_num_heads), learned_rotation
            )
        except Exception:  # noqa: BLE001 - degrade to the unrotated legal path
            pass"""
    assert src.count(old_inject) == 1
    src = src.replace(old_inject, new_inject)

    out = ATT / "candidate_v3b/solution.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8")
    print("candidate_v3b (R2b) written")


if __name__ == "__main__":
    patch_a2b()
    patch_r2b()
