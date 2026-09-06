"""L1 cost probe: measure one full hard-forward training step cost for the
structured invertible transform (FlatQuant 8x8 T1/T2) on a representative
Linear weight state.

Purpose (workpackage continuous-linear.md L1 gate):
决定训练循环是否可负担"完整硬前向每步重建"（weight GPTQ + activation GPTQ
+ output error）。若单 state 32 步成本 × 168 states 超出时间模型预算，则
按工作包要求记录 COST/DESIGN_HOLD 并提出不同求解机制，不私下换成 operand loss。

This is a LOCAL diagnostic only. It imports the archived L4 solution module
for its private helpers; the official candidate remains a single file.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import torch

L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)

spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device} torch={torch.__version__}")


def nvfp4_approx(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate NVFP4 (E2M1-like 4-bit) encoding: per-16-block scale."""
    m = dense.reshape(*dense.shape[:-1], -1, 16)
    amax = m.abs().amax(dim=-1)
    scale = (amax / 6.0).clamp_min(1e-8)
    q = torch.round(m / scale.unsqueeze(-1)).clamp_(-6.0, 6.0)
    return q.flatten(-2, -1), scale


def make_state(in_features: int, out_features: int) -> tuple:
    w_dense = (torch.randn(out_features, in_features, device=device) * 0.02).to(
        torch.float32
    )
    weight_q, weight_s = nvfp4_approx(w_dense)
    calib = []
    for _ in range(2):
        x_dense = torch.randn(128, in_features, device=device) * 0.5
        x_q, x_s = nvfp4_approx(x_dense)
        calib.append((x_q, x_s))
    return weight_q, weight_s, calib


def probe_one(in_features: int, out_features: int, label: str) -> None:
    weight_q, weight_s, calib = make_state(in_features, out_features)
    t0 = time.perf_counter()
    out = sol.hif4_calibration_and_quantize_weight(weight_q, weight_s, calib)
    t_full = time.perf_counter() - t0
    wp = out["weight_params"]
    astate = out["activation_state"]
    print(
        f"[{label}] full calibrate one state: {t_full:.3f}s "
        f"wp_keys={sorted(wp.keys())} state_keys={sorted(astate.keys())}"
    )

    # Emulate the L1 training step: rebuild weight in transformed coord with a
    # candidate T, re-run GPTQ weight + activation GPTQ on sampled rows.
    weight = sol._dequantize_nvfp4_float32(weight_q, weight_s)
    identity_d = torch.ones(in_features, dtype=torch.float32, device=device)
    weight_smooth = sol._linear_pair_transform(
        weight,
        identity_d,
        sol._identity_permutation(in_features, device),
        0,
        0,
        weight_side=True,
    )
    # identity T step baseline: just re-run GPTQ weight + act quantize
    t1 = time.perf_counter()
    gram = astate.get("gram")
    h_inv = astate.get("h_inv")
    t2 = time.perf_counter()
    print(f"    preprocessing step (grams/h_inv fetch): {t2 - t1:.3f}s")

    # --- fast-path weights: plain legal HiF4 encode (no GPTQ loop) ---
    t1 = time.perf_counter()
    w_params_fast = sol._dense_to_hif4(
        weight_smooth,
        importance=astate["importance"],
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=float(astate["error_threshold"]),
        accept_margin=float(astate["accept_margin"]),
        max_refine_ratio=0.0,
        max_refine_blocks=0,
    )
    t_wfast = time.perf_counter() - t1
    print(f"    [#{label}] one fast-path weight encode: {t_wfast:.3f}s")

    # --- fast-path activation encode ---
    dense = sol._dequantize_nvfp4_float32(calib[0][0], calib[0][1])
    dense = sol._sample_rows(dense, 128)
    t1 = time.perf_counter()
    a_params_fast = sol._dense_to_hif4(
        dense,
        importance=astate["importance"],
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=float(astate["error_threshold"]),
        accept_margin=float(astate["accept_margin"]),
        max_refine_ratio=0.0,
        max_refine_blocks=0,
    )
    t_afast = time.perf_counter() - t1
    print(f"    [#{label}] one fast-path activation encode (128 rows): {t_afast:.3f}s")

    t1 = time.perf_counter()
    # weight GPTQ with the stored gram rebuilt (identity T)
    blocks = in_features // 64
    gram_dev = (
        astate["gram"].to(device=device, dtype=torch.float32)
        if astate.get("gram") is not None
        else None
    )
    honly = (
        torch.eye(in_features, device=device)
        if gram_dev is None
        else None
    )
    wgram = None
    if gram_dev is not None:
        wgram = gram_dev.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
            int(weight.shape[0]), blocks, 8, 2, 4, 4
        )
    w_params = sol._gptq_quantize_weight(
        weight_smooth,
        gram_dev if gram_dev is not None else honly,
        importance=astate["importance"],
        group_gram=wgram,
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=float(astate["error_threshold"]),
        accept_margin=float(astate["accept_margin"]),
        max_refine_ratio=float(astate["max_refine_ratio"]),
        max_refine_blocks=int(astate["max_refine_blocks"]),
        full_sweep_top_k=0,
        regularization=0.2,
    )
    t_w = time.perf_counter() - t1
    print(f"    [#{label}] one weight GPTQ step: {t_w:.3f}s")

    t1 = time.perf_counter()
    x = calib[0][0][:128]
    dense = sol._dequantize_nvfp4_float32(calib[0][0], calib[0][1])
    # apply rank + T (identity) then activation GPTQ
    if astate.get("residual_u") is not None and astate.get("residual_v") is not None:
        u = astate["residual_u"].to(device=device, dtype=torch.float32)
        v = astate["residual_v"].to(device=device, dtype=torch.float32)
        dense = dense + (dense @ u) @ v.transpose(0, 1)
    nrows = int(dense.shape[0])
    a_gram = None
    if gram_dev is not None:
        a_gram = gram_dev.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
            nrows, blocks, 8, 2, 4, 4
        )
    a_params = sol._activation_gptq_quantize(
        dense,
        h_inv.to(device=device, dtype=torch.float32),
        importance=astate["importance"],
        group_gram=a_gram,
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=float(astate["error_threshold"]),
        accept_margin=float(astate["accept_margin"]),
        max_refine_ratio=float(astate["max_refine_ratio"]),
        max_refine_blocks=int(astate["max_refine_blocks"]),
    )
    t_a = time.perf_counter() - t1
    print(f"    [#{label}] one activation GPTQ step: {t_a:.3f}s")


probe_one(768, 768, "narrow-qkv")
probe_one(4864, 768, "wide-proj-in")
probe_one(768, 4864, "wide-fc-out")