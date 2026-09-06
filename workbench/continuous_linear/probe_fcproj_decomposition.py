"""fc/proj 定向误差分解：L4 部署坐标下 W-only/A-only/Both/interaction。

依据：官方 P3 探针（2026-09-05）显示 Linear 官方增益 100% 落在 fc+proj
大形状桶；本地 L4 分组 fc_up 0.502 / fc_gate 0.554 / proj 0.564 最差。
本探针在真实 L4 部署路径上量化为四臂，定位 fc/proj 剩余误差来自
W 量化、A 量化还是交互，为下一张机制卡提供方向。

LOCAL diagnostic only；结构与 evaluator E00/E10/E01/E11 一致。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

CACHE = Path(r"d:\工作内容\AI竞赛\artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt")
L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)
spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

device = "cuda" if torch.cuda.is_available() else "cpu"


def nvfp4_pair(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = dense.reshape(*dense.shape[:-1], -1, 16)
    amax = m.abs().amax(dim=-1)
    scale = (amax / 6.0).clamp_min(1e-8)
    q = torch.round(m / scale.unsqueeze(-1)).clamp_(-6.0, 6.0)
    return q.flatten(-2, -1), scale


def decompose(layer: int, role: str):
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    w_dense = cache["weights"][layer][role].to(torch.float32)
    wq, ws = nvfp4_pair(w_dense)
    calib_pairs = []
    for sample in cache["calibration_activations"][role][:2]:
        act = sample[layer].to(torch.float32)
        calib_pairs.append(nvfp4_pair(act))
    out = sol.hif4_calibration_and_quantize_weight(wq, ws, calib_pairs)
    astate = out["activation_state"]
    weight = sol._dequantize_nvfp4_float32(wq, ws)
    d_inv = astate.get("smooth_inv")
    if d_inv is not None:
        d_inv = d_inv.to(device=device, dtype=torch.float32)
    else:
        d_inv = torch.ones(weight.shape[1], device=device)
    perm = astate["permutation"]
    if perm is not None:
        perm = perm.to(device=device, dtype=torch.int64)
    else:
        perm = sol._identity_permutation(weight.shape[1], device)
    bs = int(astate.get("block_smooth_size", 0))
    bss = int(astate.get("block_smooth_seed", 0))
    W_s = sol._linear_pair_transform(
        weight.to(device), d_inv.reciprocal(), perm, bs, bss, weight_side=True
    )
    Xs = []
    for (xq, xs) in calib_pairs:
        x = sol._dequantize_nvfp4_float32(xq, xs).to(device)
        x = x * d_inv.unsqueeze(0)
        x = x.index_select(-1, perm)
        Xs.append(x)
    X = torch.cat(Xs, dim=0)
    Y = X @ W_s.t()  # continuous reference in deploy coords
    en = (Y - Y.mean()).square().mean()

    # 四臂：standard/std codec 用 _dense_to_hif4（合法，无 GPTQ）为"STD"侧。
    # 注意：L4 部署 Weight 走 GPTQ、激活走 h_inv GPTQ；此处 E10/E01 使用同
    # 一套候选编码器分别量化 W 或 A，另一侧保持 continuous，以便定位来源。
    wp = sol._gptq_quantize_weight(
        W_s,
        (X.t() @ X) / max(float(X.shape[0]), 1e-9),
        importance=None,
        group_gram=None,
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=0.0,
        accept_margin=0.0,
        max_refine_ratio=0.0,
        max_refine_blocks=0,
        full_sweep_top_k=0,
        regularization=0.2,
    )
    W_q = sol._dequantize_hif4(wp).to(torch.float32)
    # activation 用 h_inv 路径近似：直接用 importance 引导的简单编码
    x_importance = X.square().mean(dim=0).clamp_min(1e-9)
    x_enc = sol._dense_to_hif4(
        X,
        importance=x_importance,
        group_gram=None,
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=0.0,
        accept_margin=0.0,
        max_refine_ratio=float(astate["max_refine_ratio"]),
        max_refine_blocks=int(astate["max_refine_blocks"]),
    )
    X_q = sol._dequantize_hif4(x_enc).to(torch.float32)

    E00 = (Y - Y).square().mean() / en              # both std = 0 by construction
    Y10 = X @ W_q.t()                                # W-only
    E10 = (Y - Y10).square().mean() / en
    Y01 = X_q @ W_s.t()                              # A-only
    E01 = (Y - Y01).square().mean() / en
    Y11 = X_q @ W_q.t()                              # both
    E11 = (Y - Y11).square().mean() / en
    interact = E11 - E10 - E01 - E00
    print(
        f"[{role}-L{layer}] E10(W-only)={E10:.6f} E01(A-only)={E01:.6f} "
        f"E11(Both)={E11:.6f} interaction={interact:+.6f} "
        f"contrib_e10={100*E10/E11:.1f}% contrib_e01={100*E01/E11:.1f}%"
    )
    return E10, E01, E11, interact


for layer, role in ((0, "fc_up"), (2, "fc_up"), (0, "proj"), (2, "proj"),
                    (0, "fc_gate"), (16, "o")):
    try:
        decompose(layer, role)
    except Exception as exc:  # noqa: BLE001
        print(f"[{role}-L{layer}] ERROR {type(exc).__name__}: {exc}")