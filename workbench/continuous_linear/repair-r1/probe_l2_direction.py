"""L-R2 方向探针（修正版）：真实输入 + 部署一致编码 + 正确 STE/梯度。

上一轮 probe_l1_direction* 因 round(x/scale) 自制输入 + 简化 codec 被
R0 判为 PROBE_INVALID_FOR_DEPLOYMENT。本探针在 L-R1 闭环可信口径上重测：
- 输入：evaluator nvfp4_encode （同 L-R1，12/12 逐位一致）
- 硬前向：部署一致 weight GPTQ（真实 offsets/refine）+ activation GPTQ
- 梯度：矩阵指数 32 步 Adam，norm>1 裁剪到 1（norm<1 不放大）
- 损失：||Q(XT)Q(WT^{-T})^T − XW^T||² / 同 fold 标准 HiF4 输出 MSE
- T=I 恒等基线：连续 zero，量化输出应等于父（已在 L-R1 验证）

只测代表 state（窄 o / 宽 proj / fc_up），32 步 × 部署一致硬前向
（0.6-3.2s/step）≈ 每 state 20-100s，属于诊断成本，不提交官方。

LOCAL diagnostic only。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

CACHE = Path(r"d:\工作内容\AI竞赛\artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt")
L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)
EVAL_DIR = Path(r"d:\工作内容\AI竞赛\evaluator")

sys.path.insert(0, str(EVAL_DIR))
import proxy_v3_eval as pv3  # noqa: E402
import official_eval as v2  # noqa: E402

spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BLOCK = 64
_LOG2_4 = math.log(2.0) / 4.0
print(f"device={device.type} nvfp4={v2.NVFP4_MODE}")


def sym_zero_trace_exp(B: torch.Tensor) -> torch.Tensor:
    B = (B + B.t()) / 2.0
    B = B - torch.eye(B.shape[0], device=B.device) * (B.diag().mean())
    if float(B.detach().norm()) < 1e-12:
        return torch.matrix_exp(B)
    ev, evec = torch.linalg.eigh(B)
    ev2 = ev.clamp(-_LOG2_4, _LOG2_4)
    B2 = (evec * ev2).mm(evec.t())
    return torch.matrix_exp(B2)


def apply_T_block(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    prefix = list(x.shape[:-1])
    c = x.shape[-1]
    y = x.reshape(*prefix, c // _BLOCK, _BLOCK)
    y = torch.matmul(y, T)
    return y.reshape(*prefix, c)


def deployment_forward(W_s, X, T, reg=0.2):
    """部署一致硬前向：T 变换坐标下 weight GPTQ + activation GPTQ。

    返回 (player 输出, ref 输出)。STE 直通：量化结果 + (连续输入 − detach)，
    使梯度流回 T（声明的替代梯度；梯度检查见 probe_l1_correctness_cost）。
    """
    T_inv = torch.linalg.inv(T).t()
    W_T = apply_T_block(W_s, T_inv)
    X_T = apply_T_block(X, T)
    W_T_d = W_T.detach()
    X_T_d = X_T.detach()
    blocks = int(W_T.shape[1]) // _BLOCK
    gram_full = (X_T_d.t() @ X_T_d) / max(float(X_T_d.shape[0]), 1e-9)
    H = gram_full.clone()
    H.diagonal().add_(reg * float(H.diagonal().mean()))
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        H.diagonal().add_(reg * float(H.diagonal().mean()) * 10.0)
        L = torch.linalg.cholesky(H)
    wgram = sol._flat_group_gram(gram_full, int(W_T.shape[1])).reshape(
        blocks, 8, 2, 4, 4
    ).unsqueeze(0).expand(int(W_T.shape[0]), blocks, 8, 2, 4, 4).contiguous()
    wp = sol._gptq_quantize_weight(
        W_T_d, H,
        importance=X_T_d.square().mean(dim=0).clamp_min(1e-9),
        group_gram=wgram,
        search_offsets=sol._WEIGHT_OFFSETS,
        error_threshold=float(sol._WEIGHT_REFINE_ERROR_THRESHOLD),
        accept_margin=float(sol._WEIGHT_REFINE_ACCEPT_MARGIN),
        max_refine_ratio=1.0,
        max_refine_blocks=int(sol._WEIGHT_REFINE_MAX_BLOCKS),
        full_sweep_top_k=0, regularization=reg,
    )
    wh = sol._dequantize_hif4(wp).to(torch.float32)
    wh_ste = wh + (W_T - W_T_d)
    wout_gram = wh.detach().t() @ wh.detach()
    H_a = wout_gram.clone()
    reg_a = 0.2
    H_a.diagonal().add_(reg_a * float(H_a.diagonal().mean()))
    try:
        L_a = torch.linalg.cholesky(H_a)
    except RuntimeError:
        H_a.diagonal().add_(reg_a * float(H_a.diagonal().mean()) * 10.0)
        L_a = torch.linalg.cholesky(H_a)
    a_gram = sol._flat_group_gram(wout_gram, int(X_T.shape[1])).reshape(
        blocks, 8, 2, 4, 4
    ).unsqueeze(0).expand(int(X_T.shape[0]), blocks, 8, 2, 4, 4).contiguous()
    ap = sol._activation_gptq_quantize(
        X_T_d, torch.cholesky_inverse(L_a),
        importance=X_T_d.square().mean(dim=0).clamp_min(1e-9),
        group_gram=a_gram,
        search_offsets=sol._DYNAMIC_OFFSETS,
        error_threshold=float(sol._ACTIVATION_REFINE_ERROR_THRESHOLD),
        accept_margin=float(sol._ACTIVATION_REFINE_ACCEPT_MARGIN),
        max_refine_ratio=float(sol._ACTIVATION_REFINE_MAX_RATIO),
        max_refine_blocks=int(sol._ACTIVATION_REFINE_MAX_BLOCKS),
    )
    ah = sol._dequantize_hif4(ap).to(torch.float32)
    ah_ste = ah + (X_T - X_T_d)
    return ah_ste @ wh_ste.t(), X @ W_s.t()


def run_state(layer: int, role: str):
    raw = v2.load_pack(CACHE)
    shard = layer % pv3.SHARD_COUNT
    pack = pv3.prepare_shard(raw, shard, scenario="linear", ood=False)
    calib_pairs = [
        v2._move_pair(pack.linear_calibration_activations[role][sample][layer], device)
        for sample in pack.metadata["linear_calibration_indices"]
    ]
    weight_pair = v2._move_pair(pack.weights[layer][role], device)
    result = sol.hif4_calibration_and_quantize_weight(
        weight_pair[0], weight_pair[1], calib_pairs
    )
    astate = result["activation_state"]
    ref_w = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(torch.float32).to(device)
    d_inv = (
        astate["smooth_inv"].to(device=device, dtype=torch.float32)
        if astate.get("smooth_inv") is not None else torch.ones(ref_w.shape[-1], device=device)
    )
    perm = astate["permutation"]
    if perm is not None:
        perm = perm.to(device=device, dtype=torch.int64)
    else:
        perm = torch.arange(ref_w.shape[-1], dtype=torch.int64, device=device)
    bs = int(astate.get("block_smooth_size", 0))
    bss = int(astate.get("block_smooth_seed", 0))
    W_s = sol._linear_pair_transform(
        ref_w, d_inv.reciprocal(), perm, bs, bss, weight_side=True
    )
    X_list = []
    for (xq, xs) in calib_pairs:
        x = v2.dequantize_nvfp4(xq, xs).to(torch.float32).to(device)
        x = x * d_inv.unsqueeze(0)
        x = x.index_select(-1, perm)
        if bs:
            x = sol._block_hadamard_transform(x, bs, bss)
        X_list.append(x[:128])
    X = torch.cat(X_list, dim=0)

    T0 = torch.eye(_BLOCK, device=device)
    player0, ref0 = deployment_forward(W_s, X, T0)
    std_mse = float((ref0 - ref0.mean()).square().mean().clamp_min(1e-12))
    loss0 = float(((player0 - ref0).square().mean() / std_mse))

    B1 = torch.zeros(8, 8, device=device, requires_grad=True)
    B2 = torch.zeros(8, 8, device=device, requires_grad=True)
    opt = torch.optim.Adam([B1, B2], lr=0.01)
    reg = 1e-3
    print(f"[{role}-L{layer}] baseline(T=I) norm loss = {loss0:.6f}")
    t_start = time.perf_counter()
    for step in range(32):
        T1 = sym_zero_trace_exp(B1)
        T2 = sym_zero_trace_exp(B2)
        T = torch.kron(T1, T2)
        player, ref = deployment_forward(W_s, X, T)
        loss = (player - ref).square().mean() / std_mse
        total = loss + reg * (B1.square().mean() + B2.square().mean())
        opt.zero_grad()
        total.backward()
        g = torch.cat([B1.grad.flatten(), B2.grad.flatten()])
        norm = g.norm().clamp_min(1e-12)
        # 正确裁剪：norm>1 才缩放
        if norm > 1.0:
            B1.grad.mul_(1.0 / norm)
            B2.grad.mul_(1.0 / norm)
        opt.step()
        if step in (0, 7, 15, 31):
            print(f"    step {step + 1}: norm loss = {float(loss.detach()):.6f}")
    dt = time.perf_counter() - t_start
    print(f"[{role}-L{layer}] wall {dt:.1f}s ({dt/32:.3f}s/step)")


for layer, role in ((0, "o"), (11, "proj"), (0, "fc_up")):
    try:
        run_state(layer, role)
    except Exception as exc:  # noqa: BLE001
        print(f"[{role}-L{layer}] ERROR {type(exc).__name__}: {exc}")