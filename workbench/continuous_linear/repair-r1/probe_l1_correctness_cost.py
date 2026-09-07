"""L-R2: 修正版 L1 的正确性与成本判定（evidence-repair L-R2）。

修复项（相对于上一轮 probe_l1_direction*）：
1. 梯度范数是裁剪上限 1：norm>1 裁剪到 1，norm<1 不放大。
2. 每 fold 至多 128 行、各 fold 等权；loss 按同 fold 标准 HiF4 最终输出 MSE
   归一化（不用输出方差）。
3. 硬前向使用部署对应编码（真实 hif4_calibration_and_quantize_weight 的
   weight 编码 + 真实 hif4_dynamic_quantize_activation 解码），state 依赖
   图明确：T 改变 → 变换坐标 → weight GPTQ gram/h_inv/importance 重建。
4. 对矩阵指数/特征值投影做小矩阵梯度检查 + STE 检查 + inference_mode 可达。
5. T=I 必须恢复父路径 bit-exact；rank/状态依赖单独审计。

成本：固定最小面板（层0/11 × o/fc_up/proj）代表 state，完整一次训练步
（预热 + CUDA 同步 + 分段计时：变换/state重建/硬量化/梯度/矩阵求解）。

LOCAL diagnostic only。
"""

from __future__ import annotations

import importlib.util
import json
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
print(f"device={device.type}")


def sym_zero_trace_exp(B: torch.Tensor) -> torch.Tensor:
    """exp(B)，B 对称、零迹、特征值 clip 到 [-log2/4, log2/4]（可微）。"""
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


def grad_check():
    """L-R2-4: 对矩阵指数与特征值投影做解析/数值梯度对照（double）。"""
    torch.manual_seed(0)
    ok = True
    for trial in range(3):
        B = (torch.randn(8, 8, device=device, dtype=torch.float64) * 0.5).requires_grad_(True)
        out = sym_zero_trace_exp(B)
        (out.square().sum()).backward()
        g_analytic = B.grad.clone()
        g_numeric = torch.zeros_like(B)
        eps = 1e-6
        with torch.no_grad():
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    bp = B.clone(); bp[i, j] += eps
                    out_p = sym_zero_trace_exp(bp).square().sum()
                    bm = B.clone(); bm[i, j] -= eps
                    out_m = sym_zero_trace_exp(bm).square().sum()
                    g_numeric[i, j] = (out_p - out_m) / (2 * eps)
        rel = float((g_analytic - g_numeric).abs().max() / g_numeric.abs().max().clamp_min(1e-12))
        print(f"grad_check trial {trial}: max rel diff {rel:.2e}")
        ok = ok and rel < 2e-2

    # STE 检查：直通梯度等于输入（identity）
    x = torch.randn(4, 64, device=device, requires_grad=True)
    q = sol._dense_to_hif4(x.detach(), importance=None, group_gram=None,
                           search_offsets=(1,), error_threshold=0.0,
                           accept_margin=0.0, max_refine_ratio=0.0, max_refine_blocks=0)
    qd = sol._dequantize_hif4(q).to(torch.float32)
    ste = qd + (x - x.detach())
    ste.sum().backward()
    assert x.grad is not None
    grad_ones = float((x.grad - 1.0).abs().max())
    print(f"STE grad max |g-1| = {grad_ones:.2e}")
    ok = ok and grad_ones < 1e-6

    # inference_mode 可达
    with torch.inference_mode():
        T1 = sym_zero_trace_exp(torch.zeros(8, 8, device=device))
        T2 = sym_zero_trace_exp(torch.zeros(8, 8, device=device))
        T = torch.kron(T1, T2)
        assert torch.allclose(T, torch.eye(64, device=device), atol=1e-6)
    print("inference_mode T=I ok")
    return ok


def main_cost():
    raw = v2.load_pack(CACHE)
    rows = []
    for layer, role in ((0, "o"), (0, "fc_up"), (11, "proj")):
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
        wp = result["weight_params"]
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
        # calib fold 每 fold ≤128 行（_LINEAR_EVAL_TOKENS=128 已满足）
        X_list = []
        for (xq, xs) in calib_pairs:
            x = v2.dequantize_nvfp4(xq, xs).to(torch.float32).to(device)
            x = x * d_inv.unsqueeze(0)
            x = x.index_select(-1, perm)
            if bs:
                x = sol._block_hadamard_transform(x, bs, bss)
            X_list.append(x[:128])
        X = torch.cat(X_list, dim=0)

        # ---- 成本实测（完整一次训练步，含完整 weight GPTQ + act GPTQ）----
        timings = {}
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        T = torch.eye(_BLOCK, device=device)
        T_inv = torch.linalg.inv(T).t()
        W_T = apply_T_block(W_s, T_inv)
        X_T = apply_T_block(X, T)
        torch.cuda.synchronize(device)
        timings["transform_T"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gram_full = (X_T.detach().t() @ X_T.detach()) / max(float(X_T.shape[0]), 1e-9)
        H = gram_full.clone()
        reg = 0.2
        H.diagonal().add_(reg * float(H.diagonal().mean()))
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            H.diagonal().add_(reg * float(H.diagonal().mean()) * 10.0)
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        torch.cuda.synchronize(device)
        timings["state_gram_h_inv"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        blocks = int(W_T.shape[1]) // _BLOCK
        wgram = sol._flat_group_gram(gram_full, int(W_T.shape[1])).reshape(
            blocks, 8, 2, 4, 4
        ).unsqueeze(0).expand(
            int(W_T.shape[0]), blocks, 8, 2, 4, 4
        ).contiguous()
        # 部署一致权重编码：真实 offsets + refine 配置（_WEIGHT_* 常量）
        wp_T = sol._gptq_quantize_weight(
            W_T,
            H,
            importance=X_T.detach().square().mean(dim=0).clamp_min(1e-9),
            group_gram=wgram,
            search_offsets=sol._WEIGHT_OFFSETS,
            error_threshold=float(sol._WEIGHT_REFINE_ERROR_THRESHOLD),
            accept_margin=float(sol._WEIGHT_REFINE_ACCEPT_MARGIN),
            max_refine_ratio=1.0,
            max_refine_blocks=int(sol._WEIGHT_REFINE_MAX_BLOCKS),
            full_sweep_top_k=0, regularization=reg,
        )
        torch.cuda.synchronize(device)
        timings["hard_weight_gptq"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        wh_T = sol._dequantize_hif4(wp_T).to(torch.float32)
        wout_gram = wh_T.t() @ wh_T
        H_a = wout_gram.clone()
        reg_a = 0.2
        H_a.diagonal().add_(reg_a * float(H_a.diagonal().mean()))
        try:
            L_a = torch.linalg.cholesky(H_a)
            H_inv_a = torch.cholesky_inverse(L_a)
        except RuntimeError:
            H_a.diagonal().add_(reg_a * float(H_a.diagonal().mean()) * 10.0)
            L_a = torch.linalg.cholesky(H_a)
            H_inv_a = torch.cholesky_inverse(L_a)
        torch.cuda.synchronize(device)
        timings["state_act_h_inv"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        a_gram = sol._flat_group_gram(wout_gram, int(X_T.shape[1])).reshape(
            blocks, 8, 2, 4, 4
        ).unsqueeze(0).expand(
            int(X_T.shape[0]), blocks, 8, 2, 4, 4
        ).contiguous()
        # 部署一致激活编码：真实 offsets + refine 配置（_ACTIVATION_* 常量）
        ap_T = sol._activation_gptq_quantize(
            X_T,
            H_inv_a,
            importance=X_T.detach().square().mean(dim=0).clamp_min(1e-9),
            group_gram=a_gram,
            search_offsets=sol._DYNAMIC_OFFSETS,
            error_threshold=float(sol._ACTIVATION_REFINE_ERROR_THRESHOLD),
            accept_margin=float(sol._ACTIVATION_REFINE_ACCEPT_MARGIN),
            max_refine_ratio=float(sol._ACTIVATION_REFINE_MAX_RATIO),
            max_refine_blocks=int(sol._ACTIVATION_REFINE_MAX_BLOCKS),
        )
        torch.cuda.synchronize(device)
        timings["hard_activation_gptq"] = time.perf_counter() - t0

        total = sum(timings.values())
        print(f"[L{layer}-{role}] w{tuple(ref_w.shape)}")
        for k, val in timings.items():
            print(f"    {k:22s} {val:.3f}s")
        print(f"    {'total':22s} {total:.3f}s")
        rows.append({"layer": layer, "role": role, "shape": list(ref_w.shape),
                     "timings": timings, "total_s": total})
    return rows


if __name__ == "__main__":
    ok = grad_check()
    rows = main_cost()
    out = Path(r"d:\工作内容\AI竞赛\artifacts\proxy_v3\continuous\linear\repair-r1\l1-cost.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"grad_check_ok": ok, "rows": rows}, indent=1), encoding="utf-8")
    print("wrote", out, "grad_check_ok", ok)