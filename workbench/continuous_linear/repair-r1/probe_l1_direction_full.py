"""L1 方向判定（完整部署一致）：插 T 的校准变体，T=I 逐位恢复父。

工作包 L1/L-R2 要求：
- 硬前向 = 部署对应的权重/激活编码（含 rank-2 gram 修正、static-actorder、
  importance 规范化）；不能以简化 codec 冒充。
- T=I 必须恢复父路径；loss 按同 fold 标准 HiF4 最终输出 MSE 归一化。
- 梯度 norm>1 裁到 1；每 fold ≤128 行、各 fold 等权；inference_mode 可达。

实现：以 L4 父为白盒，重建"T 变换坐标下的完整校准"：
  1. 父校准得 base state（rank u/v、smooth/perm/hadamard、offsets 等）。
  2. T 作用于最终连续坐标：X_T = rank后的X 每64块右乘 T；
     W_T = W_smooth 每64块右乘 T^{-T}。
  3. gram_full = X_TᵀX_T（含 rank 修正，与父 rank gram 语义一致），
     weight GPTQ → weight_hat → importance（列能量规范化）→
     actorder（列能量 64 块序）→ h_inv → activation GPTQ（真实 offsets/refine）。
  4. loss = ||A_hat W_hatᵀ − ref||² / mse_standard（std codec 输出误差）。
  5. 32 步 Adam lr=0.01，norm>1 裁剪，reg=1e-3·mean(B²)，T=I 起步。

T=I 时 path 1-4 应等于父（rank 不变、T 恒等、actorder 由同一 weight_hat 得出）。
LOCAL diagnostic only（代表 state，非全量，不提交）。
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


def sym_zero_trace_exp(B):
    B = (B + B.t()) / 2.0
    B = B - torch.eye(B.shape[0], device=B.device) * (B.diag().mean())
    if float(B.detach().norm()) < 1e-12:
        return torch.matrix_exp(B)
    ev, evec = torch.linalg.eigh(B)
    ev2 = ev.clamp(-_LOG2_4, _LOG2_4)
    B2 = (evec * ev2).mm(evec.t())
    return torch.matrix_exp(B2)


def apply_T_block(x, T):
    prefix = list(x.shape[:-1])
    c = x.shape[-1]
    y = x.reshape(*prefix, c // _BLOCK, _BLOCK)
    y = torch.matmul(y, T)
    return y.reshape(*prefix, c)


def get_state(layer, role):
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
    wp = result["weight_params"]
    ref_w = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(torch.float32).to(device)
    case = next(c for c in pack.linear_cases
                if c.layer == layer and c.role == role)
    ref_x = v2.dequantize_nvfp4(
        *pack.test_activations[role][case.test_window][layer]
    ).to(torch.float32).to(device)
    return astate, wp, ref_w, ref_x, pack, case.test_window


def build_coords(astate, ref_w, ref_x):
    in_f = int(ref_w.shape[-1])
    d_inv = (
        astate["smooth_inv"].to(device=device, dtype=torch.float32)
        if astate.get("smooth_inv") is not None else torch.ones(in_f, device=device)
    )
    perm = astate["permutation"]
    if perm is not None:
        perm = perm.to(device=device, dtype=torch.int64)
    else:
        perm = torch.arange(in_f, dtype=torch.int64, device=device)
    bs = int(astate.get("block_smooth_size", 0))
    bss = int(astate.get("block_smooth_seed", 0))
    W_s = sol._linear_pair_transform(
        ref_w, d_inv.reciprocal(), perm, bs, bss, weight_side=True
    )
    x = ref_x * d_inv.unsqueeze(0)
    x = x.index_select(-1, perm)
    if bs:
        x = sol._block_hadamard_transform(x, bs, bss)
    return W_s, x, astate


def rank_apply(x, astate):
    u = astate["residual_u"].to(device=device, dtype=torch.float32)
    v = astate["residual_v"].to(device=device, dtype=torch.float32)
    return x + (x @ u) @ v.transpose(0, 1)


def full_forward(W_s, x, astate, T, ref_w, ref_x):
    """T 变换坐标完整硬前向：返回 (player, ref, standard_mse)。"""
    T_inv = torch.linalg.inv(T).t()
    W_T = apply_T_block(W_s, T_inv)
    X_rank = rank_apply(x, astate)
    X_T = apply_T_block(X_rank, T)
    X_d = X_T.detach()
    W_d = W_T.detach()

    # gram_full = X_TᵀX_T（含 rank 已在 X_rank 中；与父 rank gram 语义一致）
    gram_full = (X_d.t() @ X_d) / max(float(X_d.shape[0]), 1e-9)
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

    blocks = int(W_d.shape[1]) // _BLOCK
    wgram = sol._flat_group_gram(gram_full, int(W_d.shape[1])).reshape(
        blocks, 8, 2, 4, 4
    ).unsqueeze(0).expand(int(W_d.shape[0]), blocks, 8, 2, 4, 4).contiguous()
    importance_w = X_d.square().mean(dim=0).clamp_min(1e-9)
    wp = sol._gptq_quantize_weight(
        W_d, H,
        importance=importance_w,
        group_gram=wgram,
        search_offsets=sol._WEIGHT_OFFSETS,
        error_threshold=float(sol._WEIGHT_REFINE_ERROR_THRESHOLD),
        accept_margin=float(sol._WEIGHT_REFINE_ACCEPT_MARGIN),
        max_refine_ratio=1.0,
        max_refine_blocks=int(sol._WEIGHT_REFINE_MAX_BLOCKS),
        full_sweep_top_k=0, regularization=reg,
    )
    wh = sol._dequantize_hif4(wp).to(torch.float32)
    wh_ste = wh + (W_T - W_d)

    # importance（父语义：weight_hat 列能量，规范化）与 h_inv
    imp = sol._normalize_importance(
        wh.square().sum(dim=0), int(wh.shape[1])
    )
    if imp is None:
        imp = torch.ones(wh.shape[1], device=device)
    wout_gram = wh.t() @ wh
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

    a_gram = sol._flat_group_gram(wout_gram, int(X_d.shape[1])).reshape(
        blocks, 8, 2, 4, 4
    ).unsqueeze(0).expand(int(X_d.shape[0]), blocks, 8, 2, 4, 4).contiguous()
    ap = sol._activation_gptq_quantize(
        X_d, H_inv_a,
        importance=imp,
        group_gram=a_gram,
        search_offsets=sol._DYNAMIC_OFFSETS,
        error_threshold=float(sol._ACTIVATION_REFINE_ERROR_THRESHOLD),
        accept_margin=float(sol._ACTIVATION_REFINE_ACCEPT_MARGIN),
        max_refine_ratio=float(sol._ACTIVATION_REFINE_MAX_RATIO),
        max_refine_blocks=int(sol._ACTIVATION_REFINE_MAX_BLOCKS),
    )
    ah = sol._dequantize_hif4(ap).to(torch.float32)
    ah_ste = ah + (X_T - X_d)
    player = ah_ste @ wh_ste.t()
    ref = X_rank.detach() @ W_s.detach().t()  # 输出参考（部署坐标 rank 后）

    # 标准 HiF4 输出 MSE（同 fold 标准 codec）
    std_w = v2.decode_standard_hif4(v2.encode_standard_hif4(ref_w)).to(
        dtype=torch.float32, device=device
    )
    std_x = v2.decode_standard_hif4(v2.encode_standard_hif4(ref_x)).to(
        dtype=torch.float32, device=device
    )
    ref_orig = ref_x @ ref_w.T
    std_out = std_x @ std_w.T
    mse_std = float(((std_out - ref_orig).square().mean()))
    return player, ref, mse_std


def run_state(layer, role):
    astate, wp, ref_w, ref_x, pack, tw = get_state(layer, role)
    W_s, x, _ = build_coords(astate, ref_w, ref_x)
    T0 = torch.eye(_BLOCK, device=device)
    player0, ref0, mse_std = full_forward(W_s, x, astate, T0, ref_w, ref_x)
    # T=I 与父逐位核对
    ref_x2 = v2.dequantize_nvfp4(*pack.test_activations[role][tw][layer]).to(
        dtype=torch.float32, device=device
    )
    player_w = v2.dequantize_hif4(dict(wp), ref_w.shape).to(
        dtype=torch.float32, device=device
    )
    act_pair = v2._move_pair(pack.test_activations[role][tw][layer], device)
    act_params = sol.hif4_dynamic_quantize_activation(
        act_pair[0], act_pair[1], astate,
    )
    player_x = v2.dequantize_hif4(v2._cpu_params(act_params), ref_x2.shape).to(
        dtype=torch.float32, device=device
    )
    parent = player_x @ player_w.T
    ref_orig = ref_x2 @ ref_w.T
    parent_mse = float(((parent - ref_orig).square().mean()))
    t0_mse = float(((player0 - ref_orig).square().mean()))
    ratio = t0_mse / parent_mse if parent_mse > 0 else 0
    print(f"[{role}-L{layer}] T=I player/parent MSE ratio = {ratio:.3f} "
          f"{'PARITY' if abs(ratio - 1.0) < 0.02 else 'MISMATCH'}")

    loss0 = float(((player0 - ref_orig).square().mean()) / mse_std)
    print(f"[{role}-L{layer}] baseline(T=I) norm loss = {loss0:.6f}")

    B1 = torch.zeros(8, 8, device=device, requires_grad=True)
    B2 = torch.zeros(8, 8, device=device, requires_grad=True)
    opt = torch.optim.Adam([B1, B2], lr=0.01)
    reg = 1e-3
    t_start = time.perf_counter()
    for step in range(32):
        T1 = sym_zero_trace_exp(B1)
        T2 = sym_zero_trace_exp(B2)
        T = torch.kron(T1, T2)
        player, ref, mse_std = full_forward(W_s, x, astate, T, ref_w, ref_x)
        loss = ((player - ref_orig).square().mean()) / mse_std
        total = loss + reg * (B1.square().mean() + B2.square().mean())
        opt.zero_grad()
        total.backward()
        g = torch.cat([B1.grad.flatten(), B2.grad.flatten()])
        norm = g.norm().clamp_min(1e-12)
        if norm > 1.0:
            B1.grad.mul_(1.0 / norm)
            B2.grad.mul_(1.0 / norm)
        opt.step()
        if step in (0, 7, 15, 31):
            print(f"    step {step + 1}: norm loss = {float(loss.detach()):.6f}")
    print(f"[{role}-L{layer}] wall {time.perf_counter()-t_start:.0f}s")


for layer, role in ((0, "o"), (11, "proj")):
    try:
        run_state(layer, role)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
