"""低维 A@W 拟合 v2：每块子空间闭式解 + 完整重量化（不再 snap 父格点）。

v1 探针（snap 回父 scale/lv2/lv3 格点）rb=8：o −1.3%、proj +10.5%、fc_up +5.9%。
本版关键改动：
1. 低维闭式解 ΔB 后得到连续块权重 W_D = W_B + ΔB·Uᵀ；
2. 对 W_D 走父完整编码器（_dense_to_hif4 / _gptq_quantize_weight 路径的重量化），
   scale/lv2/lv3 随连续候选重新生成 → 合法五字段（非固定父层级）；
3. 块两臂用真实校准目标 L（teacher = X·W_origᵀ）严格改善才替换；
4. 过拟合对照：fold-0 训练、fold-1 验证，test holdout 判定。

LOCAL diagnostic；3 代表 state；不提交官方。
"""

from __future__ import annotations

import importlib.util
import math
import sys
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
RB = 8


def re_quantize_block(Wd, Xb_calib, lam=0.2):
    """对连续块候选重新完整量化（走父合法编码语义，返回连续解码值）。"""
    # 用块激活 Gram 重建合法 scale（模拟父 _dense_to_hif4 层级）
    # 直接调用父 _dense_to_hif4 的完整路径（无 group_gram 时标准层级 + adaround）
    x = Wd.detach().to(torch.float32)
    wp = sol._dense_to_hif4(
        x,
        importance=None,
        search_offsets=sol._WEIGHT_OFFSETS,
        error_threshold=float(sol._WEIGHT_REFINE_ERROR_THRESHOLD),
        accept_margin=float(sol._WEIGHT_REFINE_ACCEPT_MARGIN),
        max_refine_ratio=0.0,
        max_refine_blocks=0,
    )
    return sol._dequantize_hif4(wp).to(torch.float32)


def block_lowdim_v2(Xb, Yt, Wb0, rb):
    """低维闭式解 + 完整重量化。返回连续解码的块候选 [64, o]。"""
    N = int(Xb.shape[0])
    G = Xb.t() @ Xb
    G = 0.5 * (G + G.t())
    ev, evec = torch.linalg.eigh(G)
    idx = torch.argsort(ev, descending=True)[:rb]
    U = evec[:, idx].contiguous()
    A = Xb @ U                                    # [N, rb]
    R = Yt - Xb @ Wb0                             # [N, o] 块残差
    ATA0 = A.t() @ A
    lam = 0.2
    ATA = ATA0 + lam * float(ATA0.diagonal().mean().clamp_min(1e-12).item()) \
        * torch.eye(rb, device=A.device)
    DB = torch.linalg.solve(ATA, A.t() @ R)       # [rb, o]
    Wd = Wb0 + U @ DB                             # 连续低维候选 [64, o]
    return Wd, re_quantize_block(Wd.t().contiguous(), Xb).t().contiguous()


def main(layer, role, rb=RB):
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
    w_orig = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(
        dtype=torch.float32, device=device
    )
    W0 = v2.dequantize_hif4(dict(wp), w_orig.shape).to(
        dtype=torch.float32, device=device
    )
    o, in_f = int(w_orig.shape[0]), int(w_orig.shape[1])

    x_pairs = []
    for (xq, xs) in calib_pairs:
        x = v2.dequantize_nvfp4(xq, xs).to(torch.float32).to(device)[:128]
        act_params = sol.hif4_dynamic_quantize_activation(xq, xs, astate)
        xh = v2.dequantize_hif4(v2._cpu_params(act_params), x.shape).to(
            dtype=torch.float32, device=device
        )
        x_pairs.append((x, xh))
    F = len(x_pairs)
    std_pairs = []
    for (x, xh) in x_pairs:
        std_w = v2.decode_standard_hif4(v2.encode_standard_hif4(w_orig)).to(
            dtype=torch.float32, device=device
        )
        std_x = v2.decode_standard_hif4(v2.encode_standard_hif4(x)).to(
            dtype=torch.float32, device=device
        )
        mse_std = float(((std_x @ std_w.T - x @ w_orig.T).square().mean()))
        std_pairs.append(max(mse_std, 1e-12))
    omegas = [1.0 / (F * std_pairs[f] * float(x_pairs[f][0].shape[0]) * float(o))
              for f in range(F)]
    Xhw = torch.cat([math.sqrt(omegas[f]) * xh for f, (_, xh) in enumerate(x_pairs)], dim=0)
    Yw = torch.cat([math.sqrt(omegas[f]) * (x @ w_orig.T) for f, (x, _) in enumerate(x_pairs)], dim=0)
    Xval, Yval = x_pairs[1][1], x_pairs[1][0] @ w_orig.T

    n_blocks = in_f // _BLOCK

    def L(W, Xd, Yd):
        return float(((Xd @ W.t() - Yd) ** 2).mean())

    W = W0.clone()
    R = Yw - Xhw @ W.t()
    changed = 0
    for b in range(n_blocks):
        sl = slice(b * _BLOCK, (b + 1) * _BLOCK)
        Xb = Xhw[:, sl]
        Wb0 = W[:, sl].t().contiguous()
        Yt = R + Xb @ Wb0
        Wd, Wq = block_lowdim_v2(Xb, Yt, Wb0, rb)
        W_cand = W.clone()
        W_cand[:, sl] = Wq.t()
        if L(W_cand, Xhw, Yw) < L(W, Xhw, Yw):
            W = W_cand
            R = Yw - Xhw @ W.t()
            changed += 1

    case = next(c for c in pack.linear_cases if c.layer == layer and c.role == role)
    act_pair = v2._move_pair(pack.test_activations[role][case.test_window][layer], device)
    act_params = sol.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], astate)
    ref_x = v2.dequantize_nvfp4(*pack.test_activations[role][case.test_window][layer]).to(
        dtype=torch.float32, device=device
    )
    player_x = v2.dequantize_hif4(v2._cpu_params(act_params), ref_x.shape).to(
        dtype=torch.float32, device=device
    )
    ref_out = ref_x @ w_orig.T
    mse_p = float(((player_x @ W0.t() - ref_out).square().mean()))
    mse_n = float(((player_x @ W.t() - ref_out).square().mean()))
    rel = mse_n / mse_p if mse_p > 0 else float("inf")
    print(f"[{role}-L{layer}] rb={rb} changed={changed}/{n_blocks} "
          f"L_val_end={L(W, Xval, Yval):.3e} L_val_parent={L(W0, Xval, Yval):.3e} "
          f"holdout parent={mse_p:.4e} v2={mse_n:.4e} rel={rel:.4f} "
          f"{'IMPROVE' if rel < 0.99 else ('DEGRADE' if rel > 1.01 else 'FLAT')}")


if __name__ == "__main__":
    targets = [(0, "o"), (11, "proj"), (0, "fc_up")]
    if len(sys.argv) > 1:
        targets = [(int(sys.argv[1]), sys.argv[2])]
    rb = int(sys.argv[3]) if len(sys.argv) > 3 else RB
    for layer, role in targets:
        try:
            main(layer, role, rb)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()