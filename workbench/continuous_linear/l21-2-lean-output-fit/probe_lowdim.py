"""低维 A@W 拟合探针：每块 r_b 维子空间拟合（非全维 L21 风格）。

思路（用户 21071 机制线索）：
- 每 64 列块 B 取块内校准激活 Gram G_B = Xh_B^T Xh_B 的 top-r_b 特征
  方向 U_B ∈ [64, r_b]；
- 低维激活 A = Xh_B U_B ∈ [N, r_b]，低维修正 ΔB ∈ [r_b, o]；
- 解闭式 min ||A ΔB^T − R_B||²，R_B 为块目标残差（其他列固定）；
- 连续候选 W_new_D = W_B + ΔB U_B^T → 父 scale 格点化回合法五字段；
- 块两臂：真实校准目标 L 严格改善才替换。

自由度每 64 块仅 r_b×o 而非 64×o —— 目标减过拟合与分布外退化。
r_b 固定（4/8/16 取一预注册，不扫）。

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
RB = 8  # 预注册低维子空间秩（不扫）


def snap_block(Z, scale_comb):
    codes = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                         device=Z.device)
    k, oc = Z.shape
    grid_vals = codes.reshape(15, 1, 1) * scale_comb.reshape(1, k, oc)
    dist = (Z.reshape(1, k, oc) - grid_vals).abs()
    return grid_vals.gather(0, dist.argmin(dim=0, keepdim=True))[0]


def block_lowdim_solve(Xb, Yt, Wb0, scale_comb, lam, rb):
    """每块低维子空间拟合：ΔW = ΔB U^T，闭式 ΔB = RᵀA(AᵀA+λI)⁻¹。"""
    N = int(Xb.shape[0])
    oc = int(Yt.shape[1])
    G = Xb.t() @ Xb                              # [64,64]
    G = 0.5 * (G + G.t())
    ev, evec = torch.linalg.eigh(G)
    idx = torch.argsort(ev, descending=True)[:rb]
    U = evec[:, idx].contiguous()                # [64, rb]
    A = Xb @ U                                   # [N, rb]
    # 权重低维修正：W ≈ W0 + ΔB Uᵀ，拟合残差
    # R_B = Yt - A @ (W0 U)ᵀ? 分块：贡献 X_B W_B = X_B U (Uᵀ? no)
    # 目标 min ||A ΔBᵀ - (Yt - X_B W_B)||²
    R = Yt - Xb @ Wb0                            # [N, oc] 块残差（相对目标）
    ATA0 = A.t() @ A
    ATA = ATA0 + lam * float(ATA0.diagonal().mean().clamp_min(1e-12).item()) \
        * torch.eye(rb, device=A.device)
    DB = torch.linalg.solve(ATA, A.t() @ R)      # [rb, oc] solve(AᵀA, AᵀR)
    DB = DB.contiguous()                          # [rb, oc]
    # 目标是关于 ΔB 的：A ΔBᵀ ≈ R → ΔBᵀ = (AᵀA)⁻¹AᵀR，ΔB = RᵀA (AᵀA)⁻¹
    delta = U @ DB                                # [64, oc] 连续低维修正
    W_new = snap_block(Wb0 + delta, scale_comb)
    return W_new


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

    n_blocks = in_f // _BLOCK
    sf = wp["scale_factor"].to(torch.float32)
    lv2v = wp["scale_lv2"].to(torch.float32)
    lv3v = wp["scale_lv3"].to(torch.float32)
    scale_full = (sf * lv2v * lv3v).expand(-1, -1, 8, 2, 4).reshape(o, n_blocks, 64)
    lam = 0.2

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
        scale_comb = scale_full[:, b, :].t().contiguous()
        Wb_new = block_lowdim_solve(Xb, Yt, Wb0, scale_comb, lam, rb)
        W_cand = W.clone()
        W_cand[:, sl] = Wb_new.t()
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
          f"holdout parent={mse_p:.4e} lowdim={mse_n:.4e} rel={rel:.4f} "
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