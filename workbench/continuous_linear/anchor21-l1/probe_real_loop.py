"""L21-1 真实闭环探针（维度修正版）：契约验证 + 代表 state 逐列条件求解。

权重 w_orig 形状为 [o, in]（evaluator 中 X [N,in] @ W.T = output [N,o]）。
块沿 in 维（部署列顺序）切 64 列；每块解出的合法码写回 [:, slice]。

流程（工作包 §3.1-3.3）：
- Xh_f = 真实 dynamic activation（student 激活，保留部署坐标），
  teacher Y_f = X_f W_original^T（原 NVFP4 解码）。
- L(W)=Σ_f ω_f ||Xh_f W^T − Y_f||²，ω_f=1/(F·max(MSE_STD_f,eps)·N_f·o)。
- 固定父 scale/lv2/lv3，逐列条件求解（OBQ 型 F 更新），写回合法码。
- 块级两臂：块提案 vs 保持，用真实校准目标 L 比较，严格改善才替换。
- 过拟合对照：fold-0 训练、fold-1 独立验证、test case holdout。

LOCAL diagnostic only。
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
print(f"device={device.type}")


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
    return result, pack, weight_pair, calib_pairs


def real_Xh_f(calib_pairs, astate, n_rows=128):
    outs = []
    for (xq, xs) in calib_pairs:
        x = v2.dequantize_nvfp4(xq, xs).to(torch.float32).to(device)[:n_rows]
        act_params = sol.hif4_dynamic_quantize_activation(xq, xs, astate)
        xh = v2.dequantize_hif4(v2._cpu_params(act_params), x.shape).to(
            dtype=torch.float32, device=device
        )
        outs.append((x, xh))
    return outs


def block_cond_solve(Xb, Yt, Wb0, scale_comb, lam):
    """一块 [N,64]：连续 LS + 逐列合法格点固定（OBQ 型 F 更新）。

    Xb: [N, 64]；Yt: [N, o]（块目标=R 源列 + 本块贡献）；Wb0: [64, o]（父码）；
    scale_comb: [64, o] 固定层级 scale。
    返回合法块权重 [64, o]（值=scale·code）。
    """
    N = int(Xb.shape[0])
    o = int(Yt.shape[1])
    G = Xb.t() @ Xb
    G = 0.5 * (G + G.t())
    diagm = float(G.diagonal().mean().clamp_min(1e-12))
    H = G + lam * diagm * torch.eye(64, device=G.device)
    D = Xb.t() @ Yt + lam * diagm * Wb0           # Dλ = XhᵀYt + λ·reg·W0
    F = torch.linalg.inv(H)
    Z = torch.linalg.solve(H, D)                  # [64, o] 连续最优（含父锚）

    codes = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                         device=G.device)
    rem = list(range(64))
    fixed = {}
    while rem:
        i = 0
        scale = scale_comb[rem[i]].reshape(1, o)   # [1, o]
        grid_vals = codes.reshape(15, 1) * scale   # [15, o]
        Qi = grid_vals.gather(0, (Z[i].reshape(1, o) - grid_vals).abs().argmin(dim=0).reshape(1, -1))[0]
        delta = Z[i] - Qi
        if delta.abs().max().item() > 0:
            Z = Z - (F[:, i] / F[i, i]).reshape(-1, 1) * delta.reshape(1, -1)
        Z[i] = Qi
        fixed[rem[i]] = Qi
        kcur = F.shape[0]
        idx = [j for j in range(kcur) if j != i]
        F = F - torch.outer(F[:, i], F[i, :]) / F[i, i]
        F = F[idx][:, idx]
        Z = Z[idx]
        rem.pop(i)
    Wb = torch.zeros(64, o, device=G.device)
    for j, v in fixed.items():
        Wb[j] = v
    return Wb


def main(layer, role):
    result, pack, weight_pair, calib_pairs = get_state(layer, role)
    astate = result["activation_state"]
    wp = result["weight_params"]
    w_orig = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(
        dtype=torch.float32, device=device
    )
    W0 = v2.dequantize_hif4(dict(wp), w_orig.shape).to(
        dtype=torch.float32, device=device
    )
    o, in_f = int(w_orig.shape[0]), int(w_orig.shape[1])
    print(f"[{role}-L{layer}] o={o} in={in_f}")

    x_pairs = real_Xh_f(calib_pairs, astate)
    Xh0 = x_pairs[0][1]; X0 = x_pairs[0][0]
    Xh1 = x_pairs[1][1]; X1 = x_pairs[1][0]
    Y0 = X0 @ w_orig.T
    Y1 = X1 @ w_orig.T
    print(f"    Xh0 {tuple(Xh0.shape)} Y0 {tuple(Y0.shape)} (Xh 列 {in_f} == W 行 via W.T)")

    # STD 分母（每 fold 完整标准 HiF4 A@W MSE）
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
    F = len(x_pairs)
    omegas = [1.0 / (F * std_pairs[f] * float(x_pairs[f][0].shape[0]) * float(o))
              for f in range(F)]
    print(f"    F={F} omega={['%.3e' % w for w in omegas]} mse_std={['%.3e' % s for s in std_pairs]}")

    # 固定层级合法 scale：[o, blocks, 64]
    n_blocks = in_f // _BLOCK
    sf = wp["scale_factor"].to(torch.float32)
    lv2v = wp["scale_lv2"].to(torch.float32)
    lv3v = wp["scale_lv3"].to(torch.float32)
    scale_full = (sf * lv2v * lv3v).expand(-1, -1, 8, 2, 4).reshape(o, n_blocks, 64)
    lam = 0.2

    # 全部 fold 按 ω_f 加权拼接（工作包 §3.1：L=Σ_f ω_f ||Xh_f Wᵀ−Y_f||²）
    Xhw = torch.cat([math.sqrt(omegas[f]) * xh for f, (_, xh) in enumerate(x_pairs)], dim=0)
    Yw = torch.cat([math.sqrt(omegas[f]) * (x @ w_orig.T) for f, (x, _) in enumerate(x_pairs)], dim=0)
    Xtr, Ytr = Xhw, Yw
    # 独立验证：fold-1（未参与训练？不——在校准 fold 上学习合法；正式验证用 test holdout）
    Xval, Yval = x_pairs[1][1], x_pairs[1][0] @ w_orig.T
    print(f"    train rows={Xtr.shape[0]} (weighted all folds)")

    def L(W, Xd, Yd):
        return float(((Xd @ W.t() - Yd) ** 2).mean())

    W = W0.clone()
    R = Ytr - Xtr @ W.t()
    changed = 0
    for b in range(n_blocks):
        sl = slice(b * _BLOCK, (b + 1) * _BLOCK)
        Xb = Xtr[:, sl]                          # [N, 64]（in 列块）
        Wb0 = W[:, sl].t().contiguous()          # [64, o]（in 列 x 输出）
        Yt = R + Xb @ Wb0                        # 块目标（其他列残差 + 本块贡献）
        scale_comb = scale_full[:, b, :].t().contiguous()  # [64, o]
        Wb_new = block_cond_solve(Xb, Yt, Wb0, scale_comb, lam)
        W_cand = W.clone()
        W_cand[:, sl] = Wb_new.t()
        L_cur = L(W, Xtr, Ytr)
        L_new = L(W_cand, Xtr, Ytr)
        if L_new < L_cur:
            W = W_cand
            R = Ytr - Xtr @ W.t()
            changed += 1
        if b == 0 or (b + 1) % 7 == 0:
            print(f"    blk{b}: L_tr={L_cur:.3e}->{L_new:.3e} L_val={L(W_cand, Xval, Yval):.3e}")
    print(f"    changed={changed}/{n_blocks}")
    print(f"    L_tr_end={L(W, Xtr, Ytr):.3e} L_val_end={L(W, Xval, Yval):.3e} "
          f"L_val_parent={L(W0, Xval, Yval):.3e}")

    # test case holdout（真实动态激活）
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
    print(f"[{role}-L{layer}] holdout parent={mse_p:.4e} new={mse_n:.4e} rel={rel:.4f} "
          f"{'IMPROVE' if rel < 0.99 else ('DEGRADE' if rel > 1.01 else 'FLAT')}")


if __name__ == "__main__":
    targets = [(0, "o"), (11, "proj"), (0, "fc_up")]
    if len(sys.argv) > 1:
        targets = [(int(sys.argv[1]), sys.argv[2])]
    for layer, role in targets:
        try:
            main(layer, role)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()