"""低维 A@W 拟合 v3：权重 SVD 主导方向全局低秩修正（r 维）。

v1（逐块激活 Gram 方向+snap）o 层改善 −1.3%；v2（全重量化）更差。
本版：对每个 weight state，取 W_orig 的 SVD top-r**输出方向** V_r（V 是右奇异
向量，w_orig ∈ [o,in]，W U 奇异方向），解低维闭式：

  ΔW = U_r B V_rᵀ，B ∈ r×r，
  min ||Xh ΔWᵀ − R||²，R = Y − Xh W0ᵀ（校准残差）

闭式（U_r⊤ 正交）：B = U_rᵀ R A (AᵀA+λI)⁻¹，A = Xh V_r ∈ [N,r]。
随后 W0+ΔW 走父完整编码器重量化；块/全局两臂用真实校准目标 L 接受。

这对应"低维空间拟合"最自然的数学形式：权重变化被限制在相对原权重的
低秩子空间内（r=8/16 预注册，不扫）。

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


def re_quantize(Wd):
    """对连续候选完整重量化（父合法编码语义），返回合法解码。"""
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


def lowrank_solve(W_orig, Xhw, Yw, W0, rb, lam=0.2):
    """全局低秩修正：ΔW = U_r B V_rᵀ。

    W_orig ∈ [o,in]；Xhw ∈ [N,in]；Yw ∈ [N,o]。
    V_r = W_orig SVD 右奇异 top-r（in 方向）；U_r = W_orig 左奇异 top-r（o 方向）。
    A = Xhw V_r ∈ [N,r]；R = Yw − Xhw W0ᵀ ∈ [N,o]。
    B = U_rᵀ R A (AᵀA+λI)·固定(直接小矩阵)。
    返回连续 ΔW。
    """
    Uu, _, Vv = torch.linalg.svd(W_orig.double().detach(), full_matrices=False)
    U_r = Uu[:, :rb].to(torch.float32)              # [o, r]
    V_r = Vv[:rb, :].t().to(torch.float32).contiguous()  # [in, r]
    A = Xhw @ V_r                                    # [N, r]
    R = Yw - Xhw @ W0.t()                            # [N, o]
    ATA0 = A.t().double() @ A.double()
    ATA = ATA0 + lam * float(ATA0.diagonal().mean().clamp_min(1e-12).item()) \
        * torch.eye(rb, dtype=torch.float64, device=A.device)
    # min_B ||A Bᵀ − R U_r||² → Bᵀ = (AᵀA)⁻¹ Aᵀ R U_r；ΔW = U_r B V_rᵀ
    Bt = torch.linalg.solve(ATA, A.t().double() @ (R.double() @ U_r.double()))
    B = Bt.t().contiguous().to(torch.float32)        # [r, r]
    delta = (U_r.double() @ B.double() @ V_r.t().double()).to(torch.float32)
    return delta


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

    def L(W, Xd, Yd):
        return float(((Xd @ W.t() - Yd) ** 2).mean())

    # 全局低秩候选
    delta = lowrank_solve(w_orig, Xhw, Yw, W0, rb)
    Wd = W0 + delta
    Wq = re_quantize(Wd)
    W_cand = Wq
    print(f"[{role}-L{layer}] rb={rb} global delta_norm={float(delta.norm()):.3e} "
          f"L_tr parent={L(W0, Xhw, Yw):.3e} cand={L(W_cand, Xhw, Yw):.3e} "
          f"L_val parent={L(W0, Xval, Yval):.3e} cand={L(W_cand, Xval, Yval):.3e}")

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
    mse_n = float(((player_x @ W_cand.t() - ref_out).square().mean()))
    rel = mse_n / mse_p if mse_p > 0 else float("inf")
    print(f"[{role}-L{layer}] holdout parent={mse_p:.4e} lowrank={mse_n:.4e} rel={rel:.4f} "
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