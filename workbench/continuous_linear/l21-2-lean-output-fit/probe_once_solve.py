"""L21-2 探针：块级一次输出求解（无逐列 F 更新循环）vs L21-1 vs L4 父。

对照（config.md §4）：
- 精简父 = L4 父本身（真实 activation_state）作为基线（block rename:parent）。
- 精简父+拟合 = 在父权重 W0 上做块级一次 LS + 格点化（每 64 块：
  Z=solve(Hλ,Dλ) 直接格点化，不进逐列 OBQ 循环），块两臂接受。
- 与 L21-1 逐列结果对比：判断 holdout 退化是"输出感知拟合固有"还是
  "逐列循环过拟合"。

局部诊断；3 个代表 state；不提交官方。
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


def snap_block(Z, scale_comb):
    """一次性格点化（无逐列循环）：Z/scale -> 最近 0.25 code -> 合法值。"""
    codes = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                         device=Z.device)
    k, oc = Z.shape
    scale = scale_comb                                     # [k, oc]
    grid_vals = codes.reshape(15, 1, 1) * scale.reshape(1, k, oc)
    dist = (Z.reshape(1, k, oc) - grid_vals).abs()          # [15, k, oc]
    return grid_vals.gather(0, dist.argmin(dim=0, keepdim=True))[0]


def block_once_solve(Xb, Yt, Wb0, scale_comb, lam):
    """块级一次求解：LS + 一次格点化（无逐列 F 循环）。"""
    N = int(Xb.shape[0])
    o = int(Yt.shape[1])
    G = Xb.t() @ Xb
    G = 0.5 * (G + G.t())
    diagm = float(G.diagonal().mean().clamp_min(1e-12))
    H = G + lam * diagm * torch.eye(64, device=G.device)
    D = Xb.t() @ Yt + lam * diagm * Wb0
    Z = torch.linalg.solve(H, D)                            # [64, o]
    return snap_block(Z, scale_comb)


def main(layer, role):
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
        Wb_new = block_once_solve(Xb, Yt, Wb0, scale_comb, lam)
        W_cand = W.clone()
        W_cand[:, sl] = Wb_new.t()
        if L(W_cand, Xhw, Yw) < L(W, Xhw, Yw):
            W = W_cand
            R = Yw - Xhw @ W.t()
            changed += 1
    print(f"[{role}-L{layer}] once-solve changed={changed}/{n_blocks} "
          f"L_val_end={L(W, Xval, Yval):.3e} L_val_parent={L(W0, Xval, Yval):.3e}")

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
    print(f"[{role}-L{layer}] holdout parent={mse_p:.4e} once={mse_n:.4e} rel={rel:.4f} "
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