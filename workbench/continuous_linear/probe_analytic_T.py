"""解析 T 研究：对 fc/proj 代表 state，用解析（非训练）方式构造 T。

依据：官方 P3（fc/proj 唯一官方增益桶）+ FlatQuant L1（可逆变换数学方向
保留，训练式 COST_HOLD）。本探针验证"解析求解 T"（不做 32 步 Adam、不改变
硬前向语义）在真实部署坐标上是否有可量化增益，为注册新的解析卡提供方向。

解析 T 方案（低自由度、闭式）：
- 对每 state 校准激活 X（部署坐标：smooth/perm/hadamard 后、rank 前）取每个
  64 块样本，构造块内 Gram 平均 G = mean_b (X_b^T X_b) ∈ R^{64×64}；
- 特征分解 G = Q Λ Q^T，令 T = Q（正交）；若需要 Kronecker 因子化
  T1⊗T2 ≈ Q 可用 SVD（本探针先验整块 Q 的平坦化方向）。
- 比较 T=I vs T=Q 下：
  (a) 激活经标准 HiF4 编码的块内相对量化误差（分布平坦度代理）；
  (b) 权重侧 W T^{-T} 的标准编码误差；
  (c) 最终 X_T @ W_T^T 的量化输出误差（标准 codec，非 GPTQ —— 仅方向代理，
      明确不冒充部署）。

LOCAL diagnostic only；不训练、不扫参、不改父。
"""

from __future__ import annotations

import importlib.util
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


def apply_T_block(x, T):
    prefix = list(x.shape[:-1])
    c = x.shape[-1]
    y = x.reshape(*prefix, c // _BLOCK, _BLOCK)
    y = torch.matmul(y, T)
    return y.reshape(*prefix, c)


def analytic_T(X, T1=None, T2=None):
    """解析 T：块内 Gram 平均的特征基（正交平坦化方向）。"""
    n = int(X.shape[0])
    c = int(X.shape[-1])
    blocks = c // _BLOCK
    Xb = X.reshape(n, blocks, _BLOCK)
    G = torch.einsum("nbi,nbj->ij", Xb, Xb) / max(n * blocks, 1)
    G = 0.5 * (G + G.t())
    ev, evec = torch.linalg.eigh(G)
    # 特征基方向：能量由大到小 -> 用全部特征向量（正交）
    return evec.detach(), ev.detach()


def std_codec_mse(X, W, T):
    """标准 HiF4 编码下 X_T @ W_T^T 的输出 MSE（方向代理）。"""
    T_inv = torch.linalg.inv(T).t()
    X_T = apply_T_block(X, T)
    W_T = apply_T_block(W, T_inv)
    xq = v2.decode_standard_hif4(v2.encode_standard_hif4(X_T)).to(
        dtype=torch.float32, device=device
    )
    wq = v2.decode_standard_hif4(v2.encode_standard_hif4(W_T)).to(
        dtype=torch.float32, device=device
    )
    ref = X @ W.t()
    out = xq @ wq.t()
    return float((out - ref).square().mean())


def analyze(layer: int, role: str):
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
    Xs = []
    for (xq, xs) in calib_pairs:
        x = v2.dequantize_nvfp4(xq, xs).to(torch.float32).to(device)
        x = x * d_inv.unsqueeze(0)
        x = x.index_select(-1, perm)
        if bs:
            x = sol._block_hadamard_transform(x, bs, bss)
        Xs.append(x[:128])
    X = torch.cat(Xs, dim=0)

    T0 = torch.eye(_BLOCK, device=device)
    mse0 = std_codec_mse(X, W_s, T0)
    Tq, ev = analytic_T(X)
    mse1 = std_codec_mse(X, W_s, Tq)
    # 能量集中度：特征值前 1/2/4 占比（平坦度代理：越集中越不平坦）
    e = torch.sort(ev.abs(), descending=True).values
    top2 = float(e[:2].sum() / e.sum().clamp_min(1e-12))
    top8 = float(e[:8].sum() / e.sum().clamp_min(1e-12))
    print(
        f"[{role}-L{layer}] G eig top2={top2:.3f} top8={top8:.3f} "
        f"std_mse T=I={mse0:.4e} T=Q={mse1:.4e} rel={mse1/mse0:.4f} "
        f"{'IMPROVE' if mse1 < mse0 * 0.98 else ('DEGRADE' if mse1 > mse0 * 1.02 else 'FLAT')}"
    )


for layer, role in ((0, "fc_up"), (0, "fc_gate"), (11, "proj"), (0, "proj"), (11, "fc_up")):
    try:
        analyze(layer, role)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()