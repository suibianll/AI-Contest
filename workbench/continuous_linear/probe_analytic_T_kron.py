"""fc/proj Kronecker 解析 T：T=T1⊗T2（8×8 因子）的解析求解方向诊断。

特征基解析 T（完整 64×64 正交 Q = eigh(G)）已测得 5/5 fc/proj state 退化。
本探针测试 Kronecker 结构（FlatQuant 卡部署形式 T=T1⊗T2）的解析解是否不同：
对块内 Gram 平均 G，将 G 映射到 Kronecker 空间 min||T1⊗T2 − G||_F² 求闭式
（SVD-based），T1/T2 保持可逆；随后比较 T=I 与 T=T1⊗T2 下标准 codec
输出误差（方向代理，非部署 GPTQ；不扫参、不训练）。

若 Kronecker 结构也无正向，则 fc/proj 解析旋转族（正交/Kronecker 均退化）
证据闭合，与 L1 训练式 COST_HOLD 互证；不再在该族继续投本地算力。
LOCAL diagnostic only。
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


def apply_T_block(x, T):
    prefix = list(x.shape[:-1])
    c = x.shape[-1]
    y = x.reshape(*prefix, c // _BLOCK, _BLOCK)
    y = torch.matmul(y, T)
    return y.reshape(*prefix, c)


def kron_fit(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """min ||T1⊗T2 − G||_F² 的闭式近似（基于 8×8 分块的 SVD）。"""
    n = 8
    d = _BLOCK
    # G ∈ [64,64] → reshape [8, 8, 8, 8]，按 [i,k,j,l] 重排后求秩-1 张量分解
    Gq = G.reshape(n, n, n, n)          # [i,j,k,l] 其中 G[(i,k),(j,l)]
    M = Gq.permute(0, 2, 1, 3).reshape(n * n, n * n)  # [i,k] x [j,l]
    U, S, Vh = torch.linalg.svd(M)
    s1 = float(S[0])
    u1 = U[:, 0].reshape(n, n)
    v1 = Vh[0, :].reshape(n, n).t()
    T1 = torch.sqrt(torch.tensor(s1, device=G.device)) * u1 / u1.norm().clamp_min(1e-12)
    T2 = torch.sqrt(torch.tensor(s1, device=G.device)) * v1 / v1.norm().clamp_min(1e-12)
    return T1.contiguous(), T2.contiguous()


def std_codec_mse(X, W, T):
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

    n = int(X.shape[0])
    c = int(X.shape[-1])
    blocks = c // _BLOCK
    Xb = X.reshape(n, blocks, _BLOCK)
    G = torch.einsum("nbi,nbj->ij", Xb, Xb) / max(n * blocks, 1)
    G = 0.5 * (G + G.t())

    T0 = torch.eye(_BLOCK, device=device)
    mse0 = std_codec_mse(X, W_s, T0)
    T1, T2 = kron_fit(G)
    Tk = torch.kron(T1, T2)
    # 条件数检查（可逆性）
    cond = float(torch.linalg.cond(T1)) * float(torch.linalg.cond(T2))
    mse1 = std_codec_mse(X, W_s, Tk)
    rel = mse1 / mse0
    print(
        f"[{role}-L{layer}] kron_fit cond={cond:.2f} "
        f"std_mse T=I={mse0:.4e} T=T1xT2={mse1:.4e} rel={rel:.4f} "
        f"{'IMPROVE' if rel < 0.98 else ('DEGRADE' if rel > 1.02 else 'FLAT')}"
    )


for layer, role in ((0, "fc_up"), (0, "fc_gate"), (11, "proj"), (0, "proj")):
    try:
        analyze(layer, role)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()