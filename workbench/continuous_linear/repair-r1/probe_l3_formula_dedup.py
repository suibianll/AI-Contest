"""L-R3: GPTAQ vs JDRQ 公式级去重（单步矩阵对照）。

依据（evidence-repair L-R3）：
"验证'残差引导逐坐标选择'是否真的等价于论文含逆Hessian与非对称残差项的
逐列更新。给出一个小型确定性矩阵的单步更新对照，使用相同输入和初始权重。
若公式可证明等价或完整覆盖，则维持 DUPLICATE_CLOSED。"

JDRQ 目标（源码 _jdrq_make_target）：
   target = base + η · Rᵀ Z (ZᵀZ + λI)⁻¹      （R = Y − Z·baseᵀ）
即岭回归残差补偿，随后走合法离散化（mantissa ±0.25/0 坐标下降）。

GPTAQ 理想（论文 2504.02692）逐列解：
   W* = (ZᵀZ + λI)⁻¹ Zᵀ Y          （H = ZᵀZ, B = ZᵀY, 每列 w_j = H⁻¹ b_j）

推导：η=1, λ=0 时
   base + RᵀZ(ZᵀZ)⁻¹ = base + (Y − Z baseᵀ)ᵀ Z (ZᵀZ)⁻¹
                      = base + YᵀZ H⁻¹ − base (ZᵀZ)(ZᵀZ)⁻¹
                      = YᵀZ H⁻¹ = W*
本探针用小型确定性矩阵直接数值对照两者。

LOCAL diagnostic only。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)
spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")


def close_form(z, y, lam):
    """GPTAQ: W* = Yᵀ Z (ZᵀZ + λI)⁻¹ （输出 Y = Z Wᵀ，W ∈ [o,k]）"""
    H = z.t() @ z
    H = H + lam * H.diagonal().mean() * torch.eye(H.shape[0], device=z.device)
    return y.t() @ z @ torch.linalg.inv(H)


def gptaq_one_step(w, z, y, lam=0.0):
    """GPTAQ Eq.(5)-(8) 逐列 OBS 式更新：min ||Δw X − r||²，r = wX̃ − wX。

    取 X̃=Y 目标（全精度输出），X=Z 输入；对单个输出通道 w（一行），
    单步 Δw = (δ/ (H⁻¹_qq)) H⁻¹_q,: 的闭式解为：
        Δw = r H⁻¹   （在连续域一次全列更新，等价于对残差的最优线性补偿）
    这里 H = ZᵀZ（特征空间 Hessian）。用于公式对照，非论文完整实现。
    """
    H = z.t() @ z
    H = H + lam * H.diagonal().mean() * torch.eye(H.shape[0], device=z.device)
    H_inv = torch.linalg.inv(H)
    r = y - z @ w.t()          # [n, o] 输出残差（per output channel）
    delta = r.t() @ z @ H_inv  # [o, k] 连续最优 Δw
    return w + delta


def jdrq_target(base, z, y, lam, eta):
    """JDRQ: base + η Rᵀ Z (ZᵀZ + λI)⁻¹，与源码同语义"""
    residual = y - z @ base.t()
    proj = sol._jdrq_ridge_projection(z, residual, lam)
    return base + eta * residual.t().mm(proj)


rows = []
gptaq_vs_jdrq = []
for trial in range(3):
    n, k, o = 16, 8, 5
    z = torch.randn(n, k, device=device)
    y = torch.randn(n, o, device=device)
    base = torch.randn(o, k, device=device)
    for lam in (0.0, 0.1):
        w_star = close_form(z, y, lam)
        target1 = jdrq_target(base, z, y, lam, eta=1.0)
        target0 = jdrq_target(torch.zeros_like(base), z, y, lam, eta=1.0)
        gptaq_w = gptaq_one_step(base, z, y, lam=lam)
        diff_base = float((target1 - w_star).abs().max())
        diff_zero = float((target0 - w_star).abs().max())
        rel = float((target1 - w_star).abs().max() / w_star.abs().max().clamp_min(1e-12))
        diff_gvj = float((gptaq_w - target1).abs().max())
        gptaq_vs_jdrq.append(diff_gvj)
        rows.append((trial, lam, diff_base, diff_zero, rel))
        print(
            f"trial{trial} lam={lam}: JDRQ(base≠0,η=1) vs W* maxdiff={diff_base:.3e} "
            f"JDRQ(base=0,η=1) vs W* maxdiff={diff_zero:.3e} rel={rel:.2e} "
            f"GPTAQ-single vs JDRQ-target maxdiff={diff_gvj:.3e}"
        )

    # GPTAQ 非对称（仅一阶）残差项对照：JDRQ 是 full-dual（Zᵀ 侧 + residual），
    # 若论文是逐列 w_j = H⁻¹b_j，则单步即收敛到 W*，上面已对照。

# 结论：若 rel<1e-5 且 diff_zero≈0，则 JDRQ target(η=1) == GPTAQ 闭式解（λ=0），
# λ>0 时为岭回归正则化版本（同一族、更保守）。
ok = all(r[4] < 1e-5 for r in rows)
print("formula-equivalent:", ok)
print("max|GPTAQ_single − JDRQ_target| =", max(gptaq_vs_jdrq))