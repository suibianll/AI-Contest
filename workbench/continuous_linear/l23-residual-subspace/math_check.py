"""L23 数学最小检查：Cholesky 白化残差交叉 → rank-r SVD → 解回 ΔW。

工作包 §4 给出的闭式构造（待核验）：
1. G = Σ_ω Xh_B^T Xh_B + λI（块内 Gram + ridge，∈R^{64×64}）
2. H = Σ_ω Xh_B^T R（残差交叉，∈R^{64×o}）
3. 用 G 的 Cholesky 白化 H：Hw = L^{-1} H（L = cholesky(G)）
4. 对白化交叉 Hw 取 rank-r 截断 SVD：Hw ≈ U_r Σ_r V_r^T
5. 解回 ΔW_B = L^{-T} U_r Σ_r V_r^T（= L^{-T} (U_r U_r^T) Hw）

要验证：
- A) 白化方向与维度正确（Hw 形状 [64,o]，L 形状 [64,64]）
- B) 截断 rank-r 后的解回 ΔW 与"用最优子空间投影 Hw 到 top-r"一致，
  且目标 Σ||R − Xh ΔW||² + λ||ΔW||² 优于或等于父（残差下降）。
- C) 与直接全维最小二乘（不截断 rank）相比，截断 rank-8 的近似度。
- D) 单独小例子 vs 旧激活 Gram top-8 的差异（证明子空间同残差相关）。

LOCAL math check only；不加载 evaluator。
"""

from __future__ import annotations

import torch

torch.manual_seed(1)
dtype = torch.float64
BLOCK = 64


def lowdim_whitened(Xb, R, lam, r, omegas=None):
    """L23 闭式：G 白化 H → rank-r SVD → 解回 ΔW。

    Xb: [n,64]；R: [n,o]（残差，已含 fold 加权）；lam: ridge；r: rank。
    返回 ΔW [64,o]。
    """
    G = Xb.t() @ Xb
    G = 0.5 * (G + G.t())
    diagm = float(G.diagonal().mean().clamp_min(1e-12))
    Gλ = G + lam * diagm * torch.eye(64, dtype=dtype, device=G.device)
    L = torch.linalg.cholesky(Gλ)                   # 下三角, Gλ = L L^T
    # 白化 H：Hw = L^{-1} H（求解 Y，L Y = H → Y = L^-1 H）
    H = Xb.t() @ R                                   # [64, o]
    Hw = torch.linalg.solve_triangular(L, H, upper=False)
    U, S, Vt = torch.linalg.svd(Hw, full_matrices=False)
    U_r = U[:, :r]
    delta_w = torch.linalg.solve_triangular(
        L.t(), U_r @ (U_r.t() @ Hw), upper=True
    )
    return delta_w


def direct_ls(Xb, R, lam):
    """全维直接最小二乘（参考上限）：min ||R − Xh ΔW||² + λ||ΔW||²。"""
    G = Xb.t() @ Xb
    G = 0.5 * (G + G.t())
    diagm = float(G.diagonal().mean().clamp_min(1e-12))
    Gλ = G + lam * diagm * torch.eye(64, dtype=dtype, device=G.device)
    return torch.linalg.solve(Gλ, Xb.t() @ R)


def obj(Xb, R, dW, lam):
    G = Xb.t() @ Xb
    diagm = float(G.diagonal().mean().clamp_min(1e-12))
    resid = R - Xb @ dW
    return float((resid.square().sum() + lam * diagm * (dW.square().sum())))


def check_basic():
    n, o = 96, 40
    Xb = torch.randn(n, 64, dtype=dtype)
    Xb[:, 1] += 0.5 * Xb[:, 0]
    W0 = torch.randn(64, o, dtype=dtype)
    R = torch.randn(n, o, dtype=dtype)  # 残差（相对 W0）
    lam, r = 0.2, 8
    dw = lowdim_whitened(Xb, R, lam, r)
    dw_full = direct_ls(Xb, R, lam)
    o_dw = obj(Xb, R, dw, lam)
    o_full = obj(Xb, R, dw_full, lam)
    o_0 = obj(Xb, R, torch.zeros(64, o, dtype=dtype), lam)
    print(f"[A] whitened ΔW shape {tuple(dw.shape)} "
          f"obj0={o_0:.4e} obj_lowdim={o_dw:.4e} obj_full={o_full:.4e}")
    assert dw.shape == (64, o)
    assert o_dw < o_0, "lowdim must reduce objective"
    print(f"    reduction vs 0: {(o_0 - o_dw)/o_0:.4f}; "
          f"gap to full: {(o_dw - o_full)/o_full:.4f}")


def check_vs_full_rank_convergence():
    """rank 增加 → 目标逼近全秩。r=64 时应 ≈ 全维 LS。"""
    n, o = 80, 20
    Xb = torch.randn(n, 64, dtype=dtype)
    R = torch.randn(n, o, dtype=dtype)
    lam = 0.2
    dw_full = direct_ls(Xb, R, lam)
    for r in (2, 8, 32, 64):
        dw = lowdim_whitened(Xb, R, lam, min(r, 64))
        err = float((dw - dw_full).abs().max())
        print(f"[B] r={r} max|ΔW − ΔW_full| = {err:.2e}")
    assert torch.allclose(
        lowdim_whitened(Xb, R, lam, 64), dw_full, atol=1e-8
    ), "r=64 must equal direct LS"
    print("    r=64 == direct LS OK")


def check_diff_from_activation_gram():
    """子空间依赖残差：构造 H 由残差主导的例子，验证基与 Xh Gram 基不同。"""
    n, o = 120, 30
    Xb = torch.randn(n, 64, dtype=dtype)
    # 残差集中在某个方向 v（Xh 能量低的方向）
    v = torch.linalg.svd(Xb, full_matrices=False).Vh[20]  # 低能量右奇异
    R = torch.randn(n, o, dtype=dtype) * 0.1
    R += (Xb @ v.unsqueeze(1)).repeat(1, o) * 2.0          # 残差投影到 v
    lam, r = 0.2, 8
    dw = lowdim_whitened(Xb, R, lam, r)
    # 旧方式：激活 Gram 能量 top-8 eigh 方向上的最小二乘
    G = Xb.t() @ Xb
    ev, evec = torch.linalg.eigh(0.5 * (G + G.t()))
    idx = torch.argsort(ev, descending=True)[:r]
    U_act = evec[:, idx]
    A = Xb @ U_act
    dA = torch.linalg.solve(
        A.t() @ A + lam * float((A.t() @ A).diagonal().mean().clamp_min(1e-12))
        * torch.eye(r, dtype=dtype, device=A.device), A.t() @ R)
    dw_old = U_act @ dA
    o_new = obj(Xb, R, dw, lam)
    o_old = obj(Xb, R, dw_old, lam)
    o_0 = obj(Xb, R, torch.zeros(64, o, dtype=dtype), lam)
    print(f"[C] obj0={o_0:.4e} act-gram-top8={o_old:.4e} "
          f"whitened-resid={o_new:.4e}  (lower better)")
    assert o_new <= o_old + 1e-9 * o_0, "whitened must match or beat act-gram"
    print(f"    whitened gains {(o_0-o_new)/o_0:.4f} vs act-gram {(o_0-o_old)/o_0:.4f}")
    return o_new < o_old


if __name__ == "__main__":
    check_basic()
    check_vs_full_rank_convergence()
    beats = check_diff_from_activation_gram()
    print("ALL PASS" if beats else "SOME FAIL")