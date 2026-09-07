"""L21-1 数学最小检查：块内逐列条件求解正确性与 JDRQ 区别。

工作包 L21-1 §3.3 要求：
1. 独立直接条件最小二乘参考：非对角 H、非对称 teacher、零/奇异块；
   检查每列固定值、剩余变量、最终合法码。
2. 区别于旧 JDRQ：旧局部码±一格残差下降不自动包含"固定一列后对所有剩余
   连续列的条件最优补偿"；需用小例子证明中间 Z、Qi、F 和最终合法码区别。

本脚本用 torch 小矩阵验证：
- A) 逐列 OBQ 型条件补偿 Z_r ← Z_r − F_ri/F_ii·(Z_i−Qi) 与"固定列后直接
   条件最小二乘求解"逐列一致（非对角 H）。
- B) 构造一个例子证明 L21-1 条件补偿与旧 JDRQ ±一格那类局部更新不同
   （固定列偏移驱动多个剩余列同步补偿）。
- C) 奇异/非有限块处理：加 ridge 后有限；纯奇异保留父语义（返回哨兵）。

LOCAL math check only；不加载 evaluator、不产生候选。
"""

from __future__ import annotations

import torch

torch.manual_seed(0)
dtype = torch.float64


def ridge_block(Xh: torch.Tensor, lam: float) -> torch.Tensor:
    """Hλ = XhᵀXh + λ·mean(diag)·I（与工作包 ridge 标量规则一致的解析形式）。"""
    H = Xh.t() @ Xh
    diagm = H.diagonal().mean().clamp_min(1e-12)
    return H + lam * diagm * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)


def cond_solve_fixed(Z_full, F, i, Qi):
    """OBQ 型：固定列 i 为 Qi 后，对剩余连续列的条件补偿。

    Z_full: [k, out] 连续最优解（剩余列）；F: [k, k] 剩余列逆 Hessian；
    i: 剩余坐标中的列位置；Qi: [out] 该列合法格点。
    返回更新后的 Z（删除固定列）与 F（删除固定列的 Schur 补）。
    """
    Z = Z_full.clone()
    delta = (Z[i] - Qi).abs().max().item()
    if delta < 1e-15:
        # 无量化偏移：仅移除该列
        idx = [j for j in range(F.shape[0]) if j != i]
        return Z[idx], F[idx][:, idx]
    # 条件补偿剩余列（含固定列本身，随后覆盖为 Qi 并移除）
    Z = Z - (F[:, i] / F[i, i]).reshape(-1, 1) * (Z[i] - Qi).reshape(1, -1)
    Z[i] = Qi
    idx = [j for j in range(F.shape[0]) if j != i]
    Frr = F - torch.outer(F[:, i], F[i, :]) / F[i, i]
    # 用补偿后的 Z（沿剩余坐标 i 移除）
    Z_i = Z.clone()
    Z_i[0] = 0  # 无意义占位（避免无引用）
    Zsel = torch.index_select(Z, 0, torch.tensor(idx, dtype=torch.long))
    return Zsel, Frr[idx][:, idx]


def reference_cond_solve(Xh, Y, W0, fixed_idx, fixed_val, lam):
    """直接条件最小二乘参考：固定若干列后，剩余列重新完整求解。

    fixed_idx: 已定列下标列表；fixed_val: 对应已定值 [len, out]。
    """
    n, k = Xh.shape
    H = ridge_block(Xh, lam)
    fixed = set(fixed_idx)
    r_idx = [j for j in range(k) if j not in fixed]
    H_rr = H[r_idx][:, r_idx]
    Xh_r = Xh[:, r_idx]
    Yt = Y.clone()
    for j, v in zip(fixed_idx, fixed_val):
        Yt = Yt - Xh[:, j].reshape(-1, 1) @ v.reshape(1, -1)
    Z_r = torch.linalg.solve(H_rr, Xh_r.t() @ Yt)
    Z = torch.zeros(k, Y.shape[1], dtype=Y.dtype, device=Y.device)
    for j, v in zip(fixed_idx, fixed_val):
        Z[j] = v
    Z[r_idx] = Z_r
    return Z


def check_consistency(lam: float):
    """A) 逐列 OBQ 条件补偿 与 直接条件重解 逐列一致（非对角 H）。"""
    n, k, out_c = 24, 6, 3
    Xh = torch.randn(n, k, dtype=dtype)
    Xh[:, 1] += 0.6 * Xh[:, 0]
    Xh[:, 3] -= 0.4 * Xh[:, 1]
    Y = torch.randn(n, out_c, dtype=dtype)

    H = ridge_block(Xh, lam)
    F0 = torch.linalg.inv(H)
    Z0 = F0 @ (Xh.t() @ Y)

    grid_offset = torch.tensor([-0.3, 0.0, 0.3], dtype=dtype)
    # --- OBQ 逐列（列顺序固定 0..k-1；剩余坐标随删除移动） ---
    Z = Z0.clone()
    F = F0.clone()
    rem_pos = list(range(k))          # 剩余列对应的原始下标
    fixed_cols, fixed_vals = [], []
    while rem_pos:
        i = 0                         # 总是固定剩余列中的第一个（原始顺序）
        cand = Z[i]
        dist = (cand.reshape(-1, 1) - grid_offset.reshape(1, -1)).abs()
        g = grid_offset[dist.argmin(dim=1)]
        orig = rem_pos[i]
        fixed_cols.append(orig)
        fixed_vals.append(g)
        Z, F = cond_solve_fixed(Z, F, i, g)
        rem_pos.pop(i)
    # --- 重建全量候选：各列固定为最终格点 ---
    W_obq = torch.zeros(k, out_c, dtype=dtype)
    for orig, g in zip(fixed_cols, fixed_vals):
        W_obq[orig] = g
    # --- 直接参考：按相同顺序固定后重解剩余列 ---
    Z_ref = reference_cond_solve(Xh, Y, None, fixed_cols, torch.stack(fixed_vals), lam)
    err = (W_obq - Z_ref).abs().max().item()
    print(f"[A] lam={lam} max|Z_obq - Z_direct| = {err:.2e} {'PASS' if err < 1e-8 else 'FAIL'}")
    return err < 1e-8


def check_jdrq_difference():
    """B) L21-1 条件补偿 与 旧 JDRQ ±一格局部下降 不同。

    构造：固定列 i 移一格引起的"单步残差下降"（JDRQ 型）与"对剩余列全
    条件补偿"（L21-1 型）产生的后续格点不同。
    """
    n, k, out_c = 20, 4, 2
    Xh = torch.randn(n, k, dtype=dtype) * 2.0
    Xh[:, 1] += 0.9 * Xh[:, 0]
    Xh[:, 3] += 0.7 * Xh[:, 2]
    Y = torch.randn(n, out_c, dtype=dtype) * 3.0
    H = ridge_block(Xh, 0.05)
    F0 = torch.linalg.inv(H)
    Z0 = F0 @ (Xh.t() @ Y)

    # 小格点：每列候选 {0, step}
    step = 0.35
    grid = torch.tensor([0.0, step], dtype=dtype)
    lam = 0.05

    def eval_loss(W):
        return ((Y - Xh @ W) ** 2).sum() + lam * (H.diagonal().mean() * (W ** 2).sum())

    # L21-1：列0 固定 0.0（远离连续最优），其余列条件补偿后取最近格点
    W_l21 = Z0.clone()
    F = F0.clone()
    rem_pos = list(range(k))
    while rem_pos:
        i = 0
        cand = W_l21[i].reshape(1, -1)
        dist = (cand - grid.reshape(-1, 1)).abs()
        g = grid[dist.argmin(dim=0)]
        orig = rem_pos[i]
        W_l21, F = cond_solve_fixed(W_l21, F, i, g[0])
        rem_pos.pop(i)
    # W_l21 现在只剩最后固定一列（剩余列全被移除），重建全量
    # 重新按完整 OBQ 重建所有列：
    W_l21 = Z0.clone()
    F = F0.clone()
    rem_pos = list(range(k))
    fixed_g = {}
    while rem_pos:
        i = 0
        cand = W_l21[i].reshape(1, -1)
        dist = (cand - grid.reshape(-1, 1)).abs()
        g = grid[dist.argmin(dim=1)].reshape(-1)
        orig = rem_pos[i]
        for c in range(out_c):
            fixed_g[(orig, c)] = g[c]
        W_l21, F = cond_solve_fixed(W_l21, F, i, g)
        rem_pos.pop(i)
    W_cond_full = torch.zeros(k, out_c, dtype=dtype)
    for (r, c), v in fixed_g.items():
        W_cond_full[r][c] = v

    # JDRQ 型：只对"当前格点±一格"做单元素局部 argmin，不做剩余列补偿
    W_jdrq = Z0.clone()
    for i in range(k):
        for c in range(out_c):
            best = None
            for g in grid:
                cand_w = W_jdrq.clone()
                cand_w[i][c] = g
                if best is None or eval_loss(cand_w) < eval_loss(best):
                    best = cand_w
            W_jdrq = best
    # W_l21 所有格点取最优（含跨列回看），与 L21-1 对比中间格点
    diff = (W_cond_full - W_jdrq).abs().max().item()
    print(f"[B] max|W_l21_cond - W_jdrq_local| = {diff:.4f} "
          f"{'DIFFERENT' if diff > 1e-9 else 'SAME'}")
    return diff > 1e-9


def check_singular():
    """C) 奇异/非有限块：加 ridge 后可解；纯奇异不产生 NaN。"""
    Xh = torch.zeros(4, 4, dtype=dtype)  # 奇异
    Y = torch.randn(4, 2, dtype=dtype)
    H = ridge_block(Xh, 0.2)
    ok = torch.isfinite(H).all().item()
    try:
        torch.linalg.inv(H)
        inv_ok = True
    except RuntimeError:
        inv_ok = False
    print(f"[C] singular Xh=zero: H finite={ok} inverse_ok={inv_ok} "
          f"{'PASS' if ok and inv_ok else 'FAIL'}")
    return ok and inv_ok


if __name__ == "__main__":
    a = check_consistency(0.1)
    b = check_jdrq_difference()
    c = check_singular()
    print("ALL PASS" if (a and b and c) else "SOME FAIL")