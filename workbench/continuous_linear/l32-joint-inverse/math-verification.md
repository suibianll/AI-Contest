# L32 数学验证记录（2026-09-08）

依据 [持续研究循环](../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md) §7 L-R4。

## 验证内容

`R = I + U Vᵀ`（rank-8 每块），Woodbury 逆 `R⁻¹ = I - U(I_r + VᵀU)⁻¹Vᵀ`，
部署 `A' = AR`、`W' = W R^{-T}`，目标 `||A Wᵀ − Q(AR)Q(WR^{-T})ᵀ||²`。

| 检查 | 结果 |
|---|---|
| (1) Woodbury 逆 `R R⁻¹ ≈ I` | fp32 max err 6.9e-4 |
| (2) 乘积不变性 `(AR)(W R^{-T})ᵀ ≈ A Wᵀ` | fp32 max err 1.4e-1（相对 6.8e-5）；**fp64 3.4e-10** |
| (3) 低维方向良态性 | `(Aᵀ dY)(Aᵀ dY)ᵀ` top-8 谱隙 eigh 可解，U 正交 |

## 结论

- 数学上**可行**：fp64 下乘积精确（3.4e-10），fp32 相对误差 6.8e-5 在量化噪声（~1e-2）量级内可接受。
- 但 L32 需**修改动态激活路径**（新增 `A' = A + (AU)Vᵀ` 两次 O(Ndr) 低秩乘法）和**权重路径**（Woodbury 逆）。
- **时间风险高**：L30（联合坐标/复杂机制）已超时（api 322.9s），L32 更复杂。
- 用户核心诉求为**时间安全**；L28（286s）无法组合，L4（247s）时间安全。

## 处置

L32 机制卡已登记（`workbench/continuous_linear/l32-joint-inverse/mechanism-card.md`），
数学验证通过，但**暂缓实现**——在确认组合时间预算与 L32 动态 API 时间安全前不启动，
避免重蹈 L30 超时。组合 Linear 侧当前用 L4（时间安全）。