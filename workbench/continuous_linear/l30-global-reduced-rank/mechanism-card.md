# L30 机制卡：全局 reduced-rank 输出残差拟合（收益候选一）

依据 [持续研究循环](../docs/superpowers/plans/workpackages/2026-09-08-continuous-research-loop.md) §7 L-R2。

## 1. ID / 侧 / 父 / 创建轮次

- ID: L30
- 侧: Linear
- 父: **L4**（`solutions/v162_linear_l4-v189-linear-exact_officialNA_timeNA/solution.py`，ACB16F76，4607/247s）
  为时间父（L28 286s 无组合余量，L30 从 L4 构建以保时间安全）
- 创建: R4（2026-09-08）

## 2. 靶点

- 账本格: E1（连续低维拟合残差）+ E3_fit（最终合法部署 fit_gain）
- 现状: L28（逐块 rank-8）fit_gain 0.9453（shard0），官方 +4 vs L4 但 +39s
- 占比: E1 极小（0.0074），但 L28 的 rank-8 是**每块局部**，拼接后整体可高秩——
  不是真正全局低维；L30 用一次全局 rank-8 更新可能同精度但更快/更简单

## 3. 改变什么（调用图差异）

L28: 逐块（每 64 列块）rank-8 白化残差交叉求解 + 逐块接受（顺序增量）。
L30: **一次全局 reduced-rank regression**：

```text
R0 = Y − A·B_parent        (A 为全校准加权量化激活 [N,d], B_parent 为父部署权重转置 [d,o])
min_{rank(ΔB)≤8} ||R0 − A·ΔB||² + λ||ΔB||²
```

在校准行空间（A 的列空间）求解 reduced-rank regression，避免构造完整 d×d 逆：
用 `A = UΣVᵀ` 的 thin SVD（A 是 [N,d]，N=138 → 廉价）或 AᵀA 的 top-8 特征分解。
低维基来自真实量化激活与输出残差，不是原权重 SVD 或激活能量基。
只做**一次全局合法投影** + **一次全校准输出接受判定**。
调用图：L28 的 144 块循环 → L30 的一次性 SVD + 一次投影 + 一次接受。

## 4. 为什么可能有效

L28 逐块 rank-8 拼接后整体秩可持续增长（144 块 × rank-8 = 最高 1152 秩），
但每块只优化局部输出坐标，块间不协调。L30 用一次全局 rank-8 更新：
- 低维基捕捉全校准行的主导残差方向（SVD 的 top-8 左奇异向量）
- 一次投影到合法五字段（格点一致性比逐块投影更协调）
- 可能达到与 L28 相当或更好的 fit_gain，且调用图大幅简化（时间更安全）

## 5. 固定配置

- rank=8（固定，不扫）
- 正则 λ = 0.2/0.3（继承 L28 宽/窄层规则，不扫）
- 全部校准行（fold0+fold1，N=138），不缩样本
- 一次全局合法投影（round/clamp 五字段，scale/lv2/lv3 固定父值）
- 一次全校准输出接受判定（fit_gain 不下降才接受）
- 冻结自由度: 块序、五字段 scale、动态 activation 路径、Attention control

## 6. 证伪判据

- 连续解相对 L4 无材料收益（E1 不降）→ 关闭
- 合法投影抹掉收益（fit_gain 不显著优于 L4 的 0.72）→ 关闭
- 与旧 CAT/GPTAQ reduced-rank 数学等价 → 取消注册
- fit_gain < L28 的 0.9453 且更慢 → 关闭（L30 目标是等价精度 + 更简单）

## 7. 去重声明（四项比对）

| 维度 | L30 | 已关闭族 | 结论 |
|---|---|---|---|
| 目标 | 全局 reduced-rank 输出残差拟合 | L4 Kronecker CAT（v174）、权重 SVD 低秩、L21 逐列/块一次 | **不同**（L30 在 A 列空间 reduced-rank regression，非 Kronecker/权重 SVD） |
| 变量 | rank-8 ΔB 全局 | rank-3/残差系数/fold 扩展（v182 后） | **不同**（非残差系数扩展，是全新全局 reduced-rank 构造） |
| 插入点 | 校准末一次性 global 更新 | L28 逐块循环、L23 逐块残差交叉 | **不同**（一次性 vs 逐块） |
| 编码 | 一次全局合法投影 | L28 逐块投影、旧 4×4 Gram rounding | **不同**（全局投影 vs 逐块） |

不与已关闭族等价 → 注册。

## 8. 关闭粒度

失败只关闭"全局 reduced-rank 输出残差拟合这一实现"，不扩写为 reduced-rank 族；
L30 目标（全局低维）仍 OPEN 于其他构造方式。