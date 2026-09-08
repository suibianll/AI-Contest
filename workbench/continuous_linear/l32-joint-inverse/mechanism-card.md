# L32 机制卡：真正联合的 A@W 低维互逆拟合（收益候选三）

依据 [持续研究循环](../docs/superpowers/plans/workpackages/2026-09-08-continuous-research-loop.md) §7 L-R4。

## 1. ID / 侧 / 父 / 创建轮次

- ID: L32
- 侧: Linear
- 父: **L4**（`solutions/v162_linear_l4-v189-linear-exact_officialNA_timeNA/solution.py`，ACB16F76，4607/247s）
  时间父（组合需 Linear 时间安全）
- 创建: R5（2026-09-08）

## 2. 靶点

- 账本格: E3_fit（最终合法部署 fit_gain）
- 现状: L28（只改写合法权重，激活路径冻结）fit_gain 0.9486 但官方 286s 无法组合；
  L30 全局 reduced-rank REJECTED（fit_gain 0.707，api 322s 超时）
- L32 进入**真正联合坐标**：同时学 A/W 的低维互逆残余，目标直接为最终合法输出

## 3. 改变什么（调用图差异）

L23b/L28 只改权重，激活路径冻结。L32 在每个固定 64 维块学习 rank-8、零初始化的
可逆残余 `R = I + U Vᵀ`：

```text
A' = A R          (部署时 Woodbury: A + (A U) Vᵀ)
W' = W R^{-T}     (Woodbury 小逆: R^{-1} = I - U (I + VᵀU)^{-1} Vᵀ)
浮点乘积 A Wᵀ = A' W'ᵀ 严格不变
目标: ||A Wᵀ − Q(A R) Q(W R^{-T})ᵀ||²  (全校准最终合法输出)
```

调用图：L4 校准末尾新增一次 Gauss-Newton 方向 + 一次合法硬编码投影。
动态 activation 部署时新增两次 `O(Ndr)` 低秩乘法（替换而非叠加）。

## 4. 为什么可能有效

L28 只优化权重侧，activation 的量化误差未补偿。L32 通过可逆残余 R 同时微调
A 和 W（乘积不变），让 activation 量化在部署坐标下更有利，直接最小化最终
`Q(AR)Q(WR^{-T})ᵀ` 与浮点 `AWᵀ` 之差。这是用户确认的"真正逐列非对称量化"/
"Q/K 互逆 scale 学习"在 Linear 侧的对应（A@W 联合坐标）。

## 5. 固定配置

- 每 64 块 rank-8、零初始化 R=I，不扫 rank
- 一轮 Gauss-Newton 方向 + 一次合法硬编码投影
- 全校准行共同求解，不拆 fit/select，不使用独立窗口否决
- 固定正则、块序、五字段 scale；动态新增两次低秩乘法替换等价旧变换
- 冻结: 动态 activation 原路径、Attention control、V/Linear 非目标侧

## 6. 证伪判据

- R⁻¹、最终五字段输出、动态 activation 不可达 → 关闭
- 与父 `residual_u/v`、rank 补偿、旧联合坐标/JDRQ、Householder 数学等价 → 取消注册
- fit_gain 未保持 ≥0.9 或官方不正 → 关闭该联合残余实现

## 7. 去重声明

| 维度 | L32 | 已关闭/已有 | 结论 |
|---|---|---|---|
| 目标 | 每块 rank-8 最终合法输出 | L4 rank-2 连续域 | 不同 |
| 变量 | 每块可逆残余 R=I+UVᵀ，A/W 联合 | L4 全局 rank-2 权重残余 | 不同（块级+联合） |
| 插入点 | 校准末 Gauss-Newton + 硬投影 | L28 逐块权重投影 | 不同 |
| 编码 | Q(AR)Q(WR^{-T})ᵀ 联合 | 旧联合坐标/JDRQ | 需数学验证不等价 |

## 8. 关闭粒度

失败只关闭"每块 rank-8 联合互逆残余这一实现"；联合坐标目标仍 OPEN。
