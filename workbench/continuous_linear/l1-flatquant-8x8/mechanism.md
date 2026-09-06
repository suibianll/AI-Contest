# L1 机制卡：HiF4 层级结构化可逆变换（FlatQuant 8×8 T1⊗T2）

> run_id：`l1-flatquant-8x8`（侧：Linear）
> 日期：2026-09-07
> 契约：[持续优化总计划](../docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
> + [L 工作包](docs/superpowers/plans/workpackages/continuous-linear.md) L1。
> 直接父：L4 侧包 `ACB16F764DB80EDA94EB77FE497A7965C716B527571C241C79B2319529FF5263`
> （官方 4607/247s，v189 Linear 侧逐位复现）。

## 1. 机制登记（固定原型，不扫邻域）

- 依据：FlatQuant（arXiv 2410.09426）——每线性层学习仿射变换 P 使
  weight/activation 分布更平坦；推理期用 Kronecker 分解 P=P1⊗P2 控制在线开销。
- 本侧固定原型（工作包 L1）：
  - 每个 weight state 学习两个 8×8 可逆因子 T1、T2，构造 64 通道 T=T1⊗T2，
    该 state 全部 64 块共享同一 T（无 per-block/role 手工路由）。
  - 因子为矩阵指数 `Ti=exp(Bi)`，Bi 对称零迹、特征值投影至
    `[-log(2)/4, +log(2)/4]`，保证 cond(T) ≤ 2；初始 T=I。
  - 部署：最终连续坐标每 64 通道 X 右乘 T、W 右乘 T^-T；逆只在校准计算。
  - 训练：32 步 Adam，lr=0.01，梯度范数 1，正则 1e-3·(mean(B1²)+mean(B2²))，
    固定配置不扫描；只用官方校准 fold、每 fold ≤128 行；独立 holdout 不学参数。
  - 损失：真实输出误差 `||Q(XT)Q(WT^{-T})^T − XW^T||²` / 同 fold 标准输出 MSE。

## 2. 成本探针（完整硬前向每步重建）

在真实 Qwen 权重/激活 + 本机 CUDA 上实测单 state 单次完整硬前向的代表成本
（`probe_l1_cost.py`，合法五字段编码器）：

| state | weight GPTQ 单步 | activation GPTQ 单步 | 每步合计 |
|---|---|---|---|
| 窄层 768×768（q/k/v/o） | 0.43s | 0.16s | ~0.6s |
| 宽入 4864×896（proj） | 0.84s | 0.68s | ~1.5s |
| 宽出 768×4864（fc） | 1.52s | 0.18s | ~1.7s |

- 32 步 × ~1.2s ≈ 38s/state；168 weight states ≈ **6451s 本地**。
- 时间模型 `T ≈ 170.3 + 0.115·W_calib + ...`：+6451s 校准 →
  官方预测 +742s，远超 `<280s` 提交门与 `300s` 硬限。
- **结论：完整硬前向每步重建不可承受**（按工作包 L1 记 COST 分支）。

## 3. 方向探针（可承受代理，部署一致快速编码）

因完整硬前向不可承受，用部署同坐标、同五字段编码器的**快速合法编码**
（`_dense_to_hif4`，无 GPTQ 补偿/h_inv 重建）作为 STE 替代模型做方向验证。
不把该代理当晋级证据，只回答"是否值得为完整路径投入"。

方法（`sweep_l1_direction.py` / `probe_l1_direction_real.py`）：
- 真实 Qwen proxy-v2 cache 权重 + 校准激活（NVFP4 真实输入）；
- 走 L4 完整校准取真实 smooth/perm/hadamard/rank 部署坐标；
- 插入 T=T1⊗T2，32 步 STE Adam 训练（与工作包固定配置一致）；
- 损失 = 真实量化输出误差（快速编码）/ 标准输出 MSE。

结果（35 个真实 state = 5 层 × 7 role）：

| 结论 | 计数 |
|---|---|
| 改善（rel Δ < −1%） | **1** |
| 退化（rel Δ > +1%） | **34** |
| 持平 | 0 |

代表性：L0-o `0.004234→0.003693`（−12.8%，唯一改善）；L11-proj
`0.011520→0.014481`（+25.7%）；L17-o `+28.8%`；L23-proj `+34.9%`。
退化集中在 proj/fc（宽层）与深层（L17/L23）；无一致方向。

补充（`probe_l1_direction.py`，随机 NVFP4 输入）：窄层 `0.011018→0.011224`、
宽入 `0.011023→0.011140`，同样无一致改善。

## 4. 裁决：REJECTED（LOCAL_NEGATIVE）＋ COST/DESIGN_HOLD

- 完整硬前向每步重建：**不可承受**（§2，6451s 本地 → 官方 +742s）。
- 可承受代理方向：**35 个真实 state 中 34 个退化**，机制在 L4 已优化的
  smooth/perm/hadamard/rank/GPTQ 坐标上无可迁移的本地余量；快速代理虽非
  部署一致，但无任何一致正向信号可支撑投入完整路径。
- 不进入六 shard 完整评测，不创建候选 solution.py，不注册 official 探索。
- 不重试 offset/seed/step/正则邻域；该卡后续不再以"再扫 8×8 因子参数"重开。

## 5. 证据索引

- `workbench/continuous_linear/probe_l1_cost.py`（成本）
- `workbench/continuous_linear/probe_l1_direction.py`（随机输入方向）
- `workbench/continuous_linear/probe_l1_direction_real.py`（真实单 state 方向）
- `workbench/continuous_linear/sweep_l1_direction.py`（35 state 扫描）
- 本机输出/时间戳见 `logs/execution/continuous-linear-l1-flatquant-8x8.md`

## 6. 下一步（queue 补充）

L1 关闭后按总计划 §3.7 继续：L2（GPTAQ 非对称输出残差补偿）执行前必须先
完成 JDRQ 去重（JDRQ 目标 = min||Xh Wh^T−Y||²，冻结激活 + 合法编码候选，
已在 v189 上 J1 关闭）。若确认同目标同更新则标 DUPLICATE_CLOSED，跳过并转
L3/新假设。