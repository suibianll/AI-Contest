# R1 执行记录：L23b fit_gain 达标 + Linear 误差账本 v0（2026-09-08）

执行者：Linear 侧执行代理。依据 [持续研究循环](../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md)。

## P0：L23b fit 表（fit_gain 目标达成）

脚本 `workbench/continuous_linear/l23-residual-subspace/fit_gain_table.py`（CPU，两遍复用同 SHA 校准产物，
无新前向、无 API 重跑）。每 weight state × 每校准 fold（窗口 0/1，与拟合同一数据）：
`gain = 1 − MSE_PLAYER/MSE_STD`，分母为同 NVFP4 解码输入的标准 HiF4 完整输出 vs dense `X@Wᵀ`；
player 用真实 `hif4_dynamic_quantize_activation` + 最终五字段解码（与 evaluator `_score` 一致）。

| 指标 | 值 |
|---|---|
| **Linear fit_gain（候选 L23b 13639FB2）** | **0.9486 ≥ 0.9 ✅** |
| Linear fit_gain（父 L4 ACB16F76） | 0.7219 |
| state 覆盖 | 168/168（24 层 × 7 role） |
| fold-rows | 336（2 folds/state，fold 0 = 10 行、fold 1 = 128 行） |
| 候选 vs STD | 336W / 0L / 0T |
| 候选 vs 父 | 336W / 0L |
| activation_state 候选==父 | True（control） |
| 逐 role 最低 | fc_gate 0.9179、fc_up 0.9005（其余 role 均 > 0.94） |

产物：`workbench/continuous_linear/l23-residual-subspace/fit_gain_table.{json,md}`。
注意：fit_gain 是**校准数据上的拟合增益**（研究目标）；336 例独立窗口面板 gain 0.3395（父 0.5243）仍
record-only，两者口径不同（前者同校准输入、后者独立窗口）。

## R1：Linear 误差账本 v0（shard0 28 states，最小运行）

脚本 `workbench/continuous_linear/error_ledger.py`。E1/E2 在拟合折叠 [0,1] 上计算（分母同 MSE_STD）；
E3 取 4B 面板同 state 的 `1 − mean gain`；E4 = E3 − E1 − E2。

| 格 | 均值 | 读法 |
|---|---|---|
| E1 连续低维拟合残差 | **0.0074** | 连续求解能力接近完美（基/秩/目标选对） |
| E2 合法量化投影损失 | **0.0320** | 编码格点损失小 |
| E3 最终部署误差 | **0.6537** | 面板 1−gain（candidate） |
| **E4 部署失配残差** | **0.6143** | **主导：占 E3 的 94%** |

结论：校准↔部署分布失配（独立窗口 dynamic activation scale 分布 ≠ 校准折叠）是最大可改变误差源，
与计划 §7 先验一致 → **下一张卡 L24-C（按动态激活 scale 分桶加权拟合，靶点 E4）**。

产物：`artifacts/continuous/linear/error_ledger_linear_2026-09-08.{json,md}`。

## 交付状态

- L23b 归档：`solutions/continuous_linear_l23b-residual-subspace/`（SHA `13639FB2…10FE0`）
  → **READY_FOR_OFFICIAL**（本环境无官方上传入口；交付包路径 + SHA；官方 300s 仍为唯一时间裁决）。
- fit_gain 0.9 研究目标已达成（S1 的 Linear 部分），但 S1 还需官方 <300s 与 Attention 侧 0.9。
- 下一轮：注册 L24-C 卡（8 字段）→ 实现 → 数学/可达 → 4B shard0 paired → 归档 commit/push。