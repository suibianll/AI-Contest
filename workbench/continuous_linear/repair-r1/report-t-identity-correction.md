# 证据修正：probe_l2_direction 的 T=I 前向与父不一致 → 3/3 退化结论作废

> 日期：2026-09-07。侧：Linear。run_id：`repair-r1/t-identity-parity`。

## 1. 发现问题

工作包 L-R2 第 5 点要求："T=I 应恢复父路径；不能恢复时先查 rank/状态依赖，
不能把父和新目标之间的结构差异归为优化收益。"

`probe_t_identity_parity.py` 核对发现：`probe_l2_direction` 的
`deployment_forward`（T=I 时）输出 **不等于** 父真实 API 输出：

| state | parent_mse（真实 API） | T=I simplified_mse | ratio |
|---|---|---|---|
| L0-o w2 | 2.081e-07 | 1.666e-07 | 0.800 |
| L0-o w7 | 2.153e-07 | 1.691e-07 | 0.785 |
| L11-proj w1 | 1.832e-04 | 8.328e-05 | 0.455 |
| L11-proj w6 | 1.884e-04 | 8.446e-05 | 0.448 |
| L0-fc_up w2 | 9.443e-04 | 7.361e-04 | 0.780 |
| L0-fc_up w7 | 9.097e-04 | 7.121e-04 | 0.783 |

简化前向缺失：rank-2 残差 gram 修正（residual_u/v）、static-actorder hdiag
块序（`gptq_block_order` 重排）、importance 的规范化（`h_x_smooth` /
data-driven 语义）、以及父校准的完整候选项选择。因此简化前向的 MSE 结构
与父不同（甚至更低），`deployment_forward` 不是部署一致的 T=I 基线。

## 2. 结论修正

- `probe_l2_direction` 报告的"3/3 代表 state 退化（o +4.2% / proj +5.5% /
  fc_up +2.4%）"**无效**：退化量是相对一个与父不一致的简化前向，不能
  归因于 FlatQuant T 在真实部署上的效果。
- FlatQuant 状态回退为 **DIRECTION_UNKNOWN**：
  - 正确性方法（矩阵指数梯度/STE/inference_mode/T=I 恒等）PASS；
  - 完整硬前向成本 COST_HOLD（0.62-3.24s/step，32 步 × 168 state 不可承受）；
  - 真实部署方向证据不足（无可信 T=I 基线的 32 步训练结果）。
- 此前"3/3 退化 + COST_HOLD → 不注册候选"应修正为：
  **COST_HOLD，方向未判定；FlatQuant 不因该探针关闭**。

## 3. 若要判定 FlatQuant 方向（供下一张卡参考）

需要"插 T 的完整校准变体"：在父校准流内（weight_smooth 后 / rank 后）插入
T，使 T=I 时逐位等于父（含 rank-2 gram、static-actorder、importance 规范、
data-driven ratio），再跑 32 步。每步成本 ≈ 一次完整校准（~2s/state），
仅代表 state 可做方向诊断（3×32×2s ≈ 200s），全量仍 COST_HOLD。
该实现属于新的候选代码（不能改父 solution.py），需另立机制卡。

## 4. 产物

- `workbench/continuous_linear/repair-r1/probe_t_identity_parity.py`
- 本修正同时更新 `report-l-r2.md` §7 结论。