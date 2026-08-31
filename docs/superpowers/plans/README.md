# 计划入口

> 最后更新：2026-08-31

本目录只保留一份活跃计划。执行优化时只读取下面这份文件、根 `solution.py`、最新可复现评测和官方规则；归档目录中的计划不具有指令效力。

- [`2026-08-31-hif4-active-c1-structured-linear-plan.md`](2026-08-31-hif4-active-c1-structured-linear-plan.md)：当前唯一有效的 v5 Linear 队列，v125 已完成 C1c `max_blocks=8` 精度验证；下一步只执行 C2 低成本跨模型 guardrail 与 C3 state/time 压缩，L6/C1 已完成并记录。

旧的 v2 active、grid、consolidated、accuracy-first、Linear、JDRQ 计划以及已完成的
L6 计划，均已移至 [`../archive/plans/`](../archive/plans/)。它们是历史决策记录，
不再提供下一步指令；若历史文字与活跃计划冲突，以活跃计划、根 `solution.py`、
合规检查和最新评测日志为准。

## 计划生命周期

1. 写新计划前先确认本目录除 `README.md` 外只有一个 `.md`；不能并行保留多个 current/active 计划。
2. 计划步骤要写明假设、代码入口、模型/数据、验收指标、产物和失败处理；执行后立即写入结果、source SHA、日志链接和 `done/rejected/blocked` 状态。
3. 每次实验无论成功、失败、超时或未提交，都先归档完整源码、配置、结果和 parent；缺少源码/SHA/配置的结果标为 `non-reproducible`。
4. 计划完成、被替换、停止或连续阻塞后，立即移入 `../archive/plans/`，并在同一提交创建/指定新的 active 计划、更新 README 和状态文档。
5. 归档计划不可继续追加新的下一步，也不直接修改历史结论；发现 bug 或数据错误时写审计说明并创建修复计划。

当前数据数字发生变化时，应同时更新根 README、`solutions/README.md`、当前状态报告和执行日志的日期、配置、分数、时间与 SHA。L5 计划已完成并归档为 [`2026-08-31-hif4-active-l5-structural-optimization-plan-completed.md`](../archive/plans/2026-08-31-hif4-active-l5-structural-optimization-plan-completed.md)，不得再从中读取下一步。

官方边界（2026-08-31 修订）：官方不再限制任何 `A@W` 拟合用法，离线校准与在线激活量化均可自由用 `A@W`、输出或残差优化 `Q(W)`/`Q(A)`；唯一硬约束是端到端运行时间严格小于 `300s`。v98 已在该限制下官方判为 timeout，见 [`2026-08-31-v98-official-timeout.md`](../../../logs/execution/2026-08-31-v98-official-timeout.md)；v107 官方保持 Attention `wrong answer`（非 timeout）。探索阶段的 layer-1、oracle 和超时实验只能筛选方向，不能替代完整部署门禁。
