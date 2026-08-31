# 计划入口

> 最后更新：2026-08-31

本目录只保留一份活跃计划。执行优化时只读取下面这份文件、根 `solution.py`、最新可复现评测和官方规则；归档目录中的计划不具有指令效力。

- [`2026-08-31-hif4-active-c1-structured-linear-plan.md`](2026-08-31-hif4-active-c1-structured-linear-plan.md)：当前唯一有效的 v5 Linear 队列，基于 v124 C1c rank-8 parent 继续验证 max-blocks=8、跨模型泛化与 C1 压缩；L6、C1a、C1b 已完成并归档。

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

官方边界保持不变：离线校准可以用 `A@W` 优化离线 `Q(W)`，但不得用输出或残差选择在线 `Q(A)`；最终官方时间严格小于 420 秒。探索阶段的 layer-1、oracle 和超时实验只能筛选方向，不能替代完整部署门禁。
