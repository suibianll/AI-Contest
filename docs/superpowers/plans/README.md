# 计划入口

本目录只保留一份活跃计划：

- [`2026-08-30-hif4-active-optimization-plan.md`](2026-08-30-hif4-active-optimization-plan.md)：当前唯一有效的优化队列，按“实现审计 → 修复验证 → 全层门禁 → 时间压缩”的顺序推进。

旧的 grid、consolidated、accuracy-first、Linear、JDRQ 计划以及归档流程，均已移至 [`../archive/plans/`](../archive/plans/)。它们是历史决策记录，不再提供下一步指令；若历史文字与活跃计划冲突，以活跃计划、根 `solution.py`、合规检查和最新评测日志为准。

官方边界保持不变：离线校准可以用 `A@W` 优化离线 `Q(W)`，但不得用输出或残差选择在线 `Q(A)`；最终官方时间严格小于 420 秒。探索阶段的 layer-1、oracle 和超时实验只能筛选方向，不能替代完整部署门禁。
