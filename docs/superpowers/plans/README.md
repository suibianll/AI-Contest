# 计划入口

> 最后更新：2026-09-02

本目录只保留一份活跃计划。执行优化时只读取下面这份文件、根 `solution.py`、最新
`proxy-v2` 评测和官方规则；归档目录中的计划不具有指令效力。本地主评测只调用
`evaluator/official_eval.py`，跨模型泛化调用 `evaluator/cross_model_eval.py`；旧
`real_model_suite.py` 已退役。

- [`2026-09-02-linear-attention-dual-track-generalization-plan.md`](2026-09-02-linear-attention-dual-track-generalization-plan.md)：当前唯一有效计划。Linear 与 Attention 分支隔离优化；每个方向通过 Qwen compact/default 后，必须再通过其他模型真实前向父子配对，防止继续对 Qwen 过拟合。17816 源码不可得，不再作为等待项。

快速机制迭代使用 `--compact-panel`：Linear 为 28 个 selected Weight state + 56 个跨
validation/test holdout case，读取 median、尾部分布、负 case、cross-holdout 一致性和 W/A
interaction；不再用 mean 单独晋级。完整 default panel 仅作单侧低频审计。

旧的 v159 GPU 单线、hierarchy/encoder、17816-anchor、v2 active、grid、consolidated、
accuracy-first、Linear、JDRQ 计划以及已完成的 L6 计划，均已移至
[`../archive/plans/`](../archive/plans/)。它们是历史决策记录，
不再提供下一步指令；若历史文字与活跃计划冲突，以活跃计划、根 `solution.py`、
合规检查和最新评测日志为准。

## 计划生命周期

1. 写新计划前先确认本目录除 `README.md` 外只有一个 `.md`；不能并行保留多个 current/active 计划。
2. 计划步骤要写明假设、代码入口、模型/数据、验收指标、产物和失败处理；执行后立即写入结果、source SHA、日志链接和 `done/rejected/blocked` 状态。
3. 每次实验无论成功、失败、超时或未提交，都先归档完整源码、配置、结果和 parent；缺少源码/SHA/配置的结果标为 `non-reproducible`。
4. 计划完成、被替换、停止或连续阻塞后，立即移入 `../archive/plans/`，并在同一提交创建/指定新的 active 计划、更新 README 和状态文档。
5. 归档计划不可继续追加新的下一步，也不直接修改历史结论；发现 bug 或数据错误时写审计说明并创建修复计划。

当前数据数字发生变化时，应同时更新根 README、`solutions/README.md`、当前状态报告和执行日志的日期、配置、分数、时间与 SHA。此前 C1 structured Linear 计划已归档为 [`2026-08-31-hif4-active-c1-structured-linear-plan-superseded.md`](../archive/plans/2026-08-31-hif4-active-c1-structured-linear-plan-superseded.md)，不得再从中读取下一步。

官方边界（2026-08-31 修订）：官方不再限制任何 `A@W` 拟合用法，离线校准与在线激活量化均可自由用 `A@W`、输出或残差优化 `Q(W)`/`Q(A)`；唯一硬约束是端到端运行时间严格小于 `300s`。v98 已在该限制下官方判为 timeout，见 [`2026-08-31-v98-official-timeout.md`](../../../logs/execution/2026-08-31-v98-official-timeout.md)；v107 官方保持 Attention `wrong answer`（非 timeout）。探索阶段的 layer-1、oracle 和超时实验只能筛选方向，不能替代完整部署门禁。
