# 计划入口

> 更新：2026-09-10。只保留一个活动总计划；官方待回传不阻塞研发。

**[完整根下一轮：Linear度量预编译与Attention训练分解复用](2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md)**

当前完整根v231：18518/291s，SHA `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1`。回退v233：18428/288s。

## 下一步

1. Attention A-SR1：审计并复用每步SVD投影分解，减少训练内重复求逆；保持32步与原目标/gate，明确浮点差异。
2. Linear L-MC1：审计并将固定度量重建前移至校准；保持K=2，核验state内存、设备语义与真实输出。
3. Attention精度归因：复用A-GR2已有证据，明确训练不降与gate拒绝的不同原因，不重试参数邻域。

两线从最高分完整根独立构建，不互等官方；单GPU执行串行。本次仅制定计划，尚未启动实现或评测。

## 上轮开发已完成

| 候选 | 开发结果 | 官方状态 |
|---|---|---|
| v237 L-TF2 | 当前K=2父首遍梯度复用；336例输出相同 | unregistered/NA |
| v238 A-CT1 | gate复用；同父A-GR1六层state/72例相同 | unregistered/NA |
| v239 A-CT2 | 训练尾部复用；204→198次调用，同父六层state/72例相同 | unregistered/NA |
| v235 / v236 | 各自实现关闭，不晋级 | TIMEOUT |

局部降时不能预测完整官方秒数。侧隔离时间不能与完整根相加，撤销“需要省12s/差60倍/必然超时”的入口推断；三张已归档卡无需重新开发。

## 证据入口

- [上轮已完成计划](2026-09-10-retained-root-gradient-reuse-plan.md)
- [当前状态](../../current-solution-status.md)、[版本索引](../../../solutions/README.md)
- [更早已完成计划](../archive/plans/2026-09-10-linear-correctness-and-runtime-plan-completed.md)

规则顺序：AGENTS → 4B测试指引 → 本入口指定计划。旧分析、旧卡与workbench不提供当前执行指令。
