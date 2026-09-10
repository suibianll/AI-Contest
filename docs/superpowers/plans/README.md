# 计划入口

> 更新：2026-09-10。唯一活动总计划如下；待官方候选不另立开发队列。

**[当前完整根梯度复用与已归档候选整合计划](2026-09-10-retained-root-gradient-reuse-plan.md)**

当前根：v231 Linear K=2 + v195 Attention，**18518/291s**，余量9s。
SHA256：`ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1`。
回退根：v233，18428/288s；梯度复用只在旧K=1父上获得官方验证，尚未并入当前根。

## 当前执行顺序

1. **L-TF2**：在启动时最高分完整根移植首遍梯度复用，保留K=2及全部精度机制，独立验证与官方定价。不混入v232或v235。
2. **v235回传后整合**：若官方同分更快，从届时最优完整父重建未落地的另一项提速机制，单独验证组合，不叠加时间。
3. **v236独立裁决**：正向且300s内才晋级；超时则关闭其完整代表。后续A-GR1降时须先有明确计算冗余证据，不缩步/缩窗重试。

## 已归档结果

| 对象 | 官方状态 | 处理 |
|---|---|---|
| v231 Linear K=2 | 18518/291s | 当前完整根 |
| v233 梯度复用 | 18428/288s，相对v230同分快4s | 回退根；首卡移植其机制，不复制整个旧父 |
| v235 编码器物化优化 | unregistered/NA | 已在v231上开发完成，等待一次回传；不重复已完成评测 |
| v236 A-GR1-on-v231 | unregistered/NA | 不能由v234超时预填其官方结果 |
| v234 A-GR1 | 完整包TIMEOUT；侧隔离+29 | 精度与成本证据分开，侧时间不能预测完整时间 |
| v232公式修正、v230 Attention A-FIX1 | TIMEOUT | 不再次注册原实现 |
| A-GR2 | NO_EFFECT，无正式版本 | 不以调学习率/归一化/步数重试 |

## 历史证据

- [上一轮完整计划及结果](../archive/plans/2026-09-10-linear-correctness-and-runtime-plan-completed.md)
- [上一轮入口及更早索引](../archive/plans/2026-09-10-pre-tf2-entry-snapshot.md)
- [官方状态](../../current-solution-status.md)、[版本索引](../../../solutions/README.md)

## 执行纪律

AGENTS → [4B测试指引](../../4b-panel-testing-guide.md) → 当前计划。此目录只保留一份活动总计划；阶段进展写在该计划。
六shard必须显式传 `--stop-after-nonpositive 6`，检查records=6、stopped_early=false；截断不作为完整结论。
官方未知score/time为null；不预测287s等时间，不相加375/90/29分，不重复同SHA。新候选以启动时最高分完整根为父，版本号串行登记。
