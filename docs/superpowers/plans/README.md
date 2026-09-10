# 计划入口

> 更新：2026-09-10。唯一活动总计划如下；待官方候选不另立开发队列。

**[完整根双线优化：Linear梯度复用与Attention校准复用](2026-09-10-retained-root-gradient-reuse-plan.md)**

当前根：v231 Linear K=2 + v195 Attention，**18518/291s**，余量9s。
SHA256：`ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1`。
回退根：v233，18428/288s；梯度复用只在旧K=1父上获得官方验证，尚未并入当前根。

## 当前执行安排（两线均可立即开发，不等官方）

- **Linear L-TF2**：在最高分完整根移植首遍梯度复用，保留K=2，单独验证。**已完成并归档 v237**
  （`ecb1f9e5…`，六 shard 336/336 精确零，算子 38→36；本地墙钟低于本机分辨力，带 sham null 0.52–1.11×）。
  待官方回传。
- **Attention A-CT1**：在同一完整根装配A-GR1，复用父/候选gate共用的参考Attention与V量化结果；保留32步、目标、窗口、候选数和接受逻辑。独立验证对A-GR1旧实现等输出、对正式父有实际机制变化。
- **Attention后继 A-CT2**：A-CT1开发结束即审计训练末尾重复的loss/统计计算；确认可复用后注册固定实现，不等待v236/A-CT1官方回传，不把待官方候选当父。
- v235/v236回传只决定晋级和后续组合，不阻止上述开发。v235已回传TIMEOUT，其裁决按计划§3"不自动移植、L-TF2独立裁决"执行；v236仍待回传。单GPU测试串行，源码/归档/版本登记隔离，不相加分数或时间。
## 已归档结果

| 对象 | 官方状态 | 处理 |
|---|---|---|
| v231 Linear K=2 | 18518/291s | 当前完整根 |
| v233 梯度复用 | 18428/288s，相对v230同分快4s | 回退根；首卡移植其机制，不复制整个旧父 |
| v235 编码器物化优化 | TIMEOUT（>300s）REJECTED | 本轮唯一本地测出可分辨提速的Linear候选（34–52×null）仍超时；不缩步/缩窗重试，不自动移植 |
| v237 Linear L-TF2 | unregistered/NA | 已完成并归档，等待一次回传；本地墙钟不可分辨，不主张官方可分辨差异 |
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
