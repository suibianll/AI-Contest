# 计划入口

> 更新：2026-09-10。本轮开发已结束，官方待回传项与新开发计划分开管理。

**唯一活动总计划：** [Linear 局部代价修正与运行成本优化](2026-09-10-linear-correctness-and-runtime-plan.md)。

当前完整父：**v230 Linear L-EM2 + v195 Attention，18428/292s**；SHA256
`0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc`。回退根v202为18053/281s，官方余量8s。

下一步固定顺序：

1. **L-QF1**：修正局部二次代价的einsum收缩错误，固定K及其他配置，独立官方定价。
2. **L-TF1**：复用首遍梯度，减少重复矩阵乘，独立验证输出与执行成本。
3. **架构降时**：分清父GPTQ、新增下降、求逆/拷贝成本后，只选一个机制实施。

Attention没有新活动卡；已归档结果保留，后续方向须有具体新机制和成本依据，不重开K定心或扫描rotation/scale邻域。

## 已完成与待回传

| 对象 | 当前状态 |
|---|---|
| v230 Linear L-EM2 | RETAINED，官方18428/292s，当前根 |
| v231 Linear L-EM3 K=2 | 开发完成；六shard相对当前根+0.027507（286/0/50）；官方PENDING |
| v230 Attention A-FIX1 | 开发完成；本地−0.004884；官方未知，与Linear同编号分开绑定 |
| v229 Attention A-MC1 | 完整包TIMEOUT；标准Linear侧14424/245s，相对同口径基准−2/+2s；关闭 |
| A-QC1 | NO_EFFECT，未提交 |
| L-EM4 / L-DD1 | 侦察完成；未发现DD1实施候选。旧卡已被新计划取代 |

[本轮总结与证据修正](../../optimization-round-summary-2026-09-10.md)明确记录CPU/GPU计时口径、错误二次型、整API计数归因和时间外推的边界。
本地结果不能换算官方分数/时间；单次官方292s不证明时间模型，295.2s不是v231官方结果。

## 历史计划（只读证据）

- [L-EM1总协调计划](../archive/plans/2026-09-10-linear-exact-metric-refinement-plan-superseded.md)
- [L-EM2 / v230](../archive/plans/2026-09-10-linear-groupstep-schedule-plan-superseded.md)
- [L-EM3 / v231](../archive/plans/2026-09-10-linear-k2-timed-arm-plan-superseded.md)
- [L-DD1旧派发卡](../archive/plans/2026-09-10-linear-dynamic-dispatch-plan-superseded.md)
- [上轮完整入口快照与更早计划索引](../archive/plans/2026-09-10-round-entry-snapshot.md)

以上文件原有ACTIVE/预测/门禁描述均为历史快照，不提供下一步指令。parallel中的已执行Attention文件同样仅为结果证据。

## 执行纪律

AGENTS → [4B测试指引](../../4b-panel-testing-guide.md) → 本入口指定的活动计划。此目录除README外只保留一份活动总计划，子步骤写入该计划，不再并列多份ACTIVE。
官方无限次、硬限300s；不同执行成本的等输出实现可作一次正式提速验证，同SHA或同执行路径不重复提交估计噪声。
新候选从当时最高分完整根构建；已有候选保留原父和SHA，不机械组合、不继承未评测分数。版本号归档时串行登记，避免Linear/Attention重号。
