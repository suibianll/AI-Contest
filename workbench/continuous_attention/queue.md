# Attention 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。
2. A23：设计A23父坐标Q/K块scale乘积目标；父A22-2，高分对照A2；保留完整回退，先去重和训练/选择隔离审读，不使用本地时间预测。
3. 条件后继：仅在本卡误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。

当前父与全部历史官方结果见[state.json](state.json)。4B目标侧全量paired，平均负向损失<0.02，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
本轮只计划，没有新候选或官方提交。具体旧实现负结果保留，不以此关闭成功机制整族。
