# Linear 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。
2. L23：R0核对旧证据身份并去重；设计L23实际A@W误差驱动的成对可逆对角坐标，先证明完整L4零点与合法编译，不重跑旧L21配置。
3. 条件后继：仅在本卡误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。

当前父与全部历史官方结果见[state.json](state.json)。4B目标侧全量paired，平均负向损失<0.02，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
本轮只计划，没有新候选或官方提交。具体旧实现负结果保留，不以此关闭成功机制整族。
