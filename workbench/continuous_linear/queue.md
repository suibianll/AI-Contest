# Linear 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。
2. L23：用户指定A@W低维拟合；首卡设计为残差监督子空间，保持L4激活路径，先与旧Gram/权重SVD拟合去重并检查合法投影。撤销上一稿可逆对角变换方案。
3. 条件后继：仅在本卡误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。

当前父与全部历史官方结果见[state.json](state.json)。4B目标侧全量paired，平均负向损失<0.02，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
本轮只计划，没有新候选或官方提交。具体旧实现负结果保留，不以此关闭成功机制整族。
