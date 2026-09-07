# Attention 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。A侧部分已完成：父A22-2的4B基线由A23 paired run首跑补齐（72case mean 0.536715）。
2. **A23 已执行并关闭（2026-09-07）**：Q/K联合块scale乘积目标，父A22-2，4B paired 72case **Δmean −0.004750、test split −0.010253 → LOCAL_NEGATIVE**；乘积比降45–55%而readout不跟随（预登记失败模式）；L8 gate接受后9/12负向暴露单校准窗口gate过拟合。按§5分支关闭此目标，不扫参数；候选归档 solutions/continuous_attention_anchor23-a1/。执行记录见 logs/execution/continuous-attention-anchor23-a1.md。
3. 条件后继：仅在新误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。已知教训：单窗口gate可过拟合（L8）；实际量化QK输出目标须先与旧Jacobian/动态Gram关闭族去重。

当前父与全部历史官方结果见[state.json](state.json)。4B目标侧全量paired，平均负向损失<0.02，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
本轮只计划，没有新候选或官方提交。具体旧实现负结果保留，不以此关闭成功机制整族。
