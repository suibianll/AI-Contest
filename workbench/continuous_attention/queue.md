# Attention 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。A侧部分已完成：父A22-2的4B基线由A23 paired run首跑补齐（72case mean 0.536715）。
2. **A23 官方回传（2026-09-07）：14437 / 276s → SCORE_BEST，新侧父**。vs A22-2 **+13/+5s**；vs A2 −3/+2s（A2 14440/274s 仍 official_best，近平局 Pareto）；vs R3 +32/+38s。**4B paired 负向（−0.00475）但官方正向——paired 面板仅风险记录的二代确认，本地判读不裁决官方**。乘积目标卡由官方重新开放；research_parent → anchor23-a1。
3. **A24 已提前停止（2026-09-07）**：互逆参数化+STE 误差目标无作用通道（logits=x_q·exp(S)·exp(−S)·x_kᵀ≡x_q·x_kᵀ，float64 FD 仲裁梯度精确为零；float32 matrix_exp autograd 数值损坏警告）。REJECTED_BEFORE_EVALUATION / NO_SUPPORTED_MECHANISM；关闭边界=仅互逆+STE 组合，码级离散方法（去重后）与 A22-2/A23 结构仍开放。
4. 条件后继：仅在新误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。已知教训：单窗口gate可过拟合（L8）；4B paired 负向不否决官方探索（A23 实证）也不预测官方正向。

当前父与全部历史官方结果见[state.json](state.json)。4B目标侧全量paired，平均负向损失<0.02，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
本轮只计划，没有新候选或官方提交。具体旧实现负结果保留，不以此关闭成功机制整族。
