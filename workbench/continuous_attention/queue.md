# Attention 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。A侧部分已完成：父A22-2的4B基线由A23 paired run首跑补齐（72case mean 0.536715）。
2. **A23 官方回传（2026-09-07）：14437 / 276s → SCORE_BEST，新侧父**。vs A22-2 **+13/+5s**；vs A2 −3/+2s（A2 14440/274s 仍 official_best，近平局 Pareto）；vs R3 +32/+38s。**4B paired 负向（−0.00475）但官方正向——paired 面板仅风险记录的二代确认，本地判读不裁决官方**。乘积目标卡由官方重新开放；research_parent → anchor23-a1。
3. **A24 已提前停止（2026-09-07）**：互逆参数化+STE 误差目标无作用通道（logits=x_q·exp(S)·exp(−S)·x_kᵀ≡x_q·x_kᵀ，float64 FD 仲裁梯度精确为零；float32 matrix_exp autograd 数值损坏警告）。REJECTED_BEFORE_EVALUATION / NO_SUPPORTED_MECHANISM；关闭边界=仅互逆+STE 组合，码级离散方法（去重后）与 A22-2/A23 结构仍开放。
4. 条件后继：仅在新误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。已知教训：单窗口gate可过拟合（L8）；4B paired 负向不否决官方探索（A23 实证）也不预测官方正向。
5. **循环框架切换（2026-09-08）**：执行改为[误差账本驱动的持续研究循环](../../docs/superpowers/plans/workpackages/2026-09-08-continuous-research-loop.md)（R-1..R-6）。R1账本已建：`artifacts/continuous/attention/error_ledger_2026-09-08.json`——F4=0.468=F1(QK)0.189+F5-V 0.275；F1−F2≈−0.002；**F4下界=0.275（V冻结下gain 0.9结构不可达）**。
6. **A26-A 已本地证伪（2026-09-08，LOCAL_REJECTED 不提交官方）**：靶F2，锯齿符号×三角幅度网格残差一阶代理。数学6/6过、码翻转42%真实可达、模型loss降20-64%，但gate 0/6拒绝、logits级qk_mse 6/6恶化~10x——独立舍入残差一阶代理与真实自适应层级编码器（MSE-optimal lv2/lv3）系统性反向；两轮修正（半点伪吸引子、amax→base网格）不翻转。关闭范围=该实现+首项正弦+amax网格变体；根因同时解释A25官方失败。归档`workbench/continuous_attention/anchor26-a1/`。
7. **F5-V=0.275 最大格 FAMILY_CLOSED（AGENTS§7）待用户决策**：旧关闭证据仅覆盖 per-head/per-channel/per-token/bias 四类机制，**V 码分配类从未尝试**；V 冻结下 Attention gain 上限 0.725。
8. **next_card = A27-B**（预注册A26-B的修正实例化）：靶F2(0.192, OPEN)；低维S参数（每组谱系数共~32参数）+ 真实五字段读出MSE有限差分梯度（校准期合法）；与A24（全维STE梯度恒零）、A26-A（全维解析代理反向）去重=梯度来源与参数维度不同；证伪判据=gate接受<3/6或72例F2未降。

当前父与全部历史官方结果见[state.json](state.json)。4B目标侧全量paired，平均负向损失<0.02，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
本轮只计划，没有新候选或官方提交。具体旧实现负结果保留，不以此关闭成功机制整族。
