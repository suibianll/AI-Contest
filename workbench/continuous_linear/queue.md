# L 侧机制队列（continuous-linear）

> 更新：2026-09-07。契约：[持续优化总计划](../docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
> + [L 工作包](workpackages/continuous-linear.md)。

## 已裁决

| 状态 | 卡 | 依据 | 结果 |
|---|---|---|---|
| CLOSED / REJECTED | L1 FlatQuant 8×8 T1⊗T2 | 35 真实 state 34 退化 + 完整硬前向不可承受 | [l1-flatquant-8x8](l1-flatquant-8x8/mechanism.md) |
| CLOSED / DUPLICATE_CLOSED | L2 GPTAQ 输出残差补偿 | JDRQ 同目标同更新规则，J1 已负 | [l2-gptaq-jdrq-dedup](l2-gptaq-jdrq-dedup/dedup.md) |
| CLOSED / DUPLICATE_CLOSED | L3 合法共享层级输出选择 | 修正版合法离散网格 R1 同目标全负 | [l3-legal-hierarchy-dedup](l3-legal-hierarchy-dedup/dedup.md) |

## 队列（新假设，待研究）

L1–L3 全部关闭后，按总计划 §3."初始队列耗尽"路径：

1. 先整理剩余误差分组与已测空间（从 L4 eval-v3 decomposition 归纳真实
   输出误差的 W/A 来源、role/layer/shape 分组）。
2. 检索原始论文（FlatQuant 已测否决；可测 Rotation/SVD-family 之外的
   低自由度结构与新目标）。
3. 提出有实质新自由度/目标/求解规则的机制卡（候选数固定、配置预注册）。

## 关闭边界备忘（本侧）

- 块序族（动/静 actorder、能量/协方差块序）REJECTED（时间或分数）。
- Householder 全族、cross-fold minimax、A@W 耦合坐标拟合关闭。
- rank-3/系数/fold 扫描关闭；JDRQ 关闭；合法编码候选集合同目标已测。
- L1 FlatQuant 旋转族关闭（34/35 退化）；不扫 8×8 因子邻域。
- 本侧只从 L4 `ACB16F76...F5263` 构造候选；运行需 gpu.lock 串行。

## next_action

误差分组分析 → 论文检索 → 新机制卡。