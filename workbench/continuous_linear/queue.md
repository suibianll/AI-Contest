# L 侧机制队列（continuous-linear）

> 更新：2026-09-07，审查后修订。契约：[持续优化总计划](../../docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
> + [下一轮工作包](../../docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md)。
> 当前状态EVIDENCE_REPAIR_REQUIRED。旧探针与去重文档保留历史结论，当前裁决以下表为准。

## 已裁决

| 状态 | 卡 | 依据 | 结果 |
|---|---|---|---|
| PROBE_INVALID_FOR_DEPLOYMENT / COST_UNVERIFIED | L1 FlatQuant 8×8 T1⊗T2 | 非标准输入、简化编码、训练配置偏差；不支持完整机制关闭 | L-R1闭环→L-R2成本 |
| DEDUP_UNRESOLVED | L2 GPTAQ 输出残差补偿 | 同目标不证明同更新；原JDRQ固定负结果保留 | L-R3公式级核验 |
| SCOPE_CORRECTION | L3 合法共享层级输出选择 | 旧固定求解器总体负、10/112正，不证明全空间覆盖 | 本轮仅修证据边界，不启动枚举 |

## 队列（新假设，待研究）

先完成L-R1/R2/R3，不再使用“L1–L3全关闭”作为等待依据；之后按总计划补充队列：

1. 先整理剩余误差分组与已测空间（从 L4 eval-v3 decomposition 归纳真实
   输出误差的 W/A 来源、role/layer/shape 分组）。
2. 检索原始论文（FlatQuant 已测否决；可测 Rotation/SVD-family 之外的
   低自由度结构与新目标）。
3. 提出有实质新自由度/目标/求解规则的机制卡（候选数固定、配置预注册）。

## 关闭边界备忘（本侧）

- 已关闭机制仅按AGENTS与对应原始实验的具体边界解释；不扩展成所有块序、A@W或合法编码空间关闭。
- Householder与既有rank/参数邻域不重开；JDRQ原固定实现负结果保留。
- L1先修已定位实现/证据问题，不扫8×8因子参数，不以简化探针关闭完整变换族。
- 本侧只从 L4 `ACB16F76...F5263` 构造候选；运行需 gpu.lock 串行。

## next_action

L-R1真实输入/API闭环 → L-R2固定L1正确性/成本；L-R3公式去重可穿插进行。详见下一轮工作包。
