# A 侧机制队列（continuous-attention）

> 更新：2026-09-07。契约：[持续优化总计划](../../docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
> + [Attention 工作包](../../docs/superpowers/plans/workpackages/continuous-attention.md)
> + [下一轮工作包 §6 A-R1](../../docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md)。
> 冻结：两个 Linear API = v162 standard；V = R2c 既有 V 路径（不注册新 V 机制）。

## 官方已登记（全为官方实测）

| 状态 | 卡 | 官方 | 证据 |
|---|---|---|---|
| OFFICIAL_PASS | R2c 栈内旋转（手工梯度） | 14387.8 / 240s，C_A=13386.8 | `solutions/v162_attention_r2c-rotation-in-stack-manual_oodblocked/` |
| OFFICIAL_PASS / **best** | R3 旋转 + 学习 K-center | **14405 / 238s，C_A=13404** | `solutions/v162_attention_r3-rotation-center_allgates/` |
| OFFICIAL_PASS | A2c 独立旋转（手工梯度） | 9538 / 175s，C_A=8537 | `solutions/v162_attention_a2c-rotation-manual-trainer_oodblocked/` |
| OFFICIAL_PASS | R1 v189 栈 + 标准 Linear | 14009 / 211s，C_A=13008 | `solutions/v162_attention_r1-v189-attnstack-recovery_officialNA_timeNA/` |

## 当前执行（evidence-repair §6 A-R1 → continuous-attention A0）

1. **A0 部署目标可验证化**：核对 R2c/R3 归档 SHA 与 manifest 身份；从源码逐步枚举 Q/K
   实际执行链（decode → K-center(mode4) → multiplier → signs·H64 → pair-transform →
   learned rotation → encode/refinement），记录旋转插入前坐标 U_Q/U_K；
   写出训练目标 vs 实际 V vs 完整 K/V 的矩阵等式差异（R2c 训练用标准编码器代理 +
   `_ste_encode` V，与部署的 v189 精化编码不一致——这是 A1 要修的目标错位）。
2. **A0 最小测试**：I/父 R、GQA 共享、Lq<Lkv、inference_mode/no_grad、有限输出、state 合法、
   手工梯度 vs autograd/有限差分对照（STE 只对声明的替代模型验证）。
3. **A0 逐位对照**：新训练前向（部署路径复刻）vs 六 API 部署解码输出逐位一致或报告可解释
   浮点差异；通过前不跑优化器实验。

## 队列（A0/A1 通过后依序）

| 卡 | 假设 | 配置边界 | 关闭/依赖 |
|---|---|---|---|
| A1 部署对齐旋转训练 | R2c 收益受训练/部署坐标与编码器错位限制 | 配置从 R2c 源码提取（32步/lr0.01/采样/gate 划分），训练硬前向改真实坐标 + 父实际编码 + 量化 V；不同时改 importance/offset/gate/V | 依赖 A0 逐位对照通过 |
| A2 完整 K/V 支持集 | 仅 128 K/V 训练改变 softmax 分母/竞争 | 保持 Q 抽样与全部编码规则；全长 K/V、Q≤32 行分块；先测一层每步成本 | 依赖 A1；成本不可行记 TIME_HOLD |
| A3 受约束非正交 Q/K 变换 | FlatQuant 思想：旋转后对称跨通道伸缩 T=exp(S)，cond(T)≤2 | 每 KV 组一个对称零迹 S；Q 乘 T、K 乘 T^-T；父优化器预算 + 1e-3·mean(S²) | 先与 QK balance/pair-transform/Householder 关闭族去重 |

## 关闭边界备忘（本侧）

- Q/K 坐标（旋转）+ 平移（center）联合自由度在栈内已被吸收（R3 default −0.0028 vs R2c）——
  同族再加 DOF 不重开。
- OOD 超阈值为风险提示（2026-09-07 门禁修订），不再单独禁止官方探索；排序信号有效
  （过门 R3 官方 > 未过 R2c）。
- 手工梯度是官方环境唯一可行训练路径（autograd/inference 逃逸全被击败，A2b/R2b 官方回退为证）；
  新训练机制一律解析梯度 + `fuzz_official_contract.py` 提交前必跑。
- 每态值官方映射点（仅记录不拟合）：A2c 0.4185→8537；R2c 0.7678→13386.8；R3 0.7650→13404。
