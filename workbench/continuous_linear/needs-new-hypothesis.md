# L 侧 NEEDS_NEW_HYPOTHESIS 记录

> 日期：2026-09-07。侧：Linear。
> 依据：总计划 §3"初始队列耗尽时，先整理剩余误差分组与已测空间，再检索
> 原始论文，提出有实质新自由度/目标/求解规则的一张卡。不得用固定任务列表
> 结束证明"没有剩余算法"。若确实没有符合约束的新机制，记 NEEDS_NEW_HYPOTHESIS
> 并给出具体被阻断的选项与必要信息。"

## 已测空间（本侧，全部无正向）

1. L1 FlatQuant 8×8 Kronecker 旋转：35 真实 state 中 34 退化；完整硬前向
   每步 0.6-1.7s，32 步 × 168 state 官方时间 +742s 超预算 → 关闭。
2. L2 GPTAQ 输出残差补偿：与 JDRQ 同目标（min||Xh Wh^T−Y||²）同更新（残差
   引导逐坐标下降）→ DUPLICATE_CLOSED（JDRQ 已在 v189 上 J1_REJECTED）。
3. L3 合法共享层级：修正版合法离散网格（合法 scale×lv2×lv3×mantissa 联合
   output oracle）holdout 全负 → DUPLICATE_CLOSED。
4. 历史既有关闭族（AGENTS §7）：块序全族、Householder、cross-fold minimax、
   rank-3/系数/fold、JDRQ、A@W 耦合坐标、合法编码搜索、per-call 动态、
   dynamic offsets/swap/R64/adaptive-reg 等常量默认关闭。

## 剩余误差分组（L4 真实 336 case）

- fc_up 0.502 / o 0.534 / fc_gate 0.554 / proj 0.564 / qkv 0.75+（gain）。
- 官方 P3 探针：Linear 官方增益 100% 落在 fc+proj 大形状桶（W2/W3）；
  hidden_to_hidden（q/o）与小输出（k/v）零收益 → 本地 o 的低分不可迁移。
- fc/proj 部署坐标 E00/E10/E01/E11 分解：A-only 占 61-79%，W-only 21-39%，
  interaction ≈ 0 → 激活侧是 fc/proj 剩余误差主来源。

## 被阻断的具体选项与新假设方向

- 激活侧静态精化：h_inv GPTQ + importance + offsets + actorder hdiag 已是
  当前栈；per-call 动态、gram 引导变体、块序变体均已关闭；importance 槽位
  被 pair-smooth 占据（v188）。
- fc_gate/fc_up 中间误差传播不可体现：eval-v3 每层激活为真实前向捕获，
  gate⊙up 量化误差不进入 proj 输入，联合优化目标无评测对应。
- 新机制准入需满足：合法五字段、单文件、动态预算 <280s、非已关闭族、
  校准/holdout 隔离、真实输出误差目标。当前无可识别的一张卡。
- 候选新方向（需进一步研究）：(a) fc/proj 激活侧的低成本解析平坦化，
  与已有 Hadamard/CAT64 不同参数化；(b) 输出误差交叉项的解析闭式补偿，
  非 JDRQ 的坐标下降形式；(c) 官方 P3 桶级证据与 21765 榜首差距的归因
  审计后再定。

## 下一步

若协调者批准新假设或用户重开某一关闭边界，则从本侧 official_best
(L4, ACB16F76...F5263) 构造新候选；否则本侧保持 READY 等待新卡。不空转、
不重跑已关闭族。