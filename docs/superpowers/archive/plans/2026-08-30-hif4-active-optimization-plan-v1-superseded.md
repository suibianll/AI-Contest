# HiF4 已归档优化计划 v1：实现审计后的 Linear 优先路线

> 状态：**ARCHIVED / SUPERSEDED**
> 建立日期：2026-08-30
> 最后更新：2026-08-30（计划治理规则固化）
> 适用根：`D:/工作内容/AI竞赛/solution.py`
> 目标：在不违反 HiF4 API/状态/监督边界的前提下，最大化真实官方分数；官方接口不可用时只做可复现本地 A/B。
>
> 归档说明：本计划因公式方向、单层门禁、显式 role-id API 假设和 Linear/Attention
> 队列混排问题，于 2026-08-30 被 v2 完整计划取代。本文仅保留历史，不再提供执行指令。

## 1. 现状和硬事实

当前根是 v100 B2 PAWV diag-only + B1 GQRB，v101 只是五模型确认。规范 LF SHA256 为 `617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`。

| 指标 | 当前值 |
|---|---:|
| Qwen Linear mean | 0.501558 |
| Qwen Attention mean | 0.842039 |
| Qwen panel `250gL+200gA` | **293.797301** |
| Qwen native total | 417.882506 |
| API 时间 | 392.423565 s（C0 复测 401.130873 s） |
| 官方成绩 | 未提交、不可推断 |

本地 panel 不是官方分数。若只把 Linear mean 提到 0.9、Attention 保持当前值，则 panel 约为 `393.4`；这只是代理数学值，不能声称等于官方 36000。

官方硬约束：输出字段/API/state 合法；在线 `Q(A)` 不得由 `A@W` 输出、输出残差或测试监督拟合/选择；最终时间严格小于 420s。探索阶段可超时，但进入提交前必须重新压缩。

## 2. 已实现算法和结论

### 当前保留

1. **BOAT**：RMS 对角平衡 + 4/8/16/64 signed-Hadamard。等价变换为
   \[
   X'=XD^{-1}R,\qquad W'=WDR,
   \]
   因为 \(R^TR=I\)，所以 \(X'W'^T=XW^T\)。候选只用两侧 operand-local 误差。
2. **cross-fold Weight-HSDQ**：用校准折的 `AᵀA` 做二阶离散增量，另一折确认，只改离线 `weight_params`。
3. **Gram-hierarchy Activation-HSDQ**：用静态 `WᵀW` block Gram 选择 E6M2 hierarchy/offset，并限制 block 数和 sweep 数；state 只含合法静态信息。
4. **Attention B1 GQRB margin**：GQA group-local orthogonal mixing，完整部署复评后才替换 parent。
5. **Attention B2 PAWV diag-only**：token-row 对角 Hessian 的 V refinement；低秩跨 token 项关闭。

### 关键已验证历史组件

- MHA-only K-center、GQA head-local rotation、共享 Hadamard。
- 多折、尺寸上限、自适应 headroom 的离线 A@W→Q(W)；JDRQ hierarchy residual 的稳定子集。
- clean 单路径重写：v086 panel `267.307909` → v100 `293.797301`，是最大正向变化。

### 已实现并归档的负结果

E1 progressive、A2 FFN sparse-row、A3 rowwise leverage、A4 BOAT-2、A5 joint-fold A@W、CAT-inspired BOAT-2、frozen-Q(A) Qronos、Global-LRH、PAWV rank-8、GALS-C sparse/shape-proxy、量化后权重 Gram 等均未超过当前根，部分还超时。具体数值和源码审计见 [`docs/archive-implementation-audit.md`](../../archive-implementation-audit.md) 与 [`docs/current-solution-status.md`](../../current-solution-status.md)。

## 3. 归档审计带来的结论修正

不能把所有负结果都当作算法上限：

- v092 LRH 重新选择的 scale/lv2/lv3 在 `_write_codes(parent, codes)` 中被丢弃，属于明确写回 bug。
- v095 Global-LRH 用元素 MSE 而非部署 Gram 二次型作最终 gate，属于目标错位。
- v099 PAWV rank-8 的 metric 在最终 Q/K 变换之前生成，属于坐标系错位。
- v102/v103 的 GALS 使用浮点权重 Gram和 shape proxy；final-weight Gram + 显式 role 仍未验证。
- v087–v091 无完整源和 SHA，属于不可复现归档，不作为严格上限。

以上细节不得在新文档中被简化为“LRH/GALS/PAWV rank 已被证明不可能”。

## 4. 当前实验队列（按优先级）

### P0：真实官方校准

官方接口恢复时，不改代码，先提交当前根 v100；记录提交 SHA、官方分数、时间和日期。这个结果决定本地 panel 到官方分数的兑换率，也是判断 36000 是否现实的必要信息。

### P1：修复 v092 full-hierarchy LRH

目标：验证“正确写回的跨 block weight LRH”是否有信号。

实现要求：

1. 候选对象同时携带 mantissa、sign、E6M2 scale、lv2、lv3；禁止用旧 parent denominator 写回新 mantissa。
2. 目标使用
   \[
   \Delta J = 2\,\mathrm{tr}(E^T H Q)+\mathrm{tr}(E^T H E),
   \]
   其中 \(H=A^TA\) 或其受限 block/low-rank 近似；每次坐标更新都用离散 15-level 增量。
3. fold-1 生成、fold-2 验证；先 layer-1、OPT/Pythia/Qwen 三模型筛选，再允许一次全层。
4. 若全层 panel 不低于 v100 且时间可压缩，再保留；否则归档为“修复后仍失败”。

### P2：修复 v095 Global-LRH gate

把接受条件改成部署一致的 Gram 二次型：

\[
J_H(Q)=\sum_b \operatorname{tr}\left((Q_b-X_b)^T G_b(Q_b-X_b)\right),
\quad G_b=W_b^TW_b.
\]

低秩近似 `G≈Gdiag+UUᵀ` 时，必须在最终离散化结果上重新计算完整 `J_H`，而不是只检查 surrogate 或逐元素 MSE。用 fold-2 和真实部署 Linear gain 作为第二道门。

### P3：重做 PAWV rank/position metric

先固定最终 Q/K 变换 \(T_Q,T_K\)，再计算 attention probability：

\[
P=\operatorname{softmax}\left((Q T_Q)(K T_K)^T/\sqrt d\right),
\quad H_V=P^TP.
\]

对 \(H_V\) 做 diag + rank-r 或位置 bucket 近似；每轮更新 V 后重新量化 Q/K/V 并以真实 attention 输出复评。任何跨 token 项都必须和最终 Q/K 坐标一致。

### P4：final-weight Gram + 显式 role 的 GALS 小预算复筛

流程：先完成最终 `weight_params`，解码得到 `Wq`，再计算 `G=WqᵀWq`；API 只接受静态 role id（q/k/v/o/fc_gate/fc_up/proj），不再从 shape 猜角色。只在 E0-G/D0 显示高 gap 的 v/attention-shaped 子集上测试 1–4 个 block，并做三折/异构模型复筛。

### P5：未完成的验证性工作

- E1 progressive 的完整 beam/checkpoint、合成宽度矩阵与三折验证。
- C0 元策略路由（role、shape、RMS、kurtosis、condition number）的小模型决策树。
- 外部 `youxilee/hif4` 的逐组件实现差异审计；不得把外部输出监督迁移到在线 `Q(A)`。
- 在正向机制确认后做计算重分配、缓存、批量化和 CPU/GPU 边界压缩。

## 5. 明确停止或降级的方向

以下方向在当前实现下不再占用主线预算：全局 scale-code 扩张、无门禁 full-H、全宽/逐块大自由度 A@W、未修复的 v092/v095 LRH、未对齐坐标系的 PAWV rank-8、shape-proxy GALS。它们的历史结果仍保留，但不把“当前实现失败”写成理论不可能。

## 6. 实验与归档规则

本文件是唯一可执行计划。执行顺序只从本文件读取；历史计划、旧日志中的“下一步”
只能作为背景，不能绕过本文件直接实施。每完成一个队列项，先把实际状态和证据写回
本文件，再开始下一个队列项；如果主线改变，先归档本文件并创建新的 active 文件。

每个候选必须包含：

1. 完整 `solution.py` 源快照；
2. 规范化 LF SHA256；
3. 固定配置、模型/层范围、耗时和 Linear/Attention 分项；
4. 代码合规扫描结果；
5. 明确的 parent、接受/拒绝理由和下一步。

层-1 结果只能筛选，不能替代全层门禁；Qwen panel 是主排序，其他模型用于发现结构性回退；官方不可用时 `official_score/time` 保持 `NA`。所有旧实施计划已移入 `docs/superpowers/archive/plans/`，本文件是唯一活跃计划。
