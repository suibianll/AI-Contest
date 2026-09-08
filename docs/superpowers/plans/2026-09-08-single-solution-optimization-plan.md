# 单一完整方案优化计划

> ACTIVE，2026-09-08。取代 Linear / Attention 双侧独立持续循环。当前目标只有一个：
> 在单一完整父上提升官方总分，并保持官方时间 `<300s`。

## 1. 唯一父版本

- 根 `solution.py` 与归档 `solutions/20260908_linear-current-r3-attention_candidate/solution.py`
  逐位一致，SHA256 `12352EFDD4E23CC5E1E17953008664FBAA5EA5D693373635FDAFC4D28CE4E24E`。
- 仓库记录的用户官方回传为 `18032/280s`，相对上一根 compiled sample-energy `+396/+16s`，
  且 `280s<300s`。这是当前唯一工作父和最优已知可复现完整 solution；官方平台未单独返回计分
  SHA 的事实继续透明记录。
- 上一根 compiled sample-energy、v189、L28、A2、AC0 只作为历史对照或机制证据，不再形成并行父线。

## 2. 指标裁决

- 唯一晋级指标：官方总分更高且官方时间 `<300s`；同分时取更快版本。
- `calibration_fit_gain = mean_case(1-MSE_player/MSE_standard)` 公式正确，但只描述本地校准集内
  拟合，不能预测官方分数，也不是候选门禁。L28 `0.948587` 只保留为诊断事实。
- 4B paired 只检查接口、合法 state、finite、机制可达、非目标 API control 和明显灾难性回归；
  不用 Δmean、holdout、L1、误差账本或 probe 排序官方候选。
- 本地 `api_seconds`、按层外推和 FLOPs 只标记时间风险；官方 300s 是唯一时间裁决。

## 3. 唯一执行流程

1. **先做机制—代码一致性门。** 每张卡先列出目标公式、允许修改的函数/状态字段、明确禁止的
   附加自由度及预期 changed/attempted 计数；逐项映射到候选 diff。若根没有该机制的同构入口，
   记 `DESIGN_BLOCKED / NOT_APPLICABLE`，不得换成另一个求解器继续沿用原卡名称。
2. 用小矩阵或合成输入验证实际搜索粒度、零点/符号边界、接受判据和回退；文档声称“逐元素”、
   “只改 loss”或“逐位恢复父”时，测试必须直接覆盖该性质。失败时不启动模型评测。
3. 从当前根复制一个候选，只加入通过一致性门的一个机制和一个固定配置；不同时修改 Linear 与
   Attention。clean-room/侧隔离实现只可作因果诊断，不成为正式提交父。
4. 执行六 API 独立导入、reference 合法性、随机形状 smoke；目标侧 shard0 验证
   reachable/control，并核对实际 diff 未超出机制卡。明显灾难性回归用于否定实现；本地微小正负
   不预测官方排序。
5. 合法、可达、非 no-op、实现忠实且动态复杂度有界的唯一代表才提交官方。提交次数无限制，
   但不重复相同 SHA 或逐位等价实现。
6. 官方正向后才运行必要的 4B 六 shard 归档与完整双侧 interaction audit，并把根切到新父；
   官方负向、TIMEOUT、wrong answer分别只关闭已实际执行的具体机制、复杂度实现或正确性实现。
7. probe 可在实现前回答机制可执行性，也可在官方失败后回答会改变下一张卡的问题；必须绑定一个
   决策问题且优先零 API/合成检查，不建立永久误差账本，不产生新的独立父线。

## 4. 当前停止项

- 暂停 L31/L32、A30/A31 旧队列；如需回访，必须重新写成基于当前完整父的一张机制卡。
- 取消 Linear/Attention 各自 `gain≥0.9` 的停止目标；它们与官方成绩缺乏可用映射。
- 不再维护 E1–E4/F1–F5 持续误差账本、强制 `next_card`、多层 probe 漏斗或 side-score 组合预测。
- 不把 L28 `4611/286s` 与 A2 `14440/274s` 机械组合；两者的侧隔离时间已接近上限，且组合
  没有官方时间保证。

## 5. 下一步约束

当前没有自动继承的 L31/A30 队列。下一张卡除“改变什么、为何可能提升、如何证明
reachable/control、失败后关闭什么”外，必须增加第五项“代码映射与禁止改动”。缺少可执行机制时
保持当前根，不用探针制造进度。

## 6. 2026-09-08 方法审计与证据更正

- L-C1 卡写的是仅把根的 fold 权重改为 `MSE_STD+numel` 归一化，但实际归档实现加入了
  rank-8 residual-subspace 求解器。该运行只关闭“根上叠加这套 rank-8 后处理”，原卡未被执行；
  不得据此否定目标归一化本身。
- LC2 声称逐元素 `±1` 合法邻域，实际只尝试了每个 64-block 内全部输出元素同步 `+1/-1`；
  零值因沿用 `sign=0` 也没有发生 sign flip。`accepted=0` 只关闭这两个整块同步提案，不证明逐元素
  局部最优，不证明根的 A@W 坐标饱和。
- 历史源码和 manifest 保留原样作审计证据；当前解释以
  [`2026-09-08 LC1/LC2 方法审计`](../../../logs/execution/2026-09-08-lc1-lc2-method-audit.md)为准。

## 7. 当前顺序（Attention 优先，不得跳步）

当前根已经集成 R3 Attention，完整官方结果 `18032/280s`，相对上一根 `+396/+16s`。这证明
Attention 刚取得一次材料进展；当前问题是正向链没有继续，而不是 Attention 完全无效。A22-2 相对
R3 官方 `+19`，A23 相对 A22-2 官方 `+13`，但两段尚未进入当前完整根。因此暂停 Linear L-C3，
先沿已确认正向链继续 Attention。现有未归档 `workbench/continuous_linear/lc3-objective-only/` 保留原样，
不删除、不评测、不提交；Attention 获得完整官方裁决后再恢复。

### A-R0：R3 → A22-2 → A23 最小差异审计（零 API）

1. 先验证当前根四个 Attention API 与归档 R3 的最终定义及 state 语义一致；组合日志和官方 `+396`
   是外部证据，仍须把实际源码边界写清。
2. 审计 R3 → A22-2，只提取父坐标上的 Q/K 互逆残余变换、center 同步编译、选择与父回退；不得
   混入 A23 目标、A2 full-K/V 训练、最终残差 CG、码级代理或新主干。
3. 单独审计 A22-2 → A23，只提取 scale-product 目标变化；该段不能提前合入第一候选。
4. 若 A22-2 增量不能在当前 R3 根上保持单义，记 `DESIGN_BLOCKED` 并列出冲突字段，不用近似机制
   代替。A2 `14440/274s` 只作高分对照，本轮不混入。

### A-R1：A-C1 root reciprocal-residual-scale

仅在 A-R0 通过后，从当前完整根构建一个候选，只移植 R3 → A22-2 的最小差异。机制卡新增明确的
代码映射与禁止改动。合成测试必须覆盖：

- 连续路径 `Q'K'^T = QK^T` 的互逆关系；
- 零残余和强制拒绝时逐位恢复当前根；
- center 同步、GQA 映射与部署 trainer parity；
- 实际量化后的 Q/K scale、code、attempted/accepted 非零；
- V 与完整 Linear control 不变。

通过一致性门和 Attention shard0 后提交唯一完整根候选。官方正向且 `<300s` 才切根；负向只关闭
“A22-2 增量向当前 R3 根的这次移植”，TIMEOUT/wrong answer 只关闭对应复杂度/正确性实现，不扩大到
Q/K 互逆 scale 全族。

### A-R2：A-C2 scale-product（仅在 A-C1 官方正向后）

保持 A-C1 的参数化、父回退、训练窗口、求解器和部署路径不变，只替换为 A23 已获官方 `+13` 的
scale-product 目标。不得同时增加候选、调整 gate、移植 A2 或改变训练主干。按同一一致性门、
Attention shard0 和唯一完整官方候选裁决。

### L-R0/L-R1：恢复 Linear objective-only（Attention 裁决后）

Attention A-C1（以及被触发时的 A-C2）完成官方裁决后，才恢复已经完成入口审计的 L-C3：只替换
fold 权重，先做合成等价性、contract smoke 与 Linear shard0；不得加入 rank、残差子空间、邻域搜索、
新正则或新候选循环。

本顺序不是用本地分数预测官方，而是连续利用已发生的官方正证据。下一轮除 A-C1、条件触发的 A-C2
和顺延的 L-C3 外，不注册其他邻域或替代机制。
