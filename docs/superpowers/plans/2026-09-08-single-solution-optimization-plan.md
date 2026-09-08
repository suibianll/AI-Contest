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

## 7. 当前顺序（不得跳步）

### R0：零 API 根入口审计

在根 `solution.py` 中定位是否存在与 L-C1 原卡同构的 fold 加权拟合/选择入口，输出公式、函数、
调用图和最小 diff 边界：

- 若存在：注册 **L-C3 objective-only**，只替换 fold 权重；不得增加 rank、残差子空间、邻域搜索、
  新正则或新候选循环。
- 若不存在：记 `L-C3 DESIGN_BLOCKED / NOT_APPLICABLE`，说明缺少何种接口；不得用 rank-8、
  Full-64 或合法邻域代替这张卡。

### R1：L-C3 最小复现（仅在 R0 可执行时）

先用合成测试证明 uniform `MSE_STD` 时与父目标等价、非 uniform 时只有 fold 权重改变；再跑 contract
smoke 和 Linear shard0。必须记录 diff 范围、attempted/accepted、权重实际变化及 activation/Attention
control。明显灾难性回归关闭该移植实现；通过后只提交这一个完整根候选。无论结果如何，不把它扩写成
整个 A@W 拟合族的结论。

### R2：Q/K 互逆 scale 的官方正证据移植

完成 R0/R1 后，对 A23 相对其直接父做零 API 最小差异审计，分离“Q/K 互逆参数化、scale 乘积目标、
父回退”与侧隔离脚手架。只有差异可分离时，才在当前完整根上移植同一机制；不得同时换成最终残差
CG、码级代理或新的训练主干。验证连续 QK 互逆关系、实际量化 scale/code 变化、attempted/accepted、
V 与全部 Linear control，再做 Attention shard0 和唯一代表官方提交。

R0/R1/R2 的目的不是重新用本地分数预测官方，而是确保每个官方结果回答一个单义问题。下一轮只有
这两条已知成功方向，不注册 L-C3 objective-only 与 A23-root-port 之外的邻域或替代机制。
