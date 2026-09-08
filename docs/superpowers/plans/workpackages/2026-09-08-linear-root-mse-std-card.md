# 机制卡：根上应用 MSE_STD+numel 归一化 A@W 校准目标（L-C1，未执行）

> **INVALIDATED / NOT EXECUTED，2026-09-08 方法审计。** 本卡要求“只改 fold 权重”，但以
> LC1 名义归档的源码实际加入 rank-8 residual-subspace 求解器，机制与本卡不一致。该归档结果只
> 关闭实际 rank-8 实现，不裁决本卡。后续按活动总计划 R0 审计根是否存在同构入口；若可执行，以
> L-C3 objective-only 新卡重新注册，不能继续复用 L-C1 名称。

原依据为[单一完整方案优化计划](../2026-09-08-single-solution-optimization-plan.md)旧 §5 四项机制卡模板；
当前五项模板及执行顺序见该计划 §5–§7。

## 1. 改变什么

在根 `solution.py`（SHA `D66128A6…`，官方 17636/264s）的 Linear 权重校准中，
把残差/拟合的 fold 权重从 `1/||Y_f||²` 改为 LC0 已验证的
`ω_f = 1/(F · numel(Y_f) · MSE_STD_f)`（MSE_STD = 同 NVFP4 输入的标准 HiF4 输出误差）。
只改权重校准的 fold 归一化，不动 activation、Attention、动态 API。

## 2. 为何可能提升完整官方输出

LC0（L28 + 本归一化）官方 **+3**（4611→4614，294s），证明该目标归一化
在官方口径下有效。根是完整方案（官方 17636），其 Linear 校准的 fold 权重
若同样失配（10 行 fold 主导），应用该归一化可带来类似正向。这是
**已官方验证机制的直接移植**，非新探索。

## 3. 如何证明 reachable/control

- 根候选复制自 `solution.py`，只改 Linear 校准 fold 权重一处；
- 六 API 独立导入、reference 合法性、随机形状 smoke；
- Linear shard0 paired：reachable（attempted>0）、合法 state、finite、
  activation 路径逐位不变（control）、Attention 逐位不变（control）；
- 官方 300s 唯一时间裁决。

## 4. 失败后关闭什么

- 官方负向/同分：关闭"根上应用 MSE_STD+numel 归一化"这一实现；
  不重试（LC0 已证明在 L28 上 +3，若根上无效说明根的 fold 结构不同）；
- TIMEOUT：关闭该复杂度实现；
- wrong answer：关闭该正确性实现。
