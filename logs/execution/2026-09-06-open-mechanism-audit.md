# v189 后开放机制审计执行记录

日期：2026-09-06

计划：`2026-09-06-open-mechanism-audit-plan`

状态：**CLOSED / NO_OPEN_MECHANISM**

## 审计结果

对 `workbench/linear_static_actorder_hdiag_recovered_solution.py` 的当前禁用开关逐项
核对源码注释、归档计划、执行日志和候选结果：

| 开关/族 | 当前判定 | 证据结论 |
| --- | --- | --- |
| Q/K headwise permutation、reciprocal temperature | 已执行 | 2026-09-06 分别 no-op，不能继续调参数 |
| Q/K Fisher/Jacobian importance | 已关闭 | v188 官方 `-4`；logit-Fisher 本轮在 shard4 负向 |
| 4×4/cross-pair、pair/rotation/Householder | 已关闭 | cross-pair OOD `-0.015858`，rotation/Householder 族已有负向/关闭证据 |
| V importance/multiplier 与 source-scale | 已关闭 | V 侧、A3、source-scale 均有负向或 no-op 证据 |
| Linear JDRQ/cross-block/full64/conditional curvature | 已关闭 | 负向、迁移回归或运行时/泛化门禁失败 |
| Linear carrier/calibration act-order | 已关闭 | 本地分数虽高于 v189，但时间预测 `280.622s/281.401s` 超 `<280s` 提交门 |
| quadratic16/sample-importance/hierarchy permutation | 已关闭 | 历史真实数据负向或 no-op |
| adaptive reg/offset、候选数、fold、threshold 等 | 禁止重开 | 属于已明确禁止的参数邻域，不是新独立机制 |
| verify/routing/provenance 开关 | 非算法 | 仅诊断、形状路由或不满足在线审计，不产生精度候选 |

结论：没有发现同时满足“未关闭、独立、合法、在线复杂度受控”的下一算法机制。继续
生成候选将重复已关闭族，故不再运行本地评测；根 `solution.py` 仍为 v186，官方父
`17599/272s` 不变。

关联证据：

- `logs/execution/2026-09-06-attention-logit-fisher-pair-plan.md`
- `logs/execution/2026-09-06-attention-crosspair-4x4-plan.md`
- `logs/execution/2026-09-06-attention-headwise-permutation-plan.md`
- `logs/execution/2026-09-06-attention-reciprocal-head-temperature-plan.md`
- `logs/execution/2026-09-06-attention-residual-pressure-order-plan.md`
