# C1a structured proposal vectorization — synthetic and equivalence check

> 日期：2026-08-31
> parent：v118 L6d structured block-circulant factor
> candidate：v119 C1a vectorized proposal

## 代码等价性

- 保留 `_refine_activation_structured_reference` 作为 v118 对照实现。
- `_refine_activation_structured_vectorized` 只把独立的 row/block proposal 和 15-level
  候选评估批量化；64 个 coordinate 仍按升序串行更新，`torch.min` 首个 index tie-break、
  `_write_codes` 和最终完整部署 `G_q` row gate 均保持不变。
- 固定随机种子 `714`、`dense=[4,256]`、`deployment=[96,256]`、`max_blocks=2`，两条
  路径返回的全部字段逐项 `rtol=0, atol=1e-6` 相等。
- 结构化 matmul 的显式 circular block reference 测试仍通过；C1a 定向集合共
  **37 passed**。

## 小型宽层时间对照

固定随机种子 `715`、`rows=32`、`channels=4864`、`deployment rows=128`、同一
`deployment_gram`/parent/state、`max_blocks=4`：

| 路径 | dynamic proposal time |
|---|---:|
| v118 reference | `0.6074403s` |
| C1a vectorized | `0.0628309s` |
| speedup | `9.67x` |

该 benchmark 只证明候选循环的局部加速，不能替代 Qwen full-layer 门禁。

## 合规

`guard_solution_file('solution.py')`：`violations=[]`、`static_violations=[]`、
`contraction_count=22`；review 项仍是允许的离线 `A@W` 人工复核提示。

