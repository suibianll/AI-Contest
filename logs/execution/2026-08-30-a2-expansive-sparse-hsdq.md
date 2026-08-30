# A2 Expansive-FFN Sparse-Row HSDQ — rejected

日期：2026-08-30  
状态：`archived-rejected`；评测结束后主代码恢复到 stable parent。

## 目的与实现

针对 `fc_gate/fc_up` 的 `rows > 2 × channels` 形状，按校准 product residual
选取输出行的 `1%/2%/5%` 三档候选；每档在选中行子矩阵上复用已验证的
fixed-hierarchy HSDQ，随后以两折 cross-fit admission 和
`mean + 0.5 × max` 鲁棒评分选一个候选。未选行保持 parent 不变。

## Qwen2.5-0.5B 结果

| 范围 | Linear mean | Attention mean | panel | API time |
| --- | ---: | ---: | ---: | ---: |
| layer-1 A2 | `0.599777` | `0.926339` | `335.212015` | `16.46s` |
| 24-layer A2 | `0.497865` | `0.841829` | `292.831952` | `385.48s` |
| 24-layer stable parent | `0.501558` | `0.841829` | `293.755106` | `382.15s` |

全层 role 均值：`fc_gate 0.353886`、`fc_up 0.425646`，均低于 parent 的
`0.375126`、`0.430255`；其他 role 与 parent 相同。候选降低 panel
`0.923153`，因此不能进入主线。

## 裁决

A2 证明“按总 product residual 选高损行 + 子矩阵 HSDQ”与最终 Linear
评分不一致，不能直接扩大行数。下一步 A3 改为逐行 block leverage 和更小的
block budget，继续保留 parent 作为硬基线；若 A3 仍回退，则停止 FFN HSDQ
扩张，转 BOAT-2/激活重标定。
