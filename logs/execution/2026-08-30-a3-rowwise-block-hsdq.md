# A3 Expansive-FFN Rowwise Block-Leverage HSDQ — rejected

日期：2026-08-30  
状态：`archived-rejected`；主代码已恢复 stable parent。

## 实现

在 A2 的稀疏行方案上进一步限制耦合：对每个 expansive 输出行独立计算
`z_block.T @ residual_row`，从 `0.5%/1%/2%` 行候选中为每行选择自己的最高
64-value block，再执行一轮 fixed-hierarchy mantissa HSDQ。每行默认只允许一个
block；两折 cross-fit admission 和 `mean + 0.5 × max` 评分保留 parent。

## Qwen2.5-0.5B 结果

| 范围 | Linear mean | Attention mean | panel | API time |
| --- | ---: | ---: | ---: | ---: |
| layer-1 A3 | `0.604138` | `0.926339` | `336.302323` | `16.34s` |
| 24-layer A3 | `0.499539` | `0.841829` | `293.250467` | `384.83s` |
| 24-layer stable parent | `0.501558` | `0.841829` | `293.755106` | `382.15s` |

全层 `fc_gate=0.364259`、`fc_up=0.426992`，较 A2 的 `0.353886/0.425646`
有所恢复，但仍低于 parent 的 `0.375126/0.430255`；其他 role 保持 parent。

## 裁决

A3 比 A2 少回退但仍降低 panel `0.504639`，因此不进入主线。该结果说明
expansive FFN 的校准 product residual 与最终 Linear case gain 仍不一致；本轮
停止继续扩大 HSDQ，转 A4 blockwise BOAT-2，先改变量化坐标系再重新评测。
