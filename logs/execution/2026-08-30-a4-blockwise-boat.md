# A4 Blockwise BOAT-2 — rejected

日期：2026-08-30  
状态：`archived-rejected`；主代码已恢复 stable parent。

## 实现

在现有 BOAT 的全局指数候选之外加入两个低维 blockwise 指数 schedule：按校准
`log(a_rms / w_rms)` 的 block spread 或 magnitude，为每个 64-value block 选择
`alpha ∈ {0.25, 0.5, 0.75}`，再做全局几何均值归一化；旋转和后续 HSDQ 保持不变。

## Qwen2.5-0.5B 结果

| 范围 | Linear mean | Attention mean | panel | API time |
| --- | ---: | ---: | ---: | ---: |
| layer-1 A4 | `0.603074` | `0.926339` | `336.036268` | `17.98s` |
| 24-layer A4 | `0.498449` | `0.841829` | `292.978009` | `368.23s` |
| 24-layer stable parent | `0.501558` | `0.841829` | `293.755106` | `382.15s` |

全层 q/k/v/o 分别为 `0.598103/0.615187/0.566194/0.482905`，均出现小幅
回退；`fc_gate/fc_up` 基本等同 parent。虽然运行时间下降，但 panel 下降
`0.777097`，不符合精度门禁。

## 裁决

停止当前 blockwise BOAT-2 变体，不再保留其 schedule。下一步转 A5 FS-JDRQ：
先冻结在线激活量化，再用 calibration-only `A@W` 生成静态权重候选，所有状态
仍只含合法 quantized weight 与 operand-local Gram。
