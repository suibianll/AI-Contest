# v108 L4a final-weight Gram 消融（screen rejected）

父版本：v107 Global Activation-LRH Gram-gated precision parent  
候选：仅对静态结构条件 `rows > channels` 且输入宽度 ≤1024 的 expansive FFN，
把 Activation-HSDQ 的 block Gram 从校准浮点 `WᵀW` 换为最终量化权重
`W_qᵀW_q`；q/k/v/o 和宽形状保持父路径。

五层 `{0,5,11,17,23}` × 七 role、固定 Qwen cache 的 screen：

| 指标 | v107 parent | v108 L4a | 差值 |
|---|---:|---:|---:|
| selected-layer Linear mean | 0.5289493081 | 0.5289493081 | 0 |
| layer/role entries changed | — | 0/35 | — |

没有正向信号，按 active plan 不运行 24 层 full-layer；候选不进入 parent。

证据：`l4a-final-gram-stratified-qwen.json`、`l4a-final-gram-stratified.md`。
归档源 LF SHA256：`9917a95bafc48576b6abd5bf7f658f2d541515e6d6c7dc983139ddd43f833f38`。
