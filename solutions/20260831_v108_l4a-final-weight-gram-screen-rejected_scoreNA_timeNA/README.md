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

## 审计更正（2026-08-31）

该 screen 不能作为 L4a 的算法否定证据。首次实现把 dynamic activation 的
`dense.shape[0]`（token 数 128）误当成离线权重的 output-row 数来判断
`rows > channels`，因此 final-Gram 路由在真实调用中从未触发，结果只是 v107
parent 的 no-op。修复后的 v109 使用 calibration 阶段写入的结构路由，并通过
完整部署 Gram 的逐行 gate 获得正向 full-layer 结果；v108 源码和原始 no-op
结果保持不变，仅修正文档解释。
