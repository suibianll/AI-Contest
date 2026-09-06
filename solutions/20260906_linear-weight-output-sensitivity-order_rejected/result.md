# Linear 权重输出敏感度 GPTQ 块序候选

- 父：v189；根正式版本仍为 v186。
- 状态：**REJECTED**。
- SHA256：`dd16211e798d83c03d044b9102d4b42013dbbc7212ceafaf29039d1ad7abd2ce`。

## 机制

只在离线 weight-GPTQ 中按
`sum(diag(Gram_X) * column_energy)` 对 64-channel 块排序，使用同一 Hessian inverse
补偿，并将编码结果恢复到自然块布局；不改变在线 state 或 Attention。

## 本地 eval-v3

W1 的 Linear 112 cases 相对 v189 的两个 shard delta mean 为
`+0.001381/+0.001197`。W2 扩展到 shard 4 后连续回归，五个已完成 shard 的 delta mean
为 `+0.001381、+0.001197、+0.000726、-0.001063、-0.000992`，回退集中于深层 `o`
role；因此按计划关闭，未运行 OOD/default，也没有官方提交。

证据：`artifacts/proxy_v3/linear-weight-output-sensitivity-order-20260906/`。
