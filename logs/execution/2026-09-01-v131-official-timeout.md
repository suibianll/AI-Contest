# v131 official timeout update

用户于 2026-09-01 确认：v131 官方评测超时，耗时 `>300s`，没有返回官方分数。

本地 `official-shape-v1`：Linear `0.473131`、Attention `0.836579`、API `294.835s`、
wall `317.708s`。v131 的 Attention calibration `115.178s`、动态 Q/K/V `35.211s`，
与已经官方 timeout 的 v129/v130 属于同一高复杂度 Attention 家族；不能据此单独否定
v131 新增的 Linear `Q(W)`-Gram。

当前处理：继续以 v138 为根。v138 保留 v131 之后的 Linear 精度链，但将 Attention
calibration/动态 Q/K/V 压缩到约 `36.8s/5.1s`，两次本地总 API `187.935–192.996s`。
官方是否通过仍以平台回传为准。
