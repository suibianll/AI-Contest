# 2026-09-06 Linear 权重输出敏感度 GPTQ 块序执行记录

候选 `workbench/linear_weight_output_sensitivity_order_solution.py`，基于 v189，SHA256
`dd16211e798d83c03d044b9102d4b42013dbbc7212ceafaf29039d1ad7abd2ce`。

## 结果

- W0：通过。六 API、`py_compile`、合法 state、非自然块序 `[1,0]` 及自然块布局恢复
  均正常。
- W1：Linear 112 cases 相对 v189 为正，两个 shard delta mean
  `+0.001381/+0.001197`，L1 `0.003430/0.001993`。
- W2：eval-v3 在 shard 4 后因连续两个非正 shard 提前停止；已完成 shard delta mean
  `+0.001381、+0.001197、+0.000726、-0.001063、-0.000992`。shard 3/4 的主要
  回退来源是 `o` role，未运行第 5 shard、OOD 或 default。

## 裁决

状态：**CLOSED / W2_REJECTED**。该权重侧固定输出敏感度块序不具跨深度稳定性；不扫
邻域、不提交官方。候选源码已归档到
`solutions/20260906_linear-weight-output-sensitivity-order_rejected/solution.py`，
根 `solution.py` 未修改。

证据：

- W1 manifest：`artifacts/proxy_v3/linear-weight-output-sensitivity-order-20260906/w1/candidate/manifest.json`
- W2 manifest：`artifacts/proxy_v3/linear-weight-output-sensitivity-order-20260906/full/candidate/manifest.json`
- 归档计划：`docs/superpowers/archive/plans/2026-09-06-linear-weight-output-sensitivity-order-plan-rejected.md`
