# 2026-09-06 Linear fused carrier-energy act-order 执行记录

候选 `workbench/linear_carrier_energy_fused_solution.py`，SHA256
`5fda25e6771434ddce66616b0c78d00151de15066aa0b220404a44b3f2a2593f`，基于 v189。

- F0：通过。fused energy 与 pair-wise reference 得到相同 block order；六 API、合法 state、
  reachability 正常。
- F1：通过。shard 0/1 delta `+0.000207/+0.000201`，数值与上一候选一致。
- F2：通过。六 shard Linear `0.637750` vs parent `0.636799`，delta `+0.001045`；
  OOD `0.649639` vs `0.648735`，delta-gap change 约 `+0.000046`。
- F3：fresh default Linear `0.641778372`、Attention `0.752173407`、Overall
  `0.687776303`，高于 v189 `0.686889609`；输出有限、覆盖完整。

时间分解为 `W_calib=282.014884s`、`A_calib=57.234458s`、`dyn_act=59.431035s`、
`dyn_qkv=2.958341s`，预测 `281.400628s`，超过 `<280s` 门禁。因此最终状态为
`REJECTED_TIME`；未提交官方，根 `solution.py` 未修改。

证据：

- [fresh default JSON](../../artifacts/official_eval/linear-carrier-energy-fused-fresh-default.json)
- [fresh default report](../../logs/official_eval/linear-carrier-energy-fused-fresh-default.md)
- [six-shard manifest](../../artifacts/proxy_v3/linear-carrier-energy-fused-r2/candidate/manifest.md)
- [OOD manifest](../../artifacts/proxy_v3/linear-carrier-energy-fused-ood/candidate/manifest.md)
