# 2026-09-06 Linear carrier-energy act-order 执行记录

## 结果

候选 `workbench/linear_carrier_energy_actorder_solution.py`，SHA256
`c382bd4fe4acdbb21b4aacd6b009051093d3fd92a264ae645956f96def26958c`，基于 v189。

- R0：通过。六 API 可导入，state 合法，`gptq_block_order` reachability 正常。
- R1：通过。shard 0/1 delta 分别为 `+0.000207/+0.000201`。
- R2：通过。Linear 336 cases `0.637750` vs parent `0.636799`，delta `+0.001045`；
  输出有限、覆盖完整。
- R3 OOD：通过。candidate `0.649639` vs parent `0.648735`，delta-gap change 约
  `+0.000046`，小于 `0.01`。
- R4 fresh default：Linear `0.641778372`，Attention `0.752173407`，Overall
  `0.687776303`，高于 v189 `0.686889609`。

## 时间裁决

分解为 `W_calib=271.089500s`、`A_calib=57.589949s`、`dyn_act=59.702568s`、
`dyn_qkv=2.938077s`，代入时间模型得 `280.622241s`。因未满足 `<280s`，最终状态为
`REJECTED_TIME`；未提交官方，根 `solution.py` 未修改。

证据：

- [fresh default JSON](../../artifacts/official_eval/linear-carrier-energy-actorder-fresh-default.json)
- [fresh default report](../../logs/official_eval/linear-carrier-energy-actorder-fresh-default.md)
- [six-shard manifest](../../artifacts/proxy_v3/linear-carrier-energy-actorder-r2/candidate/manifest.md)
- [OOD manifest](../../artifacts/proxy_v3/linear-carrier-energy-actorder-ood/candidate/manifest.md)
