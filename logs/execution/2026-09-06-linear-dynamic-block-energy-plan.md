# Linear 动态块能量块序执行记录

日期：2026-09-06。父版本为 v189，官方 `17616/275s`；根 `solution.py` 未改。

## R0：合法性与可达性

候选从 v189 派生，六 API 独立导入、`py_compile`、合成 Linear 校验和 state 检查通过。
动态排序使用完整 64-block permutation，实际执行可达，输出有限。

## R1/R2：eval-v3 配对

固定 `proxy-v2` dense cache、CUDA 和 v189 baseline。R1 shard 0/1 delta mean 为
`+0.001688`、`+0.001893`，L1 为 `0.002400`、`0.002144`。

R2 六 shard 全部完成且 `reasonableness_issues=0`：

- mean delta `+0.003011996`；
- shard delta 分别为 `+0.001688/+0.001893/+0.006137/+0.002561/+0.002865/+0.002926`；
- OOD mean delta `+0.003559695`，`Δ(in−ood)` gap change 约 `-0.000548`，通过 `0.01` 门；
- 动态块序实际可达，未发现有限性或 control 异常。

原始证据保存在归档的 `proxy-r1/`、`proxy-r2/` 和 `proxy-ood/`。

## R3：fresh default 时间审计

兼容后端 `official_eval.py` 的 168 Linear + 120 Attention default：

- Linear `0.643820278430`；Attention `0.752173407020`；Overall `0.688967415343`；
- 相对本地最高 `0.687776303363` 为 `+0.001191111979`；
- `W_calib=275.920499s`、`A_calib=61.814679s`、`dyn_act=62.296941s`、
  `dyn_qkv=2.932736s`，预测 `286.022476s`；
- API total `402.964855s`，wall `430.540664s`。

## 裁决

候选虽然超过本地 proxy 最高，但时间模型未通过强制 `<280s` 提交门，因此标记为
`CLOSED / R3_REJECTED_TIME`，归档于
[`solutions/20260906_linear-dynamic-block-energy_time-rejected`](../../solutions/20260906_linear-dynamic-block-energy_time-rejected/)。
官方状态为 `unregistered/NA`，没有进行官方提交；根 `solution.py` 保持 v189。
