# Linear 动态 32 行块能量块序执行记录

日期：2026-09-06。父版本为 v189，官方 `17616/275s`；根 `solution.py` 未改。

## R0：合法性与可达性

候选从 v189 派生，六 API 独立导入、`py_compile`、合成 Linear 校验和 state 检查通过。
动态排序使用确定性 32 行样本的完整 64-channel block permutation，实际执行可达，
输出有限。

## R1/R2：eval-v3 配对

固定 `proxy-v2` dense cache、CUDA 和 v189 baseline。R1 shard 0/1 delta mean 为
`+0.001251586`、`+0.001237825`，L1 为 `0.001839595`、`0.001686732`。

R2 六 shard 全部完成且 `reasonableness_issues=0`：

- mean delta `+0.002531235`；
- shard delta 分别为 `+0.001251586/+0.001237825/+0.005470511/+0.002272533/`
  `+0.002597047/+0.002357905`；
- OOD mean delta `+0.003227747`，`Δ(in−ood)=-0.000696513`，通过 `0.01` 门；
- 动态块序实际可达，未发现有限性或 control 异常。

原始证据保存在归档的 `proxy-r1/`、`proxy-r2/` 和 `proxy-ood/`。

## R3：fresh default 时间审计

兼容后端 `official_eval.py` 的 168 Linear + 120 Attention default：

- Linear `0.643280557360`；Attention `0.752173407020`；Overall `0.688652578052`；
- 相对已测本地最高 `0.688994940507` 为 `-0.000342362456`；
- `W_calib=272.077215s`、`A_calib=62.651416s`、`dyn_act=62.213140s`、
  `dyn_qkv=3.295353s`，预测 `285.526750s`；
- API total `400.237124s`，wall `427.532596s`。

## 裁决

候选没有超过已测本地最高，且时间模型未通过强制 `<280s` 提交门，因此标记为
`CLOSED / R3_REJECTED_TIME`，归档于
[`solutions/20260906_linear-dynamic-block-energy32_time-rejected`](../../solutions/20260906_linear-dynamic-block-energy32_time-rejected/)。
官方状态为 `unregistered/NA`，没有进行官方提交；根 `solution.py` 保持 v189。
