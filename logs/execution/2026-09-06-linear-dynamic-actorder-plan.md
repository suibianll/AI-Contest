# Linear 动态样本能量 GPTQ 块序执行记录

日期：2026-09-06。父版本为 v189，官方 `17616/275s`；根 `solution.py` 未改。

## R0：合法性与可达性

从 v189 复制候选 `workbench/linear_dynamic_actorder_solution.py`，六 API 独立导入、
`py_compile`、合成 Linear 校验和 state 检查通过；动态块序为完整合法 permutation，
`[STATIC-ACTORDER-HDIAG] reachable=1`。

## R1/R2：eval-v3 配对

固定 `proxy-v2` dense cache、CUDA 和 v189 baseline。R1 两片 delta mean 为
`+0.001906528`、`+0.001949470`，L1 为 `0.002375702`、`0.002170024`。

R2 六 shard 全部完成且 `reasonableness_issues=0`：

- mean delta `+0.002981296`，median delta `+0.001424445`，L1 `0.003258954`；
- 正/负/零 case `292/44/0`，所有输出有限，动态顺序实际可达；
- OOD mean delta `+0.003518763`，`Δ(in−ood)=-0.000537467`，通过 `0.01` 门。

原始证据保存在归档的 `proxy-r2/` 与 `proxy-ood/`。

## R2 fresh default

兼容后端 `official_eval.py` 的 168 Linear + 120 Attention default：

- Linear `0.643867464427`；Attention `0.752173407020`；Overall `0.688994940507`；
- 相对本地最高 `0.687776303363` 为 `+0.001218637144`；
- `W_calib=277.132982s`、`A_calib=62.261667s`、`dyn_act=60.914819s`、
  `dyn_qkv=2.991263s`，预测 `285.365171s`，时间门失败。

## R3：实现级降时复核

不改变样本能量公式，仅把 `argsort` 产生的块序保留在算法设备，去掉 CPU permutation
校验/往返。R0 和 R1 通过，R1 输出与 R2 候选一致。R3 fresh default 仍为：

- Linear `0.643867464427`；Attention `0.752173407020`；Overall `0.688994940507`；
- `W_calib=277.721119s`、`A_calib=61.447755s`、`dyn_act=60.802674s`、
  `dyn_qkv=2.997517s`，预测 `284.775756s`，仍未通过 `<280s`。

## 裁决

候选虽然超过本地 proxy 最高，但时间模型未通过强制提交门，因此标记为
`REJECTED_TIME`，归档于
[`solutions/20260906_linear-dynamic-actorder_time-rejected`](../../solutions/20260906_linear-dynamic-actorder_time-rejected/)。
官方状态为 `unregistered/NA`，没有进行官方提交；根 `solution.py` 保持 v189。
