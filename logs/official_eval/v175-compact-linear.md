# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `33b25ebeda87dc1d09f08531a68e056c6d6587a43a336a5dd93b2b642f8a8f88`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.705628231 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.705628231 |
| Linear role macro mean | 0.705628231 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 59.227s |
| Candidate API total | 53.748s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.705628 | 0.686647 | 0.591391 | 0.539187 | 0.418855 | 56/0/0 | 0.313353 |
| family:fc | 16 | 0.563059 | 0.558638 | 0.544484 | 0.494431 | 0.418855 | 16/0/0 | 0.441362 |
| family:o | 8 | 0.687667 | 0.638188 | 0.619135 | 0.610030 | 0.601682 | 8/0/0 | 0.361812 |
| family:proj | 8 | 0.650429 | 0.608097 | 0.577909 | 0.525740 | 0.524968 | 8/0/0 | 0.391903 |
| family:qkv | 24 | 0.825061 | 0.828690 | 0.771441 | 0.746027 | 0.708551 | 24/0/0 | 0.171310 |
| role:fc_gate | 8 | 0.595191 | 0.585042 | 0.559213 | 0.556967 | 0.556445 | 8/0/0 | 0.414958 |
| role:fc_up | 8 | 0.530926 | 0.539361 | 0.519031 | 0.462712 | 0.418855 | 8/0/0 | 0.460639 |
| role:k | 8 | 0.833127 | 0.833334 | 0.789933 | 0.756076 | 0.753313 | 8/0/0 | 0.166666 |
| role:o | 8 | 0.687667 | 0.638188 | 0.619135 | 0.610030 | 0.601682 | 8/0/0 | 0.361812 |
| role:proj | 8 | 0.650429 | 0.608097 | 0.577909 | 0.525740 | 0.524968 | 8/0/0 | 0.391903 |
| role:q | 8 | 0.820863 | 0.819656 | 0.759385 | 0.722192 | 0.708551 | 8/0/0 | 0.180344 |
| role:v | 8 | 0.821193 | 0.823833 | 0.792353 | 0.762617 | 0.752392 | 8/0/0 | 0.176167 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.016655`、max `0.163949`；成对 minimum-gain median `0.682381`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.000000 | 0.000000 | 0/0/56 | 1.000000 | no_effect |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -226.153832 | -151.210999 | 0.705628 | 378.070460 | 2.239325e+00 | 1.537921e+00 |
| family:fc | 16 | -190.202203 | -153.222678 | 0.563059 | 343.987940 | 2.135941e+00 | 1.642051e+00 |
| family:o | 8 | -197.563623 | -155.622072 | 0.687667 | 353.873362 | 2.098863e+00 | 1.661822e+00 |
| family:proj | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |
| family:qkv | 24 | -259.645594 | -147.535344 | 0.825061 | 408.005999 | 2.389206e+00 | 1.399101e+00 |
| role:fc_gate | 8 | -235.270245 | -197.467870 | 0.595191 | 433.333305 | 2.173177e+00 | 1.657154e+00 |
| role:fc_up | 8 | -145.134162 | -108.977486 | 0.530926 | 254.642574 | 2.098705e+00 | 1.626947e+00 |
| role:k | 8 | -290.156469 | -188.702362 | 0.833127 | 479.691958 | 2.387937e+00 | 1.430816e+00 |
| role:o | 8 | -197.563623 | -155.622072 | 0.687667 | 353.873362 | 2.098863e+00 | 1.661822e+00 |
| role:proj | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |
| role:q | 8 | -301.219343 | -153.829758 | 0.820863 | 455.869964 | 2.395559e+00 | 1.391844e+00 |
| role:v | 8 | -187.560970 | -100.073911 | 0.821193 | 288.456074 | 2.384123e+00 | 1.374644e+00 |
| shape:hidden_to_hidden | 16 | -249.391483 | -154.725915 | 0.754265 | 404.871663 | 2.247211e+00 | 1.526833e+00 |
| shape:hidden_to_wide | 32 | -214.530461 | -148.805407 | 0.695110 | 364.030978 | 2.260985e+00 | 1.522390e+00 |
| shape:wide_to_hidden | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
