# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.556323994 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.556323994 |
| Linear role macro mean | 0.556323994 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 199.943s |
| Candidate API total | 193.124s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.556324 | 0.528262 | 0.450215 | 0.377804 | 0.314129 | 56/0/0 | 0.471738 |
| family:fc | 16 | 0.454386 | 0.456486 | 0.434872 | 0.403318 | 0.331451 | 16/0/0 | 0.543514 |
| family:o | 8 | 0.424503 | 0.362170 | 0.328863 | 0.320275 | 0.314129 | 8/0/0 | 0.637830 |
| family:proj | 8 | 0.526917 | 0.497440 | 0.437961 | 0.360309 | 0.358315 | 8/0/0 | 0.502560 |
| family:qkv | 24 | 0.678025 | 0.648782 | 0.612535 | 0.583461 | 0.538057 | 24/0/0 | 0.351218 |
| role:fc_gate | 8 | 0.461031 | 0.449176 | 0.434109 | 0.423819 | 0.413756 | 8/0/0 | 0.550824 |
| role:fc_up | 8 | 0.447741 | 0.459760 | 0.448695 | 0.383276 | 0.331451 | 8/0/0 | 0.540240 |
| role:k | 8 | 0.697702 | 0.683886 | 0.647966 | 0.601384 | 0.594046 | 8/0/0 | 0.316114 |
| role:o | 8 | 0.424503 | 0.362170 | 0.328863 | 0.320275 | 0.314129 | 8/0/0 | 0.637830 |
| role:proj | 8 | 0.526917 | 0.497440 | 0.437961 | 0.360309 | 0.358315 | 8/0/0 | 0.502560 |
| role:q | 8 | 0.684581 | 0.688578 | 0.605070 | 0.558461 | 0.538057 | 8/0/0 | 0.311422 |
| role:v | 8 | 0.651792 | 0.625051 | 0.617177 | 0.590539 | 0.572902 | 8/0/0 | 0.374949 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.018402`、max `0.149223`；成对 minimum-gain median `0.519957`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -232.803534 | -43.972270 | 0.556324 | 277.332128 | 5.002393e-01 | 5.189846e-01 |
| family:fc | 16 | -96.670932 | -26.609750 | 0.454386 | 123.735067 | 3.192482e-01 | 3.595230e-01 |
| family:o | 8 | -180.564692 | -59.656822 | 0.424503 | 240.646017 | 6.373244e-01 | 5.041610e-01 |
| family:proj | 8 | -689.108564 | -128.885631 | 0.526917 | 818.521112 | 1.655222e+00 | 1.476616e+00 |
| family:qkv | 24 | -188.869872 | -22.014646 | 0.678025 | 211.562544 | 1.902110e-01 | 3.110231e-01 |
| role:fc_gate | 8 | -115.031357 | -24.066401 | 0.461031 | 139.558788 | 6.774676e-02 | 1.417086e-01 |
| role:fc_up | 8 | -78.310507 | -29.153099 | 0.447741 | 107.911346 | 5.707496e-01 | 5.773373e-01 |
| role:k | 8 | -253.307157 | -28.528132 | 0.697702 | 282.532992 | 1.794806e-01 | 2.861413e-01 |
| role:o | 8 | -180.564692 | -59.656822 | 0.424503 | 240.646017 | 6.373244e-01 | 5.041610e-01 |
| role:proj | 8 | -689.108564 | -128.885631 | 0.526917 | 818.521112 | 1.655222e+00 | 1.476616e+00 |
| role:q | 8 | -158.716620 | -21.357847 | 0.684581 | 180.759049 | 1.983038e-01 | 3.248553e-01 |
| role:v | 8 | -154.585839 | -16.157960 | 0.651792 | 171.395591 | 1.928485e-01 | 3.220726e-01 |
| shape:hidden_to_hidden | 16 | -169.640656 | -40.507335 | 0.554542 | 210.702533 | 4.178141e-01 | 4.145082e-01 |
| shape:hidden_to_wide | 32 | -150.308715 | -24.476398 | 0.564567 | 175.349680 | 2.527064e-01 | 3.318150e-01 |
| shape:wide_to_hidden | 8 | -689.108564 | -128.885631 | 0.526917 | 818.521112 | 1.655222e+00 | 1.476616e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
