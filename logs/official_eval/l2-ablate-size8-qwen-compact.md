# ablate_size8 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `a8f36b90c955a24cd5b0c8a000fa6fc3704931b8cfb945dc7717bf50ef01c1bd`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.701647769 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.701647769 |
| Linear role macro mean | 0.701647769 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 49.467s |
| Candidate API total | 44.389s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.701648 | 0.685426 | 0.581479 | 0.534848 | 0.420862 | 56/0/0 | 0.314574 |
| family:fc | 16 | 0.561927 | 0.558070 | 0.536169 | 0.495830 | 0.420862 | 16/0/0 | 0.441930 |
| family:o | 8 | 0.668634 | 0.639062 | 0.590721 | 0.556414 | 0.544354 | 8/0/0 | 0.360938 |
| family:proj | 8 | 0.651222 | 0.609458 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.390542 |
| family:qkv | 24 | 0.822608 | 0.827681 | 0.766286 | 0.740429 | 0.707392 | 24/0/0 | 0.172319 |
| role:fc_gate | 8 | 0.591684 | 0.585536 | 0.559141 | 0.546520 | 0.538197 | 8/0/0 | 0.414464 |
| role:fc_up | 8 | 0.532169 | 0.540037 | 0.520363 | 0.464348 | 0.420862 | 8/0/0 | 0.459963 |
| role:k | 8 | 0.829713 | 0.830941 | 0.786902 | 0.751131 | 0.746451 | 8/0/0 | 0.169059 |
| role:o | 8 | 0.668634 | 0.639062 | 0.590721 | 0.556414 | 0.544354 | 8/0/0 | 0.360938 |
| role:proj | 8 | 0.651222 | 0.609458 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.390542 |
| role:q | 8 | 0.821677 | 0.819506 | 0.761582 | 0.724036 | 0.707392 | 8/0/0 | 0.180494 |
| role:v | 8 | 0.816434 | 0.821762 | 0.785936 | 0.746120 | 0.732744 | 8/0/0 | 0.178238 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.017628`、max `0.164689`；成对 minimum-gain median `0.684904`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`ablate_size8`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.003860 | 0.000000 | 1/9/46 | 1.000000 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -225.260443 | -150.266037 | 0.701648 | 376.228127 | 2.245026e+00 | 1.542459e+00 |
| family:fc | 16 | -186.289049 | -151.016756 | 0.561927 | 337.867732 | 2.141425e+00 | 1.634810e+00 |
| family:o | 8 | -198.866406 | -152.268225 | 0.668634 | 351.803266 | 2.118155e+00 | 1.673459e+00 |
| family:proj | 8 | -228.040060 | -154.600192 | 0.651222 | 383.291474 | 2.137810e+00 | 1.630026e+00 |
| family:qkv | 24 | -259.112845 | -147.653442 | 0.822608 | 407.588895 | 2.392122e+00 | 1.408036e+00 |
| role:fc_gate | 8 | -227.460891 | -193.068755 | 0.591684 | 421.121330 | 2.184246e+00 | 1.642837e+00 |
| role:fc_up | 8 | -145.117208 | -108.964757 | 0.532169 | 254.614134 | 2.098605e+00 | 1.626783e+00 |
| role:k | 8 | -289.992354 | -188.542112 | 0.829713 | 479.364179 | 2.386521e+00 | 1.431389e+00 |
| role:o | 8 | -198.866406 | -152.268225 | 0.668634 | 351.803266 | 2.118155e+00 | 1.673459e+00 |
| role:proj | 8 | -228.040060 | -154.600192 | 0.651222 | 383.291474 | 2.137810e+00 | 1.630026e+00 |
| role:q | 8 | -300.376428 | -153.801534 | 0.821677 | 454.999639 | 2.395202e+00 | 1.391875e+00 |
| role:v | 8 | -186.969751 | -100.616681 | 0.816434 | 288.402867 | 2.394643e+00 | 1.400843e+00 |
| shape:hidden_to_hidden | 16 | -249.621417 | -153.034879 | 0.745156 | 403.401452 | 2.256679e+00 | 1.532667e+00 |
| shape:hidden_to_wide | 32 | -212.385051 | -147.798076 | 0.692500 | 360.875628 | 2.266004e+00 | 1.525463e+00 |
| shape:wide_to_hidden | 8 | -228.040060 | -154.600192 | 0.651222 | 383.291474 | 2.137810e+00 | 1.630026e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。

## Decision: REJECTED (2026-09-03)

- Qwen compact 56 cases: mean Δgain -3.9e-3, 1 positive / 9 negative / 46 zero.
- o role is 4/8 negative with mean -1.95e-2; worst case o -0.0803, v -0.0204.
- Block size 4 carries a large selected gain on o; removing it is a serious
  regression. API 44.39s vs parent 47.1s does not justify. Reverted.
