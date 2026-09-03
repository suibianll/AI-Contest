# householder-x-qwen-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `ced55d60c8f0fe8e1f1f4a03b94712e9abfc1b89ad134adb1430ca4dff11136f`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.699896272 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.699896272 |
| Linear role macro mean | 0.699896272 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 52.665s |
| Candidate API total | 47.478s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.699896 | 0.686869 | 0.582373 | 0.532854 | 0.409588 | 56/0/0 | 0.313131 |
| family:fc | 16 | 0.555753 | 0.550142 | 0.537755 | 0.490249 | 0.409588 | 16/0/0 | 0.449858 |
| family:o | 8 | 0.688023 | 0.639763 | 0.621201 | 0.606899 | 0.597043 | 8/0/0 | 0.360237 |
| family:proj | 8 | 0.647360 | 0.607915 | 0.572114 | 0.523678 | 0.522807 | 8/0/0 | 0.392085 |
| family:qkv | 24 | 0.817462 | 0.816654 | 0.763730 | 0.738618 | 0.701080 | 24/0/0 | 0.183346 |
| role:fc_gate | 8 | 0.585277 | 0.571451 | 0.550638 | 0.545851 | 0.542551 | 8/0/0 | 0.428549 |
| role:fc_up | 8 | 0.526228 | 0.533304 | 0.517815 | 0.458012 | 0.409588 | 8/0/0 | 0.466696 |
| role:k | 8 | 0.824350 | 0.821924 | 0.778972 | 0.745482 | 0.743465 | 8/0/0 | 0.178076 |
| role:o | 8 | 0.688023 | 0.639763 | 0.621201 | 0.606899 | 0.597043 | 8/0/0 | 0.360237 |
| role:proj | 8 | 0.647360 | 0.607915 | 0.572114 | 0.523678 | 0.522807 | 8/0/0 | 0.392085 |
| role:q | 8 | 0.815856 | 0.813836 | 0.752820 | 0.717230 | 0.701080 | 8/0/0 | 0.186164 |
| role:v | 8 | 0.812180 | 0.808405 | 0.777717 | 0.756095 | 0.746982 | 8/0/0 | 0.191595 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.018115`、max `0.171000`；成对 minimum-gain median `0.677553`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -213.590690 | -150.449429 | 0.699896 | 364.740016 | 2.235542e+00 | 1.536795e+00 |
| family:fc | 16 | -187.201197 | -154.748829 | 0.555753 | 342.505778 | 2.133442e+00 | 1.649397e+00 |
| family:o | 8 | -206.006210 | -154.986315 | 0.688023 | 361.680548 | 2.098673e+00 | 1.664941e+00 |
| family:proj | 8 | -208.018487 | -153.589633 | 0.647360 | 362.255481 | 2.136777e+00 | 1.619021e+00 |
| family:qkv | 24 | -235.569246 | -145.024134 | 0.817462 | 381.410841 | 2.382152e+00 | 1.391603e+00 |
| role:fc_gate | 8 | -232.377466 | -200.126911 | 0.585277 | 433.089655 | 2.169133e+00 | 1.662731e+00 |
| role:fc_up | 8 | -142.024927 | -109.370747 | 0.526228 | 251.921901 | 2.097752e+00 | 1.636062e+00 |
| role:k | 8 | -252.511646 | -181.197254 | 0.824350 | 434.533251 | 2.378191e+00 | 1.418854e+00 |
| role:o | 8 | -206.006210 | -154.986315 | 0.688023 | 361.680548 | 2.098673e+00 | 1.664941e+00 |
| role:proj | 8 | -208.018487 | -153.589633 | 0.647360 | 362.255481 | 2.136777e+00 | 1.619021e+00 |
| role:q | 8 | -279.733616 | -153.535294 | 0.815856 | 434.084766 | 2.389743e+00 | 1.380379e+00 |
| role:v | 8 | -174.462476 | -100.339852 | 0.812180 | 275.614508 | 2.378522e+00 | 1.375575e+00 |
| shape:hidden_to_hidden | 16 | -242.869913 | -154.260805 | 0.751939 | 397.882657 | 2.244208e+00 | 1.522660e+00 |
| shape:hidden_to_wide | 32 | -200.344129 | -147.758691 | 0.687009 | 348.789829 | 2.255900e+00 | 1.523306e+00 |
| shape:wide_to_hidden | 8 | -208.018487 | -153.589633 | 0.647360 | 362.255481 | 2.136777e+00 | 1.619021e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
