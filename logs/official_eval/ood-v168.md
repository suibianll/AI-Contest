# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `ood-generalization-panel` / `overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `5988ae47eac2e7dde7488e06b8f91939f5660a585034280a6d68a8fb6701ac79`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.000000000 |
| Attention mean | 0.719469828 |
| Overall mean (all captured cases) | 0.299779095 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.719469828 |
| Candidate wall | 104.227s |
| Candidate API total | 75.156s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | code | 50 | 0.000000 | 0.000000 | 0.000000 | 0/0/50 |
| linear | news | 68 | 0.000000 | 0.000000 | 0.000000 | 0/0/68 |
| linear | zh | 50 | 0.000000 | 0.000000 | 0.000000 | 0/0/50 |
| linear | **overall** | 168 | 0.000000 | 0.000000 | 0.000000 | 0/0/168 |
| attention | code | 40 | 0.711406 | 0.733901 | 0.522894 | 40/0/0 |
| attention | news | 40 | 0.723720 | 0.743407 | 0.527996 | 40/0/0 |
| attention | zh | 40 | 0.723283 | 0.743262 | 0.550542 | 40/0/0 |
| attention | **overall** | 120 | 0.719470 | 0.740744 | 0.533553 | 120/0/0 |

OOD 均值不参与 proxy 排名；候选是否过拟合看 `gain_in − gain_ood`（与同 solution 的 in-dist proxy-v2 运行相减）。

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/168 | 1.000000 |
| family:fc | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/48 | 1.000000 |
| family:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| family:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/72 | 1.000000 |
| role:fc_gate | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:fc_up | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:k | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:q | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:v | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.835814e-03 | 8.673785e-03 |
| family:fc | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.726051e-03 | 8.627197e-03 |
| family:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.679961e-03 | 8.207920e-03 |
| family:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657928e-03 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.951741e-03 | 9.198751e-03 |
| role:fc_gate | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.629023e-03 | 8.536047e-03 |
| role:fc_up | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.823080e-03 | 8.718346e-03 |
| role:k | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.999664e-03 | 9.224184e-03 |
| role:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.679961e-03 | 8.207920e-03 |
| role:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657928e-03 |
| role:q | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.885952e-03 | 9.172487e-03 |
| role:v | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.969608e-03 | 9.199584e-03 |
| shape:hidden_to_hidden | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.782956e-03 | 8.690203e-03 |
| shape:hidden_to_wide | 96 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.855344e-03 | 8.919540e-03 |
| shape:wide_to_hidden | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657928e-03 |

Linear overall interpretation: `mixed_or_neutral`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 120 | -18.608303 | -39.568673 | -0.005955 | 0.714311 | 0.719470 | 58.891286 | 0.011113 |
| layer:0 | 15 | -0.647197 | -0.961403 | -0.001185 | 0.866324 | 0.867831 | 2.474924 | 0.002692 |
| layer:10 | 15 | -15.937017 | -29.044706 | -0.008184 | 0.713270 | 0.719716 | 45.694993 | 0.014630 |
| layer:13 | 15 | -21.208915 | -24.951187 | -0.015110 | 0.740440 | 0.750218 | 46.900542 | 0.024888 |
| layer:16 | 15 | -29.154540 | -16.784805 | 0.002177 | 0.513172 | 0.514103 | 46.452516 | -0.001246 |
| layer:20 | 15 | -30.424284 | -42.470412 | -0.006505 | 0.610151 | 0.612370 | 73.504846 | 0.008725 |
| layer:23 | 15 | -19.151645 | -167.753998 | 0.012902 | 0.609707 | 0.624975 | 187.515350 | 0.002366 |
| layer:3 | 15 | -17.567348 | -14.515003 | -0.026136 | 0.843062 | 0.845104 | 32.925412 | 0.028178 |
| layer:7 | 15 | -14.775477 | -20.067868 | -0.005595 | 0.818362 | 0.821441 | 35.661708 | 0.008675 |
| length:10 | 24 | -30.173909 | -55.993344 | -0.007653 | 0.717779 | 0.727034 | 86.885032 | 0.016908 |
| length:1024 | 48 | -14.254576 | -34.799585 | -0.005771 | 0.712702 | 0.715998 | 49.766862 | 0.009067 |
| length:128 | 24 | -18.466702 | -35.888835 | -0.004984 | 0.715189 | 0.721387 | 55.070726 | 0.011183 |
| length:512 | 24 | -15.891752 | -36.362016 | -0.005594 | 0.713183 | 0.716932 | 52.966951 | 0.009342 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550911e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.182128e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 2.949183e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
