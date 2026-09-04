# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `ood-generalization-panel` / `overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `4469b85b53f5adefc6cfe4fbf136bdd4d7ff9ffc48a815592c95864a7287a844`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.000000000 |
| Attention mean | 0.719198660 |
| Overall mean (all captured cases) | 0.299666108 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.719198660 |
| Candidate wall | 104.348s |
| Candidate API total | 76.304s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | code | 50 | 0.000000 | 0.000000 | 0.000000 | 0/0/50 |
| linear | news | 68 | 0.000000 | 0.000000 | 0.000000 | 0/0/68 |
| linear | zh | 50 | 0.000000 | 0.000000 | 0.000000 | 0/0/50 |
| linear | **overall** | 168 | 0.000000 | 0.000000 | 0.000000 | 0/0/168 |
| attention | code | 40 | 0.711301 | 0.730818 | 0.526406 | 40/0/0 |
| attention | news | 40 | 0.723088 | 0.737972 | 0.528072 | 40/0/0 |
| attention | zh | 40 | 0.723207 | 0.742762 | 0.548428 | 40/0/0 |
| attention | **overall** | 120 | 0.719199 | 0.735148 | 0.533954 | 120/0/0 |

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
| overall | 120 | -18.618764 | -39.678870 | -0.003060 | 0.713415 | 0.719199 | 59.011048 | 0.008844 |
| layer:0 | 15 | -0.648547 | -0.942027 | 0.000027 | 0.866810 | 0.868506 | 2.457384 | 0.001668 |
| layer:10 | 15 | -15.944324 | -29.037636 | -0.004452 | 0.713063 | 0.718855 | 45.695023 | 0.010244 |
| layer:13 | 15 | -21.208831 | -25.019567 | -0.011312 | 0.739765 | 0.749832 | 46.968163 | 0.021379 |
| layer:16 | 15 | -29.187550 | -16.773204 | 0.005650 | 0.510803 | 0.513182 | 46.471557 | -0.003270 |
| layer:20 | 15 | -30.432870 | -42.469221 | -0.001985 | 0.615681 | 0.619532 | 73.517772 | 0.005836 |
| layer:23 | 15 | -19.170455 | -168.620018 | 0.012530 | 0.602471 | 0.619076 | 188.392944 | 0.004075 |
| layer:3 | 15 | -17.582000 | -14.518449 | -0.022474 | 0.839685 | 0.841151 | 32.940134 | 0.023940 |
| layer:7 | 15 | -14.775532 | -20.050840 | -0.002464 | 0.819038 | 0.823456 | 35.645410 | 0.006882 |
| length:10 | 24 | -30.201434 | -56.431483 | -0.004353 | 0.720889 | 0.731003 | 87.353806 | 0.014467 |
| length:1024 | 48 | -14.260219 | -34.859789 | -0.003172 | 0.709583 | 0.713536 | 49.829592 | 0.007125 |
| length:128 | 24 | -18.471083 | -35.878404 | -0.001811 | 0.715715 | 0.722314 | 55.065202 | 0.008410 |
| length:512 | 24 | -15.900862 | -36.364886 | -0.002793 | 0.711302 | 0.715604 | 52.977050 | 0.007095 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550905e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.168548e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 2.917478e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
