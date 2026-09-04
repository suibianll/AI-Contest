# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `ood-generalization-panel` / `overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.419648106 |
| Attention mean | 0.719134713 |
| Overall mean (all captured cases) | 0.544434192 |
| Linear role macro mean | 0.419648106 |
| Attention layer macro mean | 0.719134713 |
| Candidate wall | 328.538s |
| Candidate API total | 301.986s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | code | 50 | 0.445222 | 0.506838 | 0.145232 | 47/3/0 |
| linear | news | 68 | 0.435929 | 0.478217 | 0.107598 | 64/4/0 |
| linear | zh | 50 | 0.371932 | 0.410625 | 0.028480 | 45/5/0 |
| linear | **overall** | 168 | 0.419648 | 0.458571 | 0.088307 | 156/12/0 |
| attention | code | 40 | 0.714027 | 0.735601 | 0.533125 | 40/0/0 |
| attention | news | 40 | 0.723290 | 0.742598 | 0.532850 | 40/0/0 |
| attention | zh | 40 | 0.720087 | 0.744353 | 0.544525 | 40/0/0 |
| attention | **overall** | 120 | 0.719135 | 0.741441 | 0.536474 | 120/0/0 |

OOD 均值不参与 proxy 排名；候选是否过拟合看 `gain_in − gain_ood`（与同 solution 的 in-dist proxy-v2 运行相减）。

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.419648 | 0.458571 | 0.307772 | 0.088307 | -0.488905 | 156/12/0 | 0.541429 |
| family:fc | 48 | 0.390136 | 0.395510 | 0.332808 | 0.289828 | 0.231105 | 48/0/0 | 0.604490 |
| family:o | 24 | 0.310544 | 0.263448 | 0.209798 | 0.032077 | -0.421691 | 23/1/0 | 0.736552 |
| family:proj | 24 | 0.051527 | 0.039138 | -0.117950 | -0.313558 | -0.488905 | 13/11/0 | 0.960862 |
| family:qkv | 72 | 0.598398 | 0.600796 | 0.538676 | 0.505204 | 0.450720 | 72/0/0 | 0.399204 |
| role:fc_gate | 24 | 0.382797 | 0.392888 | 0.325811 | 0.273499 | 0.231105 | 24/0/0 | 0.607112 |
| role:fc_up | 24 | 0.397475 | 0.395510 | 0.353089 | 0.309105 | 0.283533 | 24/0/0 | 0.604490 |
| role:k | 24 | 0.597648 | 0.599531 | 0.563312 | 0.516755 | 0.498106 | 24/0/0 | 0.400469 |
| role:o | 24 | 0.310544 | 0.263448 | 0.209798 | 0.032077 | -0.421691 | 23/1/0 | 0.736552 |
| role:proj | 24 | 0.051527 | 0.039138 | -0.117950 | -0.313558 | -0.488905 | 13/11/0 | 0.960862 |
| role:q | 24 | 0.580847 | 0.554503 | 0.535128 | 0.485621 | 0.450720 | 24/0/0 | 0.445497 |
| role:v | 24 | 0.616699 | 0.611711 | 0.576847 | 0.527407 | 0.497966 | 24/0/0 | 0.388289 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -171.439995 | -34.710889 | 0.419648 | 206.570533 | 4.112577e-01 | 4.508950e-01 |
| family:fc | 48 | -112.740204 | -24.539489 | 0.390136 | 137.669829 | 2.311660e-01 | 3.014555e-01 |
| family:o | 24 | -209.393522 | -46.039843 | 0.310544 | 255.743909 | 8.764757e-01 | 7.023482e-01 |
| family:proj | 24 | -368.566187 | -95.105661 | 0.051527 | 463.723375 | 1.021495e+00 | 1.003154e+00 |
| family:qkv | 72 | -132.213283 | -17.583915 | 0.598398 | 150.395596 | 1.728337e-01 | 2.826175e-01 |
| role:fc_gate | 24 | -166.480144 | -27.763991 | 0.382797 | 194.626932 | 7.073658e-02 | 1.592514e-01 |
| role:fc_up | 24 | -59.000264 | -21.314987 | 0.397475 | 80.712726 | 3.915954e-01 | 4.436595e-01 |
| role:k | 24 | -138.479390 | -23.398160 | 0.597648 | 162.475198 | 1.767590e-01 | 2.228733e-01 |
| role:o | 24 | -209.393522 | -46.039843 | 0.310544 | 255.743909 | 8.764757e-01 | 7.023482e-01 |
| role:proj | 24 | -368.566187 | -95.105661 | 0.051527 | 463.723375 | 1.021495e+00 | 1.003154e+00 |
| role:q | 24 | -150.380425 | -18.947707 | 0.580847 | 169.908979 | 1.969030e-01 | 2.901211e-01 |
| role:v | 24 | -107.780035 | -10.405876 | 0.616699 | 118.802610 | 1.448392e-01 | 3.348579e-01 |
| shape:hidden_to_hidden | 48 | -179.886973 | -32.493775 | 0.445696 | 212.826444 | 5.366893e-01 | 4.962347e-01 |
| shape:hidden_to_wide | 96 | -117.934958 | -20.720754 | 0.498655 | 139.154367 | 1.959826e-01 | 2.901606e-01 |
| shape:wide_to_hidden | 24 | -368.566187 | -95.105661 | 0.051527 | 463.723375 | 1.021495e+00 | 1.003154e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 120 | -18.713558 | -39.511154 | -0.005955 | 0.713798 | 0.719135 | 58.938510 | 0.011291 |
| layer:0 | 15 | -0.645524 | -0.961372 | -0.001185 | 0.866381 | 0.867971 | 2.473278 | 0.002775 |
| layer:10 | 15 | -15.926426 | -29.022728 | -0.008184 | 0.714904 | 0.721489 | 45.664058 | 0.014768 |
| layer:13 | 15 | -21.213857 | -24.960457 | -0.015110 | 0.739826 | 0.749751 | 46.914140 | 0.025035 |
| layer:16 | 15 | -29.482245 | -16.766394 | 0.002177 | 0.515877 | 0.517968 | 46.764516 | -0.000086 |
| layer:20 | 15 | -30.436587 | -42.361164 | -0.006505 | 0.615063 | 0.617502 | 73.412814 | 0.008944 |
| layer:23 | 15 | -19.150570 | -167.485880 | 0.012902 | 0.609206 | 0.624714 | 187.245656 | 0.002605 |
| layer:3 | 15 | -17.566128 | -14.515651 | -0.026136 | 0.843233 | 0.845243 | 32.925012 | 0.028147 |
| layer:7 | 15 | -15.287122 | -20.015589 | -0.005595 | 0.805895 | 0.808440 | 36.108607 | 0.008140 |
| length:10 | 24 | -30.372689 | -56.159081 | -0.007653 | 0.711699 | 0.721529 | 87.243468 | 0.017484 |
| length:1024 | 48 | -14.332082 | -34.686875 | -0.005771 | 0.712978 | 0.716350 | 49.731935 | 0.009143 |
| length:128 | 24 | -18.555328 | -35.871517 | -0.004984 | 0.717477 | 0.723427 | 55.144322 | 0.010934 |
| length:512 | 24 | -15.975607 | -36.151425 | -0.005594 | 0.713858 | 0.718018 | 52.840891 | 0.009753 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550917e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.188689e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 2.946390e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
