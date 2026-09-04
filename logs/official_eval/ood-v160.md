# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `ood-generalization-panel` / `overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `33b1d061ce6bfcd92659c597be4830bb9b910e646ff518433da67b925ae8680d`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.617209512 |
| Attention mean | 0.720900254 |
| Overall mean (all captured cases) | 0.660413988 |
| Linear role macro mean | 0.617209512 |
| Attention layer macro mean | 0.720900254 |
| Candidate wall | 323.844s |
| Candidate API total | 295.521s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | code | 50 | 0.630155 | 0.674483 | 0.411677 | 49/1/0 |
| linear | news | 68 | 0.631681 | 0.623128 | 0.427651 | 68/0/0 |
| linear | zh | 50 | 0.584583 | 0.548054 | 0.379698 | 48/2/0 |
| linear | **overall** | 168 | 0.617210 | 0.620827 | 0.405281 | 165/3/0 |
| attention | code | 40 | 0.715054 | 0.735601 | 0.533125 | 40/0/0 |
| attention | news | 40 | 0.725488 | 0.742598 | 0.532850 | 40/0/0 |
| attention | zh | 40 | 0.722159 | 0.744353 | 0.544525 | 40/0/0 |
| attention | **overall** | 120 | 0.720900 | 0.741441 | 0.536474 | 120/0/0 |

OOD 均值不参与 proxy 排名；候选是否过拟合看 `gain_in − gain_ood`（与同 solution 的 in-dist proxy-v2 运行相减）。

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.617210 | 0.620827 | 0.509938 | 0.405281 | -0.069320 | 165/3/0 | 0.379173 |
| family:fc | 48 | 0.530993 | 0.530046 | 0.459934 | 0.418882 | 0.380294 | 48/0/0 | 0.469954 |
| family:o | 24 | 0.468418 | 0.481161 | 0.371144 | 0.222305 | -0.010608 | 22/2/0 | 0.518839 |
| family:proj | 24 | 0.511638 | 0.523488 | 0.501647 | 0.375298 | -0.069320 | 23/1/0 | 0.476512 |
| family:qkv | 72 | 0.759475 | 0.757036 | 0.716904 | 0.685012 | 0.643514 | 72/0/0 | 0.242964 |
| role:fc_gate | 24 | 0.561311 | 0.564569 | 0.511905 | 0.448511 | 0.415717 | 24/0/0 | 0.435431 |
| role:fc_up | 24 | 0.500674 | 0.512133 | 0.425623 | 0.402837 | 0.380294 | 24/0/0 | 0.487867 |
| role:k | 24 | 0.771332 | 0.767184 | 0.728157 | 0.700831 | 0.672515 | 24/0/0 | 0.232816 |
| role:o | 24 | 0.468418 | 0.481161 | 0.371144 | 0.222305 | -0.010608 | 22/2/0 | 0.518839 |
| role:proj | 24 | 0.511638 | 0.523488 | 0.501647 | 0.375298 | -0.069320 | 23/1/0 | 0.476512 |
| role:q | 24 | 0.745666 | 0.743825 | 0.689640 | 0.663172 | 0.643514 | 24/0/0 | 0.256175 |
| role:v | 24 | 0.761428 | 0.762615 | 0.731598 | 0.699229 | 0.667122 | 24/0/0 | 0.237385 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -392.560229 | -132.144045 | 0.617210 | 525.321484 | 2.193137e+00 | 1.541002e+00 |
| family:fc | 48 | -166.117236 | -136.368824 | 0.530993 | 303.017053 | 2.086877e+00 | 1.577266e+00 |
| family:o | 24 | -189.417179 | -114.250073 | 0.468418 | 304.135670 | 2.179465e+00 | 1.619258e+00 |
| family:proj | 24 | -1541.156802 | -156.806474 | 0.511638 | 1698.474913 | 1.860912e+00 | 1.685521e+00 |
| family:qkv | 72 | -228.371051 | -127.071373 | 0.759475 | 356.201899 | 2.379277e+00 | 1.442568e+00 |
| role:fc_gate | 24 | -200.632205 | -178.687978 | 0.561311 | 379.881495 | 2.162678e+00 | 1.627537e+00 |
| role:fc_up | 24 | -131.602267 | -94.049671 | 0.500674 | 226.152611 | 2.011075e+00 | 1.526996e+00 |
| role:k | 24 | -288.815358 | -169.783443 | 0.771332 | 459.370133 | 2.534959e+00 | 1.463523e+00 |
| role:o | 24 | -189.417179 | -114.250073 | 0.468418 | 304.135670 | 2.179465e+00 | 1.619258e+00 |
| role:proj | 24 | -1541.156802 | -156.806474 | 0.511638 | 1698.474913 | 1.860912e+00 | 1.685521e+00 |
| role:q | 24 | -239.219131 | -119.452791 | 0.745666 | 359.417588 | 2.338834e+00 | 1.438648e+00 |
| role:v | 24 | -157.078664 | -91.977884 | 0.761428 | 249.817976 | 2.264040e+00 | 1.425533e+00 |
| shape:hidden_to_hidden | 48 | -214.318155 | -116.851432 | 0.607042 | 331.776629 | 2.259150e+00 | 1.528953e+00 |
| shape:hidden_to_wide | 96 | -194.532124 | -133.624744 | 0.648686 | 328.805554 | 2.243188e+00 | 1.510897e+00 |
| shape:wide_to_hidden | 24 | -1541.156802 | -156.806474 | 0.511638 | 1698.474913 | 1.860912e+00 | 1.685521e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 120 | -18.650151 | -39.520617 | -0.005955 | 0.715507 | 0.720900 | 58.886275 | 0.011348 |
| layer:0 | 15 | -0.645524 | -0.961372 | -0.001185 | 0.866381 | 0.867971 | 2.473278 | 0.002775 |
| layer:10 | 15 | -15.926426 | -29.022728 | -0.008184 | 0.714904 | 0.721489 | 45.664058 | 0.014768 |
| layer:13 | 15 | -21.213857 | -24.960457 | -0.015110 | 0.739826 | 0.749751 | 46.914140 | 0.025035 |
| layer:16 | 15 | -29.482245 | -16.766394 | 0.002177 | 0.515877 | 0.517968 | 46.764516 | -0.000086 |
| layer:20 | 15 | -30.436587 | -42.361164 | -0.006505 | 0.615063 | 0.617502 | 73.412814 | 0.008944 |
| layer:23 | 15 | -19.150570 | -167.485880 | 0.012902 | 0.609206 | 0.624714 | 187.245656 | 0.002605 |
| layer:3 | 15 | -17.566128 | -14.515651 | -0.026136 | 0.843233 | 0.845243 | 32.925012 | 0.028147 |
| layer:7 | 15 | -14.779867 | -20.091290 | -0.005595 | 0.819565 | 0.822565 | 35.690722 | 0.008595 |
| length:10 | 24 | -30.304597 | -55.970069 | -0.007653 | 0.712869 | 0.722720 | 86.987534 | 0.017504 |
| length:1024 | 48 | -14.276175 | -34.772251 | -0.005771 | 0.714506 | 0.717872 | 49.762932 | 0.009137 |
| length:128 | 24 | -18.483205 | -35.899711 | -0.004984 | 0.719793 | 0.726104 | 55.102710 | 0.011295 |
| length:512 | 24 | -15.910601 | -36.188802 | -0.005594 | 0.715861 | 0.719934 | 52.815264 | 0.009667 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550922e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.177689e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 2.934905e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
