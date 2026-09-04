# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `ood-generalization-panel` / `overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `f3e39e993a436e217cb4811525c81239f82a6ec58845a0646e183a824c33a438`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.620706130 |
| Attention mean | 0.721238578 |
| Overall mean (all captured cases) | 0.662594650 |
| Linear role macro mean | 0.620706130 |
| Attention layer macro mean | 0.721238578 |
| Candidate wall | 441.736s |
| Candidate API total | 413.400s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | code | 50 | 0.629836 | 0.681942 | 0.413607 | 49/1/0 |
| linear | news | 68 | 0.631807 | 0.622882 | 0.427753 | 68/0/0 |
| linear | zh | 50 | 0.596479 | 0.548550 | 0.425032 | 50/0/0 |
| linear | **overall** | 168 | 0.620706 | 0.620416 | 0.419926 | 167/1/0 |
| attention | code | 40 | 0.711659 | 0.733797 | 0.523692 | 40/0/0 |
| attention | news | 40 | 0.728499 | 0.744960 | 0.545776 | 40/0/0 |
| attention | zh | 40 | 0.723557 | 0.743679 | 0.549104 | 40/0/0 |
| attention | **overall** | 120 | 0.721239 | 0.741542 | 0.539414 | 120/0/0 |

OOD 均值不参与 proxy 排名；候选是否过拟合看 `gain_in − gain_ood`（与同 solution 的 in-dist proxy-v2 运行相减）。

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.620706 | 0.620416 | 0.514129 | 0.419926 | -0.025515 | 167/1/0 | 0.379584 |
| family:fc | 48 | 0.531023 | 0.529234 | 0.460334 | 0.417309 | 0.379836 | 48/0/0 | 0.470766 |
| family:o | 24 | 0.472623 | 0.497447 | 0.374076 | 0.226292 | -0.025515 | 23/1/0 | 0.502553 |
| family:proj | 24 | 0.532639 | 0.523440 | 0.499161 | 0.463546 | 0.335563 | 24/0/0 | 0.476560 |
| family:qkv | 72 | 0.759212 | 0.756596 | 0.716918 | 0.684896 | 0.644622 | 72/0/0 | 0.243404 |
| role:fc_gate | 24 | 0.562003 | 0.565562 | 0.514742 | 0.450626 | 0.413820 | 24/0/0 | 0.434438 |
| role:fc_up | 24 | 0.500043 | 0.512303 | 0.426794 | 0.401504 | 0.379836 | 24/0/0 | 0.487697 |
| role:k | 24 | 0.771165 | 0.770015 | 0.728959 | 0.701524 | 0.682095 | 24/0/0 | 0.229985 |
| role:o | 24 | 0.472623 | 0.497447 | 0.374076 | 0.226292 | -0.025515 | 23/1/0 | 0.502553 |
| role:proj | 24 | 0.532639 | 0.523440 | 0.499161 | 0.463546 | 0.335563 | 24/0/0 | 0.476560 |
| role:q | 24 | 0.745117 | 0.743719 | 0.691356 | 0.663358 | 0.644622 | 24/0/0 | 0.256281 |
| role:v | 24 | 0.761353 | 0.761983 | 0.733912 | 0.699033 | 0.654766 | 24/0/0 | 0.238017 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -395.023925 | -132.159025 | 0.620706 | 527.803657 | 2.193338e+00 | 1.540890e+00 |
| family:fc | 48 | -166.212604 | -136.331211 | 0.531023 | 303.074838 | 2.087120e+00 | 1.576348e+00 |
| family:o | 24 | -189.801092 | -114.241173 | 0.472623 | 304.514889 | 2.179579e+00 | 1.619688e+00 |
| family:proj | 24 | -1557.399703 | -156.830189 | 0.532639 | 1714.762531 | 1.860782e+00 | 1.685635e+00 |
| family:qkv | 72 | -228.513825 | -127.126464 | 0.759212 | 356.399501 | 2.379589e+00 | 1.442736e+00 |
| role:fc_gate | 24 | -200.797070 | -178.629579 | 0.562003 | 379.988652 | 2.162998e+00 | 1.625795e+00 |
| role:fc_up | 24 | -131.628137 | -94.032843 | 0.500043 | 226.161023 | 2.011243e+00 | 1.526902e+00 |
| role:k | 24 | -289.032536 | -169.875291 | 0.771165 | 459.678992 | 2.535365e+00 | 1.463482e+00 |
| role:o | 24 | -189.801092 | -114.241173 | 0.472623 | 304.514889 | 2.179579e+00 | 1.619688e+00 |
| role:proj | 24 | -1557.399703 | -156.830189 | 0.532639 | 1714.762531 | 1.860782e+00 | 1.685635e+00 |
| role:q | 24 | -239.467665 | -119.565194 | 0.745117 | 359.777976 | 2.339128e+00 | 1.439052e+00 |
| role:v | 24 | -157.041273 | -91.938908 | 0.761353 | 249.741534 | 2.264273e+00 | 1.425675e+00 |
| shape:hidden_to_hidden | 48 | -214.634378 | -116.903183 | 0.608870 | 332.146432 | 2.259354e+00 | 1.529370e+00 |
| shape:hidden_to_wide | 96 | -194.624754 | -133.619155 | 0.648641 | 328.892550 | 2.243470e+00 | 1.510463e+00 |
| shape:wide_to_hidden | 24 | -1557.399703 | -156.830189 | 0.532639 | 1714.762531 | 1.860782e+00 | 1.685635e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 120 | -18.617084 | -39.565728 | -0.005955 | 0.716228 | 0.721239 | 58.899040 | 0.010965 |
| layer:0 | 15 | -0.647247 | -0.962831 | -0.001185 | 0.866230 | 0.867771 | 2.476307 | 0.002726 |
| layer:10 | 15 | -15.931835 | -29.072525 | -0.008184 | 0.713295 | 0.719716 | 45.717655 | 0.014605 |
| layer:13 | 15 | -21.208478 | -24.954103 | -0.015110 | 0.742101 | 0.751777 | 46.904682 | 0.024786 |
| layer:16 | 15 | -29.228578 | -16.767500 | 0.002177 | 0.527232 | 0.526555 | 46.523310 | -0.002854 |
| layer:20 | 15 | -30.423620 | -42.493617 | -0.006505 | 0.609760 | 0.612494 | 73.526997 | 0.009239 |
| layer:23 | 15 | -19.153134 | -167.687162 | 0.012902 | 0.610263 | 0.625544 | 187.450560 | 0.002379 |
| layer:3 | 15 | -17.567272 | -14.518429 | -0.026136 | 0.842739 | 0.844761 | 32.928440 | 0.028159 |
| layer:7 | 15 | -14.776509 | -20.069657 | -0.005595 | 0.818206 | 0.821289 | 35.664372 | 0.008678 |
| length:10 | 24 | -30.188208 | -56.021774 | -0.007653 | 0.717723 | 0.726075 | 86.927706 | 0.016005 |
| length:1024 | 48 | -14.264370 | -34.787721 | -0.005771 | 0.715449 | 0.718824 | 49.767541 | 0.009145 |
| length:128 | 24 | -18.470558 | -35.857962 | -0.004984 | 0.716527 | 0.722778 | 55.045048 | 0.011235 |
| length:512 | 24 | -15.897913 | -36.373461 | -0.005594 | 0.715993 | 0.719692 | 52.987367 | 0.009293 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550892e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.182319e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 2.957651e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
