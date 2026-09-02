# ablate_alphawide1 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `4f5d6fdd9df2f8c0feded176c7aa16cc87893a5fcc1302b4d88d378faf0b0ccd`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.699698573 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.699698573 |
| Linear role macro mean | 0.699698573 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 53.674s |
| Candidate API total | 47.903s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.699699 | 0.685426 | 0.579309 | 0.532109 | 0.418283 | 56/0/0 | 0.314574 |
| family:fc | 16 | 0.552960 | 0.559890 | 0.520697 | 0.483080 | 0.418283 | 16/0/0 | 0.440110 |
| family:o | 8 | 0.688092 | 0.643534 | 0.617240 | 0.606922 | 0.598136 | 8/0/0 | 0.356466 |
| family:proj | 8 | 0.631892 | 0.608981 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.391019 |
| family:qkv | 24 | 0.823995 | 0.827681 | 0.771448 | 0.745332 | 0.707392 | 24/0/0 | 0.172319 |
| role:fc_gate | 8 | 0.581824 | 0.575578 | 0.562414 | 0.532637 | 0.510431 | 8/0/0 | 0.424422 |
| role:fc_up | 8 | 0.524096 | 0.534534 | 0.506621 | 0.455225 | 0.418283 | 8/0/0 | 0.465466 |
| role:k | 8 | 0.829713 | 0.830941 | 0.786902 | 0.751131 | 0.746451 | 8/0/0 | 0.169059 |
| role:o | 8 | 0.688092 | 0.643534 | 0.617240 | 0.606922 | 0.598136 | 8/0/0 | 0.356466 |
| role:proj | 8 | 0.631892 | 0.608981 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.391019 |
| role:q | 8 | 0.821677 | 0.819506 | 0.761582 | 0.724036 | 0.707392 | 8/0/0 | 0.180494 |
| role:v | 8 | 0.820596 | 0.821762 | 0.789166 | 0.762764 | 0.753113 | 8/0/0 | 0.178238 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.017517`、max `0.150885`；成对 minimum-gain median `0.684904`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`ablate_alphawide1`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.005809 | 0.000000 | 2/14/40 | 1.000000 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -216.578991 | -154.250922 | 0.699699 | 371.529612 | 2.204796e+00 | 1.586837e+00 |
| family:fc | 16 | -175.437127 | -161.531471 | 0.552960 | 337.521558 | 2.054280e+00 | 1.774708e+00 |
| family:o | 8 | -194.696198 | -155.666498 | 0.688092 | 351.050788 | 2.098974e+00 | 1.661824e+00 |
| family:proj | 8 | -193.555590 | -158.612555 | 0.631892 | 352.800037 | 2.060485e+00 | 1.698387e+00 |
| family:qkv | 24 | -258.975633 | -147.471486 | 0.823995 | 407.271114 | 2.388518e+00 | 1.399410e+00 |
| role:fc_gate | 8 | -215.516846 | -204.414413 | 0.581824 | 420.513083 | 2.084966e+00 | 1.759394e+00 |
| role:fc_up | 8 | -135.357408 | -118.648530 | 0.524096 | 254.530033 | 2.023594e+00 | 1.790022e+00 |
| role:k | 8 | -289.992354 | -188.542112 | 0.829713 | 479.364179 | 2.386521e+00 | 1.431389e+00 |
| role:o | 8 | -194.696198 | -155.666498 | 0.688092 | 351.050788 | 2.098974e+00 | 1.661824e+00 |
| role:proj | 8 | -193.555590 | -158.612555 | 0.631892 | 352.800037 | 2.060485e+00 | 1.698387e+00 |
| role:q | 8 | -300.376428 | -153.801534 | 0.821677 | 454.999639 | 2.395202e+00 | 1.391875e+00 |
| role:v | 8 | -186.558117 | -100.070812 | 0.820596 | 287.449524 | 2.383831e+00 | 1.374966e+00 |
| shape:hidden_to_hidden | 16 | -247.536313 | -154.734016 | 0.754885 | 403.025213 | 2.247088e+00 | 1.526849e+00 |
| shape:hidden_to_wide | 32 | -206.856181 | -152.918967 | 0.689057 | 360.464205 | 2.219728e+00 | 1.588943e+00 |
| shape:wide_to_hidden | 8 | -193.555590 | -158.612555 | 0.631892 | 352.800037 | 2.060485e+00 | 1.698387e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。

## Decision: REJECTED (2026-09-03)

- Qwen compact 56 cases: mean Δgain -5.8e-3, 14 negative / 2 positive / 40 zero.
- fc 12/16 negative (mean -1.1e-2), proj worst -0.0837/-0.0705. Wide-layer
  alpha 0.25/0.75 carry real selected gain on fc/proj. Reverted.
- L2 series conclusion: all four search dimensions (seeds, sizes, RMS smooth,
  wide alphas) are load-bearing; no ablation is safe. Batch encoding (L1)
  remains the only lossless time reduction path.
