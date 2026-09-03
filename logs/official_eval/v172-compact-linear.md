# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `67283937d8e8767fdf760afe70d4efc778228f4ed4e52af4a6f6f21769df4f65`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.657286691 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.657286691 |
| Linear role macro mean | 0.657286691 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 67.274s |
| Candidate API total | 62.242s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.657287 | 0.634725 | 0.521313 | 0.468841 | 0.350971 | 56/0/0 | 0.365275 |
| family:fc | 16 | 0.505796 | 0.506196 | 0.485335 | 0.434595 | 0.350971 | 16/0/0 | 0.493804 |
| family:o | 8 | 0.562255 | 0.494816 | 0.457559 | 0.448279 | 0.446941 | 8/0/0 | 0.505184 |
| family:proj | 8 | 0.650429 | 0.608097 | 0.577909 | 0.525740 | 0.524968 | 8/0/0 | 0.391903 |
| family:qkv | 24 | 0.792243 | 0.795637 | 0.727588 | 0.696770 | 0.652612 | 24/0/0 | 0.204363 |
| role:fc_gate | 8 | 0.538752 | 0.529360 | 0.506029 | 0.491633 | 0.488559 | 8/0/0 | 0.470640 |
| role:fc_up | 8 | 0.472841 | 0.484974 | 0.461317 | 0.397978 | 0.350971 | 8/0/0 | 0.515026 |
| role:k | 8 | 0.800093 | 0.804108 | 0.746654 | 0.708697 | 0.706525 | 8/0/0 | 0.195892 |
| role:o | 8 | 0.562255 | 0.494816 | 0.457559 | 0.448279 | 0.446941 | 8/0/0 | 0.505184 |
| role:proj | 8 | 0.650429 | 0.608097 | 0.577909 | 0.525740 | 0.524968 | 8/0/0 | 0.391903 |
| role:q | 8 | 0.790156 | 0.788726 | 0.721037 | 0.672749 | 0.652612 | 8/0/0 | 0.211274 |
| role:v | 8 | 0.786480 | 0.791628 | 0.756913 | 0.708864 | 0.698639 | 8/0/0 | 0.208372 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.016686`、max `0.172703`；成对 minimum-gain median `0.631301`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.048342 | -0.044254 | 0/48/8 | 1.157425 | consistent_regression |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -226.453610 | -151.189883 | 0.657287 | 378.300780 | 2.243165e+00 | 1.537910e+00 |
| family:fc | 16 | -190.532039 | -153.216221 | 0.505796 | 344.254057 | 2.139144e+00 | 1.641979e+00 |
| family:o | 8 | -198.118083 | -155.637711 | 0.562255 | 354.318049 | 2.106059e+00 | 1.661837e+00 |
| family:proj | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |
| family:qkv | 24 | -259.940365 | -147.485165 | 0.792243 | 408.217773 | 2.393632e+00 | 1.399118e+00 |
| role:fc_gate | 8 | -235.635649 | -197.464400 | 0.538752 | 433.638800 | 2.176469e+00 | 1.657180e+00 |
| role:fc_up | 8 | -145.428430 | -108.968042 | 0.472841 | 254.869313 | 2.101819e+00 | 1.626778e+00 |
| role:k | 8 | -289.364335 | -188.588229 | 0.800093 | 478.752658 | 2.392266e+00 | 1.430713e+00 |
| role:o | 8 | -198.118083 | -155.637711 | 0.562255 | 354.318049 | 2.106059e+00 | 1.661837e+00 |
| role:proj | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |
| role:q | 8 | -302.457676 | -153.780034 | 0.790156 | 457.027866 | 2.400140e+00 | 1.391807e+00 |
| role:v | 8 | -187.999083 | -100.087230 | 0.786480 | 288.872794 | 2.388489e+00 | 1.374836e+00 |
| shape:hidden_to_hidden | 16 | -250.287879 | -154.708873 | 0.676206 | 405.672958 | 2.253100e+00 | 1.526822e+00 |
| shape:hidden_to_wide | 32 | -214.606874 | -148.776975 | 0.649542 | 364.033391 | 2.264761e+00 | 1.522377e+00 |
| shape:wide_to_hidden | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
