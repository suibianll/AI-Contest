# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `1cc1d0b9d011f8e51bc68c89bc1ca0b08034e4ef906025a7c4d931242fd6f05b`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.682679443 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.682679443 |
| Linear role macro mean | 0.682679443 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 256.893s |
| Candidate API total | 228.700s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.682679 | 0.656077 | 0.541712 | 0.509543 | 0.399559 | 56/0/0 | 0.343923 |
| family:fc | 16 | 0.537993 | 0.535793 | 0.519710 | 0.472147 | 0.399559 | 16/0/0 | 0.464207 |
| family:o | 8 | 0.618381 | 0.568808 | 0.518474 | 0.511623 | 0.510867 | 8/0/0 | 0.431192 |
| family:proj | 8 | 0.650429 | 0.608097 | 0.577909 | 0.525740 | 0.524968 | 8/0/0 | 0.391903 |
| family:qkv | 24 | 0.811320 | 0.818298 | 0.754145 | 0.720308 | 0.679877 | 24/0/0 | 0.181702 |
| role:fc_gate | 8 | 0.568995 | 0.557996 | 0.537847 | 0.526859 | 0.526028 | 8/0/0 | 0.442004 |
| role:fc_up | 8 | 0.506992 | 0.514934 | 0.497602 | 0.439912 | 0.399559 | 8/0/0 | 0.485066 |
| role:k | 8 | 0.817707 | 0.818298 | 0.761626 | 0.727278 | 0.723361 | 8/0/0 | 0.181702 |
| role:o | 8 | 0.618381 | 0.568808 | 0.518474 | 0.511623 | 0.510867 | 8/0/0 | 0.431192 |
| role:proj | 8 | 0.650429 | 0.608097 | 0.577909 | 0.525740 | 0.524968 | 8/0/0 | 0.391903 |
| role:q | 8 | 0.808549 | 0.806342 | 0.745355 | 0.698530 | 0.679877 | 8/0/0 | 0.193658 |
| role:v | 8 | 0.807703 | 0.813042 | 0.774908 | 0.735117 | 0.717889 | 8/0/0 | 0.186958 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.015127`、max `0.159078`；成对 minimum-gain median `0.652303`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.022949 | -0.019923 | 1/47/8 | 1.062112 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -226.193979 | -151.241192 | 0.682679 | 378.117851 | 2.239262e+00 | 1.537950e+00 |
| family:fc | 16 | -190.170363 | -153.205862 | 0.537993 | 343.914218 | 2.135638e+00 | 1.641978e+00 |
| family:o | 8 | -197.702339 | -155.636297 | 0.618381 | 353.957017 | 2.099290e+00 | 1.662120e+00 |
| family:proj | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |
| family:qkv | 24 | -259.714257 | -147.612263 | 0.811320 | 408.137840 | 2.389118e+00 | 1.399117e+00 |
| role:fc_gate | 8 | -235.263297 | -197.462560 | 0.568995 | 433.294851 | 2.172929e+00 | 1.657089e+00 |
| role:fc_up | 8 | -145.077429 | -108.949165 | 0.506992 | 254.533585 | 2.098348e+00 | 1.626867e+00 |
| role:k | 8 | -289.422484 | -188.773823 | 0.817707 | 479.014014 | 2.386811e+00 | 1.430678e+00 |
| role:o | 8 | -197.702339 | -155.636297 | 0.618381 | 353.957017 | 2.099290e+00 | 1.662120e+00 |
| role:proj | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |
| role:q | 8 | -302.143035 | -153.943983 | 0.808549 | 456.895567 | 2.396246e+00 | 1.391830e+00 |
| role:v | 8 | -187.577254 | -100.118983 | 0.807703 | 288.503940 | 2.384296e+00 | 1.374843e+00 |
| shape:hidden_to_hidden | 16 | -249.922687 | -154.790140 | 0.713465 | 405.426292 | 2.247768e+00 | 1.526975e+00 |
| shape:hidden_to_wide | 32 | -214.335116 | -148.826133 | 0.675349 | 363.836598 | 2.260596e+00 | 1.522369e+00 |
| shape:wide_to_hidden | 8 | -226.172016 | -153.803537 | 0.650429 | 380.625982 | 2.136913e+00 | 1.622221e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
