# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `2a5b74161a8c606de0dc05e2735bdc51e3cd11d2a4da910211049001e6f84a7d`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.702895621 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.702895621 |
| Linear role macro mean | 0.702895621 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 68.364s |
| Candidate API total | 63.240s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.702896 | 0.682677 | 0.587781 | 0.539314 | 0.445626 | 56/0/0 | 0.317323 |
| family:fc | 16 | 0.564751 | 0.570331 | 0.528742 | 0.495667 | 0.445626 | 16/0/0 | 0.429669 |
| family:o | 8 | 0.658154 | 0.599771 | 0.594654 | 0.578679 | 0.568465 | 8/0/0 | 0.400229 |
| family:proj | 8 | 0.649250 | 0.608473 | 0.573549 | 0.526543 | 0.525392 | 8/0/0 | 0.391527 |
| family:qkv | 24 | 0.827788 | 0.829133 | 0.779789 | 0.752231 | 0.712812 | 24/0/0 | 0.170867 |
| role:fc_gate | 8 | 0.585184 | 0.591707 | 0.554915 | 0.510749 | 0.508017 | 8/0/0 | 0.408293 |
| role:fc_up | 8 | 0.544318 | 0.554942 | 0.528742 | 0.480584 | 0.445626 | 8/0/0 | 0.445058 |
| role:k | 8 | 0.831461 | 0.831687 | 0.795789 | 0.756330 | 0.752514 | 8/0/0 | 0.168313 |
| role:o | 8 | 0.658154 | 0.599771 | 0.594654 | 0.578679 | 0.568465 | 8/0/0 | 0.400229 |
| role:proj | 8 | 0.649250 | 0.608473 | 0.573549 | 0.526543 | 0.525392 | 8/0/0 | 0.391527 |
| role:q | 8 | 0.827706 | 0.827962 | 0.772311 | 0.730903 | 0.712812 | 8/0/0 | 0.172038 |
| role:v | 8 | 0.824196 | 0.824075 | 0.797195 | 0.769461 | 0.760014 | 8/0/0 | 0.175925 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.014623`、max `0.138996`；成对 minimum-gain median `0.676354`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.002733 | 0.001191 | 34/22/0 | 0.994750 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -225.277581 | -151.125225 | 0.702896 | 377.105702 | 2.230751e+00 | 1.536626e+00 |
| family:fc | 16 | -189.801922 | -153.187912 | 0.564751 | 343.554586 | 2.130972e+00 | 1.640476e+00 |
| family:o | 8 | -196.658093 | -155.304123 | 0.658154 | 352.620370 | 2.097138e+00 | 1.658482e+00 |
| family:proj | 8 | -228.685052 | -153.684329 | 0.649250 | 383.018632 | 2.134405e+00 | 1.620909e+00 |
| family:qkv | 24 | -257.332026 | -147.504099 | 0.827788 | 405.663913 | 2.373924e+00 | 1.398679e+00 |
| role:fc_gate | 8 | -234.591186 | -197.470696 | 0.585184 | 432.647067 | 2.167030e+00 | 1.654802e+00 |
| role:fc_up | 8 | -145.012659 | -108.905128 | 0.544318 | 254.462105 | 2.094913e+00 | 1.626151e+00 |
| role:k | 8 | -287.978793 | -188.512980 | 0.831461 | 477.323234 | 2.374501e+00 | 1.430304e+00 |
| role:o | 8 | -196.658093 | -155.304123 | 0.658154 | 352.620370 | 2.097138e+00 | 1.658482e+00 |
| role:proj | 8 | -228.685052 | -153.684329 | 0.649250 | 383.018632 | 2.134405e+00 | 1.620909e+00 |
| role:q | 8 | -298.725154 | -153.787821 | 0.827706 | 453.340680 | 2.382526e+00 | 1.392472e+00 |
| role:v | 8 | -185.292130 | -100.211498 | 0.824196 | 286.327824 | 2.364746e+00 | 1.373262e+00 |
| shape:hidden_to_hidden | 16 | -247.691623 | -154.545972 | 0.742930 | 402.980525 | 2.239832e+00 | 1.525477e+00 |
| shape:hidden_to_wide | 32 | -213.218692 | -148.775076 | 0.696290 | 362.690057 | 2.250297e+00 | 1.521130e+00 |
| shape:wide_to_hidden | 8 | -228.685052 | -153.684329 | 0.649250 | 383.018632 | 2.134405e+00 | 1.620909e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
