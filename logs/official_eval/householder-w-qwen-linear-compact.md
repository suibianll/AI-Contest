# householder-w-qwen-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `4364180fc40d1700dcd4adb20d229af0748fb0646ae7611c0244a5950909018a`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.699719023 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.699719023 |
| Linear role macro mean | 0.699719023 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 53.375s |
| Candidate API total | 48.019s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.699719 | 0.682387 | 0.581540 | 0.531984 | 0.413924 | 56/0/0 | 0.317613 |
| family:fc | 16 | 0.554373 | 0.550994 | 0.533892 | 0.490483 | 0.413924 | 16/0/0 | 0.449006 |
| family:o | 8 | 0.688733 | 0.637554 | 0.616040 | 0.613397 | 0.612574 | 8/0/0 | 0.362446 |
| family:proj | 8 | 0.645990 | 0.606355 | 0.568508 | 0.524057 | 0.523111 | 8/0/0 | 0.393645 |
| family:qkv | 24 | 0.818188 | 0.819533 | 0.765265 | 0.737212 | 0.695767 | 24/0/0 | 0.180467 |
| role:fc_gate | 8 | 0.583151 | 0.570045 | 0.551942 | 0.543038 | 0.536978 | 8/0/0 | 0.429955 |
| role:fc_up | 8 | 0.525595 | 0.534496 | 0.515841 | 0.458652 | 0.413924 | 8/0/0 | 0.465504 |
| role:k | 8 | 0.822355 | 0.819892 | 0.785902 | 0.745683 | 0.738101 | 8/0/0 | 0.180108 |
| role:o | 8 | 0.688733 | 0.637554 | 0.616040 | 0.613397 | 0.612574 | 8/0/0 | 0.362446 |
| role:proj | 8 | 0.645990 | 0.606355 | 0.568508 | 0.524057 | 0.523111 | 8/0/0 | 0.393645 |
| role:q | 8 | 0.818344 | 0.815023 | 0.757827 | 0.716492 | 0.695767 | 8/0/0 | 0.184977 |
| role:v | 8 | 0.813865 | 0.815055 | 0.790470 | 0.749840 | 0.734225 | 8/0/0 | 0.184945 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.016794`、max `0.163207`；成对 minimum-gain median `0.673664`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -214.699850 | -150.035683 | 0.699719 | 365.435252 | 2.234696e+00 | 1.536749e+00 |
| family:fc | 16 | -189.921585 | -154.120747 | 0.554373 | 344.596705 | 2.132701e+00 | 1.653828e+00 |
| family:o | 8 | -193.388588 | -155.608506 | 0.688733 | 349.685828 | 2.099648e+00 | 1.663586e+00 |
| family:proj | 8 | -226.052956 | -154.363923 | 0.645990 | 381.062870 | 2.136611e+00 | 1.628091e+00 |
| family:qkv | 24 | -234.538078 | -144.011953 | 0.818188 | 379.368219 | 2.380403e+00 | 1.385971e+00 |
| role:fc_gate | 8 | -236.514432 | -198.819120 | 0.583151 | 435.916703 | 2.169678e+00 | 1.682841e+00 |
| role:fc_up | 8 | -143.328738 | -109.422374 | 0.525595 | 253.276707 | 2.095725e+00 | 1.624815e+00 |
| role:k | 8 | -252.470713 | -179.768539 | 0.822355 | 433.061606 | 2.376067e+00 | 1.408310e+00 |
| role:o | 8 | -193.388588 | -155.608506 | 0.688733 | 349.685828 | 2.099648e+00 | 1.663586e+00 |
| role:proj | 8 | -226.052956 | -154.363923 | 0.645990 | 381.062870 | 2.136611e+00 | 1.628091e+00 |
| role:q | 8 | -277.530446 | -152.809033 | 0.818344 | 431.157823 | 2.380638e+00 | 1.378894e+00 |
| role:v | 8 | -173.613075 | -99.458287 | 0.813865 | 273.885227 | 2.384503e+00 | 1.370708e+00 |
| shape:hidden_to_hidden | 16 | -235.459517 | -154.208769 | 0.753539 | 390.421825 | 2.240143e+00 | 1.521240e+00 |
| shape:hidden_to_wide | 32 | -201.481739 | -146.867080 | 0.686241 | 349.035061 | 2.256493e+00 | 1.521669e+00 |
| shape:wide_to_hidden | 8 | -226.052956 | -154.363923 | 0.645990 | 381.062870 | 2.136611e+00 | 1.628091e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
