# householder-amax-qwen-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `6e9d01a84b3352fed2a722222046884c99e689311bff95296df0f6d6d33bd2ae`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.703344080 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.703344080 |
| Linear role macro mean | 0.703344080 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 49.219s |
| Candidate API total | 44.776s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.703344 | 0.685084 | 0.587413 | 0.536184 | 0.420548 | 56/0/0 | 0.314916 |
| family:fc | 16 | 0.559897 | 0.554325 | 0.539746 | 0.493409 | 0.420548 | 16/0/0 | 0.445675 |
| family:o | 8 | 0.689463 | 0.640023 | 0.619933 | 0.611839 | 0.606281 | 8/0/0 | 0.359977 |
| family:proj | 8 | 0.649997 | 0.610962 | 0.574002 | 0.522048 | 0.518666 | 8/0/0 | 0.389038 |
| family:qkv | 24 | 0.821385 | 0.820860 | 0.769066 | 0.741345 | 0.704863 | 24/0/0 | 0.179140 |
| role:fc_gate | 8 | 0.591147 | 0.581474 | 0.555087 | 0.552023 | 0.551244 | 8/0/0 | 0.418526 |
| role:fc_up | 8 | 0.528648 | 0.534941 | 0.517790 | 0.463303 | 0.420548 | 8/0/0 | 0.465059 |
| role:k | 8 | 0.828867 | 0.826196 | 0.784553 | 0.751145 | 0.746024 | 8/0/0 | 0.173804 |
| role:o | 8 | 0.689463 | 0.640023 | 0.619933 | 0.611839 | 0.606281 | 8/0/0 | 0.359977 |
| role:proj | 8 | 0.649997 | 0.610962 | 0.574002 | 0.522048 | 0.518666 | 8/0/0 | 0.389038 |
| role:q | 8 | 0.819583 | 0.817893 | 0.760085 | 0.721339 | 0.704863 | 8/0/0 | 0.182107 |
| role:v | 8 | 0.815704 | 0.815012 | 0.791157 | 0.752589 | 0.735592 | 8/0/0 | 0.184988 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.015264`、max `0.158301`；成对 minimum-gain median `0.679099`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -213.287914 | -150.096514 | 0.703344 | 364.087772 | 2.236742e+00 | 1.529956e+00 |
| family:fc | 16 | -186.352211 | -152.384442 | 0.559897 | 339.296551 | 2.132795e+00 | 1.636824e+00 |
| family:o | 8 | -193.775186 | -154.453169 | 0.689463 | 348.917818 | 2.097019e+00 | 1.662152e+00 |
| family:proj | 8 | -218.960115 | -153.322900 | 0.649997 | 372.933013 | 2.136028e+00 | 1.619325e+00 |
| family:qkv | 24 | -235.858558 | -146.043548 | 0.821385 | 382.723490 | 2.386186e+00 | 1.384856e+00 |
| role:fc_gate | 8 | -230.141513 | -195.968306 | 0.591147 | 426.700966 | 2.168430e+00 | 1.653184e+00 |
| role:fc_up | 8 | -142.562909 | -108.800579 | 0.528648 | 251.892136 | 2.097161e+00 | 1.620463e+00 |
| role:k | 8 | -250.683508 | -186.263302 | 0.828867 | 437.775677 | 2.382867e+00 | 1.408864e+00 |
| role:o | 8 | -193.775186 | -154.453169 | 0.689463 | 348.917818 | 2.097019e+00 | 1.662152e+00 |
| role:proj | 8 | -218.960115 | -153.322900 | 0.649997 | 372.933013 | 2.136028e+00 | 1.619325e+00 |
| role:q | 8 | -275.960779 | -152.547204 | 0.819583 | 429.327566 | 2.392472e+00 | 1.382119e+00 |
| role:v | 8 | -180.931387 | -99.320137 | 0.815704 | 281.067228 | 2.383218e+00 | 1.363586e+00 |
| shape:hidden_to_hidden | 16 | -234.867982 | -153.500187 | 0.754523 | 389.122692 | 2.244745e+00 | 1.522135e+00 |
| shape:hidden_to_wide | 32 | -201.079829 | -147.588081 | 0.691092 | 349.359002 | 2.257919e+00 | 1.511524e+00 |
| shape:wide_to_hidden | 8 | -218.960115 | -153.322900 | 0.649997 | 372.933013 | 2.136028e+00 | 1.619325e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
