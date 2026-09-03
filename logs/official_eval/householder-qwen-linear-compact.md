# householder-qwen-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `5d205e6ecadfacb010d19d82faccc7265c91cfe28cf0dab44b04e68d91a2804f`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.699189784 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.699189784 |
| Linear role macro mean | 0.699189784 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 52.390s |
| Candidate API total | 47.387s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.699190 | 0.680797 | 0.583086 | 0.532565 | 0.416837 | 56/0/0 | 0.319203 |
| family:fc | 16 | 0.555400 | 0.549815 | 0.535609 | 0.492165 | 0.416837 | 16/0/0 | 0.450185 |
| family:o | 8 | 0.689596 | 0.646115 | 0.620525 | 0.610086 | 0.605533 | 8/0/0 | 0.353885 |
| family:proj | 8 | 0.646652 | 0.605401 | 0.568913 | 0.520838 | 0.519049 | 8/0/0 | 0.394599 |
| family:qkv | 24 | 0.815760 | 0.810643 | 0.764275 | 0.734956 | 0.692915 | 24/0/0 | 0.189357 |
| role:fc_gate | 8 | 0.583840 | 0.573039 | 0.549608 | 0.543143 | 0.539677 | 8/0/0 | 0.426961 |
| role:fc_up | 8 | 0.526961 | 0.532354 | 0.518174 | 0.461557 | 0.416837 | 8/0/0 | 0.467646 |
| role:k | 8 | 0.821531 | 0.814086 | 0.779206 | 0.749132 | 0.739430 | 8/0/0 | 0.185914 |
| role:o | 8 | 0.689596 | 0.646115 | 0.620525 | 0.610086 | 0.605533 | 8/0/0 | 0.353885 |
| role:proj | 8 | 0.646652 | 0.605401 | 0.568913 | 0.520838 | 0.519049 | 8/0/0 | 0.394599 |
| role:q | 8 | 0.815315 | 0.811721 | 0.754167 | 0.714122 | 0.692915 | 8/0/0 | 0.188279 |
| role:v | 8 | 0.810434 | 0.807562 | 0.787167 | 0.744168 | 0.722784 | 8/0/0 | 0.192438 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.018060`、max `0.162485`；成对 minimum-gain median `0.680362`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -209.218252 | -150.868995 | 0.699190 | 360.786437 | 2.234286e+00 | 1.536440e+00 |
| family:fc | 16 | -185.148124 | -154.794630 | 0.555400 | 340.498154 | 2.133032e+00 | 1.650193e+00 |
| family:o | 8 | -202.115322 | -155.281986 | 0.689596 | 358.086904 | 2.097671e+00 | 1.664843e+00 |
| family:proj | 8 | -207.228102 | -153.214478 | 0.646652 | 361.089232 | 2.136131e+00 | 1.618655e+00 |
| family:qkv | 24 | -228.296030 | -145.999081 | 0.815760 | 375.110872 | 2.380045e+00 | 1.390398e+00 |
| role:fc_gate | 8 | -227.656523 | -199.985134 | 0.583840 | 428.225497 | 2.169175e+00 | 1.663007e+00 |
| role:fc_up | 8 | -142.639725 | -109.604125 | 0.526961 | 252.770811 | 2.096889e+00 | 1.637379e+00 |
| role:k | 8 | -242.903340 | -182.261394 | 0.821531 | 425.986265 | 2.374810e+00 | 1.411646e+00 |
| role:o | 8 | -202.115322 | -155.281986 | 0.689596 | 358.086904 | 2.097671e+00 | 1.664843e+00 |
| role:proj | 8 | -207.228102 | -153.214478 | 0.646652 | 361.089232 | 2.136131e+00 | 1.618655e+00 |
| role:q | 8 | -266.045946 | -155.342842 | 0.815315 | 422.204104 | 2.385000e+00 | 1.381082e+00 |
| role:v | 8 | -175.938804 | -100.393008 | 0.810434 | 277.142246 | 2.380325e+00 | 1.378465e+00 |
| shape:hidden_to_hidden | 16 | -234.080634 | -155.312414 | 0.752456 | 390.145504 | 2.241336e+00 | 1.522962e+00 |
| shape:hidden_to_wide | 32 | -197.284598 | -148.060915 | 0.685691 | 346.031205 | 2.255300e+00 | 1.522624e+00 |
| shape:wide_to_hidden | 8 | -207.228102 | -153.214478 | 0.646652 | 361.089232 | 2.136131e+00 | 1.618655e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
