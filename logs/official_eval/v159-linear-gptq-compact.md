# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `dbd1c2c1600cc611883a3c6bde5f8f65885edbd31643b39c6d857bb610396ca5`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.685982160 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.685982160 |
| Linear role macro mean | 0.685982160 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 169.374s |
| Candidate API total | 162.643s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.685982 | 0.685400 | 0.559369 | 0.478923 | 0.336993 | 56/0/0 | 0.314600 |
| family:fc | 16 | 0.563702 | 0.558280 | 0.545089 | 0.495832 | 0.420782 | 16/0/0 | 0.441720 |
| family:o | 8 | 0.688055 | 0.643591 | 0.617099 | 0.606922 | 0.598136 | 8/0/0 | 0.356409 |
| family:proj | 8 | 0.514400 | 0.460085 | 0.407659 | 0.341023 | 0.336993 | 8/0/0 | 0.539915 |
| family:qkv | 24 | 0.824005 | 0.827681 | 0.771448 | 0.745331 | 0.707392 | 24/0/0 | 0.172319 |
| role:fc_gate | 8 | 0.595219 | 0.585560 | 0.559369 | 0.555761 | 0.555418 | 8/0/0 | 0.414440 |
| role:fc_up | 8 | 0.532185 | 0.540112 | 0.520377 | 0.464306 | 0.420782 | 8/0/0 | 0.459888 |
| role:k | 8 | 0.829712 | 0.830941 | 0.786913 | 0.751128 | 0.746403 | 8/0/0 | 0.169059 |
| role:o | 8 | 0.688055 | 0.643591 | 0.617099 | 0.606922 | 0.598136 | 8/0/0 | 0.356409 |
| role:proj | 8 | 0.514400 | 0.460085 | 0.407659 | 0.341023 | 0.336993 | 8/0/0 | 0.539915 |
| role:q | 8 | 0.821709 | 0.819631 | 0.761581 | 0.724036 | 0.707392 | 8/0/0 | 0.180369 |
| role:v | 8 | 0.820596 | 0.821762 | 0.789166 | 0.762764 | 0.753113 | 8/0/0 | 0.178238 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.017628`、max `0.164753`；成对 minimum-gain median `0.684883`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -225.444608 | -151.196217 | 0.685982 | 377.326808 | 2.239016e+00 | 1.537970e+00 |
| family:fc | 16 | -190.072421 | -153.220688 | 0.563702 | 343.856811 | 2.135777e+00 | 1.641897e+00 |
| family:o | 8 | -194.696633 | -155.668967 | 0.688055 | 351.053656 | 2.098976e+00 | 1.661821e+00 |
| family:proj | 8 | -226.345733 | -153.849409 | 0.514400 | 380.709542 | 2.137051e+00 | 1.621957e+00 |
| family:qkv | 24 | -258.975017 | -147.471257 | 0.824005 | 407.270279 | 2.388510e+00 | 1.399406e+00 |
| role:fc_gate | 8 | -235.028564 | -197.476327 | 0.595219 | 433.100110 | 2.172947e+00 | 1.657018e+00 |
| role:fc_up | 8 | -145.116278 | -108.965049 | 0.532185 | 254.613512 | 2.098606e+00 | 1.626776e+00 |
| role:k | 8 | -289.992354 | -188.542798 | 0.829712 | 479.364864 | 2.386521e+00 | 1.431389e+00 |
| role:o | 8 | -194.696633 | -155.668967 | 0.688055 | 351.053656 | 2.098976e+00 | 1.661821e+00 |
| role:proj | 8 | -226.345733 | -153.849409 | 0.514400 | 380.709542 | 2.137051e+00 | 1.621957e+00 |
| role:q | 8 | -300.374579 | -153.800161 | 0.821709 | 454.996449 | 2.395177e+00 | 1.391864e+00 |
| role:v | 8 | -186.558117 | -100.070811 | 0.820596 | 287.449523 | 2.383831e+00 | 1.374966e+00 |
| shape:hidden_to_hidden | 16 | -247.535606 | -154.734564 | 0.754882 | 403.025052 | 2.247076e+00 | 1.526843e+00 |
| shape:hidden_to_wide | 32 | -214.173828 | -148.763746 | 0.694428 | 363.632002 | 2.260476e+00 | 1.522537e+00 |
| shape:wide_to_hidden | 8 | -226.345733 | -153.849409 | 0.514400 | 380.709542 | 2.137051e+00 | 1.621957e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
