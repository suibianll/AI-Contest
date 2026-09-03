# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `df285fcaa383bc36dc67d8edc14b772beadd07c8da3e70ece62838432db93fd1`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.705507633 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.705507633 |
| Linear role macro mean | 0.705507633 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 55.063s |
| Candidate API total | 50.601s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.705508 | 0.685426 | 0.591841 | 0.540094 | 0.420862 | 56/0/0 | 0.314574 |
| family:fc | 16 | 0.563657 | 0.558280 | 0.545012 | 0.495830 | 0.420862 | 16/0/0 | 0.441720 |
| family:o | 8 | 0.688092 | 0.643534 | 0.617240 | 0.606922 | 0.598136 | 8/0/0 | 0.356466 |
| family:proj | 8 | 0.651161 | 0.608981 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.391019 |
| family:qkv | 24 | 0.823995 | 0.827681 | 0.771448 | 0.745332 | 0.707392 | 24/0/0 | 0.172319 |
| role:fc_gate | 8 | 0.595145 | 0.585536 | 0.559427 | 0.555414 | 0.554842 | 8/0/0 | 0.414464 |
| role:fc_up | 8 | 0.532169 | 0.540037 | 0.520363 | 0.464348 | 0.420862 | 8/0/0 | 0.459963 |
| role:k | 8 | 0.829713 | 0.830941 | 0.786902 | 0.751131 | 0.746451 | 8/0/0 | 0.169059 |
| role:o | 8 | 0.688092 | 0.643534 | 0.617240 | 0.606922 | 0.598136 | 8/0/0 | 0.356466 |
| role:proj | 8 | 0.651161 | 0.608981 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.391019 |
| role:q | 8 | 0.821677 | 0.819506 | 0.761582 | 0.724036 | 0.707392 | 8/0/0 | 0.180494 |
| role:v | 8 | 0.820596 | 0.821762 | 0.789166 | 0.762764 | 0.753113 | 8/0/0 | 0.178238 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.017517`、max `0.164689`；成对 minimum-gain median `0.684904`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
