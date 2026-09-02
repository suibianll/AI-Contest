# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.556322821 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.556322821 |
| Linear role macro mean | 0.556322821 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 46.447s |
| Candidate API total | 42.095s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.556323 | 0.528142 | 0.450215 | 0.377758 | 0.314129 | 56/0/0 | 0.471858 |
| family:fc | 16 | 0.454370 | 0.456445 | 0.434830 | 0.403277 | 0.331451 | 16/0/0 | 0.543555 |
| family:o | 8 | 0.424478 | 0.361990 | 0.328863 | 0.320276 | 0.314129 | 8/0/0 | 0.638010 |
| family:proj | 8 | 0.526926 | 0.497068 | 0.438413 | 0.360174 | 0.358018 | 8/0/0 | 0.502932 |
| family:qkv | 24 | 0.678039 | 0.648780 | 0.612691 | 0.583489 | 0.537827 | 24/0/0 | 0.351220 |
| role:fc_gate | 8 | 0.461000 | 0.449136 | 0.433984 | 0.423819 | 0.413756 | 8/0/0 | 0.550864 |
| role:fc_up | 8 | 0.447740 | 0.459760 | 0.448694 | 0.383275 | 0.331451 | 8/0/0 | 0.540240 |
| role:k | 8 | 0.697769 | 0.684233 | 0.647989 | 0.601595 | 0.594122 | 8/0/0 | 0.315767 |
| role:o | 8 | 0.424478 | 0.361990 | 0.328863 | 0.320276 | 0.314129 | 8/0/0 | 0.638010 |
| role:proj | 8 | 0.526926 | 0.497068 | 0.438413 | 0.360174 | 0.358018 | 8/0/0 | 0.502932 |
| role:q | 8 | 0.684504 | 0.688493 | 0.605114 | 0.558292 | 0.537827 | 8/0/0 | 0.311507 |
| role:v | 8 | 0.651844 | 0.625107 | 0.617302 | 0.590579 | 0.572902 | 8/0/0 | 0.374893 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.018626`、max `0.149223`；成对 minimum-gain median `0.519844`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
