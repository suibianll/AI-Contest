# parent — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-default-panel` / `linear-only-proxy-ranking-within-identical-cache`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `5444817ef8e6b9e0ef8a4117bf18521a5aefec0a1eaef24de3059279fd300370`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.633526215 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.633526215 |
| Linear role macro mean | 0.633526215 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 254.128s |
| Candidate API total | 231.359s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.633526 | 0.626581 | 0.536043 | 0.434968 | -0.055986 | 167/1/0 | 0.373419 |
| family:fc | 48 | 0.541915 | 0.546906 | 0.480997 | 0.430425 | 0.381204 | 48/0/0 | 0.453094 |
| family:o | 24 | 0.516677 | 0.511542 | 0.409206 | 0.305394 | -0.055986 | 23/1/0 | 0.488458 |
| family:proj | 24 | 0.534493 | 0.560032 | 0.518117 | 0.400870 | 0.024728 | 24/0/0 | 0.439968 |
| family:qkv | 72 | 0.766561 | 0.756762 | 0.732685 | 0.698230 | 0.625682 | 72/0/0 | 0.243238 |
| role:fc_gate | 24 | 0.569145 | 0.570796 | 0.526177 | 0.452775 | 0.381204 | 24/0/0 | 0.429204 |
| role:fc_up | 24 | 0.514685 | 0.514788 | 0.456483 | 0.419644 | 0.406775 | 24/0/0 | 0.485212 |
| role:k | 24 | 0.780293 | 0.771021 | 0.744579 | 0.726400 | 0.679656 | 24/0/0 | 0.228979 |
| role:o | 24 | 0.516677 | 0.511542 | 0.409206 | 0.305394 | -0.055986 | 23/1/0 | 0.488458 |
| role:proj | 24 | 0.534493 | 0.560032 | 0.518117 | 0.400870 | 0.024728 | 24/0/0 | 0.439968 |
| role:q | 24 | 0.750268 | 0.752874 | 0.692494 | 0.666592 | 0.625682 | 24/0/0 | 0.247126 |
| role:v | 24 | 0.769122 | 0.760891 | 0.735248 | 0.718308 | 0.699571 | 24/0/0 | 0.239109 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -386.639222 | -130.530943 | 0.633526 | 517.803692 | 2.193137e+00 | 1.522076e+00 |
| family:fc | 48 | -157.299580 | -137.331994 | 0.541915 | 295.173490 | 2.086877e+00 | 1.563780e+00 |
| family:o | 24 | -184.152083 | -108.569643 | 0.516677 | 293.238404 | 2.179465e+00 | 1.586140e+00 |
| family:proj | 24 | -1554.558363 | -155.780992 | 0.534493 | 1710.873849 | 1.860912e+00 | 1.640422e+00 |
| family:qkv | 72 | -217.721649 | -124.900660 | 0.766561 | 343.388870 | 2.379277e+00 | 1.433470e+00 |
| role:fc_gate | 24 | -180.874774 | -178.031915 | 0.569145 | 359.475834 | 2.162678e+00 | 1.592024e+00 |
| role:fc_up | 24 | -133.724387 | -96.632073 | 0.514685 | 230.871145 | 2.011075e+00 | 1.535536e+00 |
| role:k | 24 | -275.293676 | -166.058867 | 0.780293 | 442.132835 | 2.534959e+00 | 1.461413e+00 |
| role:o | 24 | -184.152083 | -108.569643 | 0.516677 | 293.238404 | 2.179465e+00 | 1.586140e+00 |
| role:proj | 24 | -1554.558363 | -155.780992 | 0.534493 | 1710.873849 | 1.860912e+00 | 1.640422e+00 |
| role:q | 24 | -223.591143 | -116.000337 | 0.750268 | 340.341748 | 2.338834e+00 | 1.422896e+00 |
| role:v | 24 | -154.280130 | -92.642775 | 0.769122 | 247.692027 | 2.264040e+00 | 1.416101e+00 |
| shape:hidden_to_hidden | 48 | -203.871613 | -112.284990 | 0.633473 | 316.790076 | 2.259150e+00 | 1.504518e+00 |
| shape:hidden_to_wide | 96 | -186.043241 | -133.341408 | 0.658311 | 320.042960 | 2.243188e+00 | 1.501268e+00 |
| shape:wide_to_hidden | 24 | -1554.558363 | -155.780992 | 0.534493 | 1710.873849 | 1.860912e+00 | 1.640422e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
