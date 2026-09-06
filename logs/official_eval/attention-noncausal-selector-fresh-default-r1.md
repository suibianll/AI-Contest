# attention-noncausal-selector-fresh-default-r1 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `default-panel` / `proxy-ranking-within-identical-cache`
- proxy ranking comparable: `True`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `de1e07c2515062298a575f5e2f6a1748a60bfab948fae67f4b27d32bc97fd1fb`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.640258324 |
| Attention mean | 0.748924596 |
| Overall mean (all captured cases) | 0.685535938 |
| Linear role macro mean | 0.640258324 |
| Attention layer macro mean | 0.748924596 |
| Candidate wall | 423.830s |
| Candidate API total | 399.136s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.640258 | 0.642321 | 0.536856 | 0.449202 | -0.043193 | 167/1/0 | 0.357679 |
| family:fc | 48 | 0.543661 | 0.548559 | 0.483102 | 0.430274 | 0.384101 | 48/0/0 | 0.451441 |
| family:o | 24 | 0.527661 | 0.506474 | 0.426538 | 0.324588 | -0.043193 | 23/1/0 | 0.493526 |
| family:proj | 24 | 0.557920 | 0.567292 | 0.526751 | 0.471327 | 0.271840 | 24/0/0 | 0.432708 |
| family:qkv | 72 | 0.769635 | 0.759562 | 0.736178 | 0.702220 | 0.646951 | 72/0/0 | 0.240438 |
| role:fc_gate | 24 | 0.572191 | 0.577585 | 0.529990 | 0.456256 | 0.384101 | 24/0/0 | 0.422415 |
| role:fc_up | 24 | 0.515131 | 0.515457 | 0.455246 | 0.418888 | 0.406565 | 24/0/0 | 0.484543 |
| role:k | 24 | 0.781805 | 0.775788 | 0.748532 | 0.727233 | 0.681957 | 24/0/0 | 0.224212 |
| role:o | 24 | 0.527661 | 0.506474 | 0.426538 | 0.324588 | -0.043193 | 23/1/0 | 0.493526 |
| role:proj | 24 | 0.557920 | 0.567292 | 0.526751 | 0.471327 | 0.271840 | 24/0/0 | 0.432708 |
| role:q | 24 | 0.755477 | 0.757203 | 0.699148 | 0.674505 | 0.646951 | 24/0/0 | 0.242797 |
| role:v | 24 | 0.771622 | 0.761840 | 0.737658 | 0.720542 | 0.705889 | 24/0/0 | 0.238160 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
