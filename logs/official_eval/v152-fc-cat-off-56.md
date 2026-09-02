# v152-fc-cat-off-56 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- calibration lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 1 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `1ca4a1b0a428a17d5eb9f66fc1cd6ffa3300550ca3001bd7c27ae129d3662d69`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.542552798 |
| Attention mean | 0.942927486 |
| Overall mean (all captured cases) | 0.549576915 |
| Linear role macro mean | 0.542552798 |
| Attention layer macro mean | 0.942927486 |
| Candidate wall | 213.442s |
| Candidate API total | 200.432s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -226.952801 | -146.702996 | 0.542553 | 374.198350 | 2.116418e+00 | 1.514143e+00 |
| family:fc | 16 | -135.162187 | -143.541757 | 0.372480 | 279.076424 | 2.029080e+00 | 1.663786e+00 |
| family:o | 8 | -285.762192 | -86.930695 | 0.472956 | 373.165843 | 1.580941e+00 | 1.394948e+00 |
| family:proj | 8 | -454.936311 | -216.350899 | 0.422457 | 671.709667 | 2.806752e+00 | 1.525248e+00 |
| family:qkv | 24 | -192.548909 | -145.518622 | 0.719166 | 338.786697 | 2.123025e+00 | 1.450410e+00 |
| role:fc_gate | 8 | -147.659457 | -182.900736 | 0.393306 | 330.953499 | 2.046465e+00 | 1.666573e+00 |
| role:fc_up | 8 | -122.664918 | -104.182778 | 0.351654 | 227.199349 | 2.011695e+00 | 1.661000e+00 |
| role:k | 8 | -256.952019 | -182.704668 | 0.763374 | 440.420061 | 2.178560e+00 | 1.467954e+00 |
| role:o | 8 | -285.762192 | -86.930695 | 0.472956 | 373.165843 | 1.580941e+00 | 1.394948e+00 |
| role:proj | 8 | -454.936311 | -216.350899 | 0.422457 | 671.709667 | 2.806752e+00 | 1.525248e+00 |
| role:q | 8 | -184.639222 | -154.271528 | 0.649184 | 339.559934 | 2.169327e+00 | 1.447823e+00 |
| role:v | 8 | -136.055487 | -99.579671 | 0.744939 | 236.380097 | 2.021189e+00 | 1.435452e+00 |
| shape:hidden_to_hidden | 16 | -235.200707 | -120.601111 | 0.561070 | 356.362888 | 1.875134e+00 | 1.421385e+00 |
| shape:hidden_to_wide | 32 | -165.832970 | -142.341963 | 0.563318 | 308.738251 | 2.064477e+00 | 1.557745e+00 |
| shape:wide_to_hidden | 8 | -454.936311 | -216.350899 | 0.422457 | 671.709667 | 2.806752e+00 | 1.525248e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| layer:0 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| length:10 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 6.684101e+02 | 1.299533e+05 |
| probability MSE vs reference | 3.018351e-02 | 1.530230e-03 |
| probability KL(reference || estimate) | 9.326769e-02 | 2.934108e-03 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
