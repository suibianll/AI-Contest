# v158-smoke — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `smoke-prefix` / `interface-and-local-sanity-only`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1, 2, 3, 4]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `2 Linear + 1 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.780260040 |
| Attention mean | 0.942927486 |
| Overall mean (all captured cases) | 0.834482522 |
| Linear role macro mean | 0.222931440 |
| Attention layer macro mean | 0.942927486 |
| Candidate wall | 381.868s |
| Candidate API total | 379.321s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 2 | 0.780260 | 0.780260 | 0.754037 | 0.727814 | 0.727814 | 2/0/0 | 0.219740 |
| family:qkv | 2 | 0.780260 | 0.780260 | 0.754037 | 0.727814 | 0.727814 | 2/0/0 | 0.219740 |
| role:k | 1 | 0.832706 | 0.832706 | 0.832706 | 0.832706 | 0.832706 | 1/0/0 | 0.167294 |
| role:q | 1 | 0.727814 | 0.727814 | 0.727814 | 0.727814 | 0.727814 | 1/0/0 | 0.272186 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 2 | -240.401605 | -42.706274 | 0.780260 | 283.888139 | 1.818198e-01 | 2.702173e-01 |
| family:qkv | 2 | -240.401605 | -42.706274 | 0.780260 | 283.888139 | 1.818198e-01 | 2.702173e-01 |
| role:k | 1 | -244.152172 | -36.790827 | 0.832706 | 281.775705 | 1.766701e-01 | 3.058596e-01 |
| role:q | 1 | -236.651037 | -48.621722 | 0.727814 | 286.000574 | 1.869696e-01 | 2.345749e-01 |
| shape:hidden_to_hidden | 1 | -236.651037 | -48.621722 | 0.727814 | 286.000574 | 1.869696e-01 | 2.345749e-01 |
| shape:hidden_to_wide | 1 | -244.152172 | -36.790827 | 0.832706 | 281.775705 | 1.766701e-01 | 3.058596e-01 |

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
