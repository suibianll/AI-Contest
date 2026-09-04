# v185-cleanroom-attn-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `3ea046594fb18dd86fd8ccfd2364a391039b0112e29986c8f949f9af526c136c`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.686541396 |
| Overall mean (all captured cases) | 0.686541396 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.686541396 |
| Candidate wall | 3.927s |
| Candidate API total | 3.766s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -0.753147 | -1.980468 | -0.000974 | 0.686761 | 0.686541 | 3.420376 | 0.000754 |
| layer:0 | 1 | 0.078803 | 0.627068 | -0.002706 | 0.880734 | 0.880764 | 0.174862 | 0.002736 |
| layer:15 | 1 | -2.463128 | -7.667052 | 0.000303 | 0.435375 | 0.435457 | 10.565555 | -0.000221 |
| layer:23 | 1 | -0.366405 | -0.638548 | 0.008362 | 0.542354 | 0.540344 | 1.547307 | -0.010372 |
| layer:8 | 1 | -0.261857 | -0.243342 | -0.009855 | 0.888580 | 0.889600 | 1.393780 | 0.010875 |
| length:128 | 2 | -1.192162 | -3.519992 | -0.001201 | 0.658054 | 0.658110 | 5.370209 | 0.001257 |
| length:512 | 2 | -0.314131 | -0.440945 | -0.000746 | 0.715467 | 0.714972 | 1.470543 | 0.000252 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.472639e+04 |
| probability MSE vs reference | 5.931959e-04 | 1.074216e-04 |
| probability KL(reference || estimate) | 3.396776e-03 | 3.530268e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
