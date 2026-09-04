# v186 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.799083884 |
| Overall mean (all captured cases) | 0.799083884 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.799083884 |
| Candidate wall | 10.679s |
| Candidate API total | 10.551s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.112891 | -46.730362 | -0.000518 | 0.792823 | 0.799084 | 57.636076 | 0.006779 |
| layer:0 | 1 | -0.178164 | -0.142367 | -0.004158 | 0.923058 | 0.923362 | 1.243589 | 0.004462 |
| layer:15 | 1 | -22.578752 | -18.987073 | -0.000463 | 0.731303 | 0.735336 | 42.297127 | 0.004496 |
| layer:23 | 1 | -16.088834 | -165.812515 | 0.015243 | 0.592615 | 0.611221 | 182.493963 | 0.003363 |
| layer:8 | 1 | -1.605813 | -1.979495 | -0.012694 | 0.924317 | 0.926417 | 4.509626 | 0.014794 |
| length:128 | 2 | -11.378458 | -9.564720 | -0.002311 | 0.827181 | 0.829349 | 21.770358 | 0.004479 |
| length:512 | 2 | -8.847324 | -83.896005 | 0.001274 | 0.758466 | 0.768819 | 93.501794 | 0.009079 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391147e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.387883e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.020964e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
