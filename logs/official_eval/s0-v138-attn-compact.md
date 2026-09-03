# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `3a120beb62443ff6a5bcdb89b5fad970ac6d8d45f48f40fe31812073060c2d10`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.783370032 |
| Overall mean (all captured cases) | 0.783370032 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.783370032 |
| Candidate wall | 7.038s |
| Candidate API total | 6.906s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.014092 | -0.012992 | 0/4/0 | 1.082279 | consistent_regression |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -19.025273 | -15.768919 | -0.002674 | 0.781659 | 0.783370 | 35.575851 | 0.004385 |
| layer:0 | 1 | -0.175823 | -0.351351 | -0.004362 | 0.915347 | 0.915416 | 1.442521 | 0.004431 |
| layer:15 | 1 | -27.028910 | -16.054369 | -0.004523 | 0.727700 | 0.725658 | 43.810979 | 0.002481 |
| layer:23 | 1 | -47.171538 | -45.381501 | 0.015415 | 0.576414 | 0.584054 | 93.129453 | -0.007774 |
| layer:8 | 1 | -1.724823 | -1.288455 | -0.017227 | 0.907175 | 0.908351 | 3.920452 | 0.018404 |
| length:128 | 2 | -13.602366 | -8.202860 | -0.004442 | 0.821524 | 0.820537 | 22.626750 | 0.003456 |
| length:512 | 2 | -24.448180 | -23.334978 | -0.000906 | 0.741794 | 0.746203 | 48.524952 | 0.005315 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.470942e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.834199e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.053066e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
