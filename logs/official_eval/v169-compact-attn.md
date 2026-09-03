# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `3e9307bc45ede56e240380e09905c9fef8577c9a461fc752acdd8105ef67dae8`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.793085128 |
| Overall mean (all captured cases) | 0.793085128 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.793085128 |
| Candidate wall | 10.753s |
| Candidate API total | 10.631s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.004377 | -0.002164 | 0/4/0 | 1.011769 | consistent_regression |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.112254 | -46.375657 | -0.002609 | 0.791205 | 0.793085 | 57.279116 | 0.004490 |
| layer:0 | 1 | -0.178057 | -0.143543 | 0.010180 | 0.923383 | 0.923048 | 1.244983 | -0.010515 |
| layer:15 | 1 | -22.577934 | -19.022992 | -0.001092 | 0.731563 | 0.731828 | 42.332488 | 0.001357 |
| layer:23 | 1 | -16.087349 | -164.355329 | 0.001834 | 0.587557 | 0.593628 | 181.030236 | 0.004237 |
| layer:8 | 1 | -1.605676 | -1.980762 | -0.021359 | 0.922317 | 0.923837 | 4.508756 | 0.022879 |
| length:128 | 2 | -11.377995 | -9.583267 | 0.004544 | 0.827473 | 0.827438 | 21.788735 | -0.004579 |
| length:512 | 2 | -8.846513 | -83.168046 | -0.009763 | 0.754937 | 0.758732 | 92.769496 | 0.013558 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391785e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.340554e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.005347e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
