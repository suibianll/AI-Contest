# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `4469b85b53f5adefc6cfe4fbf136bdd4d7ff9ffc48a815592c95864a7287a844`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.797456767 |
| Overall mean (all captured cases) | 0.797456767 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.797456767 |
| Candidate wall | 10.556s |
| Candidate API total | 10.433s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.000297 | -0.000250 | 1/3/0 | 1.000792 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.109717 | -46.678391 | 0.000380 | 0.791992 | 0.797457 | 57.580099 | 0.005085 |
| layer:0 | 1 | -0.178264 | -0.149013 | -0.004313 | 0.923106 | 0.923370 | 1.250383 | 0.004577 |
| layer:15 | 1 | -22.565522 | -19.093449 | 0.000155 | 0.731257 | 0.733845 | 42.390228 | 0.002434 |
| layer:23 | 1 | -16.090191 | -165.497525 | 0.013746 | 0.590711 | 0.607514 | 182.178427 | 0.003056 |
| layer:8 | 1 | -1.604892 | -1.973575 | -0.008066 | 0.922892 | 0.925098 | 4.501358 | 0.010272 |
| length:128 | 2 | -11.371893 | -9.621231 | -0.002079 | 0.827182 | 0.828608 | 21.820305 | 0.003505 |
| length:512 | 2 | -8.847541 | -83.735550 | 0.002840 | 0.756802 | 0.766306 | 93.339893 | 0.006664 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391124e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.402296e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.023922e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
