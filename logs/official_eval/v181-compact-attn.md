# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `86b0762c99ef2ce6dd319aa551e0711a1678c4ba785ca01535e3e12f4f637800`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.798348005 |
| Overall mean (all captured cases) | 0.798348005 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.798348005 |
| Candidate wall | 13.531s |
| Candidate API total | 13.405s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | 0.000595 | 0.000643 | 4/0/0 | 0.995101 | consistent_improvement |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.109917 | -46.668177 | -0.000518 | 0.791999 | 0.798348 | 57.570093 | 0.006867 |
| layer:0 | 1 | -0.179068 | -0.142093 | -0.004158 | 0.923245 | 0.923683 | 1.244406 | 0.004596 |
| layer:15 | 1 | -22.567431 | -19.010243 | -0.000463 | 0.730808 | 0.735003 | 42.308482 | 0.004658 |
| layer:23 | 1 | -16.086535 | -165.541152 | 0.015243 | 0.589297 | 0.607949 | 182.216984 | 0.003410 |
| layer:8 | 1 | -1.606635 | -1.979218 | -0.012694 | 0.924646 | 0.926757 | 4.510499 | 0.014805 |
| length:128 | 2 | -11.373249 | -9.576168 | -0.002311 | 0.827026 | 0.829343 | 21.776444 | 0.004627 |
| length:512 | 2 | -8.846585 | -83.760185 | 0.001274 | 0.756972 | 0.767353 | 93.363741 | 0.009107 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391727e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.435468e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.024889e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
