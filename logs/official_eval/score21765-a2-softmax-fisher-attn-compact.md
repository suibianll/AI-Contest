# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `492f3d18eba354e7e2fb5d922f362a28595a96a346ad0d7b6244ea5a07d3fbd2`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.789648797 |
| Overall mean (all captured cases) | 0.789648797 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.789648797 |
| Candidate wall | 10.636s |
| Candidate API total | 10.516s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.007813 | -0.004871 | 1/3/0 | 1.050113 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.104315 | -45.971819 | -0.000518 | 0.783084 | 0.789649 | 56.859218 | 0.007083 |
| layer:0 | 1 | -0.178057 | -0.159001 | -0.004158 | 0.915976 | 0.916201 | 1.253033 | 0.004384 |
| layer:15 | 1 | -22.546507 | -19.279044 | -0.000463 | 0.736313 | 0.741588 | 42.561864 | 0.005739 |
| layer:23 | 1 | -16.088099 | -162.471753 | 0.015243 | 0.560064 | 0.578466 | 179.119916 | 0.003159 |
| layer:8 | 1 | -1.604598 | -1.977478 | -0.012694 | 0.919983 | 0.922340 | 4.502058 | 0.015052 |
| length:128 | 2 | -11.362282 | -9.719023 | -0.002311 | 0.826144 | 0.828894 | 21.907449 | 0.005061 |
| length:512 | 2 | -8.846348 | -82.224615 | 0.001274 | 0.740024 | 0.750403 | 91.810987 | 0.009105 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391461e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.835883e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.148095e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
