# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `167bf45ddb3ee9eca6e551582bca7ef1fed458af40d227f607dde2afc4e5f4dc`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.568479982 |
| Overall mean (all captured cases) | 0.568479982 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.568479982 |
| Candidate wall | 10.662s |
| Candidate API total | 10.547s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.229273 | -0.062230 | 1/3/0 | 1.511725 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.876377 | -42.881332 | -0.000518 | 0.565309 | 0.568480 | 54.323018 | 0.003690 |
| layer:0 | 1 | -2.124673 | -3.972632 | -0.004158 | 0.862904 | 0.863206 | 6.960208 | 0.004460 |
| layer:15 | 1 | -23.454422 | -25.952034 | -0.000463 | 0.663702 | 0.669626 | 50.070158 | 0.006387 |
| layer:23 | 1 | -16.208197 | -138.524003 | 0.015243 | 0.604360 | 0.617778 | 155.336560 | -0.001825 |
| layer:8 | 1 | -1.718215 | -3.076662 | -0.012694 | 0.130267 | 0.123310 | 4.925144 | 0.005737 |
| length:128 | 2 | -12.789547 | -14.962333 | -0.002311 | 0.763303 | 0.766416 | 28.515183 | 0.005424 |
| length:512 | 2 | -8.963206 | -70.800332 | 0.001274 | 0.367314 | 0.370544 | 80.130852 | 0.001956 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.400697e+04 |
| probability MSE vs reference | 5.931959e-04 | 2.247885e-04 |
| probability KL(reference || estimate) | 3.396776e-03 | 1.274387e-03 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
