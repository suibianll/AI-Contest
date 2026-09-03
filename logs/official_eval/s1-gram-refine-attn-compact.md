# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `27eee4710b0170384a17e2f3e9ab87b3437e7b224883150d70bebf8a5fb11848`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.819033405 |
| Overall mean (all captured cases) | 0.819033405 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.819033405 |
| Candidate wall | 10.520s |
| Candidate API total | 10.401s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | 0.021571 | 0.000561 | 2/2/0 | 0.984245 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.105570 | -47.009405 | -0.000518 | 0.813472 | 0.819033 | 57.928447 | 0.006080 |
| layer:0 | 1 | -0.175062 | -0.383919 | -0.004158 | 0.926229 | 0.926402 | 1.485210 | 0.004332 |
| layer:15 | 1 | -22.540111 | -18.950732 | -0.000463 | 0.823937 | 0.828504 | 42.314780 | 0.005030 |
| layer:23 | 1 | -16.093087 | -166.615009 | 0.015243 | 0.588471 | 0.604575 | 183.296568 | 0.000861 |
| layer:8 | 1 | -1.614021 | -2.087958 | -0.012694 | 0.915251 | 0.916653 | 4.617230 | 0.014096 |
| length:128 | 2 | -11.357587 | -9.667326 | -0.002311 | 0.875083 | 0.877453 | 21.899995 | 0.004681 |
| length:512 | 2 | -8.853554 | -84.351484 | 0.001274 | 0.751861 | 0.760614 | 93.956899 | 0.007479 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391174e+04 |
| probability MSE vs reference | 5.931959e-04 | 5.293157e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 1.610103e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
