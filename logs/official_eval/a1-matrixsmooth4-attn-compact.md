# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `7e7df03f4229fb390be45b25dde63447762bdef2512edc38580a71780f916a5b`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.718818653 |
| Overall mean (all captured cases) | 0.718818653 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.718818653 |
| Candidate wall | 10.953s |
| Candidate API total | 10.826s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.078643 | -0.002349 | 0/2/2 | 1.008878 | consistent_regression |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -9.921307 | -46.458202 | -0.000518 | 0.712868 | 0.718819 | 57.092378 | 0.006469 |
| layer:0 | 1 | -0.178057 | -0.143543 | -0.004158 | 0.923383 | 0.923690 | 1.244983 | 0.004465 |
| layer:15 | 1 | -21.829360 | -18.743894 | -0.000463 | 0.724944 | 0.730701 | 41.298198 | 0.006220 |
| layer:23 | 1 | -16.087349 | -164.355329 | 0.015243 | 0.587557 | 0.606165 | 181.030236 | 0.003365 |
| layer:8 | 1 | -1.590463 | -2.590042 | -0.012694 | 0.615589 | 0.614719 | 4.796094 | 0.011824 |
| length:128 | 2 | -11.003708 | -9.443719 | -0.002311 | 0.824163 | 0.827195 | 21.271590 | 0.005343 |
| length:512 | 2 | -8.838906 | -83.472685 | 0.001274 | 0.601573 | 0.610442 | 92.913165 | 0.007595 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.395328e+04 |
| probability MSE vs reference | 5.931959e-04 | 1.074877e-04 |
| probability KL(reference || estimate) | 3.396776e-03 | 4.583634e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
