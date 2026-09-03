# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `7319f00e5259fe15e7c5eca99e214a8f7482cf5cf066d6e3025e86c92d9095ec`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.861039125 |
| Overall mean (all captured cases) | 0.861039125 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.861039125 |
| Candidate wall | 21.736s |
| Candidate API total | 21.611s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | 0.063577 | 0.048867 | 3/1/0 | 0.672606 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -15.097853 | -14.165599 | -0.002674 | 0.855559 | 0.861039 | 30.119011 | 0.008155 |
| layer:0 | 1 | -4.007485 | -2.575394 | -0.004362 | 0.922272 | 0.922968 | 7.505151 | 0.005058 |
| layer:15 | 1 | -21.368086 | -18.768516 | -0.004523 | 0.798299 | 0.802976 | 40.934901 | 0.009200 |
| layer:23 | 1 | -33.191091 | -33.398419 | 0.015415 | 0.748050 | 0.763462 | 67.337559 | -0.000002 |
| layer:8 | 1 | -1.824752 | -1.920066 | -0.017227 | 0.953616 | 0.954751 | 4.698433 | 0.018363 |
| length:128 | 2 | -12.687786 | -10.671955 | -0.004442 | 0.860285 | 0.862972 | 24.220026 | 0.007129 |
| length:512 | 2 | -17.507921 | -17.659242 | -0.000906 | 0.850833 | 0.859107 | 36.017996 | 0.009180 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.472250e+04 |
| probability MSE vs reference | 5.931959e-04 | 5.365185e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 1.829004e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
