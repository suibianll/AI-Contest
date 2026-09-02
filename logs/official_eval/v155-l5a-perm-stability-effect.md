# v155-l5a-perm-stability-effect — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `effect-panel` / `paired-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 5 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `816ecbf5e253745c5ebfd04233bd04a2b772cf1510641393c7900cdafa0eb4cc`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.588162284 |
| Attention mean | 0.757433277 |
| Overall mean (all captured cases) | 0.602036955 |
| Linear role macro mean | 0.588162284 |
| Attention layer macro mean | 0.757433277 |
| Candidate wall | 219.182s |
| Candidate API total | 207.196s |

## 父版本配对效果

基线：`l3-fc-parent`；候选：`v155-l5a-perm-stability-effect`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.000139 | 0.000000 | 2/0/54 | 1.000000 | consistent_improvement |
| Linear focus:fc | 16 | 0.000487 | 0.000000 | 2/0/14 | 1.000000 | consistent_improvement |
| Linear control | 40 | 0.000000 | 0.000000 | 0/0/40 | 1.000000 | no_effect |
| Attention overall | 5 | 0.000000 | 0.000000 | 0/0/5 | 1.000000 | no_effect |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -191.917114 | -139.262893 | 0.588162 | 331.768169 | 2.062212e+00 | 1.464930e+00 |
| family:fc | 16 | -155.165519 | -137.980736 | 0.444349 | 293.590604 | 2.069563e+00 | 1.601893e+00 |
| family:o | 8 | -282.161928 | -75.941618 | 0.465986 | 358.569532 | 1.940461e+00 | 1.153801e+00 |
| family:proj | 8 | -195.967503 | -234.603821 | 0.458395 | 431.029720 | 2.067174e+00 | 1.798343e+00 |
| family:qkv | 24 | -184.986442 | -129.444446 | 0.768019 | 315.198907 | 2.096242e+00 | 1.366193e+00 |
| role:fc_gate | 8 | -181.311731 | -171.228962 | 0.472051 | 353.012744 | 2.108230e+00 | 1.602878e+00 |
| role:fc_up | 8 | -129.019308 | -104.732510 | 0.416646 | 234.168464 | 2.030896e+00 | 1.600908e+00 |
| role:k | 8 | -260.406160 | -179.078596 | 0.805386 | 440.290142 | 2.188287e+00 | 1.403072e+00 |
| role:o | 8 | -282.161928 | -75.941618 | 0.465986 | 358.569532 | 1.940461e+00 | 1.153801e+00 |
| role:proj | 8 | -195.967503 | -234.603821 | 0.458395 | 431.029720 | 2.067174e+00 | 1.798343e+00 |
| role:q | 8 | -163.171946 | -126.176050 | 0.720156 | 290.068152 | 2.115610e+00 | 1.339571e+00 |
| role:v | 8 | -131.381221 | -83.078691 | 0.778516 | 215.238428 | 1.984829e+00 | 1.355936e+00 |
| shape:hidden_to_hidden | 16 | -222.666937 | -101.058834 | 0.593071 | 324.318842 | 2.028035e+00 | 1.246686e+00 |
| shape:hidden_to_wide | 32 | -175.529605 | -134.529690 | 0.618150 | 310.677445 | 2.078061e+00 | 1.490699e+00 |
| shape:wide_to_hidden | 8 | -195.967503 | -234.603821 | 0.458395 | 431.029720 | 2.067174e+00 | 1.798343e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 5 | -9.581426 | -57.953275 | -0.002313 | 0.751428 | 0.757433 | 68.286129 | 0.008318 |
| layer:0 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| layer:12 | 1 | -7.862716 | -52.715567 | -0.004866 | 0.705221 | 0.709622 | 61.283504 | 0.009267 |
| layer:17 | 1 | -11.341867 | -24.711127 | -0.006043 | 0.759176 | 0.763079 | 36.812170 | 0.009946 |
| layer:23 | 1 | -16.768763 | -195.429871 | 0.011059 | 0.576886 | 0.592705 | 212.775520 | 0.004760 |
| layer:6 | 1 | -11.840970 | -16.903607 | -0.020006 | 0.776882 | 0.778834 | 29.521459 | 0.021958 |
| length:10 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| length:1024 | 2 | -14.055315 | -110.070499 | 0.002508 | 0.668031 | 0.677892 | 124.793845 | 0.007353 |
| length:128 | 1 | -11.840970 | -16.903607 | -0.020006 | 0.776882 | 0.778834 | 29.521459 | 0.021958 |
| length:512 | 1 | -7.862716 | -52.715567 | -0.004866 | 0.705221 | 0.709622 | 61.283504 | 0.009267 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 1.340842e+02 | 2.601625e+04 |
| probability MSE vs reference | 6.110683e-03 | 3.220119e-04 |
| probability KL(reference || estimate) | 1.893840e-02 | 6.457014e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
