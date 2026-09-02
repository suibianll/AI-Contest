# Paired mechanism effect — proxy-v2

- baseline: `pre-a3-parent-default`
- candidate: `l5a-linear-perm-stability-default`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 168 | 0.000117 | 0.000000 | 4 | 0 | 164 | 1.000000 | consistent_improvement |
| Linear focus:fc | 48 | 0.000408 | 0.000000 | 4 | 0 | 44 | 1.000000 | consistent_improvement |
| Linear control | 120 | 0.000000 | 0.000000 | 0 | 0 | 120 | 1.000000 | no_effect |
| Attention overall | 120 | 0.000000 | 0.000000 | 0 | 0 | 120 | 1.000000 | no_effect |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 48 | 0.000408 | 0.000000 | 4/0/44 | -3.842236 | -1.234397 | 5.077040 |
| family:o | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0/0/72 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 24 | 0.000604 | 0.000000 | 3/0/21 | -7.282534 | -2.213816 | 9.496954 |
| role:fc_up | 24 | 0.000212 | 0.000000 | 1/0/23 | -0.401937 | -0.254977 | 0.657126 |
| role:k | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:q | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 0 | fc_gate | validation | 1024 | 4 | 0.000000 | 1.000000 |
| 0 | fc_up | validation | 10 | 0 | 0.000000 | 1.000000 |
| 0 | k | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | o | test | 1024 | 3 | 0.000000 | 1.000000 |
| 0 | proj | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | q | validation | 10 | 0 | 0.000000 | 1.000000 |
| 0 | v | validation | 512 | 2 | 0.000000 | 1.000000 |
| 1 | fc_gate | validation | 10 | 0 | 0.000000 | 1.000000 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 151.405 | 169.916 | 18.511 | 1.122 | 168/168 |
| hif4_calibration_attention | 69.931 | 57.900 | -12.031 | 0.828 | 24/24 |
| hif4_dynamic_quantize_activation | 18.912 | 16.699 | -2.212 | 0.883 | 168/168 |
| hif4_dynamic_quantize_k | 1.270 | 1.142 | -0.129 | 0.899 | 120/120 |
| hif4_dynamic_quantize_q | 1.659 | 1.503 | -0.156 | 0.906 | 120/120 |
| hif4_dynamic_quantize_v | 1.097 | 0.962 | -0.135 | 0.877 | 120/120 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
