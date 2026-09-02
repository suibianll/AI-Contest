# Paired mechanism effect — proxy-v2

- baseline: `pre_a3_parent`
- candidate: `v153-fc-decoupled-activation-targeted`
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 14 | -0.013775 | 0.000000 | 0 | 4 | 10 | 1.000000 | consistent_regression |
| Linear focus:fc | 4 | -0.048211 | -0.049511 | 0 | 4 | 0 | 1.078387 | consistent_regression |
| Linear control | 10 | 0.000000 | 0.000000 | 0 | 0 | 10 | 1.000000 | no_effect |
| Attention overall | 1 | 0.000000 | 0.000000 | 0 | 0 | 1 | 1.000000 | no_effect |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 4 | -0.048211 | -0.049511 | 0/4/0 | 0.000000 | 0.209081 | -0.257292 |
| family:o | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |
| family:qkv | 6 | 0.000000 | 0.000000 | 0/0/6 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 2 | -0.046911 | -0.046911 | 0/2/0 | 0.000000 | 0.291500 | -0.338411 |
| role:fc_up | 2 | -0.049511 | -0.049511 | 0/2/0 | 0.000000 | 0.126661 | -0.176173 |
| role:k | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |
| role:q | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 2 | 0.000000 | 0.000000 | 0/0/2 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 1 | fc_gate | validation | 10 | 0 | -0.054502 | 1.081375 |
| 1 | fc_up | test | 128 | 1 | -0.049874 | 1.079423 |
| 0 | fc_up | validation | 10 | 0 | -0.049148 | 1.077351 |
| 0 | fc_gate | validation | 1024 | 4 | -0.039320 | 1.073314 |
| 0 | k | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | o | test | 1024 | 3 | 0.000000 | 1.000000 |
| 0 | proj | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | q | validation | 10 | 0 | 0.000000 | 1.000000 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 146.508 | 143.423 | -3.085 | 0.979 | 168/168 |
| hif4_calibration_attention | 53.363 | 52.897 | -0.466 | 0.991 | 24/24 |
| hif4_dynamic_quantize_activation | 1.366 | 1.317 | -0.049 | 0.964 | 14/14 |
| hif4_dynamic_quantize_k | 0.006 | 0.006 | -0.000 | 0.970 | 1/1 |
| hif4_dynamic_quantize_q | 0.007 | 0.007 | -0.000 | 0.962 | 1/1 |
| hif4_dynamic_quantize_v | 0.007 | 0.007 | -0.000 | 0.956 | 1/1 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
