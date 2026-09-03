# Paired mechanism effect — cross-model-probe-v1

- baseline: `parent`
- candidate: `v174`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | 0.009767 | 0.006314 | 56 | 16 | 0 | 0.983699 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | 0.005753 | 0.007229 | 10/2/0 | -0.645129 | 0.026131 | 0.624751 |
| family:o | 12 | 0.008455 | 0.011253 | 9/3/0 | -0.246378 | 0.224408 | 0.030424 |
| family:proj | 12 | 0.029100 | 0.002894 | 7/5/0 | 2.995967 | 4.969576 | -7.936443 |
| family:qkv | 36 | 0.005098 | 0.005409 | 30/6/0 | -0.462338 | 0.116309 | 0.351127 |
| role:ffn_in | 12 | 0.005753 | 0.007229 | 10/2/0 | -0.645129 | 0.026131 | 0.624751 |
| role:k | 12 | 0.005603 | 0.006727 | 10/2/0 | 0.200076 | 0.229054 | -0.423528 |
| role:o | 12 | 0.008455 | 0.011253 | 9/3/0 | -0.246378 | 0.224408 | 0.030424 |
| role:proj | 12 | 0.029100 | 0.002894 | 7/5/0 | 2.995967 | 4.969576 | -7.936443 |
| role:q | 12 | 0.005530 | 0.003175 | 11/1/0 | -1.021047 | 0.035807 | 0.990771 |
| role:v | 12 | 0.004162 | 0.005409 | 9/3/0 | -0.566042 | 0.084066 | 0.486138 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 11 | proj | test | 128 | 1 | -0.066667 | 1.102838 |
| 7 | o | validation | 10 | 0 | -0.038752 | 1.095431 |
| 4 | o | validation | 512 | 2 | -0.026319 | 1.050828 |
| 4 | proj | validation | 1024 | 4 | -0.024772 | 1.053443 |
| 11 | ffn_in | validation | 10 | 0 | -0.020392 | 1.036844 |
| 5 | proj | validation | 10 | 0 | -0.011289 | 1.031565 |
| 3 | v | validation | 10 | 0 | -0.010304 | 1.036228 |
| 8 | v | validation | 10 | 0 | -0.007303 | 1.017941 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 76.027 | 98.028 | 22.000 | 1.289 | 72/72 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 30.942 | 27.962 | -2.980 | 0.904 | 72/72 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
