# Paired mechanism effect — proxy-v2

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.027151 | 0.000000 | 24 | 0 | 32 | 1.000000 | consistent_improvement |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | 0.026667 | 0.020368 | 16/0/0 | -0.033614 | -0.008048 | 0.068328 |
| family:o | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 8 | 0.136727 | 0.149298 | 8/0/0 | 0.000000 | 0.014208 | 0.122519 |
| family:qkv | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 8 | 0.029074 | 0.020688 | 8/0/0 | -0.040800 | -0.019869 | 0.089743 |
| role:fc_up | 8 | 0.024259 | 0.017643 | 8/0/0 | -0.026428 | 0.003773 | 0.046914 |
| role:k | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 8 | 0.136727 | 0.149298 | 8/0/0 | 0.000000 | 0.014208 | 0.122519 |
| role:q | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 0 | k | validation | 512 | 2 | 0.000000 | 1.000000 |
| 0 | k | test | 512 | 7 | 0.000000 | 1.000000 |
| 0 | o | validation | 512 | 2 | 0.000000 | 1.000000 |
| 0 | o | test | 512 | 7 | 0.000000 | 1.000000 |
| 0 | q | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | q | validation | 128 | 6 | 0.000000 | 1.000000 |
| 0 | v | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | v | validation | 128 | 6 | 0.000000 | 1.000000 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 142.626 | 131.693 | -10.933 | 0.923 | 28/28 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 32.921 | 35.877 | 2.956 | 1.090 | 56/56 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
