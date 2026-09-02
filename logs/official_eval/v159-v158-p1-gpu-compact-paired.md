# Paired mechanism effect — proxy-v2

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.149185 | 0.138682 | 56 | 0 | 0 | 0.602388 | consistent_improvement |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | 0.109287 | 0.109840 | 16/0/0 | 0.000000 | 0.000000 | 0.000000 |
| family:o | 8 | 0.263614 | 0.281398 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 8 | 0.124236 | 0.116130 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| family:qkv | 24 | 0.145957 | 0.148546 | 24/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 8 | 0.134146 | 0.138415 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_up | 8 | 0.084429 | 0.082175 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:k | 8 | 0.131944 | 0.140783 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 8 | 0.263614 | 0.281398 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 8 | 0.124236 | 0.116130 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:q | 8 | 0.137174 | 0.140875 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 8 | 0.168752 | 0.175134 | 8/0/0 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 0 | fc_up | test | 512 | 7 | 0.071314 | 0.869573 |
| 0 | k | test | 512 | 7 | 0.072313 | 0.573491 |
| 0 | fc_up | validation | 512 | 2 | 0.072733 | 0.871245 |
| 8 | fc_up | test | 128 | 1 | 0.074283 | 0.863500 |
| 8 | fc_up | validation | 128 | 6 | 0.074940 | 0.857245 |
| 0 | k | validation | 512 | 2 | 0.079208 | 0.542559 |
| 23 | proj | validation | 512 | 2 | 0.087210 | 0.592684 |
| 23 | fc_up | test | 128 | 1 | 0.089411 | 0.866261 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 33.398 | 35.432 | 2.034 | 1.061 | 28/28 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 8.697 | 16.888 | 8.191 | 1.942 | 56/56 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
