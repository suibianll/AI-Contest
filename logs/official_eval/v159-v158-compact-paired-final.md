# Paired mechanism effect — proxy-v2

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.149191 | 0.138823 | 56 | 0 | 0 | 0.602279 | consistent_improvement |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | 0.109316 | 0.109892 | 16/0/0 | -93.401489 | -126.610938 | 220.121743 |
| family:o | 8 | 0.263552 | 0.281451 | 8/0/0 | -14.131941 | -96.012145 | 110.407639 |
| family:proj | 8 | 0.124209 | 0.115945 | 8/0/0 | 462.762831 | -24.949570 | -437.689051 |
| family:qkv | 24 | 0.145980 | 0.148712 | 24/0/0 | -70.105144 | -125.456610 | 195.707735 |
| role:fc_gate | 8 | 0.134188 | 0.138346 | 8/0/0 | -119.997207 | -173.409926 | 293.541321 |
| role:fc_up | 8 | 0.084444 | 0.082175 | 8/0/0 | -66.805771 | -79.811950 | 146.702165 |
| role:k | 8 | 0.132010 | 0.141113 | 8/0/0 | -36.685197 | -160.014665 | 196.831872 |
| role:o | 8 | 0.263552 | 0.281451 | 8/0/0 | -14.131941 | -96.012145 | 110.407639 |
| role:proj | 8 | 0.124209 | 0.115945 | 8/0/0 | 462.762831 | -24.949570 | -437.689051 |
| role:q | 8 | 0.137127 | 0.140929 | 8/0/0 | -141.657959 | -132.442314 | 274.237399 |
| role:v | 8 | 0.168803 | 0.175245 | 8/0/0 | -31.972277 | -83.912851 | 116.053932 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 0 | fc_up | test | 512 | 7 | 0.071333 | 0.869538 |
| 0 | k | test | 512 | 7 | 0.072015 | 0.574504 |
| 0 | fc_up | validation | 512 | 2 | 0.072728 | 0.871255 |
| 8 | fc_up | test | 128 | 1 | 0.074354 | 0.863369 |
| 8 | fc_up | validation | 128 | 6 | 0.075019 | 0.857094 |
| 0 | k | validation | 512 | 2 | 0.078918 | 0.543471 |
| 23 | proj | validation | 512 | 2 | 0.087186 | 0.592798 |
| 23 | fc_up | test | 128 | 1 | 0.089332 | 0.866380 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 129.945 | 131.693 | 1.748 | 1.013 | 28/28 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 63.178 | 35.877 | -27.302 | 0.568 | 56/56 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
