# Paired mechanism effect — proxy-v2

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.129658 | 0.136473 | 50 | 6 | 0 | 0.606305 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | 0.109316 | 0.109892 | 16/0/0 | -93.401489 | -126.610938 | 220.121743 |
| family:o | 8 | 0.263552 | 0.281451 | 8/0/0 | -14.131941 | -96.012145 | 110.407639 |
| family:proj | 8 | -0.012517 | -0.024082 | 2/6/0 | 462.762831 | -24.963778 | -437.811570 |
| family:qkv | 24 | 0.145980 | 0.148712 | 24/0/0 | -70.105144 | -125.456610 | 195.707735 |
| role:fc_gate | 8 | 0.134188 | 0.138346 | 8/0/0 | -119.997207 | -173.409926 | 293.541321 |
| role:fc_up | 8 | 0.084444 | 0.082175 | 8/0/0 | -66.805771 | -79.811950 | 146.702165 |
| role:k | 8 | 0.132010 | 0.141113 | 8/0/0 | -36.685197 | -160.014665 | 196.831872 |
| role:o | 8 | 0.263552 | 0.281451 | 8/0/0 | -14.131941 | -96.012145 | 110.407639 |
| role:proj | 8 | -0.012517 | -0.024082 | 2/6/0 | 462.762831 | -24.963778 | -437.811570 |
| role:q | 8 | 0.137127 | 0.140929 | 8/0/0 | -141.657959 | -132.442314 | 274.237399 |
| role:v | 8 | 0.168803 | 0.175245 | 8/0/0 | -31.972277 | -83.912851 | 116.053932 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 15 | proj | validation | 128 | 6 | -0.060038 | 1.123534 |
| 15 | proj | test | 128 | 1 | -0.034651 | 1.064549 |
| 0 | proj | test | 128 | 1 | -0.028168 | 1.055709 |
| 0 | proj | validation | 128 | 6 | -0.026843 | 1.053740 |
| 8 | proj | test | 512 | 7 | -0.021322 | 1.033227 |
| 8 | proj | validation | 512 | 2 | -0.017251 | 1.027052 |
| 23 | proj | validation | 512 | 2 | 0.032341 | 0.848952 |
| 23 | proj | test | 512 | 7 | 0.055792 | 0.788048 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 129.945 | 129.448 | -0.497 | 0.996 | 28/28 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 63.178 | 33.195 | -29.984 | 0.525 | 56/56 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
