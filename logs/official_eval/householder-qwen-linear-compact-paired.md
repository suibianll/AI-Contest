# Paired mechanism effect — proxy-v2

- baseline: `nohouseholder-qwen-linear-compact`
- candidate: `householder-qwen-linear-compact`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.006318 | -0.005958 | 8 | 48 | 0 | 1.019967 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | -0.008257 | -0.008189 | 0/16/0 | 4.925311 | -1.572830 | -3.360738 |
| family:o | 8 | 0.001504 | 0.002494 | 4/4/0 | -7.419124 | 0.384512 | 7.036117 |
| family:proj | 8 | -0.004510 | -0.004623 | 1/7/0 | 19.118282 | 0.622985 | -19.745777 |
| family:qkv | 24 | -0.008235 | -0.006736 | 3/21/0 | 30.679603 | 1.472404 | -32.160242 |
| role:fc_gate | 8 | -0.011306 | -0.009737 | 0/8/0 | 7.373139 | -2.506292 | -4.878152 |
| role:fc_up | 8 | -0.005208 | -0.005887 | 0/8/0 | 2.477483 | -0.639368 | -1.843323 |
| role:k | 8 | -0.008182 | -0.009145 | 2/6/0 | 47.089015 | 6.280718 | -53.377914 |
| role:o | 8 | 0.001504 | 0.002494 | 4/4/0 | -7.419124 | 0.384512 | 7.036117 |
| role:proj | 8 | -0.004510 | -0.004623 | 1/7/0 | 19.118282 | 0.622985 | -19.745777 |
| role:q | 8 | -0.006362 | -0.004809 | 0/8/0 | 34.330482 | -1.541309 | -32.795535 |
| role:v | 8 | -0.010162 | -0.006736 | 1/7/0 | 10.619312 | -0.322196 | -10.307278 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 15 | v | test | 128 | 1 | -0.030329 | 1.122846 |
| 23 | v | test | 512 | 7 | -0.021790 | 1.129001 |
| 23 | k | test | 128 | 1 | -0.021423 | 1.131925 |
| 23 | v | validation | 512 | 2 | -0.018197 | 1.118695 |
| 23 | k | validation | 128 | 6 | -0.017709 | 1.133629 |
| 15 | fc_gate | test | 128 | 1 | -0.015514 | 1.043844 |
| 0 | fc_gate | validation | 128 | 6 | -0.015485 | 1.036619 |
| 0 | fc_gate | test | 128 | 1 | -0.015165 | 1.034067 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 28.628 | 29.469 | 0.840 | 1.029 | 28/28 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 17.423 | 17.918 | 0.495 | 1.028 | 56/56 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
