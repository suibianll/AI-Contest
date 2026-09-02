# Paired mechanism effect — proxy-v2

- baseline: `pre-a3-parent-default`
- candidate: `l5a-linear-perm-mode3-default`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 168 | 0.000199 | 0.000000 | 9 | 5 | 154 | 1.000000 | mixed |
| Linear focus:fc | 48 | 0.000696 | 0.000000 | 9 | 5 | 34 | 1.000000 | mixed |
| Linear control | 120 | 0.000000 | 0.000000 | 0 | 0 | 120 | 1.000000 | no_effect |
| Attention overall | 120 | 0.000000 | 0.000000 | 0 | 0 | 120 | 1.000000 | no_effect |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 48 | 0.000696 | 0.000000 | 9/5/34 | -12.940603 | -3.827682 | 16.768981 |
| family:o | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0/0/72 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 24 | 0.001259 | 0.000000 | 6/3/15 | -24.392953 | -7.026756 | 31.420968 |
| role:fc_up | 24 | 0.000133 | 0.000000 | 3/2/19 | -1.488252 | -0.628609 | 2.116994 |
| role:k | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:q | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 5 | fc_up | validation | 10 | 0 | -0.015006 | 1.029716 |
| 11 | fc_gate | validation | 10 | 0 | -0.005608 | 1.012629 |
| 12 | fc_gate | test | 128 | 1 | -0.002704 | 1.005835 |
| 12 | fc_up | validation | 512 | 2 | -0.001094 | 1.002079 |
| 9 | fc_gate | test | 1024 | 3 | -0.000479 | 1.000998 |
| 0 | fc_gate | validation | 1024 | 4 | 0.000000 | 1.000000 |
| 0 | fc_up | validation | 10 | 0 | 0.000000 | 1.000000 |
| 0 | k | test | 128 | 1 | 0.000000 | 1.000000 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 151.405 | 166.759 | 15.354 | 1.101 | 168/168 |
| hif4_calibration_attention | 69.931 | 59.279 | -10.652 | 0.848 | 24/24 |
| hif4_dynamic_quantize_activation | 18.912 | 17.602 | -1.309 | 0.931 | 168/168 |
| hif4_dynamic_quantize_k | 1.270 | 1.133 | -0.138 | 0.891 | 120/120 |
| hif4_dynamic_quantize_q | 1.659 | 1.561 | -0.098 | 0.941 | 120/120 |
| hif4_dynamic_quantize_v | 1.097 | 0.972 | -0.124 | 0.887 | 120/120 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
