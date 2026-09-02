# Paired mechanism effect — proxy-v2

- baseline: `v152-parent-56`
- candidate: `v152-fc-cat-off-56`
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.000186 | 0.000000 | 3 | 3 | 50 | 1.000000 | mixed |
| Linear focus:fc | 16 | 0.000653 | 0.000000 | 3 | 3 | 10 | 1.000000 | mixed |
| Linear control | 40 | 0.000000 | 0.000000 | 0 | 0 | 40 | 1.000000 | no_effect |
| Attention overall | 1 | 0.000000 | 0.000000 | 0 | 0 | 1 | 1.000000 | no_effect |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | 0.000653 | 0.000000 | 3/3/10 | 15.918874 | 1.074829 | -16.993051 |
| family:o | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| family:qkv | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 8 | 0.001871 | 0.000000 | 3/1/4 | 26.071698 | 2.267701 | -28.337528 |
| role:fc_up | 8 | -0.000565 | 0.000000 | 0/2/6 | 5.766051 | -0.118043 | -5.648573 |
| role:k | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:q | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 7 | fc_gate | test | 128 | 1 | -0.007283 | 1.014428 |
| 0 | fc_up | validation | 10 | 0 | -0.003087 | 1.004858 |
| 1 | fc_up | test | 128 | 1 | -0.001435 | 1.002285 |
| 0 | k | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | o | test | 1024 | 3 | 0.000000 | 1.000000 |
| 0 | proj | test | 128 | 1 | 0.000000 | 1.000000 |
| 0 | q | validation | 10 | 0 | 0.000000 | 1.000000 |
| 0 | v | validation | 512 | 2 | 0.000000 | 1.000000 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 142.119 | 142.246 | 0.128 | 1.001 | 168/168 |
| hif4_calibration_attention | 53.554 | 52.659 | -0.896 | 0.983 | 24/24 |
| hif4_dynamic_quantize_activation | 5.424 | 5.505 | 0.081 | 1.015 | 56/56 |
| hif4_dynamic_quantize_k | 0.006 | 0.007 | 0.001 | 1.217 | 1/1 |
| hif4_dynamic_quantize_q | 0.009 | 0.008 | -0.001 | 0.935 | 1/1 |
| hif4_dynamic_quantize_v | 0.007 | 0.006 | -0.001 | 0.907 | 1/1 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
