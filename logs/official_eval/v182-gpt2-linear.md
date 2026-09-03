# Paired mechanism effect — cross-model-probe-v1

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | 0.001171 | 0.000307 | 37 | 35 | 0 | 0.998737 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | -0.000181 | -0.000524 | 5/7/0 | -0.200083 | -0.234788 | 0.434690 |
| family:o | 12 | 0.005994 | 0.004295 | 9/3/0 | -0.039222 | -0.066727 | 0.111943 |
| family:proj | 12 | 0.002841 | 0.001396 | 7/5/0 | 0.557309 | 0.124355 | -0.678823 |
| family:qkv | 36 | -0.000544 | -0.000425 | 16/20/0 | 0.040333 | 0.042530 | -0.083407 |
| role:ffn_in | 12 | -0.000181 | -0.000524 | 5/7/0 | -0.200083 | -0.234788 | 0.434690 |
| role:k | 12 | -0.001639 | -0.001444 | 4/8/0 | 0.677408 | 0.076938 | -0.755984 |
| role:o | 12 | 0.005994 | 0.004295 | 9/3/0 | -0.039222 | -0.066727 | 0.111943 |
| role:proj | 12 | 0.002841 | 0.001396 | 7/5/0 | 0.557309 | 0.124355 | -0.678823 |
| role:q | 12 | 0.000074 | 0.000689 | 7/5/0 | -0.513053 | 0.029566 | 0.483561 |
| role:v | 12 | -0.000066 | -0.000577 | 5/7/0 | -0.043355 | 0.021086 | 0.022203 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 0 | q | validation | 10 | 0 | -0.012779 | 1.022423 |
| 6 | proj | test | 128 | 1 | -0.012407 | 1.026707 |
| 8 | o | test | 128 | 1 | -0.011407 | 1.017257 |
| 1 | o | validation | 1024 | 4 | -0.009817 | 1.025457 |
| 9 | k | validation | 10 | 0 | -0.009735 | 1.032478 |
| 3 | v | validation | 10 | 0 | -0.007311 | 1.026338 |
| 6 | ffn_in | validation | 10 | 0 | -0.004901 | 1.010872 |
| 4 | k | validation | 10 | 0 | -0.004567 | 1.037162 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 84.260 | 113.290 | 29.030 | 1.345 | 72/72 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 27.450 | 27.612 | 0.162 | 1.006 | 72/72 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
