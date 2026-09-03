# Paired mechanism effect — cross-model-probe-v1

- baseline: `parent`
- candidate: `v175`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | 0.000786 | -0.000383 | 34 | 38 | 0 | 1.000816 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | -0.000636 | -0.000337 | 5/7/0 | -0.052844 | -0.058963 | 0.111171 |
| family:o | 12 | -0.006222 | -0.004533 | 4/8/0 | 0.200169 | -0.063445 | -0.142946 |
| family:proj | 12 | 0.011003 | 0.000083 | 6/6/0 | 3.810475 | -0.022850 | -3.776622 |
| family:qkv | 36 | 0.000190 | 0.000177 | 19/17/0 | -0.323712 | 0.051487 | 0.272415 |
| role:ffn_in | 12 | -0.000636 | -0.000337 | 5/7/0 | -0.052844 | -0.058963 | 0.111171 |
| role:k | 12 | 0.002110 | 0.000748 | 10/2/0 | -0.471326 | 0.247131 | 0.226305 |
| role:o | 12 | -0.006222 | -0.004533 | 4/8/0 | 0.200169 | -0.063445 | -0.142946 |
| role:proj | 12 | 0.011003 | 0.000083 | 6/6/0 | 3.810475 | -0.022850 | -3.776622 |
| role:q | 12 | -0.000211 | -0.001068 | 5/7/0 | -0.409895 | -0.073176 | 0.482860 |
| role:v | 12 | -0.001330 | -0.001336 | 4/8/0 | -0.089916 | -0.019495 | 0.108080 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 11 | o | validation | 1024 | 4 | -0.051286 | 1.195725 |
| 9 | o | validation | 512 | 2 | -0.024267 | 1.038308 |
| 0 | proj | validation | 10 | 0 | -0.020915 | 1.043343 |
| 10 | o | test | 1024 | 3 | -0.020571 | 1.034055 |
| 5 | o | test | 1024 | 3 | -0.015871 | 1.027266 |
| 11 | ffn_in | validation | 10 | 0 | -0.013077 | 1.023627 |
| 0 | o | test | 1024 | 3 | -0.009236 | 1.033380 |
| 8 | v | validation | 10 | 0 | -0.008899 | 1.021861 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 76.027 | 84.164 | 8.136 | 1.107 | 72/72 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 30.942 | 27.618 | -3.324 | 0.893 | 72/72 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
