# Paired mechanism effect — cross-model-probe-v1

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | 0.025632 | 0.000787 | 40 | 32 | 0 | 0.998331 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | -0.000629 | -0.000715 | 3/9/0 | 0.285719 | -0.508521 | 0.222173 |
| family:o | 12 | 0.001151 | -0.000145 | 6/6/0 | -0.061197 | -0.013107 | 0.075455 |
| family:proj | 12 | 0.151213 | 0.014455 | 9/3/0 | -4.129573 | -0.252476 | 4.533261 |
| family:qkv | 36 | 0.000686 | 0.001063 | 22/14/0 | 0.160646 | 0.058078 | -0.218038 |
| role:ffn_in | 12 | -0.000629 | -0.000715 | 3/9/0 | 0.285719 | -0.508521 | 0.222173 |
| role:k | 12 | 0.000621 | 0.000607 | 7/5/0 | 0.298336 | 0.087832 | -0.385548 |
| role:o | 12 | 0.001151 | -0.000145 | 6/6/0 | -0.061197 | -0.013107 | 0.075455 |
| role:proj | 12 | 0.151213 | 0.014455 | 9/3/0 | -4.129573 | -0.252476 | 4.533261 |
| role:q | 12 | 0.000564 | 0.000987 | 7/5/0 | 0.169890 | 0.093809 | -0.263135 |
| role:v | 12 | 0.000873 | 0.001633 | 8/4/0 | 0.013711 | -0.007407 | -0.005432 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 8 | proj | test | 1024 | 3 | -0.285469 | 1.031841 |
| 9 | proj | validation | 1024 | 4 | -0.034544 | 1.009321 |
| 11 | q | test | 128 | 1 | -0.009982 | 1.020344 |
| 0 | proj | validation | 10 | 0 | -0.008241 | 1.903422 |
| 8 | o | test | 128 | 1 | -0.008184 | 1.014136 |
| 5 | o | test | 1024 | 3 | -0.005655 | 1.009308 |
| 2 | o | validation | 10 | 0 | -0.005573 | 1.007489 |
| 10 | ffn_in | validation | 1024 | 4 | -0.004523 | 1.006723 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 83.697 | 113.619 | 29.922 | 1.358 | 72/72 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 26.368 | 28.950 | 2.583 | 1.098 | 72/72 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
