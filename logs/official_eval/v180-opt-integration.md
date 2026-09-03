# Paired mechanism effect — cross-model-probe-v1

- baseline: `parent`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | -0.237432 | -0.001109 | 22 | 50 | 0 | 1.003069 | mixed |
| Attention overall | 60 | -0.000208 | -0.000593 | 28 | 32 | 0 | 1.000540 | mixed |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | -0.000112 | 0.000076 | 7/5/0 | 0.154327 | 0.268091 | -0.422530 |
| family:o | 12 | -0.005829 | -0.005561 | 2/10/0 | 0.096341 | 0.111310 | -0.213480 |
| family:proj | 12 | -1.415832 | -0.144985 | 1/11/0 | -22.001237 | -0.311452 | 20.896857 |
| family:qkv | 36 | -0.000940 | -0.000644 | 12/24/0 | -0.090269 | 0.014871 | 0.074458 |
| role:ffn_in | 12 | -0.000112 | 0.000076 | 7/5/0 | 0.154327 | 0.268091 | -0.422530 |
| role:k | 12 | -0.000556 | -0.000556 | 5/7/0 | -0.315503 | -0.095897 | 0.410845 |
| role:o | 12 | -0.005829 | -0.005561 | 2/10/0 | 0.096341 | 0.111310 | -0.213480 |
| role:proj | 12 | -1.415832 | -0.144985 | 1/11/0 | -22.001237 | -0.311452 | 20.896857 |
| role:q | 12 | -0.000066 | -0.000558 | 3/9/0 | 0.102524 | 0.173362 | -0.275952 |
| role:v | 12 | -0.002197 | -0.001341 | 4/8/0 | -0.057827 | -0.032851 | 0.088482 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 4 | proj | validation | 1024 | 4 | -9.858657 | 5.131867 |
| 3 | proj | test | 1024 | 3 | -4.076552 | 2.703166 |
| 10 | proj | validation | 10 | 0 | -1.550978 | 1.102247 |
| 8 | proj | test | 1024 | 3 | -0.892421 | 1.110544 |
| 11 | proj | test | 128 | 1 | -0.155535 | 1.082530 |
| 5 | proj | validation | 10 | 0 | -0.151072 | 1.054603 |
| 2 | proj | validation | 512 | 2 | -0.138898 | 3.242156 |
| 7 | proj | validation | 512 | 2 | -0.105523 | 1.027514 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| -0.016306 | -0.004987 | 0.000000 | -0.000122 | -0.000208 | -4.417640e-07 | -6.907138e-07 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 68.684 | 93.182 | 24.499 | 1.357 | 72/72 |
| hif4_calibration_attention | 14.313 | 17.837 | 3.524 | 1.246 | 12/12 |
| hif4_dynamic_quantize_activation | 27.144 | 28.618 | 1.474 | 1.054 | 72/72 |
| hif4_dynamic_quantize_k | 0.656 | 0.686 | 0.030 | 1.046 | 60/60 |
| hif4_dynamic_quantize_q | 0.671 | 0.694 | 0.023 | 1.034 | 60/60 |
| hif4_dynamic_quantize_v | 0.531 | 0.551 | 0.021 | 1.039 | 60/60 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
