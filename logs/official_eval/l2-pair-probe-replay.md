# Paired mechanism effect — proxy-v2

- baseline: `l3-fc-parent`
- candidate: `l2-pair-probe`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.089737 | 0.000000 | 0 | 16 | 40 | 1.000000 | consistent_regression |
| Linear focus:fc | 16 | -0.314079 | -0.314713 | 0 | 16 | 0 | 1.636124 | consistent_regression |
| Linear control | 40 | 0.000000 | 0.000000 | 0 | 0 | 40 | 1.000000 | no_effect |
| Attention overall | 5 | 0.000000 | 0.000000 | 0 | 0 | 5 | 1.000000 | no_effect |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | -0.314079 | -0.314713 | 0/16/0 | -753.978911 | 79.116720 | 674.548112 |
| family:o | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| family:proj | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| family:qkv | 24 | 0.000000 | 0.000000 | 0/0/24 | 0.000000 | 0.000000 | 0.000000 |
| role:fc_gate | 8 | -0.317206 | -0.313905 | 0/8/0 | -1095.740653 | 88.414357 | 1007.009091 |
| role:fc_up | 8 | -0.310953 | -0.316111 | 0/8/0 | -412.217168 | 69.819083 | 342.087133 |
| role:k | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:o | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:proj | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:q | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |
| role:v | 8 | 0.000000 | 0.000000 | 0/0/8 | 0.000000 | 0.000000 | 0.000000 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 16 | fc_gate | validation | 10 | 0 | -0.426692 | 2.206704 |
| 20 | fc_up | validation | 10 | 0 | -0.415433 | 1.927700 |
| 16 | fc_up | test | 128 | 1 | -0.405271 | 1.910561 |
| 13 | fc_up | test | 1024 | 3 | -0.365183 | 1.777554 |
| 20 | fc_gate | validation | 1024 | 4 | -0.344611 | 1.786567 |
| 10 | fc_gate | validation | 1024 | 4 | -0.338732 | 1.759890 |
| 10 | fc_up | validation | 10 | 0 | -0.331502 | 1.660819 |
| 13 | fc_gate | validation | 512 | 2 | -0.328706 | 1.710786 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 142.898 | 129.948 | -12.950 | 0.909 | 168/168 |
| hif4_calibration_attention | 53.287 | 50.108 | -3.179 | 0.940 | 24/24 |
| hif4_dynamic_quantize_activation | 6.003 | 5.672 | -0.331 | 0.945 | 56/56 |
| hif4_dynamic_quantize_k | 0.042 | 0.047 | 0.005 | 1.115 | 5/5 |
| hif4_dynamic_quantize_q | 0.053 | 0.052 | -0.001 | 0.979 | 5/5 |
| hif4_dynamic_quantize_v | 0.035 | 0.038 | 0.003 | 1.086 | 5/5 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
