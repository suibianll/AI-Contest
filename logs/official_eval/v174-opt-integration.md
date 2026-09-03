# Paired mechanism effect — cross-model-probe-v1

- baseline: `parent`
- candidate: `v174`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | -0.177772 | 0.005728 | 54 | 18 | 0 | 0.979781 | mixed |
| Attention overall | 60 | -0.071179 | -0.260012 | 15 | 45 | 0 | 1.351413 | mixed |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | 0.007153 | 0.007471 | 9/3/0 | -0.337909 | 0.025005 | 0.320058 |
| family:o | 12 | 0.024282 | 0.023613 | 11/1/0 | 0.027790 | 0.215078 | -0.218585 |
| family:proj | 12 | -1.113980 | -0.127905 | 4/8/0 | -7.264753 | -1.022413 | 7.173186 |
| family:qkv | 36 | 0.005304 | 0.004910 | 30/6/0 | -0.343179 | 0.116080 | 0.232404 |
| role:ffn_in | 12 | 0.007153 | 0.007471 | 9/3/0 | -0.337909 | 0.025005 | 0.320058 |
| role:k | 12 | 0.004786 | 0.005323 | 10/2/0 | -0.230612 | -0.017320 | 0.252717 |
| role:o | 12 | 0.024282 | 0.023613 | 11/1/0 | 0.027790 | 0.215078 | -0.218585 |
| role:proj | 12 | -1.113980 | -0.127905 | 4/8/0 | -7.264753 | -1.022413 | 7.173186 |
| role:q | 12 | 0.005872 | 0.003646 | 11/1/0 | -0.711357 | 0.262214 | 0.455015 |
| role:v | 12 | 0.005255 | 0.004115 | 9/3/0 | -0.087570 | 0.103345 | -0.010520 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 4 | proj | validation | 1024 | 4 | -7.106221 | 3.978292 |
| 3 | proj | test | 1024 | 3 | -4.559929 | 2.905119 |
| 8 | proj | test | 1024 | 3 | -1.005083 | 1.124500 |
| 10 | proj | validation | 10 | 0 | -0.637421 | 1.042021 |
| 2 | proj | validation | 512 | 2 | -0.190556 | 4.076049 |
| 7 | proj | validation | 512 | 2 | -0.145692 | 1.037988 |
| 9 | proj | validation | 1024 | 4 | -0.110118 | 1.030397 |
| 1 | proj | test | 128 | 1 | -0.026853 | 1.195633 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 48.349719 | 44.006931 | -0.020062 | -0.046499 | -0.071179 | 3.208070e-05 | 8.492547e-05 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 68.684 | 95.122 | 26.438 | 1.385 | 72/72 |
| hif4_calibration_attention | 14.313 | 0.001 | -14.312 | 0.000 | 12/12 |
| hif4_dynamic_quantize_activation | 27.144 | 26.715 | -0.429 | 0.984 | 72/72 |
| hif4_dynamic_quantize_k | 0.656 | 0.107 | -0.549 | 0.163 | 60/60 |
| hif4_dynamic_quantize_q | 0.671 | 0.121 | -0.550 | 0.181 | 60/60 |
| hif4_dynamic_quantize_v | 0.531 | 0.105 | -0.425 | 0.199 | 60/60 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
