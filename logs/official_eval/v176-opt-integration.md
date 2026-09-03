# Paired mechanism effect — cross-model-probe-v1

- baseline: `parent`
- candidate: `v176`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 72 | -0.059836 | -0.588537 | 9 | 63 | 0 | 2.432128 | mixed |
| Attention overall | 60 | -0.021851 | -0.001365 | 30 | 30 | 0 | 1.002361 | mixed |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 12 | -0.561079 | -0.580433 | 0/12/0 | 276.156941 | 303.205728 | -579.923748 |
| family:o | 12 | -0.390792 | -0.396307 | 0/12/0 | 155.914685 | 120.827673 | -277.133149 |
| family:proj | 12 | 2.590104 | 1.568340 | 9/3/0 | 1093.443089 | 270.427315 | -1361.280299 |
| family:qkv | 36 | -0.665750 | -0.694482 | 0/36/0 | 188.824815 | 142.092109 | -331.582674 |
| role:ffn_in | 12 | -0.561079 | -0.580433 | 0/12/0 | 276.156941 | 303.205728 | -579.923748 |
| role:k | 12 | -0.730368 | -0.727193 | 0/12/0 | 264.714593 | 195.907645 | -461.352606 |
| role:o | 12 | -0.390792 | -0.396307 | 0/12/0 | 155.914685 | 120.827673 | -277.133149 |
| role:proj | 12 | 2.590104 | 1.568340 | 9/3/0 | 1093.443089 | 270.427315 | -1361.280299 |
| role:q | 12 | -0.705139 | -0.697159 | 0/12/0 | 190.222155 | 147.706979 | -338.634272 |
| role:v | 12 | -0.561743 | -0.572096 | 0/12/0 | 111.537697 | 82.661703 | -194.761143 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 0 | proj | validation | 10 | 0 | -0.991786 | 121.743980 |
| 2 | proj | validation | 512 | 2 | -0.938052 | 16.142474 |
| 1 | proj | test | 128 | 1 | -0.862736 | 7.285225 |
| 3 | q | test | 1024 | 3 | -0.852640 | 6.786096 |
| 3 | k | validation | 1024 | 4 | -0.852513 | 6.780261 |
| 2 | k | test | 1024 | 3 | -0.834708 | 6.049906 |
| 2 | q | validation | 512 | 2 | -0.823132 | 5.653948 |
| 4 | q | validation | 1024 | 4 | -0.801700 | 5.042875 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| -0.133163 | 0.102418 | 0.000000 | -0.024583 | -0.021851 | 3.574395e-06 | 6.419712e-06 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 68.684 | 0.288 | -68.395 | 0.004 | 72/72 |
| hif4_calibration_attention | 14.313 | 15.583 | 1.270 | 1.089 | 12/12 |
| hif4_dynamic_quantize_activation | 27.144 | 0.431 | -26.713 | 0.016 | 72/72 |
| hif4_dynamic_quantize_k | 0.656 | 0.680 | 0.024 | 1.037 | 60/60 |
| hif4_dynamic_quantize_q | 0.671 | 0.705 | 0.034 | 1.051 | 60/60 |
| hif4_dynamic_quantize_v | 0.531 | 0.578 | 0.047 | 1.089 | 60/60 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
