# Paired mechanism effect — proxy-v2

- baseline: `candidate`
- candidate: `ablate_seeds2`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.000253 | 0.000000 | 17 | 17 | 22 | 1.000000 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| family:fc | 16 | -0.000119 | 0.000000 | 3/3/10 | -0.014808 | -0.023584 | 0.038272 |
| family:o | 8 | -0.000360 | 0.000397 | 5/3/0 | 0.107206 | 0.025144 | -0.132710 |
| family:proj | 8 | -0.000293 | 0.000000 | 3/3/2 | -0.017732 | -0.016799 | 0.034239 |
| family:qkv | 24 | -0.000292 | 0.000000 | 6/8/10 | -0.810732 | -0.432372 | 1.242812 |
| role:fc_gate | 8 | -0.000367 | 0.000000 | 1/3/4 | -0.017640 | -0.026655 | 0.043927 |
| role:fc_up | 8 | 0.000128 | 0.000000 | 2/0/6 | -0.011975 | -0.020514 | 0.032618 |
| role:k | 8 | -0.000000 | 0.000000 | 3/3/2 | -2.363067 | -1.267044 | 3.630110 |
| role:o | 8 | -0.000360 | 0.000397 | 5/3/0 | 0.107206 | 0.025144 | -0.132710 |
| role:proj | 8 | -0.000293 | 0.000000 | 3/3/2 | -0.017732 | -0.016799 | 0.034239 |
| role:q | 8 | 0.000142 | 0.000000 | 3/1/4 | 0.027818 | -0.007356 | -0.020320 |
| role:v | 8 | -0.001019 | -0.000666 | 0/4/4 | -0.096947 | -0.022717 | 0.118645 |

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|
| 23 | k | validation | 128 | 6 | -0.005090 | 1.038404 |
| 23 | o | validation | 128 | 6 | -0.003234 | 1.009611 |
| 0 | k | validation | 512 | 2 | -0.003217 | 1.034246 |
| 15 | v | validation | 128 | 6 | -0.002792 | 1.012267 |
| 0 | v | validation | 128 | 6 | -0.002277 | 1.019388 |
| 0 | fc_gate | validation | 128 | 6 | -0.001838 | 1.004347 |
| 15 | v | test | 128 | 1 | -0.001749 | 1.007083 |
| 15 | o | validation | 512 | 2 | -0.001720 | 1.004583 |

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 35.550 | 27.234 | -8.316 | 0.766 | 28/28 |
| hif4_calibration_attention | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_activation | 18.151 | 17.026 | -1.125 | 0.938 | 56/56 |
| hif4_dynamic_quantize_k | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_q | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_v | 0.000 | 0.000 | 0.000 | - | 0/0 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。

## Decision: REJECTED (2026-09-03)

- Mean Δgain -2.5e-4 over 56 cases; 17 positive / 17 negative / 22 zero.
- Negatives spread across all role families (fc -1.2e-4, o -3.6e-4, proj -2.9e-4,
  qkv -2.9e-4); v role 4/8 negative (mean -1.0e-3). Worst: k -0.0051, o -0.0032,
  v -0.0028. Seeds 2/3 carry real selected gain; not redundant. Reverted.
