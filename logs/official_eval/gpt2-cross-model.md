# Paired mechanism effect — cross-model-probe-v1

- baseline: `candidate`
- candidate: `v188-jacobian-port`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |
| Attention overall | 60 | -0.000950 | 0.000000 | 4 | 6 | 50 | 1.000000 | mixed |

## Linear role/family 配对差分

| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |
|---|---:|---:|---:|---:|---:|---:|---:|

W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。

## 最坏 Linear 回归

| layer | role | split | length | window | Δgain | MSE ratio |
|---:|---|---|---:|---:|---:|---:|

## Attention 控制臂差分

| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |
|---:|---:|---:|---:|---:|---:|---:|
| -0.001840 | 0.009041 | 0.000000 | -0.001136 | -0.000950 | 1.596823e-06 | 9.785759e-07 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_calibration_attention | 19.295 | 21.180 | 1.885 | 1.098 | 12/12 |
| hif4_dynamic_quantize_activation | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_k | 0.651 | 0.676 | 0.025 | 1.038 | 60/60 |
| hif4_dynamic_quantize_q | 0.641 | 0.660 | 0.019 | 1.030 | 60/60 |
| hif4_dynamic_quantize_v | 0.496 | 0.560 | 0.065 | 1.130 | 60/60 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
