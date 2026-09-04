# Paired mechanism effect — cross-model-probe-v1

- baseline: `candidate`
- candidate: `candidate`
- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)
- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay
- positive Δ gain means the candidate reduced output error on the same case

## 效果总览

| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | - | no_cases |
| Attention overall | 60 | 0.000000 | 0.000000 | 0 | 0 | 60 | 1.000000 | no_effect |

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
| 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## 同机 API 时间差分

| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |
|---|---:|---:|---:|---:|---:|
| hif4_calibration_and_quantize_weight | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_calibration_attention | 16.466 | 33.031 | 16.565 | 2.006 | 12/12 |
| hif4_dynamic_quantize_activation | 0.000 | 0.000 | 0.000 | - | 0/0 |
| hif4_dynamic_quantize_k | 0.695 | 0.640 | -0.055 | 0.920 | 60/60 |
| hif4_dynamic_quantize_q | 0.746 | 0.669 | -0.077 | 0.896 | 60/60 |
| hif4_dynamic_quantize_v | 0.554 | 0.540 | -0.014 | 0.974 | 60/60 |

该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。
