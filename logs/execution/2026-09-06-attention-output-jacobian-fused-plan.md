# Attention 输出 Jacobian 加权 pair transform 融合实现记录

日期：2026-09-06  
计划：`2026-09-06-attention-output-jacobian-fused-plan`  
状态：**CLOSED / F2_TIME_REJECTED**  
父版本：v189，SHA `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`  
候选 SHA-256：`38138cc977b3351907d8d95a4200bd249d02fb6283920316c3862bee9462d17a`

## F0/F1

融合 causal/non-causal Jacobian estimator 与上一候选的权重最大差约 `2.4e-7`，
`py_compile`、导入、有限输出、合法 state 和连续 Q·K 不变量检查通过。eval-v3 六
Attention shard 的 48 个配对 case：候选 mean `0.756914389304788`，v189 mean
`0.752772354840816`，delta `+0.00414203446397265`。

六个 shard delta mean 为：
`+0.013525590458857192`、`+0.005133423918639052`、`+0.005621091286669697`、
`+0.002732973706860764`、`-0.002160872587190721`、`0`。Linear control 未调用。

## F2

OOD 配对保持输出-Jacobian候选的门内结果。fresh default（proxy-v2 兼容后端，168
Linear + 120 Attention）为：

- Linear `0.640258324429894`，与 v189 逐位相同；
- Attention `0.753580573564532`，v189 `0.75217340702007`；
- Overall `0.687475928235993`；
- `W_calib=269.289149999153s`、`A_calib=60.1939524004702s`、
  `dyn_act=59.312791100936s`、`dyn_qkv=2.903339898446577s`；
- 时间模型 `T_pred=281.99116684437035s`，超过 `<280s`。

## 决策

融合实现没有消除时间门失败，关闭该计划，不分配 v190、不提交官方、不扫描数学参数。
候选源码和证据保存在
`solutions/20260906_attention-output-jacobian-fused_time-rejected/`。

证据：

- F1：`artifacts/proxy_v3/attention-output-jacobian-fused-20260906/f1/candidate/manifest.json`
- F2 OOD：`artifacts/proxy_v3/attention-output-jacobian-fused-20260906/f2-ood/candidate/manifest.json`
- fresh default：`artifacts/official_eval/attention-output-jacobian-fused-fresh-default.json`
