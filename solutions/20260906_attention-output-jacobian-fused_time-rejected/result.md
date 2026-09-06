# Attention 输出 Jacobian 加权 pair transform 融合实现

- 状态：**REJECTED_TIME**
- 日期：2026-09-06
- 父版本：v189
- 候选 SHA-256：`38138cc977b3351907d8d95a4200bd249d02fb6283920316c3862bee9462d17a`
- root `solution.py`：v186，未修改

融合实现复用 base calibration context，并把 causal/non-causal 权重计算合并；数学
输出与上一时间拒绝候选几乎逐位一致。eval-v3 Attention mean 为
`0.756914389304788`，父 `0.752772354840816`，delta `+0.00414203446397265`；
fresh default Attention `0.753580573564532`，Linear `0.640258324429894`，Overall
`0.687475928235993`。

时间模型使用 `W=269.289149999153`、`A=60.1939524004702`、
`dyn_act=59.312791100936`、`dyn_qkv=2.903339898446577` 得到
`T_pred=281.99116684437035s`，未通过 `<280s`。因此不分配 v190、不提交官方，
候选仅作时间拒绝归档。

证据：

- [F1 manifest](../../artifacts/proxy_v3/attention-output-jacobian-fused-20260906/f1/candidate/manifest.json)
- [fresh default JSON](../../artifacts/official_eval/attention-output-jacobian-fused-fresh-default.json)
