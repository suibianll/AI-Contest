# Attention 输出 Jacobian 加权 2×2 Q/K 变换

- 状态：**REJECTED_TIME**（准确率/OOD 正向，但时间门失败）
- 日期：2026-09-06
- 父版本：v189
- 候选源码 SHA-256：`cf99ca3ec54754bd3d2a6d8bd7d79b129e71e601d9d8a37e8b130f88d69cf7e7`
- 根 `solution.py`：v186，未修改

## 结果

在 v189 已冻结 Q/K 状态后，候选按 causal/non-causal attention 输出敏感度加权
合法 2×2 reciprocal pair transform。eval-v3 六 shard 的 48 个 Attention case：

- 候选 mean `0.7569128773065631`，v189 mean `0.7527723548408157`，delta
  `+0.00414052246574736`；
- OOD 相对父的 `Δ(in−OOD)≈-0.0010034`，通过 `0.01` 门；
- fresh default（168 Linear + 120 Attention）：Linear
  `0.6402583244298936`（逐位不变），Attention `0.7535836669915871`，Overall
  `0.6874772171639325`；
- API 分解代入时间模型：`T_pred=282.59095457584823s`，不满足 `<280s`。

因此不分配 v190、不提交官方、不把本地分数换算为官方分数。候选源码保留在本目录。

证据：

- [eval-v3 P1 manifest](../../artifacts/proxy_v3/attention-output-jacobian-pair-matrix-20260906/p1b/candidate/manifest.json)
- [eval-v3 P2 OOD manifest](../../artifacts/proxy_v3/attention-output-jacobian-pair-matrix-20260906/p2-ood/candidate/manifest.json)
- [fresh default JSON](../../artifacts/official_eval/attention-output-jacobian-pair-matrix-fresh-default.json)
