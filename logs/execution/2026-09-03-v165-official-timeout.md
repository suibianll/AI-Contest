# v165 官方超时记录

日期：2026-09-03
来源：用户回传官方评测结果。
候选：`20260903_v165_standard-linear_v161-attn_scoreNA_timeout`
源码 SHA256：`033E85D5DAF1A820BACDB14F9E35183C485E8DD489D118899A1AE3CB491D8C1D`

## 官方结果

| 候选 | Linear | Attention | 官方分数 | 官方时间 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| v165 | standard | v161 Cross-Gram64 refine | `NA` | `>300s` | `TIMEOUT` |

官方没有返回分数，不用本地 proxy 补写分数，不计算 `C_A/G_A/P_A/R_A`，也不把 timeout
解释为精度回退。

## 时间归因

v165 与 v164 都使用 standard Linear；唯一算法差异是 Attention 从 v160 换成 v161 的
Q/K Cross-Gram64 per-call 动态精化。v164 官方为 `13945 / 204s`，v165 超过 `300s`，因此在
官方评测稳定的已知条件下，这条动态路径相对 v164 增加的官方成本至少约 `>96s`。v165 本地
对应 Attention 增量约 `27s`，所以官方/本地增量倍率下界约为 `>3.6×`。超时没有返回完整
耗时，上述数字只能作为下界。

## 精度与后续裁决

- 本地 Qwen Attention default paired `+0.052502`、`106+/14-`，GPT-2 同号；这些结果不能
  推断官方精度。
- v165 不重试，不缩原 Cross-Gram64 路径的 sweep、block 或阈值。
- 当前活动计划继续执行一次同数学目标的低复杂度 rank-2 Gram 残差码本重构。
- 若该重构仍 timeout，当前动态 Gram Attention 目标关闭。
- 新搜集的 logits 偏差校正、V 质心补偿和 K/V 非对称编码放入独立排队计划；当前活动计划
  完成并归档前不得启动。

## 证据

- 本地 Attention：`artifacts/official_eval/sidecal-v165-attn-default.json`
- 本地 Linear control：`artifacts/official_eval/sidecal-v165-linear-default.json`
- 候选结果：`solutions/20260903_v165_standard-linear_v161-attn_scoreNA_timeout/result.md`
- 当前活动计划：`docs/superpowers/plans/2026-09-03-v162-official-side-isolation-optimization-plan.md`
