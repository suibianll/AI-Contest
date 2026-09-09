# v222 FIX-A2 官方结果

用户于 2026-09-09 回传 v222 官方结果：`18015/293s`。

- 候选：`solutions/20260909_v222_attention-a2-correctness-fix_scoreNA_timeNA/`
- 候选 SHA256：`8D3474BB15784E7916C212FE37E94F5D343481DE3353B559FAE35E47C0123699`
- 父：v202 Linear + v195 Attention，`18053/281s`
- 相对父：`−38/+12s`
- 决定：`REJECTED`，根保持 v202 Linear + v195 Attention

v222 修复了 A2 多窗口 mean-gradient 与异常传播，但本地 shard0/shard1 和官方结果均为负向。
该结果只关闭 v222 的具体实现；当前活动计划 R1 从实际部署父 rotation 出发重新定义 hard-event
路径，并固定噪声量级的 center，不修改 v222 归档源码。
