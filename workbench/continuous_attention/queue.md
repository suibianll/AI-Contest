# Attention 当前机制队列

唯一明细：[持续研究循环 §7](../../docs/superpowers/plans/workpackages/2026-09-08-continuous-research-loop.md)。
本文件只保存状态，不定义新门禁。

1. **当前身份**：官方最高为 A2 `14440/274s`；时间父为 R3 `14405/238s`；AC0
   `14395/258s` 只作正确性参考。
2. **A29 已裁决**：`continuous_attention_a29-boundary-output` 是 AC0 逐位骨架，分数归属 AC0；
   真正 A29 实现 SHA `D8BAAB46...126A5` 官方 TIMEOUT。只有新建、去重后的降时实现卡才可回访。
3. **下一卡**：A30。随后才是 A31；A32 共享码语义仍需用户单独授权。

4B paired 只检查合法性、可达性和风险，不预测官方符号；本地 `api_seconds` 和按层外推只标记
`time-risk`，不得恢复 `<280s` 提交门。官方分数和官方 300s 是最终裁决。
