# Attention logit-Fisher pair 执行记录

日期：2026-09-06

计划：`2026-09-06-attention-logit-fisher-pair-plan`

状态：**CLOSED / F2_REJECTED**

父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

候选源码：`workbench/attention_logit_fisher_pair_solution.py`

候选 SHA256：`6C3150FAA44020A71C69205E6F8D0B43E6DC0431F6B60B7D10C135E743B5523C`

## 执行

- F0 通过：六 API 可导入；合成 GQA 连续 `QK` 最大绝对误差 `9.54e-7`。
- F1 shard0：mean delta `+0.013007`，median `0`，L1 `0.015736`，正/负/零
  `2/2/4`。
- F1 shard1：mean delta `+0.005376`，median `+0.002051`，L1 `0.010917`，正/负/零
  `4/2/2`。
- F2 shard2：mean delta `+0.000355`，L1 `0.001861`，正/负/零 `2/2/4`。
- F2 shard3：mean delta `0`，正/负/零 `0/0/8`。
- F2 shard4：mean delta `-0.001774`，L1 `0.004926`，正/负/零 `1/1/6`；最坏
  delta `-0.026800`（layer 4、validation、length 128）。评测在此停止，shard5 未运行。

F1/F2 原始 eval-v3 结果：

- `artifacts/proxy_v3/attention-logit-fisher-pair-20260906/f1/`
- `artifacts/proxy_v3/attention-logit-fisher-pair-20260906/f2/`

结论：Attention logit-Fisher pair 机制关闭为 **REJECTED**，不运行 OOD/default，不提交
官方，不扫 Fisher 权重或其它参数邻域；根 `solution.py` 仍为 v186。
