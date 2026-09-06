# Attention logit-Fisher pair（REJECTED）

日期：2026-09-06

父版本：v189（源码 SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`）

候选源码 SHA256：`6C3150FAA44020A71C69205E6F8D0B43E6DC0431F6B60B7D10C135E743B5523C`

状态：**REJECTED / F2_REJECTED**。根 `solution.py` 未改，官方状态保持
`unregistered/NA`。

## 机制

在 v189 已冻结的 Q/K 状态之后，按 causal/non-causal softmax logit Fisher
`p * (1 - p)` 加权 Q/K 二维 pair covariance，并应用一次 GQA-local SPD 变换：Q
使用 `M`，K 使用 `M^-T`。不使用 V、输出残差或 output Jacobian；校准统计不进入在线路径。

## 证据

- F0：六 API 可导入；合成 GQA 连续 `QK` 最大绝对误差 `9.54e-7`。
- F1 shard0：均值 delta `+0.013007`，median `0`，L1 `0.015736`，正/负/零 `2/2/4`。
- F1 shard1：均值 delta `+0.005376`，median `+0.002051`，L1 `0.010917`，正/负/零
  `4/2/2`。
- F2 shard2：均值 delta `+0.000355`，L1 `0.001861`，正/负/零 `2/2/4`。
- F2 shard3：均值 delta `0`，`0/0/8`，无变化。
- F2 shard4：均值 delta `-0.001774`，L1 `0.004926`，正/负/零 `1/1/6`，最坏
  delta `-0.026800`（layer 4、validation、length 128）。评测在此停止，shard5、
  OOD、fresh default 和官方提交均未执行。

完整 eval-v3 证据：

- `artifacts/proxy_v3/attention-logit-fisher-pair-20260906/f1/`
- `artifacts/proxy_v3/attention-logit-fisher-pair-20260906/f2/`
