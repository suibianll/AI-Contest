# standard-linear + v229-attn 侧隔离官方回传（2026-09-10）

用户原话：“standard_linear_v229分数为14424，时间为245”（原文误写 v299，按唯一登记的
standard-linear_v229-attn 探针绑定）。

- 探针：`solutions/20260910_standard-linear_v229-attn_scoreNA_timeNA/solution.py`，SHA256
  `ffe736a3face76defe06fc4b66c63b09364359ad524c27bd7a7cede4e7eea931`（构建与逐位核验见
  `logs/execution/2026-09-10-attention-amc1-k-mean-recenter.md` 附录）。
- 官方侧隔离结果：**14424 / 245s**。
- 对照基准：同表 `standard-linear_v195-attn`（v195 系根 Attention，即 v229 的父 Attention）
  **14426 / 243s**；侧隔离总基线（标准 Linear + R3）为 `14405/238s`。

## 裁决

1. **A-MC1（K 侧 per-call 均值再定心）官方侧价值 = −2（14424 − 14426），无正向价值。**
   本地六 shard `+0.014923`（26/10/36）未迁移官方，方向反转；按规则该路线（K 平移 /
   per-call 定心）不再用本地 proxy 晋级。±1~4 不证明随机噪声，保留实际裁决 −2。
2. **完整包 v229 的 TIMEOUT 未在侧隔离复现**：侧隔离形态 245s < 300s（相对 v195 侧
   仅 +2s），说明完整包超时不是 A-MC1 的 per-call 成本（一次 reduce+减法）单独导致；
   同时再次确认既有规则——**侧隔离时间对完整包时间没有预测力**，不从本次 +2s 反推
   完整包的超时根因或折算系数。
3. v229 至此精度与时间两个维度都有官方裁决：完整包 TIMEOUT + 侧隔离 −2，
   **K 平移类（含 per-call 均值再定心）正式关闭**，不以 mean/midrange/中位数/trimmed
   变体或降成本 gate 重试（A-QC1 已 NO_EFFECT，两侧 per-call 定心规则均有定论）。
4. 根不变：v230 Linear（L-EM2）+ v195 Attention，`18428/292s`。

同步更新：A-MC1 执行日志附录、A-MC1 计划 §7、`solutions/README.md` 侧隔离表、
`docs/closed-mechanism-evidence.md`、`docs/current-solution-status.md`、
`docs/superpowers/plans/README.md` 与 A-QC1 计划中的 v229 引用。未重跑任何评测，
未修改任何候选源码。
