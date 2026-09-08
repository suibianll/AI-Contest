# L30：全局 reduced-rank 输出残差拟合（REJECTED）

研究循环卡 L-R2（L30），从时间父 L4 构建。2026-09-08 实现并本地验证，**关闭**。

## 机制

`min_{rank(ΔB)≤8} ||R0 − A·ΔB||²` 在校准行空间（A 的列空间）做一次全局 rank-8
reduced-rank regression，一次全局合法投影 + 一次全校准接受判定。目的：替代 L28 的
逐块 rank-8（拼接后整体可高秩），用更简单/更快的调用图达到 ≥L28 的 fit_gain。

## 证据（4B shard0 paired，父 L4）

| 指标 | L4 | L28 | L30 |
|---|---|---|---|
| fit_gain（校准折叠） | ~0.72 | 0.9453 | **0.7065** |
| delta_mean（vs L4 面板） | — | -0.2182 | **-0.000018** |
| api_total（本地 shard0） | — | 217.9s | **322.9s（>300s）** |

**关闭原因**：全局 rank-8 在 N=138 行上压缩过激（reduced-rank regression 秩受限），
连续解无材料收益（fit_gain ≈ L4），且 d×o 大矩阵（proj d=9216×o=2560）导致本地
api_total 322.9s **超 300s**。按证伪判据（"连续解无材料收益 / 时间超预算"）关闭。

## 关闭粒度

只关闭"一次全局 reduced-rank regression 这一实现"。全局低维目标仍 OPEN
（L31 输出目标合法相邻码更新、L32 联合 A/W 低维互逆拟合）。L28 维持 Linear 时间父，
L4 维持组合时间预算父。

## 产物

- 源码：`workbench/continuous_linear/l30-global-reduced-rank/candidate/solution.py`
  （SHA `3E4BBB944D8541F5…`）
- 归档：`solutions/continuous_linear_l30-global-reduced-rank_rejected/`
- 评测：`artifacts/proxy_v3/continuous/linear/l30-global-reduced-rank/shard0/`
- 机制卡：`workbench/continuous_linear/l30-global-reduced-rank/mechanism-card.md`