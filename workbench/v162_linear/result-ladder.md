# L 侧独立分支：RECOVERY 阶梯结果报告（L1–L4）

> 日期：2026-09-06。契约：[总计划](../../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> + [L 任务书](../../docs/superpowers/plans/workpackages/v162-linear.md)。
> 执行日志：`logs/execution/v162-independent-linear.md`。根 v189 未改动。

## 结论（TL;DR）

1. v162 独立 Linear 分支已通过四个 RECOVERY 版本（L1 v160 栈 → L2 rank-2 →
   L3 块序 → L4 +4 码窗）从零点 0 追平已知本地最高 **0.6368**（eval-v3 六 shard，
   n=336）；L4 与 v189 官方父的 Linear 侧**逐位一致**（median/min/max 相同，
   fresh default linear_mean 0.640258324430 与 v189 文档值完全一致）。
2. 全部门禁通过：合法性/finite/case 身份、冻结 Attention 侧逐位 control、
   28/28 state 可达、非正 case 0、OOD 全链 |Δgap| ≤ 0.0018、时间模型
   **249.4s < 280s**。
3. 未超过本地最高（持平），按用户规则未触发"超过即提交官方"的晋级；根 v189
   不变，正式提交/组合由协调者决定。后续进入 L5+ 新机制阶段，目标 0.9。

## 版本链

| 版本 | 机制（来源） | eval-v3 mean | 单步 Δ | 官方锚 | SHA |
|---|---|---|---|---|---|
| L0 零点 | v162 全标准 | 0.0 | — | 1001/146s | `56101559...C000A` |
| L1 | v160 Linear 栈整体（v163，官方 4587/202s） | 0.628182 | +0.6282 | 4587/202s | `3352BDEC...3EB612` |
| L2 | + rank-2 残差重分布（v182 Linear 侧） | 0.632762 | +0.004580 | 17598/273s（组合） | `AFD6F116...BE361A` |
| L3 | + 静态 Hdiag activation-GPTQ 64-block 块序（v189，无 +4 窗） | 0.636705 | +0.003944 | — | `7A89A87B...966433` |
| L4 | + `_DYNAMIC_OFFSETS` +4 码（v186 常量）＝v189 Linear 侧逐位复现 | 0.636799 | +0.000094 | 17616/275s（组合） | `ACB16F76...F5263` |

每版一个机制、单一预注册配置、无邻域扫描；全部标记 **RECOVERY**（已验证机制
重现，不宣称新突破）。

## 门禁明细（L4 为阶梯终点候选）

- **合法性/control**：四个 Attention API 与 v162 标准块在真实 Q/K/V 输入、
  真实调用顺序下五字段/state/输出逐位一致（shard0 全部 case + 逆序重放），
  0 failures；weight state 28/28 非空（gram/h_inv/importance/rank/residual），
  机制可达（`[L-R2] rank2 reachable=1`、块序 order 非空）。
- **case 覆盖**：336 Linear cases（六 shard），identity 唯一、输出有限、
  非正 case 0。
- **OOD**：见执行日志表，全链 |Δgap| ≤ 0.01 通过（OOD 均值整体略高于 ID，
  无退化）。
- **时间**：fresh default（compat 后端，168+120，六 API 实测）
  `T ≈ 170.3 + 0.115×292.5 + 0.694×0.0 + 0.734×63.5 − 1.58×0.74 = 249.4s`
  `< 280s` 提交门通过（硬限 300s）。注意 v189 完整栈官方 275s 为同 Linear 侧
  的官方实测上界参照；本侧以标准 Attention 替换优化 Attention，官方时间预期
  不高于 v189。
- **官方分数**：NA（本侧未单独提交官方；历史锚 v163=4587 为 L1 等价机制的
  官方测量）。本地 proxy 不换算官方分数。

## 与强对照的关系

- 强对照 v166（rank-1 + v160 栈，官方 4590）：L4 在 v166 的机制上完成 rank-2
  与块序两步已验证增量；相对 v166 的材料进展（D_strong ≥ 20% 研究目标）**
  未达成**——L4 相对 v166 的本地增益与 rank-2+块序的组合增量同量级
  （约 +0.008 级），远小于 20% 剩余误差削减。L5+ 必须提出新机制。

## 下一步（L5+ 预注册方向）

`量化感知 Weight GPTQ Hessian`：权重 GPTQ 的协方差目前从理想校准激活
`X^T X` 累积（`cov_sum += stats_sample.t().mm(stats_sample)`），而部署时与
量化权重相乘的是量化激活 `Q(X)`。将 `cov_sum` 改为 HiF4 标准回写样本的 Gram
`Q(X)^T Q(X)`（复用文件内 `_dense_to_hif4`/`_dequantize_hif4`，无在线改动、
无候选循环、单一配置）。历史去重：v104/A7 是反方向（部署权重 Gram→激活侧），
本方向未在归档摘要或 AGENTS §7 关闭族中出现。
