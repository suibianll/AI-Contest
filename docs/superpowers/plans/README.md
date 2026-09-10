# 计划入口

> 更新：2026-09-11，第三版审计修订。这里只登记唯一活动计划；执行顺序、卡片状态和停止条件统一维护在计划正文，不复制卡片快照。

**[Attention 算法研究计划（第三版修订）](proposals/2026-09-10-attention-080-algorithm-research-next.md)**（已激活，保留原路径。）

当前完整根 v237：18518/289s，SHA `ECB1F9E510B5507E1A2DC8B95F8A84E9B51864A420B3828537A328813E2CE554`。回退 v233：18428/288s。候选启动时重新核对最高分完整归档根，不使用未晋级 v240。

**官方已回传（2026-09-11）：VK-1 的两个正式候选 v241 / v242 均 TIMEOUT（>300s）REJECTED**，
见[超时记录](../../../logs/execution/2026-09-11-v241-v242-vk-official-timeout.md)。
两卡是同一机制的两种粒度（`diff` 仅 30 行），各自独立提交、各自超时；根未切换。
**关掉的是这两个实现，不是 VK-1 方向**（计划 §7）。该次回传的方法论教训已写入计划 §6.1：
v242 的"落进余量"算术（`+0.0117 s/次 × 250 用例 ≈ 2.9 s < 11 s`）三个支点只有"一 case 一次 V 调用"可靠，
**case 数取自 `SOURCE_UNBOUND` 的用户报告且与更早记录的 200 Attention 冲突**，
**单次成本是 GPU 实测而官方判题是鲲鹏 920B CPU**。

**已开侧隔离定价通道（2026-09-11，用户指令）**：完整包两次都没进限、VK 机制至今**零官方分数**，
故改用仓库既有的侧隔离口径给同一机制定价——`standard-linear_v241-attn` / `standard-linear_v242-attn`
（标准 Linear + 候选 Attention），基线 `14426/243s`（标准 Linear + v195 attn）。**只测分、不测时间**
（侧隔离时间对完整包无预测力，v194/v190 双向反例），且**侧隔离分不晋级、不与完整分相加**（AGENTS §2）。
见[构建记录](../../../logs/execution/2026-09-11-vk-side-isolation-probes.md)。

当前下一步及候选路径只见[活动计划 §6–7](proposals/2026-09-10-attention-080-algorithm-research-next.md#6-待研究方向与当前下一步)。旧 A-JC1 合同及历史 workbench 不提供当前执行指令。

[上一轮计划](2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md)仅供历史查询。规则顺序：AGENTS → 4B 指引 → 本入口指定计划。局部诊断、外部整包秒数和 0.80 里程碑均不设提交门，不证明机制整族耗尽。

本次修订依据见[修复记录](../../../logs/execution/2026-09-11-active-plan-audit-fix.md)。
