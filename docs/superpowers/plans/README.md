# 计划入口

> 最后更新：2026-09-03

本目录只保留一份活跃计划。执行优化时只读取下面这份文件、根 `solution.py`、最新
`proxy-v2` 评测和官方规则；归档目录中的计划不具有指令效力。本地主评测只调用
`evaluator/official_eval.py`，跨模型泛化调用 `evaluator/cross_model_eval.py`；旧
`real_model_suite.py` 已退役。

**当前活动计划（2026-09-03）**：
[`2026-09-03-post-v162-low-complexity-algorithm-expansion-plan.md`](2026-09-03-post-v162-low-complexity-algorithm-expansion-plan.md)
——侧向隔离计划收官后的低复杂度算法扩展，**Attention 优先**（A1 解析 logits 增益校正 →
A2 V 输出偏差质心补偿 → A3 动态 scale 静态策略编译 → A4 矩匹配 mantissa 阈值），随后
Linear L1-L4（WUSH/CAT 审计、Babai 解码、Trellis/VQ、Kronecker CAT）。候选仍从 v162
双标准零点单侧构造（`P_L = v163 4587/202s`，v166 rank-1 待官方回传；`P_A = v164
13945/204s`），官方差分按计划 §3.3 登记；每个工作包一个候选、失败换机制不扫邻域，
v165 约束（动态 API 无 Gram contraction、无候选循环、复杂计算只在 calibration）对
全部 A 包强制。

当日已归档：v162 官方侧向隔离优化计划（v165 timeout、v167 低秩 Gram 码本本地
REJECTED、v166 rank-1 已提交待回传，`-superseded`）、官方两侧分数比重校准计划
（v162 `1001/146s`、v163 `4587/202s`、v164
`13945/204s`，score interaction 为 1，当前已实现 Attention:Linear 官方贡献约 `3.61:1`）、
Attention per-call 序列自适应精化计划（v161 官方 timeout，per-call 动态族关闭）、Attention
解析式宽域计划（A1a 4×4 REJECTED、A2 无病因、A3 未启动）与 Householder 快速验证计划
（全族 REJECTED），见
[`../archive/plans/`](../archive/plans/)。Linear 侧 T<d 秩亏伪增益通道已结构性封闭，
不再从已关闭族内微调；官方证据判别器 D1/D2/D3 预注册于
[`OPA-1 Stage 1 账本`](../../../logs/execution/2026-09-03-opa1-stage1-official-evidence-ledger.md)，
绑定未来任何官方提交。

快速机制迭代使用 `--compact-panel`：Linear 为 28 个 selected Weight state + 56 个跨
validation/test holdout case，Attention 为四个深度/长度哨兵；读取 median、尾部分布、负
case、cross-holdout 一致性和 interaction；不再用 mean 单独晋级。完整 default panel 仅作
单侧低频审计。

所有历史计划（含已完成的 21765 A/B/C 计划、Householder 与 Attention 解析计划）均已移至
[`../archive/plans/`](../archive/plans/)。它们是历史决策记录，
不再提供下一步指令。

## 计划生命周期

1. 写新计划前先确认本目录除 `README.md` 外只有一个 `.md`；不能并行保留多个 current/active 计划。
2. 计划步骤要写明假设、代码入口、模型/数据、验收指标、产物和失败处理；执行后立即写入结果、source SHA、日志链接和 `done/rejected/blocked` 状态。
3. 每次实验无论成功、失败、超时或未提交，都先归档完整源码、配置、结果和 parent；缺少源码/SHA/配置的结果标为 `non-reproducible`。
4. 计划完成、被替换、停止或连续阻塞后，立即移入 `../archive/plans/`，并在同一提交创建/指定新的 active 计划、更新 README 和状态文档。
5. 归档计划不可继续追加新的下一步，也不直接修改历史结论；发现 bug 或数据错误时写审计说明并创建修复计划。

当前数据数字发生变化时，应同时更新根 README、`solutions/README.md`、当前状态报告和执行日志的日期、配置、分数、时间与 SHA。此前 C1 structured Linear 计划已归档为 [`2026-08-31-hif4-active-c1-structured-linear-plan-superseded.md`](../archive/plans/2026-08-31-hif4-active-c1-structured-linear-plan-superseded.md)，不得再从中读取下一步。

官方边界（2026-08-31 修订）：官方不再限制任何 `A@W` 拟合用法，离线校准与在线激活量化均可自由用 `A@W`、输出或残差优化 `Q(W)`/`Q(A)`；唯一硬约束是端到端运行时间严格小于 `300s`。v98 已在该限制下官方判为 timeout，见 [`2026-08-31-v98-official-timeout.md`](../../../logs/execution/2026-08-31-v98-official-timeout.md)；v107 官方保持 Attention `wrong answer`（非 timeout）。探索阶段的 layer-1、oracle 和超时实验只能筛选方向，不能替代完整部署门禁。
