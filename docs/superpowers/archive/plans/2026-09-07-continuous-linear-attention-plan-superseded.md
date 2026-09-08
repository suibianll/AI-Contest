# Linear / Attention 持续优化总计划

> 状态：**CLOSED / SUPERSEDED**，2026-09-08。当前入口为
> [单一完整方案计划](../../plans/2026-09-08-single-solution-optimization-plan.md)。下文仅保留历史设计，不提供指令。
>
> 原计划曾改为**持续研究循环**：误差账本驱动定位 → 代理自行补卡 →
> 固定配置验证 → 及时交付与官方探索 → 强制续接（每轮必留 next_card 或停止声明）。
> 用户确认目标：Linear 校准拟合 gain≥0.9；Attention 独立窗口最终输出 gain≥0.9。
> 两者均以同 NVFP4 输入的标准 HiF4 误差为分母，不换算官方成绩。

唯一当时明细：[持续研究循环（误差账本驱动）](2026-09-08-continuous-research-loop.md)。
旧[双侧0.9执行任务书](2026-09-08-dual-side-09-execution.md)降级为 L23b/A25 证据与 P0 事务来源，
不再提供队列指令。停止条件只有三条：达标 0.9、缺不可替代外部输入、去重后无可执行新假设；
后两条必须列具体阻碍。

## 当前顺序

0. 循环机制（§1–§6 见明细）：账本 E1–E4 / F1–F5 定位 → 出卡 → 一卡一配置 → 交付/官方 → 续接。

1. Linear：先核验完整候选 `17636/264s` 的官方计分 SHA；核验后在 L31/L32 中去重并只注册一张，
   从 L4 time reference 构建，以 L28 为 score target。L29-Q/G、L30 和 LC0 不在当前队列。
2. Attention：A29 实际实现已 TIMEOUT，AC0 骨架分数只归属 AC0；当前只执行 A30，A31 随后，
   A32 共享码语义仍需用户单独授权。
3. 单侧正向且官方 `<300s` 才升级侧父并重定位账本；两侧均正向后登记单侧增量组合包。
   等待官方回传期间继续独立机制推导与实现，不停循环。

根运行父仍为 v189；另有 compiled sample-energy 用户回传 `17636/264s`，须先核验官方计分 SHA，
核验前只列为“待身份绑定的更优完整候选”。Linear `score_parent=L28`，`time_reference=L4`；
Attention `score_parent=A2`，`time_parent=R3`。现有包的新旧路径/SHA必须分开，不覆盖历史源码或继承结果。
执行细节、指标、失败分支、所有权、GPU锁与交付模板均以循环明细为准。
旧[21071阶段](21071-evidence-driven-research.md)保留为机制设计依据，不再提供队列指令。
