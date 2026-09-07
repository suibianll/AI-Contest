# Linear 当前机制队列

更新2026-09-07（探针后）。当前执行入口：[Linear独立工作包](../../docs/superpowers/plans/workpackages/linear-output-followthrough.md)。
父L4官方4607/247s，冻结v162 standard Attention，尚无新候选待官方。

| 顺序 | 卡 | 状态 |
|---|---|---|
| 1 | L21-1 真实A@W固定层级逐列条件求解 | **探针负向**：数学 PASS（OBQ 逐位一致、JDRQ 区别成立），可达 PASS（合法格点/非no-op），但真实闭环 3/3 代表 state test holdout 退化（o +28%、proj +78%、fc_up +56%），校准 fold 过拟合 → 分布外不泛化 |
| 2 | L21-2 精简输出拟合主干 | 首卡本地负向，保留为后继：精简父 + 拟合收益分开；若 L21-1 无官方探索则评估 |
| 3 | L21-3 合法层级联合输出求解 | 需首卡有效或有直接机制证据后登记 |

L21-1 探针细节见 `anchor21-l1/probe-report.md`（数学检查 + 真实闭环）。按工作包失败分支
"本地 insplit 负向 → 如实标 EXPLORATORY，仍须风险/合法/control/非no-op/时间门"；
一次失败不扫列序/步数，不自动宣称该机制族关闭（3 state 不能扩展到全族否定）。

保留的具体历史结果：正交特征基5state退步、Kronecker4state两退两平、旧全栈训练成本过高、
旧JDRQ实际负结果、L21-1逐列条件求解3/3 holdout退步。
旧GPTAQ/JDRQ去重、L21-1公式区别均完成；块序/Householder/E2E offset 属已关闭边界。

Attention同时按A22-1推进，GPU共用锁、候选与控制侧分离，分别官方确认后才组合。