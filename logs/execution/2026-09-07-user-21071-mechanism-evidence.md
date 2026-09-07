# 用户确认的21071机制证据

2026-09-07。来源：用户本轮明确提供的信息；不是本仓库新增实验。

- 官方50个Linear、250个Attention场景样例。
- A@W拟合可优化Linear，约4400/5000。
- 在校准集学习量化Q/K的互逆变换，使各自scale降低，配合Linear拟合达到21071分、283s。
- 当前未提供对应源码/配置/SHA和分项原始记录，记EXTERNAL_USER_CONFIRMED / SOURCE_UNBOUND。

该事实支持继续研究上述机制，不证明本仓库任一旧实现有效。不得将4400分项与L4侧隔离整包4607直接比较；不得由样例数量推断隐藏形状、case权重或Attention满分；不得将283s实测与280s预测门混淆。

行动修订：不再使用旧A@W/JDRQ目标相似或“连续乘积已等价”推断完整机制关闭。保留旧固定算法的真实负结果，开启有明确新目标/求解规则的固定代表候选。新计划见[21071研究工作包](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。
