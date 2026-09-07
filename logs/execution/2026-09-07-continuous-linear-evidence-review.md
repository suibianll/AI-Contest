# 持续Linear探针证据复核与下一轮计划

2026-09-07。本次为只读审查后的计划/状态修订，不是新的模型实验；原始探针、日志和报告不覆盖。

确认偏差：sweep_l1_direction使用round/clamp自制输入代替evaluator NVFP4，硬前向简化codec未走实际dynamic API；梯度归一化不等于clip，输出方差不等于标准输出MSE；训练样本未提供独立holdout证据。成本脚本使用随机合成输入，非default时间不能代入官方六API模型。

fc/proj分解使用简化激活编码，61–79%不能解释为真实L4部署误差占比。GPTAQ与JDRQ同目标不证明更新公式相同，去重改为待核验。合法网格旧结果为总体负但10/112正，关闭边界只覆盖原固定实现。

因此L1当前状态为PROBE_INVALID_FOR_DEPLOYMENT/COST_UNVERIFIED；L2为DEDUP_UNRESOLVED；L3为SCOPE_CORRECTION。不推断候选一定有效，也不重开原JDRQ或其他有效关闭邻域。

下一轮按[修复工作包](../../docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md)执行。official_best仍为L4 4607/247s，根v189未修改。
