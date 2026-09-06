# L 代理任务书：从 v162 独立优化 Linear

> 状态：DESIGN_ONLY / L0_PENDING。主契约：[总计划](../2026-09-06-v162-independent-linear-attention-plan.md)。
> 本任务书负责独立实验边界与比较，不自动继承正在进行的 v189 块序实验。
> 用户将另行委托 L 代理；A 代理的算法、结果与提交不能进入本任务。

## L0. 建立真实独立分支

从总计划指定 SHA 的 v162 复制到 `workbench/v162_linear/baseline/solution.py`。
候选写 `workbench/v162_linear/candidate/solution.py`，初始也必须是 v162。
四个 Attention API、其标准 state、公共 standard codec 均冻结为 v162；只新建 Linear 专用 helper。
真实 Q/K/V 输入以及调用顺序下，冻结侧五字段、state 和输出必须逐位一致。
基线 Linear 六 shard 336 cases gain 应约 0，身份匹配的既有 baseline 只读复用。

## L1. 独立提出一个可解释机制

L 代理先交付本侧 `mechanism.md`、固定 `config.json`，包含：

1. 从 v162 缺少的哪个明确机制切入，以及与历史 GPTQ/full64/Householder/JDRQ/块序等的去重结果；
2. 变量是什么、连续变换是否保持 `XW^T`、最终解码路径是什么；
3. 校准如何学习，真实量化 X 与 W 的 interaction 如何进入最终输出损失；
4. API 插入点、在线复杂度、完整校准图成本、固定候选数量和停止门；
5. 训练、校准选择、独立 holdout 的确切拆分，禁止临时改 fold/seed/role 路由。

本次没有授权把此前失败的 v189 排序邻域重新跑一遍；换 v162 父不自动产生机制新颖性。
允许从 v162 重现一个已验证基础机制作为本侧学习/工程基准，但标 RECOVERY，不能称新突破。
没有具体机制卡时停在 DESIGN_PENDING，不能自行启动无假设的全量评测。

## L2. 目标与对照

目标必须是 `Y_ref=XW^T` 对 `Y_hat=Q(XR) Q(WR^-T)^T` 的实际输出误差。
Hessian/Gram 取最终部署坐标，校准用真实 dynamic activation 输出；不可用 W-only 胜出
掩盖 X/W interaction 回退。额外存同坐标 X-only/W-only/Both 分量只用于归因。

对每个候选按同 case 保存四种比较：

- 本侧累计效果：相对 v162；
- 单步效果：相对本侧直接父；
- 历史强对照：v166，以及已验证最新父的 Linear 目标侧输出（仅只读强对照）；
- 未修改 control：标准 Attention 必须不变。

不能用新的 Attention 高分抬高 Overall 后声称 Linear 改善，也不能要求 Linear 候选
超过某个使用不同 Attention 的“最高 Overall”。本侧晋级/时间按总计划执行。
相对 v162 正向但仍弱于历史强对照，标 RECOVERY_ONLY；突破材料目标是相对强对照
平均标准化剩余误差降低≥20%，不是从标准重现已有收益。

## L3. 评测、产物与官方

按总计划唯一命令入口评测六 shard、control、OOD 与 fresh default；所有运行先取得 GPU 锁。
案例全部 seven roles 都保留；机制局部适用性用一般 shape/state 条件表达，禁止模型层号路由。
每次修改必须记录 attempted/accepted、实际 changed codes、最终返回 state/weight SHA。
接口/环境问题记 ERROR，不当算法失败；正确执行且 gate 失败才 REJECTED，不扫邻域。

每侧官方包自身完整六 API，但 Attention 始终 standard。官方累计贡献 `C_L=S_L−1001`，
本步贡献相对同侧父；未知记 NA，不假设可从本地 gain 算出官方收益。
不要提交与 v162 或任何旧包同 SHA/逐位等价的零点复测。没有全部门禁结果就交付研究报告，
不得改根 v189；正式编号与组合由协调者安排。

最终交付 `solution.py/config.json/manifest.json/result.json/report.md/mechanism.md`，以及
一段明确区分标准恢复、现有最佳突破、时间阻断和政策阻断的结论。

可直接给代理的指令：
“执行本任务书，以指定 v162 原始源码为初始父，冻结全部 Attention。先提交一张固定配置的
Linear 机制卡并核对历史关闭边界，再按总计划独立评测。分别报告相对 v162、直接父、历史
强对照的效果；只写本侧目录，GPU 排队，不借用 A 代理的增益、不改根、不扫描旧邻域。”
