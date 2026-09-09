# 历史过拟合 Linear 机制的 4B 并行恢复计划

> 状态：并行执行附录，2026-09-09。
> 本文件从属于[唯一活动总计划](../2026-09-09-output-aware-rounding-and-joint-aw-plan.md)，
> 不建立第二条父版本、版本号或官方晋级线。当前完整父仍为 v202 Linear + v195 Attention，
> 官方 `18053/281s`。正式候选的归档、提交和根切换仍由唯一活动总计划统一处理。

## 1. 目的

旧 C34、C70、v091 以及 v107-v125 表明，Linear 的输出感知算法存在两类真实信号：

1. 直接优化 `Q(A)Q(W)^T-AW^T` 比 operand MSE 更接近最终目标；
2. 使用最终部署 `W_hat^T W_hat` 可以发现普通块内误差看不到的输出敏感方向。

旧实现同时存在自由度过高、只在少量窗口或单层改善、在线循环过重等问题。现在使用
Qwen3.5-4B 面板重新判断这些机制，但不原样运行归档源码，也不重新扫描旧 rank、fold、coverage、
Jacobi、offset 或 block 数。本计划只实现两个固定、彼此独立且不与当前舍入边界计划重复的算法。

## 2. 与活动总计划的分工

活动总计划继续独占：

- `workbench/full_solution/linear-lrb1-residual-rounding/`；
- `workbench/full_solution/attention-arb1-qk-rounding/`；
- 后续 L-JRB1 的共享 activation/weight 舍入边界；
- 根 `solution.py`、正式版本号、`solutions/` 归档和官方结果登记。

本计划只使用：

- `workbench/parallel_linear_overfit_recovery/pla1-lowrank-output-correction/`；
- `workbench/parallel_linear_overfit_recovery/plw1-c70-layer-update/`；
- `artifacts/proxy_v3/parallel-linear-overfit-recovery/`；
- `logs/execution/2026-09-09-parallel-linear-overfit-recovery.md`。

两条并行实现都从启动时的当前最高分完整根复制到各自工作目录，不读取或修改另一执行线的
workbench。研究结果先留在本计划目录；需要形成正式候选时，等待活动总计划当前卡完成写入，
再把单个机制重放到届时的最高分完整根。不得直接覆盖根文件。

## 3. PLA1：低秩输出度量的一次运行时 Activation 纠码

### 3.1 要解决的问题

C34 的单侧激活精修在旧面板上过拟合，v107-v125 使用最终部署 Gram 后恢复了正向，但在线逐行、
逐坐标、多轮候选循环导致数百至数千秒开销。当前根已有前向 block GPTQ；PLA1 不重复该编码，
只在父编码全部结束后做一次反向、低秩、样本自适应纠码。

### 3.2 固定算法

校准阶段对最终部署权重 `W_hat` 计算：

`G = W_hat^T W_hat`。

保留父编码器已有的 4×4 block-local Gram，并对去掉这些 block-local 项后的剩余矩阵做一次固定
rank-4 对称特征分解，保存 `U[channels,4]` 与带符号特征值 `lambda[4]`。不搜索 rank；所有形状都
固定使用4，不足4时使用实际可用维数。

动态阶段：

1. 先完整执行当前父 activation 编码，得到合法五字段和 `X_hat`；
2. 计算当前样本误差 `E=X_hat-X`；
3. 用 `E U diag(lambda) U^T` 加父已有的 block-local Gram 项，得到输出误差梯度近似；
4. 每个自然64通道块只比较当前码与固定 hierarchy 下的相邻 mantissa `−1/+1`，全部候选一次
   tensor 化计算；每行每块最多改一个4元素组，只接受近似二次型严格下降的组；
5. 所有块只反向处理一遍，不重新估计梯度，不做第二轮，不改变 scale/lv2/lv3；
6. 输出仍为原五字段，零 mantissa 使用规范零 sign。

这个机制与 L-JRB1 不同：L-JRB1 学习跨样本共享的舍入边界；PLA1 不学习阈值，而是根据当前输入
样本的实际量化误差执行一次编译好的低秩输出纠码。

### 3.3 执行

1. 建立父 control：关闭 PLA1 时与复制的完整父逐位一致。
2. 用合成输入确认 rank-4 路径实际产生合法 mantissa 变化，并记录 attempted/accepted、修改行数、
   修改4元素组数和输出二次型变化。
3. 运行 Linear shard0；若接口正确但所有真实 case 都没有发生修改，记录 `NO_REACHABILITY` 并结束。
4. 有真实修改时运行固定六 shard一次，保存336个配对 case、API时间和分解结果。
5. 不因局部分片或角色结果修改 rank、邻码范围、处理轮数或每块修改数量。

### 3.4 结果处理

- 六 shard 最终与父逐位相同：关闭 PLA1，不生成版本号。
- 有合法非等价输出：把源码、固定配置和结果交给活动总计划；由总计划在最新完整根上重放后决定正式
  归档与官方提交。本地正负只写诊断，不作为提交门。
- 官方超时：只关闭这次运行时纠码实现，不把 rank4改成rank2或减少覆盖重试。

## 4. PLW1：C70 的当前根单轮整层静态 A@W 重构

### 4.1 要解决的问题

C70 在旧 GPT-2 上正向、旧 Qwen2.5-0.5B 和 OPT 上负向，说明 A@W 联合残差目标存在信号，
但旧三轮逐块 Gauss-Seidel 对模型、父坐标和校准窗口高度敏感。PLW1保留“冻结真实Q(A)，按最终
输出残差重构Q(W)”的核心，去掉多轮更新和局部立即接受。

### 4.2 固定算法

1. 完整执行当前父的 Linear 校准，冻结最终 transform、activation state 和 `W_hat`。
2. 对全部4B校准 case生成父动态激活 `X_hat`，教师输出为 `Y=XW^T`，父残差为
   `R=X_hat W_hat^T-Y`；case按自身元素数归一后等权累加。
3. 对每个自然64通道权重块只生成一个 C70 型候选集合：保持父 sign 语义，固定使用旧 C70 的
   E6M2 offset 集 `{-2,-1,+1,+2,+3}`，每个 offset 完整重解合法 lv2/lv3/mantissa。
4. 用固定 `X_hat` 下的精确二次型计算每个块候选；每块只保留损失最低且优于父的一个状态。
5. 把所有保留块一次性组装成唯一整层候选，再计算一次完整整层 A@W 损失。整层严格改善才写回，
   否则整层全部恢复父状态。
6. 只做这一轮，不按写回结果更新残差，不扫 offset、块比例、fold、阻尼或更新顺序。动态 Linear API
   完全不增加计算。

PLW1 与 L-RB1/L-JRB1 不重复：现有计划改变共享 mantissa 舍入边界；PLW1保持舍入规则不变，改变的是
一个64通道权重块的完整合法 scale/lv2/lv3/mantissa 联合状态。

### 4.3 执行

1. 先确认当前父 JDRQ 是否已经生成逐位相同的候选；若所有提案与父完全等价，直接记录
   `SUBSUMED_BY_PARENT_JDRQ`，不运行模型评测。
2. 候选非等价时记录 attempted blocks、locally improved blocks、layer accepted、五字段 changed count
   和精确整层 A@W 损失变化。
3. 运行 Linear shard0排除接口和不可达问题，随后固定运行六 shard一次。
4. 不根据4B结果增加第二轮、改变offset集合或拆分role/layer专属配置。

### 4.4 结果处理

- 被当前父JDRQ完全覆盖或最终整层全部回退：关闭PLW1，不生成版本号。
- 形成合法非等价候选：交给活动总计划在最新完整根上重放并安排正式归档、官方提交。
- 官方负向或超时后关闭本实现，不从旧C70邻域继续搜索。

## 5. 并行安排与交付

PLA1与PLW1代码和缓存完全独立，可以同时实现和运行。二者不相互叠加，也不等待对方结果：

| 工作 | 目录 | 输出 |
|---|---|---|
| PLA1低秩运行时纠码 | `pla1-lowrank-output-correction/` | candidate、control、shard0、六shard结果 |
| PLW1静态整层重构 | `plw1-c70-layer-update/` | candidate、等价性记录、shard0、六shard结果 |

每条线完成后在同一执行日志中写清算法是否可达、实际改变了哪些字段、4B诊断和API时间。清理本计划产生
的死候选校准缓存时使用 `--min-age-hours 2`，不得删除4B dense主缓存，也不得删除另一执行线两小时内
产生的缓存。

本计划完成条件是PLA1、PLW1各得到一次明确裁决并把非等价代表交回活动总计划。结束后将本文件移入
`docs/superpowers/archive/plans/`；当前唯一活动总计划在整个过程中保持不变。
