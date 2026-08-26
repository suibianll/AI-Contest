# HiF4 原理驱动精度提升架构设计

日期：2026-08-26  
状态：待用户最终审阅  
目标：在官方五分钟 CPU 时限内，以当前官方 10250 分实现为 Champion，突破以局部 scale/offset 调参为主的优化上限。

## 1. 背景与问题定义

当前 `solution.py` 已实现以下能力：

- 标准 HiF4 保底与精确层级求解；
- E6M2 邻域 scale 搜索；
- SmoothQuant、通道排列和 K-centering；
- Linear 对角重要度和 Attention Jacobian 重要度；
- 困难块精修、真实 Attention 校准门控和运行预算限制。

官方结果为 10250 分、127 秒，而排行榜最高分约 26000。继续扩大 offset、refine ratio 或候选网格，只能在现有“原坐标系 + 局部加权 MSE”表示空间中寻找更精细的局部最优，无法解释或弥合这一数量级的分差。

新方案改变三个基础条件：

1. 用严格等价的结构化正交旋转改变数据分布，降低共享 scale 对离群值的敏感性；
2. 用完整 64 维二阶误差和误差反馈代替逐元素独立舍入；
3. 用合法 Linear 二阶代理和真实 non-causal Attention 输出选择策略。

官方只返回总分和运行时间，没有 Linear/Attention 分项。每天允许 30 次手动提交，因此实验必须采用单机制候选和小型因子设计，从总分差推断机制贡献。

## 2. 目标与非目标

### 2.1 目标

- 保持六个官方接口、HiF4 五字段和 state 合法性；
- 不计算 `A @ W` 并用其反推激活量化；
- 所有新增变换在未量化空间严格保持 Linear 输出或 Attention logits；
- 当前 10250 算法始终作为逐 case 回退；
- 将官方总时间控制在 220～235 秒的内部目标内，硬上限保留至 300 秒；
- 通过可归因的手动提交实验持续更新 Champion。

### 2.2 非目标

- 不继续进行大规模固定参数网格搜索；
- 第一阶段不实现完整 64×64 稠密学习旋转；
- 第一阶段不引入 temperature compensation；
- 不对 V 使用无法在输出端恢复的旋转、排列或 centering；
- 不在 A/B/C 主机制通过前消耗新的正式 holdout 预算。

## 3. 总体架构

算法由三个独立层组成。

### 3.1 StructuredTransform

对 Linear 的 A/W 和 Attention 的 Q/K 应用成对结构化正交变换。支持：

- Identity；
- 随机符号 H64；
- 大张量 H8 分组快速路径；
- 第二阶段可学习 64 维 butterfly rotation。

变换由 calibration 选择，在线只执行选中的单个变换。state 保存符号、排列、角度、shape 和版本，不保存带梯度 Tensor。

### 3.2 SecondOrderQuantizer

保留现有合法 HiF4 参数生成和标准候选，但用完整块 Hessian、误差反馈和块坐标下降重排：

- E6M2 scale；
- lv2/lv3 层级指数；
- sign/mantissa 舍入。

Weight 使用完整 64×64 Hessian；Activation 在线路径使用 Cholesky 或 `diag + rank-8` 近似，并只处理困难块。

### 3.3 OperatorSelector

外层选择器比较当前 Champion、旋转、二阶舍入及其组合：

- Linear 使用不生成 `A @ W` 的二阶展开代理；
- Attention 使用真实 non-causal softmax 输出；
- 任何非法、非有限、超时或尾部退化候选回退到当前 Champion。

## 4. 结构化旋转设计

### 4.1 Linear 不变量

对每个连续 64 维输入通道块定义正交矩阵 `R`、正对角矩阵 `D` 和排列矩阵 `P`：

\[
A'=AD^{-1}PR,\qquad W'=WDPR.
\]

因为 `R Rᵀ = I`、`P Pᵀ = I`：

\[
A'W'^T=AW^T.
\]

旋转限制在 HiF4 的 64 元素块内，使单点离群值扩散到多个坐标，降低全局 E6M2 scale 和两级共享指数的压力。

### 4.2 Attention 不变量

对每个 KV head 使用：

\[
Q'=QDPR,\qquad K'=KD^{-1}PR.
\]

GQA 中映射到同一 KV head 的所有 Query heads 共享同一 `D/P/R`，因此：

\[
Q'K'^T=QK^T.
\]

V 保持原坐标系。

### 4.3 固定旋转候选

第一阶段比较：

- Identity；
- 4～8 个确定性随机符号 H64；
- H8 分组旋转。

H64 使用六层 FWHT，仅包含加减法。候选符号种子固定，避免根据隐藏场景硬编码；最终 state 保存实际符号 Tensor，保证在线复现。

### 4.4 可学习 butterfly

第二阶段把固定 H64 升级为：

\[
R(\theta)=B_6(\theta_6)\cdots B_1(\theta_1),
\]

其中每层 `B` 由不相交的 2×2 正交旋转组成。首版每层或每个 8 维组共享少量角度，不保存完整稠密矩阵。

校准从最佳 H64 初始化，使用 STE HiF4 和算子目标优化 12～20 步。最终选择必须重新运行真实离散 HiF4，不能使用 STE 分数代替正式判定。

## 5. 完整二阶 HiF4 舍入

### 5.1 Weight 目标

变换后的校准激活构造：

\[
H_A=\frac{A^TA}{N}+\lambda I.
\]

对每个 Weight 输出行和 64 通道块最小化：

\[
L_W=(w-\hat w)^T H_A(w-\hat w).
\]

这只使用 `AᵀA`，不生成被禁止的 Linear 输出。

### 5.2 分层求解

每个 64 块依次执行：

1. 生成标准 scale、相邻 E6M2 code 和少量数据驱动 scale；
2. 以当前精确层级解初始化；
3. 固定 scale/lv2/lv3，按 Hessian 逆矩阵执行 GPTQ 式顺序 mantissa 舍入和误差反馈；
4. 逐个 8 元素组枚举 8 种 lv2/lv3 配置，按完整二次型更新；
5. 执行两轮块坐标下降；
6. 与当前 Champion 的完整候选比较，二次型不下降则回退。

该方法避免 `8^8` 全组合，同时允许不同通道误差通过 Hessian 相关性互相补偿。

### 5.3 Activation 在线二阶路径

量化后 Weight 构造：

\[
H_W=\hat W^T\hat W.
\]

calibration state 按 64 通道块保存阻尼 Cholesky，或在内存/时间较大时保存 `diag + rank-8`。在线 Activation：

- 普通块继续使用当前快速量化器；
- 仅困难块执行一轮批量顺序舍入；
- 超过运行预算时自动降低 rank、困难块比例或切换 H8。

### 5.4 Weight/Activation 交叉项

定义：

\[
\Delta A=\hat A-A,\qquad \Delta W=\hat W-W.
\]

Linear 输出误差为：

\[
A\Delta W^T+\Delta A\hat W^T.
\]

其平方损失的交叉项为：

\[
2\operatorname{tr}\left((\Delta A^TA)(\Delta W^T\hat W)\right).
\]

该项使用输入相关性、Weight 残差和 Activation 量化残差，不计算 `A @ W`。为降低规则解释风险，交叉项必须作为独立候选，与纯 Hessian 版本分别提交，不与首个二阶版本捆绑。

## 6. Attention 真实损失学习

正式目标为：

\[
O=\operatorname{softmax}(QK^T/\sqrt d)V,
\]

\[
J(\theta)=\operatorname{mean}_i r_i+\lambda\max_i(r_i-1),
\quad
r_i=\frac{MSE(O_i,\hat O_i^\theta)}{MSE(O_i,\hat O_i^{std})+\epsilon}.
\]

只使用官方代理协议中的 non-causal 输出。

### 6.1 学习和选择分离

- 至少两个 calibration 样本时，一部分学习旋转，一部分验证；
- 只有一个样本时按 token 窗口切分；
- 优化过程使用 STE，最终评分使用真正离散 HiF4；
- 任一验证窗口超过最差容忍度即回退；
- 固定 H64 和 Identity 始终参与最终选择。

### 6.2 V 偏差感知量化

V 不旋转，使用：

\[
L_V=\sum_t\|e_t\|_2^2+\lambda T\|\operatorname{mean}_t e_t\|_2^2.
\]

比较标准候选、局部 MSE 候选和均值误差抑制候选，最终仍由真实 Attention 输出门控。

## 7. 失败处理与安全回退

- Cholesky 不正定：逐级增加阻尼，仍失败则使用 `diag + rank-8`；
- 旋转学习出现 NaN/Inf：回退固定 H64；
- 固定旋转验证退化：回退 Identity；
- 二阶解未降低完整目标：回退当前 10250 候选；
- 样本不足：禁用学习型旋转，只允许固定候选；
- 预计总时间超过 270 秒：本地拒绝提交；
- 输出/state 不合法：本地合规门禁拒绝；
- 所有 state Tensor 返回前执行 `detach().cpu().contiguous()`，并确保无梯度、有限和 dense strided。

## 8. 测试策略

### 8.1 数学性质测试

- H64/H8/butterfly 的正交性；
- 未量化 Linear 输出旋转前后相等；
- 未量化 MHA/GQA logits 旋转前后相等；
- GQA Query head 与 KV head 的共享变换映射正确。

### 8.2 离散求解器测试

- 所有 HiF4 参数合法；
- 二阶目标不高于初始化候选；
- 块坐标更新单调不增；
- 标准回退逐字段保持可用；
- Cholesky 失败路径确定且有限。

### 8.3 泛化测试

- schema v2 standard/dev 配对；
- 匿名 holdout；
- 单点离群、多点离群、heavy-tail、稀疏、均值漂移；
- calibration/test 幅值漂移；
- Attention saturated logits、V outlier、MHA/GQA 和 head_dim 64/128；
- 分别记录 Linear/Attention 分项、负分率、最差十分位和候选启用率。

### 8.4 性能测试

以当前本地时间对应官方 127 秒建立比例估计。内部目标为官方 220～235 秒，预测超过 270 秒的候选不提交；官方硬上限仍为 300 秒。

## 9. 官方黑盒实验设计

当前 10250 分代码先固化为新 Champion。首日提交：

| 编号 | Linear 旋转 | Attention 旋转 | 二阶 Weight | 用途 |
|---|---:|---:|---:|---|
| C0 | 无 | 无 | 无 | 重测基线稳定性 |
| C1 | H64 | 无 | 无 | Linear 旋转主效应 |
| C2 | 无 | H64 | 无 | Attention 旋转主效应 |
| C3 | 无 | 无 | 开 | 二阶 Weight 主效应 |
| C4 | H64 | H64 | 无 | 双旋转交互作用 |
| C5 | H64 | 无 | 开 | Linear 旋转与二阶交互 |
| C6 | H64 | H64 | 开 | 完整固定组合 |

每个候选生成 manifest，记录候选编号、唯一机制、源码 SHA、本地分项、本地时间、预测官方时间。用户手动提交后回填官方总分与实际时间。

首日只使用约 7 次提交。第二阶段围绕最高分主机制比较：

- H64 与 H8；
- 固定 H64 与学习 butterfly；
- Hessian full/rank、阻尼和困难块比例；
- V 标准与偏差抑制。

每次只改变一个变量。最高分候选经复验后固化为新 Champion；后续时间比和分数增量只相对新 Champion 计算。

## 10. 实现边界与产物

官方提交仍为单文件 `solution.py`。工程内新增可测试模块和离线导出工具，最终导出自包含候选：

- 结构化旋转核心；
- 二阶 HiF4 求解核心；
- Attention 学习与选择核心；
- feature flag 候选生成器；
- submission manifest 与手动官方反馈登记；
- 数学、合法性、泛化和性能测试。

实现顺序为：

1. 固化 10250 Champion；
2. 固定 H64/H8 与不变量测试；
3. C1/C2/C4 候选；
4. Weight 完整二阶求解与 C3/C5/C6；
5. 学习 butterfly；
6. V 偏差量化；
7. 官方反馈驱动的单变量迭代。

## 11. 研究依据

- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs, NeurIPS 2024.
- SpinQuant: LLM Quantization with Learned Rotations, ICLR 2025.
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
- QuIP: 2-Bit Quantization of Large Language Models With Guarantees.
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.
- Benchmarking Post-Training Quantization of Large Language Models under Microscaling Floating Point Formats, arXiv:2601.09555.

这些工作分别支持旋转消除离群、学习旋转、二阶误差反馈、incoherence processing，以及 MXFP4 中 scale 优化的重要性。本设计只采用与本赛题六接口、HiF4 固定格式和 CPU 时限兼容的部分。
