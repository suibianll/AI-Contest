# A 代理任务书：从 v162 独立优化 Attention

> 状态：CLOSED / SUPERSEDED，2026-09-07。历史工作包，仅保留执行设计。
> 当前任务见 [Attention 持续优化](continuous-attention.md)，从R2c官方14387.8/240s继续；新任务V冻结为R2c现有路径。
> 目标：从双标准 v162 出发，构造更低完整 Attention 输出误差的合法 Q/K 编码算法。
> Linear、V 全程保持 v162。不要启动或操作另一代理；不改根 v189。

## A0. 基线、去重与输入证明

1. 复制总计划指定 SHA 的 v162 到 `workbench/v162_attention/baseline/solution.py`，原样保留。
   候选另放 `candidate/solution.py`。冻结两个 Linear API、dynamic V、standard codec。
2. 复用身份匹配的 v162 baseline；否则按总计划生成 A 侧六 shard 一次。
   v162 目标侧 gain 应逐 case 约为 0；非零超数值容差先查 codec/输入，不能继续优化。
3. 只读旧 learned butterfly/rotation 计划及实际执行源码、JSON、官方记录。重点核对
   `2026-08-26-principled-hif4-accuracy-optimization` 的 6×8 共享角度蝶形、v187/v188
   importance、09-06 Jacobian pair-matrix 的区别。写 novelty 表：参数空间、损失、硬编码器、
   V 是否量化、GQA、训练/holdout、可达计数、父与耗时。仅有旧计划文字不等于已执行。
4. 本次假设是**每 KV group 一个完整 head 的可学习正交矩阵，Q/K 两侧真实 HiF4 联合前向，
   以完整 Attention 输出为损失**。若存在数学等价且正确执行过的有效失败实验，停止为
   DUPLICATE_CLOSED，不通过换学习率、步数、seed、head/layer 路由重开。
5. 核对 pack 真实 heads/head_dim 和 evaluator mask；当前兼容 `_attention` 是 non-causal。
   本轮按实际评测语义，不重新开展 causal/non-causal 混合扫描。

## A1. 最小合法实现及测试

令 D 为运行时 head_dim，G 为 KV heads。每 group 存一个 `R_g∈R^(D×D)`，满足 R_g R_g^T=I。
同组所有 query heads 使用相同 R_g，K 使用同一个 R_g；V 不旋转。

`Q'_h = Q_h R_group(h)`，`K'_g=K_g R_g`，因此连续 Q'K'^T=QK^T。
动态路径：NVFP4 参考解码 → head reshape → 一个固定矩阵乘法 → 原 v162 HiF4 encoder。
输出只用原五字段；R 仅在 calibration state 中保存，Q/K state 都独立包含所需副本及映射。
Q API 不读取当前 K/V，不缓存跨调用 token 数据，不新增 per-call 搜索、Gram 或循环选码。

只新增目标侧 helper；不改公共 `_standard_params`，防止 Linear/V control 被改变。

先写且通过这些必要测试：

- R=I 时六 API 输出、state 语义和输出精度与 v162 对齐；
- 合成正交 R 的 GQA 连续 QK 误差 ≤ `1e-5+1e-5*max(abs(QK))`；
- 不同 token 数和 Q/K 调用先后次序互换仍一致；
- 所有五字段通过 reference 合法性，decoded 与研究前向完全一致；
- 训练 surrogate 的前向等于真实 encoder/decode；有有限非零参数梯度，不能被 no_grad/detach
  意外挡住。若评测器在 no_grad 环境，校准局部显式 enable_grad，不更改评测器；
- Linear 和 V 的状态/编码/输出在实际输入上逐位不变。

## A2. 固定一次的完整输出旋转学习实验

这是研究假设，不是声称新最优算法。借鉴可学习旋转、完整 attention readout 优化思想：
[SpinQuant](https://arxiv.org/abs/2405.16406)、[OptR](https://arxiv.org/abs/2608.02691)。
本题不使用 OptR 的 W_O、INT2 codec 或额外 V 逆变换；使用本题实际输出与 HiF4。

### 固定配置（首次读取训练结果前保存 config.json）

| 参数 | 固定值 |
|---|---|
| 小面板 | layer 0/8/15/23；全部 KV groups，禁止只选最有利 head |
| 初始化 | D 为 2 的幂时使用规范化 Sylvester Hadamard H；否则用 I；不扫 seed |
| 参数化 | A=Theta−Theta^T，C=(I−A)(I+A)^−1，R=H C；Theta 初始全 0 |
| 学习器 | Adam，lr=0.01，32 步，梯度范数上限 1.0；无 scheduler/early-best 选择 |
| 正则 | `1e-3 * mean((C-I)^2)` |
| 前向编码 | 原 v162 标准 HiF4，硬码 forward；STE 仅用于 backward |
| 参数量 | 每层每 KV group D² 存储参数；反对称有效自由度 D(D−1)/2 |
| 研究采样 | 每 calibration window 均匀取至多 128 个 K/V token；其中均匀取至多 32 个 Q token |
| fold 选择 | 依输入顺序，最后一个 calibration window 为 gate，其余训练；不按长度/效果重新排序 |
| 训练聚合 | 各 window 等权标准化输出损失，再平均 heads；每步使用全部训练 window 的累计梯度 |
| 校准 state 选择 | 仅 final step 的 learned R 与固定 H 比；gate 上 learned 严格更好才选 learned |

不显式求逆，用线性求解实现 Cayley；仅在校准阶段。H/R/candidate 的 dtype 为 FP32，
正交/不变量另用 FP64 小样本核验。若矩阵数值非有限则 ERROR，不静默挑历史最好 step。
R=I baseline 与固定 H 都记录；若固定 H 自身在 gate 比 I 差，部署回退 I。
顺序固定：先在 H/learned 中按 gate 选一个，再与 I 比，平局优先 I、其次 H。
至少两个 calibration windows 才训练；不足时只走固定 H/I gate，记录 TRAINING_UNAVAILABLE。
不使用测试文本、层号专属开关或人为 head 优先级。

### 完整训练目标

在每个训练 window 的同一采样集合，reference 使用原 NVFP4 解码值计算：

`O_ref = softmax(Q K^T / sqrt(D)) V`；
`O_hat = softmax(Qh(R) Kh(R)^T / sqrt(D)) Vh_standard`。

`loss_f = mean((O_hat-O_ref)^2) / max(MSE_STD_f,1e-12)`；
`loss = mean_f(loss_f) + regularizer`。

Qh、Kh 同时硬量化；Vh 始终标准量化，不能训练用 float V、部署再换量化 V。
STE 写为 `x+(decode(encode(x))-x).detach()`，硬前向含 E6M2、共享层级、mantissa 和 sign。
这是 surrogate 梯度，不声称真实离散梯度；候选裁决全部用真实硬编码输出。

训练 subset 内 reference 和 player 使用同一 Q/K/V 采样，不能将 subset 输出与完整 reference
比较。校准 gate 则恢复完整 K/V，上面 128/32 采样只用于训练，不冒充完整 attention。
gate/full holdout 可按 Q 行分块避免 n² 内存；每行仍使用所有 K/V。

### 独立验证和材料收益

选定 R 后在 evaluator 的独立 validation/test 执行，不能在 holdout 重选 I/H/R。
报告：v162、固定 H、learned（未 gate）、实际部署 gate 版四臂；只有部署版可晋级。
测试层固定 0/8/15/23，完整长度而非 32/128 前缀；然后扩大到六 shard，不挑正层。

- 相对 v162 mean/median 均正、两个 split mean 均正 → CLEAN_ROOM_PROGRESS。
- 相对历史 v168，以及同输入上最新完整父的 Attention 输出，也必须各自成对报告。
  最新父只用作读数强对照，不带入其 Linear。保存其来源 SHA，不能混不同 panel/协议均值。
- 若超过 v162 但未超过强对照：记 RECOVERY_ONLY，不作为填补榜首差距的新机制。
- 本轮突破研究目标：相对强对照的平均标准化剩余误差降低≥20%，至少 3/4 小面板层为正，
  独立 split 同号。未达到可交付研究结果，但不继续调学习率、步数、seed 或增加模型自由度。
- 若训练显著降损失、gate/holdout 不降：OVERFIT_REJECTED；不通过扩大校准集搜索修复。
- 若训练不降，先查梯度、状态实际部署可达和 hard/surrogate 一致；正确后仍不降则
  FIXED_HYPOTHESIS_REJECTED，而非整个旋转空间无解。

## A3. 完整评测与资源决策

材料门通过才对全部层运行上述固定校准规则、六 shard 48-case Attention、OOD、真实 control。
v162 的标准 Linear 使本次 Attention 拥有独立预算，不沿用 v189 完整组合的 15s 锚点余量。
校准 32 步、Cayley 求解与所有 dynamic R matmul 都必须进入最终提交计时。
R1/R2 原型耗时不充当官方预测；后续按总计划跑完整六 API fresh default 168+120。

先做单层计时定位是否是 calibration 或在线 matmul 主导；若明显无法进入预算，不盲目启动
全层训练。若精度通过但最终时间预测≥280，记 ACCURACY_PASS/TIME_HOLD，不能随意减步数/
减少 head/缩小层覆盖改算法重测。可提出一份数学等价的编译优化说明，由协调者决定下一阶段；
不能仅因已有 Hadamard 在线很快就声称 learned dense R 也很快。

总计划的负向损失门已由用户确认：总 L1 只记录，L1_negative<0.02，不因正收益大而拒绝。
全部门通过后，单侧官方以 `S_A−1001` 登记累计贡献；仍冻结 v162 Linear，不混入 L 代理产物。

## A4. 最终交接

交付所有 SHA/配置、总计划账本、四臂对比、真实硬编码输出/control/OOD、训练时间/
校准时间/动态时间、state 大小和官方状态。
本文件仅负责 Attention 工作，根文件、Linear 代码、全局计划入口由协调者维护。

可直接给代理的指令：
“执行本任务书，从指定 SHA 的 v162 开始，冻结 Linear 与 V。先完成 A0 历史去重、基线和
A1 正交/GQA/硬编码测试，再按固定 A2 配置验证完整输出驱动旋转学习。分别报告相对 v162、
直接父及历史强对照的效果；遵守 GPU 排队和提交政策，不改根、不启动另一代理、不扫描邻域。”
