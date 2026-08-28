# 基于当前 v2.0 代码仓的 HiF4 优化方向

日期：2026-08-26
状态：历史技术记录；不再作为执行或晋级规则。旧执行计划已归档至
`docs/superpowers/archive/plans/2026-08-27-local-progressive-hif4-optimization-plan.md`。
适用基线：`solution_b0_tmp.py` 为精确 B0；当前根 `solution.py` 为 A1-only 本地 Champion
替代关系：本文替代以官方 10250 分、127 秒版本为基线的早期优化方向

## 1. 结论

当前优化主线调整为：

> **Attention 真实目标对齐优先 → HiF4 scale 优化 → 有界块二阶优化 → 最后考虑学习旋转。**

不再把“全面实现 H64、完整 64×64 GPTQ 和学习 butterfly”作为第一阶段。当前代码已经具备有效的分块 Hadamard、Weight/Activation 二次型和接近全量的困难块精修；后续工作应当保留这些已有 winner，通过单变量候选补足 Attention 目标错位、宽层 Activation 二阶缺失和 HiF4 scale 候选不足，而不是重新搭建一套平行量化器。

### 1.1 本地唯一评测口径

当前无法进行官方评测。本文后续所有“晋级”“Champion”和“时间门禁”均只表示本地可复现结论，不再等待官方分数，也不允许用本地指标推算官方绝对分数。B0 本地门禁固定为：

- B0 源码及 SHA256；
- 相同边界下的 CUDA/CPU 分项和算法阶段时间；
- 按第 3 节统一边界重新测得的本地 CUDA/CPU 时间；
- B0 与 v002 归档、当前工作区文件的差异说明；
- 本地静态检查、state 合法性与 feature-off 等价检查。

B0 的本地精确数据与哈希已经记录，门禁关闭。历史 `~15000 / ~140s` 仅作来源记录，不参与任何当前决策。Linear sampled output 的外部规则兼容性仍未知，但不阻塞本地候选之间的相对配对。

## 2. 当前代码仓状态

### 2.1 当前 Champion 与工作区候选

- 不可变 Champion 仍是 v002 归档；已记录官方分数约 `15000`、时间约 `140s`，但缺少精确值及其与提交源码 SHA256 的证据绑定；
- `solution_b0_tmp.py` 是 GPU-compatible 的开发起点 B0，行为复现 v002 分项，但 SHA256 与 v002 归档不同，不能借用归档的官方身份；
- 根目录 `solution.py` 启用 A1，A2、A3、L1 默认关闭，当前有效配置是 **A1-only**；
- v002 是历史官方归档；从现在起，本地 Champion 只按本文件的固定配对协议确定，不与历史官方绝对分数混排。

### 2.2 已实现机制

当前实现已经包含：

- SmoothQuant 的 amax/RMS 候选；
- hierarchy-aware、Weight-only、Activation-only 等通道置换候选；
- 4/8/16 分块确定性 signed Hadamard；
- E6M2 scale offset 搜索和边缘扩展；
- 精确 lv2/lv3 层级求解；
- 以损失覆盖率决定的困难块精修比例；
- Weight 激活协方差及 4×4 二次型舍入；
- Activation Weight Gram 及 4×4 二次型舍入；
- Attention Q/K SmoothQuant、K centering、GQA 对齐置换；
- V Attention head importance；
- 标准 HiF4 和已有候选的安全回退。
- A1 真实 Attention 输出选择器及对 B0 proxy winner 的逐层终验回退；
- 默认关闭的 A2 H64、A3 V importance 候选和 L1 scale 候选实验代码。

因此，H8、Weight 二次型以及困难块精修都不再是新机制。后续候选只能作为现有路径的增量扩展。

### 2.3 2026-08-27 首轮执行裁决

- A1 在公开 dev、三个 MHA 回归窗口、两种附加 scale mode 和 GQA 上均显著提高聚合均值，晋级为本地 Champion；offset 193 的 GQA non-causal 单层 `-6.89pp` 作为 C2 的尾部优化目标，不再触发整体回退；
- A2 聚合均值虽有改善，但违反单层和 non-causal 安全轨门槛，未晋级；
- A3 与 L1 未达到均值门槛，且分别出现尾部或 Attention 退化，未晋级；
- 下一动作是在 offset 0 上实现 C2 Segment-CVaR 选择器，直接相对 A1 改善 GQA 尾部；候选定稿后再运行固定回归矩阵。

## 3. B0 开发基线评测

评测配置：GPT-2 12 层、hidden size 768、12×64 heads、sequence length 128、2 个 calibration batch、2 个 test batch、`amax6` NVFP4。

| 指标 | 当前结果 |
|---|---:|
| Linear 均值 | 0.566819 |
| q | 0.627327 |
| k | 0.682394 |
| v | 0.593020 |
| o | 0.526626 |
| fc | 0.489287 |
| proj | 0.482259 |
| Attention | 0.378546 |
| 本次 CUDA 进程墙钟时间 | 31.88 秒 |
| 另一轮同配置 GPU 进程墙钟时间 | 40.2 秒 |
| 本次 CPU 进程墙钟时间 | 66.04 秒 |

这些本地分数只用于候选配对排序，不用于预测官方绝对分数。

### 3.1 统一计时口径

历史记录中存在至少三种不可直接混用的时间：31.88 秒 CUDA 进程墙钟、40.2 秒同配置 GPU 进程墙钟，以及消融表的 18.76 秒循环阶段时间。B0 起统一记录以下两个口径：

1. **进程墙钟时间**：从评测子进程启动到退出，包含 Python 启动、模块导入、模型加载、数据采集和算法执行。该值仅用于诊断环境波动，不进入官方时间预测公式。
2. **算法阶段时间**：从首次 `hif4_calibration_and_quantize_weight` 或 `hif4_calibration_attention` 调用前一刻开始，到最后一次动态量化 API 返回后一刻结束。它包含六个正式 API、必要 NVFP4/HiF4 数据准备和候选内部计算，但排除 Python 启动、模型加载、tokenization 和前向数据捕获。

CUDA 与 CPU 必须使用相同算法阶段边界、相同样本和相同计时实现。第 7.3 节只允许使用 CPU 算法阶段时间。当前 18.76 秒来自消融循环的旧计时边界，只可用于本轮消融内部比较，不能代入官方时间预测。

从分项看，当前主要精度瓶颈依次为：

1. Attention；
2. Linear `proj`；
3. Linear `fc`；
4. Linear `o`。

## 4. 关键机制消融

消融使用相同模型、数据和评测配置，仅在运行时修改对应开关，没有修改源码。

| 候选 | Linear | Attention | CUDA 消融循环时间 | 相对基线结论 |
|---|---:|---:|---:|---|
| 当前基线 | 0.566819 | 0.378546 | 18.76 秒 | — |
| 移除 block Hadamard | 0.522326 | 0.378546 | 13.81 秒 | Linear 下降 0.044493，核心机制 |
| 移除 Weight quadratic | 0.556201 | 0.378546 | 16.11 秒 | Linear 下降 0.010618 |
| 移除 Activation quadratic | 0.558972 | 0.378546 | 17.65 秒 | Linear 下降 0.007847 |
| 移除 permutation bases | 0.566443 | 0.378596 | 16.82 秒 | 精度近似不变，可作为时间预算来源 |
| 移除 V importance | 0.566819 | 0.376736 | 18.26 秒 | Attention 下降 0.001810 |
| refine coverage 0.999→0.99 | 0.566624 | 0.375497 | 17.96 秒 | Attention 下降 0.003049 |

由消融得到以下约束：

- 必须保留现有 block Hadamard；
- 必须保留 Weight quadratic；
- 必须保留 Activation quadratic；
- 必须保留 0.999 refine coverage；
- 应保留 V importance；
- `permutation bases` 可以作为独立的时间换精度候选，但未经 CPU 和官方验证前不能直接删除。

## 5. 第一优先级：Attention 真实目标对齐

### 5.1 A1：真实 Attention 输出选择器

当前 `_attention_candidate_metrics` 只比较 Q/K 各自的加权重建误差。该代理忽略：

- Q/K 误差在 logits 中的交叉影响；
- softmax 对不同 logits 区域的非线性敏感度；
- V 的几何结构；
- 不同 Query token 对 K/V 误差的不同放大程度。

A1 只改变 calibration 选择器，不增加动态量化步骤：

1. 保留现有 Identity、K-centering、Q/K SmoothQuant 和 permutation 候选；
2. 对每个候选执行真正离散 HiF4；
3. 计算完整 Attention 输出误差；
4. 同时记录 causal 和 non-causal 结果；
5. 默认以 causal 为主目标，以 non-causal 为泛化安全轨；
6. 当前选择和 Identity 始终参与最终比较；
7. 平均改善不足 0.2 个百分点或尾部退化时回退。

A1 不引入新 state，不增加动态路径耗时，是当前最高优先级候选。

### 5.2 A2：Attention Q/K 固定 H64

只有 A1 通过本地评测后才增加 H64：

- 每个连续 64 维 head block 独立旋转；
- 首版只比较 2 个确定性 sign seed；
- head_dim 128 按两个 64 维 block 处理；
- GQA 中映射到同一 KV head 的所有 Query heads 共享同一旋转；
- 在线只执行被选中的一个 FWHT；
- H64 必须经过 A1 的真实 Attention 输出门控；
- Identity 和当前 D/P/centering winner 始终作为回退。

首版不实现学习角度，也不同时扩大 scale offset。

### 5.3 A3：V 偏差与位置感知量化

当前 V importance 是 head 级统计。对于 head_dim 64，一个 HiF4 block 正好覆盖一个 head，head 内常量 importance 主要影响困难块排序，对块内离散解的影响有限。

A3 比较以下候选：

- 当前 V quantizer；
- 均值误差抑制候选；
- calibration attention probability 推导的 token/head 权重候选。

V 保持原坐标系，不旋转、不置换、不 centering。最终候选必须通过真实 Attention 输出门控。

## 6. 第二优先级：HiF4 格式专用优化

### 6.1 L1：数据驱动 scale 候选

当前 scale 精修主要围绕标准 amax E6M2 code 搜索固定 offset。新增候选应优先针对 HiF4 scale，而不是直接扩大 GPTQ 复杂度。

对困难块生成：

- 当前标准 scale 与 offset winner；
- weighted least-squares 连续 scale；
- 截尾或分位数 scale；
- 连续最优 scale 相邻的 E6M2 code；
- 一至两轮 scale 与离散层级交替更新。

所有候选仍返回合法 E6M2 scale，并与当前完整五字段候选逐块比较。未降低对应目标时逐字段回退。

### 6.2 L2：Linear H32/H64 增量候选

现有 4/8/16 Hadamard 是当前最大单项贡献机制。H32/H64 只能作为其增量候选：

- 保留 4/8/16 当前 winner；
- 优先在 `fc` 和 `proj` 上评估；
- 每层只保存一个最终 block size 和 seed；
- tie 时选择较小 block；
- 离线校准阶段允许用真实 Linear 输出目标优化 `Q(W)`；
- 该输出目标不得用于 `Q(A)` 的 gate、拟合或 state；若目标是激活侧，必须
  使用 activation-only 统计或重构误差。

### 6.3 L3：逐级扩大 Weight 二阶块

当前 Weight quadratic 只提取 4×4 Gram。不要直接跳到全量 64×64 GPTQ，按以下顺序实验：

1. 8×8 二阶块；
2. 16×16 二阶块；
3. 仅当前两者有效时测试 64×64。

实现约束：

- 只处理当前完整损失最高的困难块；
- 使用缓存 `H·e` 的增量二次型更新；
- 单坐标或单组更新不得重新计算完整 64×64 二次型；
- 每个新 block 解与当前五字段 winner 比较；
- 二次目标未下降或出现非有限值时回退；
- 首版只用于静态 Weight calibration，不进入 Activation 在线路径。

### 6.4 L4：针对宽层 `proj` 的选择性 Activation 二次型

当前 Activation quadratic 只在 `in_features <= 1024` 时启用，因此 GPT-2 `proj` 的 3072 输入维不会使用 Weight Gram。这与 `proj` 当前最低的 Linear 分项一致。

建议只为高损失的少数 64 通道组保存二阶状态：

- calibration 选择 top-K 通道组；
- top-K 使用 8×8 或低秩 Gram；
- 其余组继续使用当前对角 importance；
- state 保存组索引和紧凑 factor；
- 在线只对 top-K 组和最困难 Activation block 执行二阶更新；
- 超过状态或时间预算时自动回退当前路径。

## 7. 时间预算策略

### 7.1 可释放的预算

`_PERMUTATION_BASES` 在当前 GPT-2 评测中只贡献约 0.000376 Linear，旧 CUDA 消融循环计时节省约 1.94 秒。可以创建单独候选 T1：

- 关闭扩展 permutation bases；
- 保留初始 hierarchy-aware permutation；
- 运行完整 CUDA、CPU 本地配对；
- 只有固定本地矩阵精度不下降时，才将节省的时间用于 A2、L2 或 L3。

### 7.2 不应释放的预算

禁止通过以下方式为新机制腾出时间：

- 删除 block Hadamard；
- 删除 Weight/Activation quadratic；
- 把 refine coverage 从 0.999 降到 0.99；
- 删除 V importance；
- 大幅减少现有困难块精修而不做独立本地配对验证。

### 7.3 本地运行时间门禁

不再预测官方耗时。候选与 B0 在同一机器、同一数据和同一算法阶段边界下配对，记录：

```text
local_time_ratio = candidate_cpu_algorithm_time / B0_cpu_algorithm_time
```

首轮单机制默认要求 `local_time_ratio <= 1.15`。超过该值必须证明精度收益足够且另立时间换精度候选，不能借用历史约 140 秒推测外部运行时间。

## 8. 暂缓方向

以下机制不进入第一轮：

- 在线 Activation 完整 64×64 Cholesky；
- 全量 Weight 64×64 GPTQ；
- 12～20 步 learned butterfly；
- 完整 64×64 稠密学习旋转；
- 继续扩大固定 offset 网格；
- 同一候选同时引入旋转、scale、二阶和 V bias；
- 未经真实 Attention 门控的学习型变换。

只有 A1/A2 或 L1/L2 已证明存在稳定主效应，且仍有明确时间余量时，才重新评估 learned butterfly。

## 9. 首轮实验矩阵

| 编号 | 唯一变化 | 目的 |
|---|---|---|
| B0 | 无 | 固化当前精确本地基线和时间 |
| T1 | 关闭 permutation bases | 验证可释放的时间预算 |
| A1 | 现有 Attention 候选改用真实输出选择 | 验证目标对齐主效应 |
| A2 | A1 + 固定 H64 | 验证 Attention 旋转增益 |
| A3 | 仅 V bias-aware | 验证 V 路径主效应 |
| L1 | 仅数据驱动 scale | 验证 HiF4 scale 主效应 |
| L2 | 仅 Linear H32/H64 候选 | 验证更大旋转块增益 |
| L3 | 仅 8×8 Weight 二阶 | 验证跨 4 元素组相关性收益 |
| M1 | 最佳 Attention + 最佳 Linear | 验证已知 winner 组合 |
| M1R | 重复 M1 | Champion 晋级复验 |

第一轮不需要完整三因子组合，因为 A1、A2、A3、L1、L2、L3 都先作为单机制候选。只有两个单机制分别通过固定本地矩阵，才评估它们的组合。

## 10. 本地晋级门槛

候选必须满足：

- 目标分项配对均值至少提升 0.2 个百分点；
- 非目标分项下降不超过 0.2 个百分点；
- 单层最差值完整记录并进入下一轮优化目标，不再单独否定综合主效应；
- 最差十分位通过 `tail_mean_delta` 进入综合晋级门槛；
- causal/non-causal Attention 均无灾难性退化；
- MHA、GQA、head_dim 64/128 均通过；
- 所有 HiF4 五字段合法、有限且 shape 正确；
- 所有 state 为有限、无梯度、CPU、contiguous、dense-strided 数据；
- 本地 CPU 算法阶段时间比默认不超过 B0 的 1.15 倍；
- 候选源码 SHA、配置、分项和时间记录完整。

本地分数不满足门槛时直接判退，不能用聚合均值掩盖尾部失败。

### 10.1 锁定的本地回归窗口

offset `0` 是唯一开发窗口。offset `97`、`193`、`389` 已锁定并在 A1 裁决中使用，因此从 2026-08-27 起它们是已知回归窗口，不再称为匿名 holdout。后续候选不得根据这三个窗口选择阈值、seed 或机制。

每个定稿候选只运行一次固定矩阵：

1. 三个 offset 的 MHA causal/non-causal；
2. offset 193 的 GQA causal/non-causal；
3. offset 0 的 `amax4`、`pow2` 格式敏感性；
4. head_dim 128、GQA 旋转不变量和 saturated-logit 合成安全测试；
5. CPU 时间与 state/静态检查。

结果必须与候选 SHA 绑定。若查看回归结果后修改候选，必须分配新候选编号并回到 offset 0 开发阶段，不能把同一次裁决解释为独立盲测。

## 11. 必须先解决的规则与评测问题

### 11.1 Linear 合规口径

当前 `_linear_output_candidate_metrics` 会计算 sampled Activation 与 sampled
Weight 的矩阵乘积。早期设计又声明 Linear calibration 不得计算 `A @ W`，这
两者冲突源于旧的过宽规则解读；当前应按数据流区分离线 `Q(W)` 目标和在线
`Q(A)` 状态。

开始 L2/L3 前必须确认官方规则：

- 若 sampled output 只优化离线 `Q(W)`，可以保留该目标；
- 若 sampled output 影响 `Q(A)` 或 `activation_state`，必须移除并替换为
  `AᵀA`、activation-only 重构误差或其他不依赖输出的目标；
- 不能同时声明“所有 calibration 都禁止 Activation×Weight”和“官方允许
  离线 `Q(W)` 输出目标”。

### 11.2 Attention mask 口径

当前 `evaluator/real_data_eval.py` 使用 causal Attention，且 v002 在当前评测器上复现了远程 `youxilee/hif4` CHANGELOG 的全部分项结果。因此裁决路径为：

- **默认主口径为 causal**，用于 A1/A2 学习、选择和主晋级指标；
- non-causal 作为泛化安全轨，要求无灾难性尾部退化；
- 只有官方任务书、判题样例或可复现的官方/本地差异明确证明 mask 不一致时，才允许触发例外裁决；
- 例外裁决必须先更新评测协议和 B0，再重新开始候选实验，不能在已有实验中途切换主口径。

### 11.3 基线源码与时间

当前只要求本地证据闭环。每个 B0/候选必须记录：

- 本地评测源码 SHA256；
- 本地 CUDA/CPU 分项和时间；
- 与父 Champion 的差异说明；
- 官方状态 `pending`、`unavailable` 或 `recorded`。未来官方结果只追加，不能覆盖本地记录。

## 12. 实施顺序（由 2026-08-27 前向计划接管）

0. **已完成**：冻结 B0 本地基线、统一计时与评测器 telemetry；
1. **已完成并晋级 C1**：A1 真实 Attention 输出选择器；
2. **当前下一候选 C2**：Segment-CVaR Attention 选择器，修复 A1 的 GQA non-causal 尾部；
3. **C3**：在 C2 Champion 上增加 top-K 8×8 Linear 二阶；
4. **C4**：只对已晋级机制做组合后的时间精简；
5. A2、A3、L1 保留历史实验结论，不与 C2/C3 同时重新启用；
6. 每个候选本地评测结束立即归档，后续官方数据只追加更新归档日志。

### 12.1 Telemetry 合规边界

Telemetry 只能存在于离线评测器、外层 wrapper 或可在导出时完全剥离的开发代码中：

- 正式 `solution.py` 不得写文件、打印日志、访问网络或读取环境外部状态；
- 不得把累计计数、时间戳或历史候选数据放入官方 state；
- 推荐由评测器包装六个正式 API，在调用前后计时，并从返回 state 的既有只读字段推导选择率和回退率；
- 如确需开发期 hook，必须位于明确标记的 dev-only 区域，由候选导出器删除；
- 导出后重新运行 AST 静态检查，并验证提交文件不包含 telemetry 标记、文件 I/O、日志输出或调试依赖；
- manifest、分层报告和回退率只写入 `artifacts/` 等评测器侧目录，不进入提交源码。

## 13. 最终成功标准

本轮优化成功不以一次偶然高分定义，而要求：

- 至少一个 Attention 单机制在固定本地矩阵上取得稳定综合正增益；
- 至少一个 Linear 旋转或二阶单机制在固定本地矩阵上取得稳定综合正增益；
- 组合候选重复本地评测后仍高于父 Champion；
- CPU algorithm-stage 时间满足预注册门槛；
- 根 `solution.py`、归档源码和结果记录的 SHA 一致；
- 保留所有历史 Champion 和失败分支，不覆盖记录；
- 每个正式候选的增益来源可归因。
