# 持续研究循环：误差账本驱动的双侧机制队列（SUPERSEDED）

> SUPERSEDED，2026-09-08。双侧独立 gain 目标、误差账本和强制 next_card 已由
> [单一完整方案优化计划](../../plans/2026-09-08-single-solution-optimization-plan.md)取代。以下只保留历史机制证据，
> 不提供执行队列或当前门禁。
> 本文件**取代** [2026-09-08 双侧 0.9 执行任务书](2026-09-08-dual-side-09-execution.md) 的"候选卡清单"形态；
> 旧文件保留为 L23b/A25 证据与 P0 事务来源，**不再提供队列指令**。
> 测试统一按 [4B 指引](../../../4b-panel-testing-guide.md)；官方 300s 硬限，无本地时间门。

## 1. 循环定义

研究以 **R 轮**推进，每轮必须闭合下列六步；任一步的产出缺失则该轮未结束，不得停在"等用户指示"。

| 步 | 动作 | 硬产出 |
|---|---|---|
| R-1 账本 | 更新本侧误差账本（E/F 各格数值 + 覆盖 + SHA） | `error_ledger_<side>_<date>.json` + md 行 |
| R-2 定位 | 先检查指标可比与用户目标，再取最大可行动 OPEN 格；已达拟合目标但超时则优先成本，写出理由 | 一行 target 记录（格 / 数值 / 占比 / 下一格为何不选） |
| R-3 出卡 | 按 §3 模板出**一张**卡（8 字段齐全），或声明 §6 停止条件 | 机制卡（含去重四项比对） |
| R-4 验证 | 固定一配置：数学/可达 → 合法 state/control → 4B shard0 → 必要时六 shard | paired JSON + report + attempted/accepted |
| R-5 交付 | 独立归档 → 显式 git add/commit/push → 官方状态登记（不等待 0.9） | 归档目录 + SHA + GIT_PUSHED/READY_FOR_OFFICIAL/RESULT |
| R-6 续接 | 写出下一张卡（完整 8 字段）或 §6 的停止声明 | `next_card` 或 阻碍清单 |

两条侧线（Linear / Attention）**各自独立跑循环**，共享 GPU 锁
`artifacts/proxy_v3/v162-independent/gpu.lock`，Git 写操作串行；非目标侧与根 `solution.py` 冻结。

## 2. 误差账本规范

### 2.1 统一口径

- `gain = 1 − MSE_PLAYER / MSE_STD`，分母固定为**同 NVFP4 解码输入的标准 HiF4 输出**。
- 账本每格折算为**相对 MSE_STD 的归一化份额**；份额之和允许不等于 `1−gain`，差值记入残差格（交叉项），
  **不强行归零、不做正交化**，交叉项本身也是定位信号。
- 每格必须记录：case 数、覆盖（shard/层/role/长度）、父与候选源码 SHA、evaluator 版本、cache 路径。
- **零 API 优先**：能由已归档 JSON / 已有 dense cache 重算的格一律重算，不重跑模型前向。

### 2.2 Linear：同校准数据上的误差账本

用户目标是全校准数据fit_gain，不是独立窗口泛化。所有分解必须在相同样本、相同
MSE_STD、相同状态/SHA和相同聚合上计算。

| 格 | 定义 | 证据边界 |
|---|---|---|
| E1 | 校准数据上当前连续提案的输出残差MSE / MSE_STD | 只代表本次连续解，不是整个合法算法空间的界 |
| E2 | 同数据上连续提案与合法权重的输出差MSE / MSE_STD | 投影影响，不能忽略与E1的交叉项 |
| E3_fit | 同校准输入上最终合法部署输出误差MSE / MSE_STD | 目标量，fit_gain=1−E3_fit |
| E_cross | E3_fit−E1−E2 | 仅在上述身份一致时可解释为交叉项，可正可负 |
| E_holdout | 已有独立窗口的1−gain | 单独记录，不与校准E1/E2相减后称“部署失配” |

旧E4=独立窗口E3−校准E1−校准E2的0.6143只是跨数据统计差，不具有纯量化误差或
结构不可达含义，不据此优先开展泛化修正。目标fit_gain已达到但官方超时，优先处理校准复杂度。
不得为改善独立窗口而恢复跨fold一致性接受、分布配比或其他用户已取消的泛化门。

### 2.3 Attention：反事实诊断与最终输出

F4=真实独立窗口最终输出MSE/MSE_STD，为主目标量；F1/F2只在明确固定其余输入、
变换/编码坐标及V路径的反事实下测量。F3 logits/probability单独报告单位与参考，
不当作与输出MSE可直接相加的份额。F1−F2、F4−F1只是对应干预的统计差，
没有正交性与同坐标证明时不能解释为独立因果贡献，更不能推出自由度耗尽。

令目标输出为PV，实际为P_hat V_hat，则误差可写为
(P_hat−P)V_hat + P(V_hat−V)。总平方误差包含两项平方及交叉项。
“精确QK/概率P + 当前V_hat”的误差仅是一个固定干预结果，不是所有合法QK方案的下界；
改变P_hat可能补偿V量化误差。旧0.2754 V-only读数不能证明冻结V时gain0.9不可达。
V禁令继续遵守，不据此自动解禁V，也不在尚未证明不可达时要求用户降低目标。

### 2.4 每格标签（决定是否"尚可改变"）

| 标签 | 含义 | 必须附带的证据 |
|---|---|---|
| `OPEN` | 存在未尝试的结构自由度 | 指出具体自由度在哪（变量/插入点/编码） |
| `FAMILY_CLOSED` | 改变它的机制族已按 AGENTS §7 关闭 | 归档 SHA + 关闭结论 |
| `STRUCTURAL_FIXED` | 数学上不可改变（如 V 侧 per-channel multiplier 破坏输出） | 一行数学论证或 AGENTS 条款 |
| `IN_FLIGHT` | 已有在途卡/在途官方包针对该格 | 卡 ID 或官方包 SHA |

定位规则：只在同数据/同分母/同目标的可比格中取 `OPEN` 最大者；若最大格非 `OPEN`，取次大并在记录中写明"最大格 X 因 Y 不可改变"。

### 2.5 账本实现

- 脚本：`workbench/continuous_linear/error_ledger.py`、`workbench/continuous_attention/error_ledger.py`。
- 输入：已有 dense cache + 同 SHA 的父/候选 JSON + 父源码；**不新增前向、不新增 0.5B/OOD/跨模型**。
- 输出：`artifacts/continuous/<side>/error_ledger_<date>.json` 与同名 `.md`（一行一格，含上轮对比列）。
- E3/F4 直接取 4B 评测 JSON；E1/E2、F1–F3 缺值时才补**最小**运行（Linear ≤ shard0，Attention ≤ 72 例）。
- 账本 v0 必须在 R1 建立，**不得**为建账本重跑整个面板或重复已有配对。

## 3. 出卡规则（代理自行补卡）

### 3.1 卡模板（8 字段，缺一不算卡）

1. **ID / 侧 / 父 SHA / 创建轮次**
2. **靶点**：账本格 + 数值 + 占比（来自 R-2）
3. **改变什么**：变量、插入点、与父的**调用图差异**（一句话可验证）
4. **为什么可能有效**：机制论证；可引文献（必要时 WebSearch/WebFetch 查原始论文，卡内记来源名或 URL）
5. **固定配置**：一个代表配置，并列出**被冻结的全部自由度**，声明不扫参
6. **证伪判据**：数值阈值，写入后不再调整
7. **去重声明**：与已关闭族做四项比对 —— **目标 / 变量 / 插入点 / 编码**；数学等价或逐位等价即**取消注册**，不改名重试
8. **关闭粒度**：失败时关闭的具体一句话（不扩写为整族）

### 3.2 补卡来源（按优先级）

1. 账本最大 `OPEN` 格；
2. 源码审读发现的**未使用结构自由度**（五字段、fold、lv2/lv3、块序、状态逻辑）；
3. 已确认的外部成功机制（21071：A@W 拟合、Q/K 互逆 scale 学习）—— 只取机制，不继承成绩；
4. 文献原始论文（GPTQ/OBC 误差补偿、AWQ、SmoothQuant、QuaRot/SpinQuant 旋转与 incoherence、
   SageAttention 类 K/V 码误差、MXFP4/NVFP4 微缩放格式）—— 需要时自查，不等用户指定。

**禁止**：把已关闭族换名重出；扫 rank/正则/步数/窗口/阈值/覆盖率邻域；为凑数出无实质区别的卡；
因某一格暂时难测而跳过整个循环。

## 4. 固定配置验证（一张卡一个配置）

顺序，前一步不过不进下一步：

1. **数学/小形状独立参考**：闭式解 vs 数值解、维度、白化方向、目标值一致；
2. **可实现性**：六 API 自包含导入、合法 state、finite、非 no-op（`attempted/accepted > 0` 且 `changed codes` 证明可达）；
3. **冻结侧 control**：非目标侧输出逐位不变；
4. **4B 目标侧 shard0 paired**：父/候选同 cache、同协议、同 device、case 身份与 `mse_standard`/`reference_energy` 精确匹配；
5. 必要时六 shard（Linear 336 / Attention 72）。

关闭粒度：失败只写"关闭 <该卡第 3 字段的具体实现>"，并标注该族剩余自由度是否仍 `OPEN`。
不扫参数、不按最坏分组加专属路由、不为负向微调配置。

## 5. 交付与官方探索

- 通过 §4 前 4 步的候选：立即**独立归档 + 显式 git add/commit/push**（origin-ssh 优先），核验 `git status`。
- 状态分列：`GIT_COMMITTED → GIT_PUSHED → READY_FOR_OFFICIAL → SUBMITTED → RESULT`；
  无上传入口时标 `READY_FOR_OFFICIAL` 并给出包路径与 SHA，**不假称已提交**。
- **官方探索不等待 gain 0.9、也不等待账本完美**；0.9 是研究目标不是提交门。
- 每侧最多一个在途官方包。等待回传期间继续**独立机制**的推导与实现：新工作另开目录新 SHA，
  **不改动在途包源码**，不重复同 SHA/逐位等价提交。
- 官方回传处置：
  - 正向且 <300s → 升级侧父，账本换父基准并**重新定位**（新一轮 R-1）；
  - 负向 → 关闭本卡，把结果写回对应格作为"已尝试"证据，继续下一张卡；
  - TIMEOUT → 只关闭该复杂度实现，机制本身仍 `OPEN`；
  - 根 `solution.py` 仅在完整官方回传更优时切换（当前 v189 17616/275s）。

## 6. 强制续接与停止条件

每轮交付必须含 **`next_card`（8 字段齐全）**，否则必须给出停止声明。**停止条件仅三条**：

| 条件 | 要求 |
|---|---|
| **S1 达标** | Linear `fit_gain ≥ 0.9` 且 Attention 最终输出 `gain ≥ 0.9`，且均满足合法输出、完整同口径覆盖、官方 <300s |
| **S2 缺不可替代外部输入** | 必须列出：① 缺什么 ② 为何不可替代 ③ 已尝试的替代方案 ④ 何时可继续 |
| **S3 去重后无可执行新假设** | 必须列出**去重比对表**：候选卡 × 已关闭族 × 四项比对结论 × 判定轮次；并说明所有 `OPEN` 格均已有在途卡或在途官方包 |

**禁止**以下作为结束理由："这几个方向都试完了"、"等待用户指示"、"等待官方回传"（回传等待属并行队列，
不构成停止）、"本地 proxy 排序不利所以该路线结束"。

## 7. 当前执行队列（L28 / A28 之后）

本节是代理唯一可执行队列，取代旧 L24–L26/A26 初始表。Linear 已有 L28 官方
`4611/286s`（相对 L4 `+4/+39s`）；Attention 官方最高为 A2 `14440/274s`，A23
`14437/276s` 仅作为保留 R3 旧训练与互逆残余机制的研究父。A2 在分数和时间上均优于 A23，
二者不得再写成 Pareto 并列。

采用双父策略：Linear 的 `score_parent=L28`、`time_parent=L4(4607/247s)`；Attention 的
`score_parent=A2`、`mechanism_reference=A23`、`time_parent=R3(14405/238s)`。数学等价的降时卡从
score parent 重构；需要新增复杂校准或动态算子的收益卡从 time parent 构建，并以 score parent 为官方超越目标。
不得为了保留几分历史增益而无条件叠加所有训练栈。

### 7.0 当前状态快照（执行以此为准）

| 侧 | 已完成 | 当前父/对照 | 唯一下一步 |
|---|---|---|---|
| Linear | L28 `4611/286s` RETAINED；完整 fit `0.948587`；L29-Q/G 前置拒绝；L30 拒绝；LC0 只作正确性审计 | score parent=L28；time reference=L4 | 先核验完整候选 `17636/264s` 的计分 SHA；随后在 L31/L32 中去重后只注册一张，从 L4 构建 |
| Attention | AC0 `14395/258s`；真实 A29 实现 TIMEOUT，AC0 骨架不得冒充 A29 得分 | score parent=A2 `14440/274s`；time parent=R3 `14405/238s` | A30；A29 只有另立“降时实现卡”才可回访 |

所有本地时间仅标注风险和安排降时优先级；不得设置 `<280s` 或按层外推提交门。Linear 独立窗口
只记录，不得否决全校准 fit 候选；Attention 的 4B paired 只作合法性/风险判读，官方结果是唯一分数裁决。

### P0：证据身份与解释修复（先做，零模型 API）

| owner | 动作 | 完成条件 |
|---|---|---|
| Linear 代理 | 核对 L28 官方计分源码、归档源码、manifest、official-result 四处 SHA；唯一有效 SHA 为实际文件哈希 `44D7E964F82646331E4509996FFFEAB84E4FE0B3870528D45F25ED953EC93AB5` | 四处一致；官方 `4611/286s` 日志、状态与版本索引提交并 push；不把旧错误 SHA 留作当前身份 |
| Linear 代理 | 为 L28 重新生成完整校准 fit 表 | 168 states、336 fold-rows、同一 STD/父/候选身份；当前 shard0 `0.9453` 只作已有证据，在完整表产生前不继承 L23b 的全量 `0.948560` |
| Linear 代理 | 读取 4B manifest，列出各 role 的 `N,d,o,rank(A)`，估算薄 QR、完整 Gram、分块 cross-Gram 的 FLOPs/峰值内存 | 零模型 API 选择 L29-Q 或 L29-G，只注册一个；不能先实现后按本地时间挑赢家 |
| Attention 代理 | 对 A26/A27 paired 数据做 F2 深度归因 | 不调用模型 API；输出 layer/length/split、L22、code-change 与校准→独立窗迁移关系；只决定下一卡结构，不按分组建专属路由 |
| Attention 代理 | 写 A28 解释纠偏日志并更新活动状态 | 明确 A28 只关闭当前码语义下的 V offset/refine/importance 码分配；删除“冻结 V 时 gain≤0.725”的当前结论；历史原始结果不改写 |

P0 不得因文档修复延迟两个侧线的独立数学推导；Git 写操作由根代理串行合入。

### L-R1A：L29-Q 精确校准行空间压缩（速度候选一）

对加权量化激活做薄 QR：

```text
A = UT
Z = UᵀY
E⊥ = Y − UZ
||Y−AB||² = ||Z−TB||² + ||E⊥||²
```

正交余项与权重无关，因此原 L28 的每个连续提案、合法投影与严格接受差值都可在 `T,Z` 上精确计算。
本卡只改变校准计算表示，在线 API 和最终码不变；不得截断小奇异值后仍称精确。只有当
`rank(A)<N` 且压缩后的循环工作量明确下降时注册；否则记 `REJECTED_BEFORE_IMPLEMENTATION`，转 L29-G。

验收要求：小矩阵恒等式、逐块 `H/提案/接受序列/最终码` 与 L28 对齐，完整 fit 不下降，官方结果不低于
L28 且更快才升级时间父。它与抽样、fold 缩减、Gram top-8、weight-SVD 均不等价。

### L-R1B：L29-G 校准充分统计量驱动的顺序低维拟合（速度候选二）

**目的**：当薄 QR 无有效压缩时，L28 只有 14s 官方余量。L29-G 保持全部校准数据、
rank、块序、正则、合法投影和顺序接受语义，消除逐块物化 `N×o` 残差及重复大乘法，先恢复时间余量，
不继续优化已达标的拟合数字。

令量化激活 `A∈R^(N×d)`、teacher 输出 `Y∈R^(N×o)`、当前部署权重转置 `B∈R^(d×o)`：

```text
G = AᵀA
C = AᵀY
H = C − GB = Aᵀ(Y−AB)
```

对第 `b` 个 64 列块，保持 L28 的 rank-8 残差交叉子空间：

```text
L_b L_bᵀ = G_bb + λ·mean(diag(G_bb))·I
Z_b = L_b⁻¹ H_b
U_b = top8_left(Z_b)
Δ_cont = L_b⁻ᵀ U_b U_bᵀ Z_b
Bq_b = legal_project(B_b + Δ_cont)
D_b = Bq_b − B_b
ΔSSE = tr(D_bᵀG_bbD_b) − 2·tr(D_bᵀH_b)
```

仅当 `ΔSSE<0` 接受，并精确更新所有未来块 `H_future ← H_future − G_future,b D_b`。
这保留“前块接受后改变后块提案”的顺序依赖，禁止把所有块同时提案后一次接受。

固定配置：继承 L28 的全校准行、64 列块、rank=8、正则 `0.2/0.3`、块序和五字段 scale；
不缩样本/rank/步数。实现前先按真实 `N,d,o` 列出 `G/H` 内存与 FLOPs；若完整 `d×d` Gram
成本更高，则改成数学等价的分块 cross-Gram 缓存，仍属本卡，不改变目标或接受序列。

验收顺序：小矩阵逐块比较 `H_b/Δ_cont/Bq_b/接受序列` → `ΔSSE` 与直接残差重算一致 →
合法 state、激活与 Attention control → L28 同身份 shard0 fit_gain 与最终码对齐 → 记录 API 时间风险 →
归档、commit、push、官方。不得预称整体更快；只有官方 `<300s` 裁决时间。

结果分支：

- 官方分数不低于 L28 且更快：L29 成为 Linear 时间父；
- 官方正向但更慢：保留 L28，关闭该统计布局；
- TIMEOUT：关闭完整 Gram/统计更新这一复杂度实现，下一卡只做精确分块融合；
- 未超时但低分：先查接受序列和最终码是否与 L28 分叉；若语义一致仍低分，则停止等价加速换版本，转收益卡。

### L-R2：L30 全局 reduced-rank 输出残差拟合（收益候选一）

L28 的 rank-8 是“每个 64 列块各自 rank-8”，多块拼接后整体秩可持续增长，因此不是真正全局低维。
L30 从时间父 L4 构建，用一次全局 rank-8 更新替代逐块更新：

```text
R0 = Y − ABparent
min_rank(ΔB)≤8 ||R0−AΔB||² + λ||ΔB||²
```

在校准行空间求 reduced-rank regression，避免构造完整 `d×d` 逆；低维基来自真实量化激活与输出残差，
不是原权重 SVD 或激活能量基。只做一次全局合法投影和一次全校准输出接受判定。

固定 rank=8、L28 正则与全部校准行，不扫 rank/阻尼。连续解无材料收益、合法投影抹掉收益或源码审计发现
与旧 CAT/GPTAQ reduced-rank 实现等价时关闭/取消注册。成本和拟合均通过后才官方探索。

### L-R3：L31 输出目标下的合法相邻码更新（收益候选二，先去重）

L28 连续求解优化输出误差，最终却用逐元素欧氏最近码投影，目标在最后一步发生变化。L31 仅在 L30
的唯一全局连续提案成立后启动；以上述固定提案为上游，对比 L30 只改变合法投影，
每个坐标只比较当前码与一个相邻合法码，并直接使用：

```text
ΔL(D) = tr(DᵀG_bbD) − 2tr(DᵀH_b)
```

批量决定真实 A@W 二次目标下降的码更新，固定一次遍历并同步梯度；不构造 `N×o` 候选输出、不扫遍数或顺序。
该方向与 AdaRound/GPTQ/JDRQ/旧 4×4 Gram rounding 的去重风险最高：只有当“完整输出残差交叉项 +
低维提案后的合法码 + 本插入点”均未被实现时才注册，否则取消，不换名重试。

### L-R4：L32 真正联合的 A@W 低维互逆拟合（收益候选三）

当前 L23b/L28 主要改写合法权重，动态 activation 路径保持不变。L32 从 L4 构建并进入真正的联合坐标：
在每个固定 64 维块学习 rank-8、零初始化的可逆残余 `R=I+UVᵀ`，用 Woodbury 小逆部署
`A' = AR`、`W' = WR^{-T}`，浮点乘积严格不变；目标直接为全 Qwen3.5-4B 校准数据上的
`||AWᵀ − Q(AR)Q(WR^{-T})ᵀ||²`。

固定一轮 Gauss–Newton 方向 + 一次合法硬编码投影，所有校准行共同求解，不拆 fit/select，
不使用独立窗口否决，不扫 rank/阻尼/覆盖/块序。必须先证明 `R^{-1}`、最终五字段输出、
动态 activation 可达和非目标侧 control；fit_gain 未保持 `≥0.9` 或官方不正即只关闭该联合残余实现。
L32 必须先与父中已有 `residual_u/v`、rank 补偿、旧联合坐标/JDRQ、Householder 去重；若变量与调用图等价，
取消注册。动态新增两次 `O(Ndr)` 低秩乘法，必须替换而非叠加等价旧变换。

### A-R0：RB-0，把 A23 残余重新拟合到 A2 高分父（低风险组合，非主线前置）

**目的**：先得到正确的高分父，不复制已学矩阵。以 A2 完整校准输出为父 `P`，在 `P` 坐标重新学习
一层 A23 的乘积目标互逆残余 `exp(±S)`；调用图为 `A2 → residual S`，`S=0`、异常或 gate 失败
均逐位恢复 A2。V 和 Linear 冻结，K-center 同步编译。

固定配置继承 A23 的参数化、32 步、学习率、谱限和乘积 loss；划分改为 fold0–2 训练、fold3 选择、
fold4 独立 holdout。三个训练 fold 各自产生 driver 后等权聚合成一个 `S`，不按层/role 调超参。
验证必须包含 `S=0==A2`、合法 state、changed codes、V/Linear control 和 72 例 4B 诊断。

选择与 holdout 最终输出 MSE 均严格改善、非 no-op、专项负向损失 `<0.02` 即归档并提交官方；
72 例均值只记录，因为 A23/A25 已证明 proxy 与官方双向反转。官方 `>14440` 且 `<300s` 才升级侧父；
负向关闭“在 A2 上叠 A23 乘积残余”，TIMEOUT 只关闭这一组合复杂度。

### A-R1：A29 最终输出残差驱动的量化边界 Q/K 互逆补偿

> **当前裁决：IMPLEMENTATION_TIMEOUT。** 实际机制实现
> `solutions/v163_attention_a29-final-residual-s/solution.py` 官方超时；
> `solutions/continuous_attention_a29-boundary-output/solution.py` 只是 AC0 逐位骨架，官方
> `14395/258s` 归属 AC0。以下保留为机制定义证据，不是当前待执行卡。F4 仍 OPEN；只有新建、
> 去重后的降时实现卡才能回访。当前队列继续 A30。

**靶点**：直接优化 F4，而非继续压 Q/K operand MSE。A28 的 V 重编码饱和不构成最终输出下界；
A29 利用 QK 项与 V 项的负交叉补偿。固定父为时间父 R3，保留其已获官方验证的旧训练，
但不叠加 A22/A23 的 32 步残余训练；`S=0` 必须逐位恢复 R3。A23 只提供互逆残余的正向机制证据，
A2 `14440` 是必须超越的 score target。若以后换成其他父，必须显式重建账本和重新验证，不能静默继承结果。

对每个训练 fold 和 GQA 组，取父坐标连续 `Q̃,K̃`、真实五字段硬解码 `Q̂,K̂,V̂`：

```text
L0 = Q̂K̂ᵀ/√d + mask
P0 = softmax(L0)
Y0 = P0V̂
Rout = Yref − Y0
J_i = diag(p_i) − p_i p_iᵀ
M_i = J_i V̂
δℓ_i* = argmin_(1ᵀz=0) ||Rout_i − zᵀM_i||²
```

`δℓ*` 用 minimum-norm pseudoinverse 求解，只按数值秩容差截断，并把 RMS 限制到父真实 logit 误差 RMS。
R3 完整父后附 `S_g=S_gᵀ,tr(S_g)=0`，Q/K 分别乘 `exp(S_g)`、`exp(−S_g)`。以父量化残差
`E_q=Q̂−Q̃,E_k=K̂−K̃` 构造边界 driver：

```text
A_g(S) = (Q̃ S E_kᵀ − E_q S K̃ᵀ)/√d
S* = argmin_(S=Sᵀ,trS=0) Σ ||δℓ* − A_g(S)||²
```

每个训练 fold 独立求 minimum-norm `S`，Frobenius 归一后等权平均；固定取绝对特征值最大的 4 个方向，
不扫 rank。该线性式只决定方向。沿唯一方向，根据父五字段合法码的相邻中点和 `dQ=Q̃S,dK=−K̃S`
计算首个正向翻码距离；固定用输出敏感度加权的 `1/64` 分位数作为唯一步长，并限制
`||αS||₂≤log(2)/2`。只生成一个 proposal，随后必须经过真实编码/解码。

真正训练与选择目标始终是：

```text
Lhard(S) = Σ ||Yref − softmax(Q̂(S)K̂(S)ᵀ/√d + mask)V̂||² / Σ ||Yref||²
```

隔离规则：fold0–2 只产生 driver；fold3 只在完整父和唯一 proposal 中严格选择；fold4 完全独立 holdout，
只有最终硬输出 MSE 严格下降才部署，否则整层回父。fold4 不得改方向、步长或配置；72 个评测窗口只作外部诊断。

固定验证：minimum-norm/Jacobian/伴随和有限差分小形状参考、互逆转置、mask/GQA/K-center、
真实 changed codes、attempted/selected/holdout_pass、六 API 单文件、合法 CPU state、finite、V/Linear 逐位 control。
专项负向损失 `<0.02` 且非 no-op 后即可固定代表并官方探索；72 例 Δmean 不预测官方符号，也不因微负自动停交。

与旧卡的区别：A23 只用 amax 乘积；A24 是连续 STE 恒等死路；A26 是失真的独立舍入代理；
A27 是删除R3旧训练的单主干、固定 Hadamard 8 谱带和 signGD；A29 保留 R3 官方有效旧训练，拟合含 V 的最终残差，
用 softmax-Jacobian 定方向、真实码边界定唯一步长。动态 API 不增加 Gram、搜索或候选循环。

失败分支：

- `δℓ*` 或 `S*` 数值为零：关闭该 Jacobian→冻结误差线性映射，F4 仍 OPEN；
- `S*` 非零但 changed codes=0：关闭 `1/64` 边界步长实现，不改分位数重试；
- 训练硬 loss 降而 fold3/4 失败：关闭固定聚合，登记 transfer 失败；
- fold3/4 通过但 72 例微负：记录风险，仍保留一个固定代表的官方探索资格；
- 官方负向：关闭 A29 具体机制；TIMEOUT：只关闭该校准复杂度实现。

### A-R2：A30 闭式逐通道互逆输出能量平衡（速度与精度候选）

A2/A23 的 32 步 dense 训练成本高，且 amax 代理不感知最终输出。A30 同样从 R3 时间父构建，在完整父后学习每 GQA 组逐通道
对角 `d`，以 Q/K 两侧量化误差传播到最终输出的能量 `a_j,b_j` 构造固定近似：

```text
L_j(d_j) = a_j exp(2d_j) + b_j exp(−2d_j)
d_j* = 1/4 · log((b_j+eps)/(a_j+eps))
```

组内去均值并固定 `cond≤2`；Q 乘 `exp(d)`、K 乘 `exp(−d)`，连续 QK 不变。统计、闭式解、一次真实
硬编码和 fold3/fold4 gate 完成全部校准，不运行 Adam 或 matrix-exp 迭代；最终矩阵合入父
`learned_rotation`，动态 API 不增加候选循环或额外矩阵乘。

硬输出未改善、changed codes=0、fold4 不通过或官方无增益，只关闭“逐通道输出能量闭式平衡”这一实现。
它不同于每头常量 temperature、A22/A23 的全矩阵 amax-Adam、A25 单主干和 v187 importance，不扫 clamp/scale。

### A-R3：A31 64 块对齐的稀疏三角误差搬运（结构候选）

A31 不再追求更小 amax，而是在 head_dim 的四个 64 块间搬运量化压力。固定非重叠块对 `(0→1,2→3)`，
构造 `T=I+N`，其中 `N` 只含两个 rank-1 上三角块，故 `N²=0`、`T^{-1}=I−N` 精确；
Q/K 分别乘 `T` 与 `T^{-T}`，连续 QK 严格不变。

每个 rank-1 方向来自父最终硬输出 loss 对 `N` 的一次梯度主奇异向量，步长由真实五字段首个码边界唯一确定；
校准期每组只做两次 64×64 SVD 和一次硬 proposal，部署时把 `T/T^{-T}` 合入父 rotation，零动态增量。
不得扫描块对、rank 或方向。连续不变量失败、无翻码、fold3/fold4 硬 F4 不降或官方负向即关闭。

本卡变量是跨 64 特征块的 nilpotent shear，须与 crosspair-4×4、headwise/residual-pressure 排列、
A23 dense SPD 和 A27 Hadamard 谱带逐项去重；任一数学等价则取消。

### A-R4：A33 折一致切空间重心（选择候选，必须单独成卡）

若 A29/A30 的训练或 fold3 改善、fold4/72 例反转，下一轮才改变聚合：fold0/1/2 分别产生同一机制 driver，
Frobenius 归一后取固定几何中位及符号一致分量形成唯一 proposal；fold3 选择、fold4 holdout，二者均严格优于
完整父才部署。它只改变校准期 driver 聚合与 gate，不能和新目标/新变量同卡，也不能按 layer/length/role 路由。

复用已有梯度统计，额外成本仅聚合和两次硬读出。accepted=0/no-op、fold4 失败或官方负向关闭该固定重心规则；
不得变成 fold、阈值或 coverage 扫描。

### A-R5：A32 共享码语义变更（表示候选，单独授权后才启动）

A28 后剩余的 V 侧自由度是修改 scale 结构、lv2/lv3 判据、mantissa 重建表或 E6M2 映射，并同步修改
Q/K/V/Linear 共享编码与 `_dequantize_hif4`。固定语义必须保持五字段合法、单调、对称，编码端和解码端
一起变化；训练目标是双侧最终输出，禁止 V 专属路由或扫码表。

该卡作用域跨两侧并改变根代码语义，当前只登记为高风险后继；没有用户明确解禁前不实现。
A28 只穷尽旧语义下的 offset/refine/importance，不关闭本方向；cb1/cb2 存在编码错误，也不能作为合法语义空间证伪。

### 方法审计：为什么不能继续只精修现有实现

| 侧 | 当前方法的问题 | 已有证据 | 计划修正 |
|---|---|---|---|
| Linear | 每个 64 块各做 rank-8，拼接后整体可高秩；并非真正全局低维 | L28 全校准局部拟合很强，官方仅 `+4` | L30 改为一次全局 rank-8；L32 同时改变 A/W 坐标 |
| Linear | 激活路径冻结，权重独自补偿权重和激活量化误差 | L23b/L28 activation state 逐位沿用父 | L32 学 `A'=AR,W'=WR^{-T}` |
| Linear | 连续输出目标后接逐元素最近码，最终投影目标不一致 | E1 连续残差小，合法投影仍有差距 | L31 直接按 A@W 二次目标选择合法相邻码 |
| Linear | 逐块残差、谱分解与候选输出物化换来 `+39s` | L4 `4607/247s` → L28 `4611/286s` | L29-Q/G 只做数学等价的计算压缩，L4继续作时间对照 |
| Attention | 互逆变换在连续域保持 QK 不变，收益只发生在离散翻码边界 | A23 乘积 ratio 大降，官方仅 `+13` | A29 由最终残差定方向、真实码边界定步长 |
| Attention | amax/scale 代理忽略码相位、层级重求、softmax、V 和交叉项 | A26 代理下降但真实 QK MSE 约恶化 10 倍 | A29/A30 最终硬输出闭环 |
| Attention | 删除旧训练会把机制收益与父损失混在一起 | A25 相对 A23 `−380` | 所有新卡保留直接父的完整已验证训练；A29/A30 从 R3 时间父构建，RB-0 才处理 A2/A23 组合；零提案逐位回到直接父 |
| Attention | 32 步 dense/FD 训练慢，且 4B 与官方方向双向反转 | A23 `276s`，A27 gate `4/6` 仍独立窗负 | A29/A30/A31 一次解析方向 + 一次硬 proposal；4B只记录风险 |
| Attention | A28 固定干预被错误当成最终下界 | `0.2754` 只对应固定 `P,V̂` | 保留交叉补偿方向；A32 才触及码语义 |

算法来源只用于机制设计，不继承论文配置：GPTQ支持一次二阶误差补偿，QuaRot/SpinQuant支持
全精度不变量下的旋转自由度，FlatQuant/OmniQuant支持可学习等价仿射变换，SageAttention2支持
把 Attention 的 Q/K 平滑与最终注意力误差分开处理。对应原始论文：
[GPTQ](https://arxiv.org/abs/2210.17323)、[QuaRot](https://arxiv.org/abs/2404.00456)、
[SpinQuant](https://arxiv.org/abs/2405.16406)、[FlatQuant](https://arxiv.org/abs/2410.09426)、
[OmniQuant](https://arxiv.org/abs/2308.13137)、[SageAttention2](https://arxiv.org/abs/2411.10958)。
论文中的随机旋转、长时间训练、格式和 kernel 不直接移植；每张卡仍服从本计划的合法 state、单配置和 300s 边界。

### 双侧组合与优先级

执行顺序为 `P0 → (Linear速度预检、L30去重、A29数学验证并行) → A29/L30各一张固定卡 → 条件分支`。
Attention 是主收益线；Linear 的 L29-Q/G 是可选等价降时卡，不阻塞从 L4 开始的 L30 收益研究。
Attention 默认顺序为 `A29 → A30 → A31`；RB-0 只作为低风险组合旁路，不阻塞大机制，A33 只处理已观察到的
transfer 失败，A32 需另行授权。
每侧最多一个在途官方包；等待期间可推导下一卡，不得修改在途源码。

为避免再次陷入微增益邻域：官方正向结果仍按事实登记，但若新卡既不降时，分数步长又不超过当前同侧最近
有效步长（Linear `+4`、Attention `+13`），该具体家族不获得参数邻域续卡；下一轮必须换到上表另一条
结构轴。这个规则只决定研究优先级，不撤销官方正向结果，也不阻止不同机制的固定代表提交。

当 Linear 与 Attention 均有正式侧父后，只登记一个组合包。高复杂度组合先以 v180 `17597/242s` 为时间预算父
核对调用图，再生成自包含根 `solution.py`；不得把侧分或侧时间相加冒充完整结果。组合只做合法性、六 API、
interaction/control 和一次官方裁决，不用本地 overall 换算官方分。

## 8. 每轮交付模板（一张简表，必填）

`side | round | 账本格与数值(target) | card_id | 父 SHA | 候选 SHA | coverage | 正确/可达/control | 靶格 Δ | 最终 gain Δ |
Git 状态 | 官方状态 | next_card（或停止条件编号 + 阻碍清单）`

## 9. 边界

- 不新增 0.5B、逐候选 OOD、跨模型 GPT-2/opt、fresh-default 计时运行；不恢复本地时间公式或 280s 门。
- 官方 300s 为唯一时间硬约束；API 秒数只记录与风险提示。
- 本文件是**循环框架**；每轮的实测细节、失败分支与证据写入 `logs/execution/`，不在此追加流水账。
- 与 AGENTS.md 冲突时以 AGENTS.md 为准；本文件不放松任何合法性、隔离、单文件与官方裁决约束。
