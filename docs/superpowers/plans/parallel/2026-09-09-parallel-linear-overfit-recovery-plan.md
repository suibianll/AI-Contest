# 历史过拟合 Linear 机制的 4B 并行恢复计划

> 状态：并行执行附录（v2，2026-09-09 审查后修订）。
> 本文件从属于[唯一活动总计划](../2026-09-09-output-aware-rounding-and-joint-aw-plan.md)，
> 不建立第二条父版本、版本号或官方晋级线。当前完整父仍为 v202 Linear + v195 Attention，
> 官方 `18053/281s`。正式候选的归档、提交和根切换仍由唯一活动总计划统一处理。
> 规则优先级：`AGENTS.md` → `docs/4b-panel-testing-guide.md` → 唯一活动总计划 → 本附录 →
> workbench 状态；任何冲突以活动总计划为准，本附录不得覆盖其卡表、门禁或目录所有权。
>
> v2 修订要点：历史证据按"失败模式 / 正向但不可部署 / 已关闭同族"重新分类；新增 §3 零 API
> 前置去重门（R0），两条线必须先通过它才允许写代码；PLA1 补接受余量、校准期精确输出交叉核对、
> 宽层处理、hierarchy 双读数与成本比值；PLW1 改为受 R0 门控的诊断线，补活动编码器自覆盖对照、
> shard0 早停与校准侧成本预注册；§6 补并发/日志/缓存/单卡规则；新增 §7 记录契约、§8 完成条件。

## 1. 目的与历史证据分类

本计划只实现两个固定、彼此独立且不与当前舍入边界计划重复的算法，并用 Qwen3.5-4B 面板重新判断
历史机制，不原样运行归档源码。历史证据按结论分为三类，不得混用：

### 1.1 失败模式证据（不得当作正向信号）

- **C34**（`_ACTIVATION_QUADRATIC16 = False`，`solution.py:572`，注释 `REJECTED 2026-08-28:
  -3.43pp real-data`）：16 组二阶精修对单个 test sample 过拟合；日志同时给出并列主因
  "单侧动编码会**打破两侧补偿**而非纯叠加受益"
  （`logs/execution/2026-08-26-optimization-execution-log.md:1684`，另见 `1657-1658`）。
- **C70**：GPT-2 `+6.270550`、OPT `−0.615957`、Qwen `−6.991865`
  （`solutions/20260829_v070_c70-joint-refine-rejected_scoreNA_timeNA/result.md:27`）；
  L5d 审计对同一路线结论为"历史 C70 已经证明跨模型不稳定，**故本路线关闭**"
  （`logs/execution/2026-08-31-l5d-external-component-audit.md:71`）。
- **v091 / A5**：唯一机制就是离线 `A@(W−Q(W))^T`，即下文"信号 1"的实现，结果为
  `284.595177` 对 stable parent `293.755106`（Δ `−9.159929`），q/k/v/o/proj 回退
  （`solutions/20260830_v091_a5-joint-aw-rejected_score284.595177_time358s/result.md:5`）。

### 1.2 正向但不可部署的证据（本计划的前提）

- **v107–v125**（L3 Global Activation-LRH / L6b 宽输入 rank-4 / L6c `G_64` / C1a/C1c）：
  使用最终部署 `G_q = W_qᵀW_q` 的 off-block 低秩提案 + 逐行精确二次型 gate，panel 正向，
  但 API `481–2653s`、v125 标注 `runtime invalid`，**从未获得官方验证**
  （`docs/algorithm-inventory-and-directions.md:97-101, 146-157, 377-386`）。
  本计划的动机正是"机制有效但太重"，不是"这些机制尚未试过"。
  第 1 条信号（直接优化 `Q(A)Q(W)^T−AW^T`）的历史正向证据是 C70 的 GPT-2 单点，
  不是 v091；第 2 条信号的正向证据是上述 `G_q` 行级 gate 链。

### 1.3 已关闭的同族（执行前必须逐项对照，见 §3 R0）

- v141–v145 BDLR（rank-4 off-block selected-column，含 dynamic-only）：v143 本地 Linear
  `0.361153657663258`，v145 明确 `The selected-column approximation is therefore closed as a
  direction`（`logs/execution/2026-09-01-v143-bdlr-dynamic-only.md:3`、
  `logs/execution/2026-09-01-v145-bdlr-damped005.md:5`、`solutions/README.md:372`）；
  09-02 清单 §6 把 BDLR 变体列入"已关闭路线"
  （`docs/stale-information-inventory-2026-09-02.md:97-100`）。
- v212/AW9（64-block 共享整数 signed-mantissa 偏移）、v215（输出感知 E6M2 `scale_factor`
  相邻码步进，shard0 `−0.150813`）、v182 之后"残差低秩族接近饱和，**禁止升 rank 或邻域扫描**"
  （`logs/execution/2026-09-04-v182-official-result.md:42`）、v204 rank-2 残差段官方贡献 0 分
  （`logs/execution/2026-09-09-v204-v205-subtraction-pricing-official.md:9`）。
- `AGENTS.md §7` 对 A@W 拟合族的关闭边界与 AW 归因分析的唯一存活形态
  （A/W 联合、低自由度、直接在码空间、校准成本可控；
  `logs/execution/2026-09-09-aw-fitting-family-analysis.md:26-32`）。
- `workbench/continuous_linear/needs-new-hypothesis.md:14,31-33`：输出残差引导的 `Q(W)` 重解
  与 JDRQ 同目标同更新 → `DUPLICATE_CLOSED`；激活侧 `per-call 动态、gram 引导变体、块序变体`
  均已关闭（该文件只作历史证据，不重定义门禁）。

## 2. 与活动总计划的分工

活动总计划继续独占：

- `workbench/full_solution/linear-lrb1-residual-rounding/`；
- `workbench/full_solution/attention-arb1-qk-rounding/`；
- `workbench/full_solution/linear-jrb1-joint-aw-boundaries/` 及 L-JRB1 的共享
  activation/weight 舍入边界；
- 根 `solution.py`、正式版本号、`solutions/` 归档和官方结果登记。

本计划只使用：

- `workbench/parallel_linear_overfit_recovery/pla1-lowrank-output-correction/`；
- `workbench/parallel_linear_overfit_recovery/plw1-c70-layer-update/`；
- `artifacts/proxy_v3/parallel-linear-overfit-recovery/`；
- `logs/execution/2026-09-09-parallel-linear-overfit-recovery-pla1.md` 与
  `logs/execution/2026-09-09-parallel-linear-overfit-recovery-plw1.md`（两条线分开，见 §6）。

两条并行实现都从启动时的当前最高分完整根复制到各自工作目录，不读取或修改另一执行线的
workbench。研究结果先留在本计划目录；需要形成正式候选时，等待活动总计划当前卡完成写入，
再把单个机制重放到届时的最高分完整根。不得直接覆盖根文件。

**回放顺序与互斥（v2 新增）**：PLA1 的正式回放必须晚于 L-JRB1 卡的裁决；若 L-JRB1 已改变根
（含 activation threshold 张量），必须在新根上重算 rank-4 基、父"当前码"基准与宽层名单。
两条线都不得与 L-JRB1 叠加进同一候选；若将来要评估叠加，必须由活动总计划另行预注册
activation-only / weight-only / joint 归因读数，本附录不自行组合。

## 3. 通用 R0：零 API 前置去重门（v2 新增，先于任何实现）

R0 只做代码与归档对照，不运行模型。结论必须写进 §3.3 表格并提交后，才允许进入 §4/§5 的实现。
判为已关闭同族或活动机制邻域时，直接关闭该线：**不写代码、不跑 shard0、不跑六 shard**。

### 3.1 R0-A（PLA1）

1. **父机制绑定**：父动态激活路径的 live 机制是 `h_inv` GPTQ + `activation_state["gram"]`
   （= `_flat_group_gram(W_hatᵀW_hat, in_features)`，即部署输出 Gram 的 4×4 块对角部分）
   + `_DYNAMIC_OFFSETS` + v202 逐样本 sample-energy 块序。`activation_state` 只含 `"gram"`
   键，**不含** `gram8/gram16/gram64`（`solution.py:501, 548, 553, 8563-8579, 8733-8757`）；
   动态路径以 `group_gram=gram_ordered` 调用 `_activation_gptq_quantize`
   （`solution.py:10993-10997`、`11738-11742`），没有 8/16/64 的 gram。
   `_refine_activation_blocks64`/`_refine_activation_hierarchy64` 只在 `group_gram64` 非空时
   运行，而唯一传入 `group_gram64` 的是无调用点的 `_jdrq_calibration_products`
   （`solution.py:6791`、`6848`），因此属未激活路径，不得当作父已有机制。
2. **宽层名单**：`activation_state["gram"]` 仅在 `in_features <= _ACTIVATION_QUADRATIC_MAX_FEATURES
   (3072)` 存在（`solution.py:8563-8579`）。逐 role 列出 4B 的 in_features 与 gram 是否存在，
   确认受影响 case 数（预期 4B 的 o/proj 为宽层）。
3. **同族差异论证**：逐项写出 PLA1 与 v141–v145、v107–v125、AW8/v211、AW14、L23/L30、
   v182 "禁止升 rank"、v204 的差异。允许声明的唯一增量是：**固定 rank-4 的块外度量项 +
   固定 hierarchy 下相邻 mantissa ±1 的候选集 + 每行每块最多一个 4 元素组**。
   若论证退化为"同族内换 rank / 换接受规则 / 换粒度"，记 `NEIGHBORHOOD_CLOSED`，关闭 PLA1。

### 3.2 R0-B（PLW1）

1. **活动编码器自覆盖**：现役权重编码器已对每个 64 块枚举 E6M2 offset
   （`_WEIGHT_OFFSETS = (-1, 1, 2, 3)`，`solution.py:502`，经 `_gptq_quantize_weight`/
   `_dense_to_hif4` 传入，`8488`/`8501`），沿 offset 维展开后一次性 `_solve_exact_hierarchy`
   精确重解 lv2/lv3/mantissa，并按块 argmin + improve-mask 接受（`3896-3957`）；
   另有默认关闭但同形状的 `_weight_e2e_refine`（offset 集 `(-3,-2,-1,1,2,3,4)`，
   `solution.py:66,68`，按真实输出 MSE 逐块接受，`5324-5385`）。
   `_JDRQ_*` 族（`_jdrq_select_weight_candidate` 等）在根里**没有调用点**，是只读残留
   （`solution.py:7519` 唯一定义；旁证 `logs/execution/2026-09-05-coordinate-error-and-probes.md:72-73`），
   不得当作"当前父 JDRQ"。
2. **已关闭对照**：v212/AW9、v215（shard0 `−0.150813`）、C70 跨模型回归、L5d 路线关闭、
   `AGENTS.md §7` 的 A@W 形态关闭、AW 归因分析的唯一存活形态（PLW1 冻结 A，不满足"A/W 联合"）。
3. **差异论证**：PLW1 相对活动编码器的可声明增量只有"整层一次性精确接受（而非逐块局部接受）"。
   若该增量落在 offset/候选数量/coverage/接受层级邻域内，或不能证明其不是
   `hierarchy 邻码步进` 的重试，记 `NEIGHBORHOOD_CLOSED`，关闭 PLW1：不写代码、不跑模型。

### 3.3 R0 结论（执行前填写）

| 线 | 对照项 | 判定（通过 / NEIGHBORHOOD_CLOSED） | 依据 文件:行号 |
|---|---|---|---|
| PLA1 | §3.1.1–3.1.3 | 待填 | 待填 |
| PLW1 | §3.2.1–3.2.3 | 待填 | 待填 |

## 4. PLA1：低秩输出度量的一次运行时 Activation 纠码

### 4.1 要解决的问题

C34 的单侧激活精修在旧面板过拟合，且日志给出的并列主因是"离线统计驱动在线编码会打破两侧
偏差抵消平衡"；v107–v125 用最终部署 Gram 恢复了 panel 正向，但逐行、逐坐标、多轮候选循环
导致 `481–2653s` 不可部署。PLA1 只做一次反向、低秩、样本自适应纠码，不做多轮。

### 4.2 固定算法

校准阶段对最终部署权重 `W_hat` 计算 `G = W_hatᵀW_hat`。保留父已有的 4×4 block-local Gram
（= `activation_state["gram"]`，仅窄层存在），并对去掉这些 block-local 项后的剩余矩阵做一次
固定 rank-4 对称特征分解，保存 `U[channels, 4]` 与带符号特征值 `lambda[4]`。不搜索 rank；
所有形状固定使用 4，不足 4 时使用实际可用维数。剩余矩阵一般**非半正定**，因此该分解只用作
梯度近似（见 §4.3 的接受与核对规则），不得当作可信任的曲率。

动态阶段：

1. 先完整执行当前父 activation 编码，得到合法五字段和 `X_hat`；
2. 计算当前样本误差 `E = X_hat − X`；
3. 用 `E U diag(lambda) Uᵀ` 加父已有的 block-local Gram 项，得到输出误差梯度近似；
4. 每个自然 64 通道块只比较当前码与固定 hierarchy 下的相邻 mantissa `−1/+1`，全部候选一次
   tensor 化计算；每行每块最多改一个 4 元素组，只接受近似二次型严格下降且超过接受余量的组
   （余量见 §4.3）；
5. 所有块只反向处理一遍，不重新估计梯度，不做第二轮，不改变 scale/lv2/lv3；
6. 输出仍为原五字段，零 mantissa 使用规范零 sign。

**宽层处理（v2 新增）**：`in_features > 3072` 的层父没有 4×4 Gram。默认对这些层**跳过**
PLA1（保持父码），只处理窄层；若确需覆盖宽层，必须作为独立读数单独预注册"补 blockdiag(G)"
的目标函数改动，并单独记录，不得与窄层结果混排。

### 4.3 接受判据与精确核对（v2 修订）

1. **余量**：代理接受门必须带余量，与父同量级（对齐 `_ACTIVATION_GRAM64_ACCEPT_MARGIN = 1e-5`
   或等价相对余量；`solution.py:590`）。L-RB1 以 ΔL ≈ 5e-7 的噪声级收益被接受后六 shard
   全负（`docs/superpowers/plans/README.md:17-19`），零余量判据不再使用。
2. **校准期精确交叉核对（kill gate）**：校准期必须用**完整** `G_full = W_hatᵀW_hat`
   （不截断、不取块对角，校准期 `weight_output_gram` 可用）对全部 4B calibration case 逐组
   计算精确二次型
   `ΔL_exact = 2·⟨E W_hat, ΔX⟩ + ΔX·G_full·ΔXᵀ`（case 等权），并与代理判据逐组比对，
   记录符号一致率、被代理接受但被精确拒绝的组数。**预注册门：一致率 < 99% 或精确拒绝率 > 5%
   时关闭该实现，不进入 shard0/六 shard。**
3. **动态侧不新增 state 张量**：除已预注册的 `U`/`lambda` 外不得新增 state 字段。若要让动态侧
   做精确接受，必须把完整 `G_full` 放进 state，属新增 state 字段与体积变更，需单独预注册；
   默认不做。
4. 块间耦合：父 `_activation_gptq_quantize` 是带误差前馈的顺序 GPTQ（`solution.py:3595-3607`），
   后块码建立在父已传播的补偿上；一次反向改前块会使补偿失配。必须记录"被改块的块序分布"
   作为该项诊断。

### 4.4 执行

1. 通过 §3.1 的 R0-A 后建立父 control：关闭 PLA1 时与复制的完整父逐位一致。
2. 用合成输入确认 rank-4 路径实际产生合法 mantissa 变化，并记录 attempted/accepted、修改行数、
   修改 4 元素组数、代理与精确二次型变化、宽层跳过计数。
3. 完成 §4.3.2 的校准期精确交叉核对；未过门即关闭，不跑 shard0。
4. 运行 Linear shard0；若接口正确但所有真实 case 都没有发生修改，记录 `NO_REACHABILITY` 并结束。
5. 有真实修改时运行固定六 shard一次，保存 336 个配对 case、API 时间、候选/父 `api_total` 比值
   （>1.5× 只标注时间风险，不设本地时间门）和分解结果。
6. **hierarchy 双读数（v2 新增）**：必须分别记录"固定 hierarchy"与"随阈值重选 hierarchy"
   两种读数下五个字段各自的 changed count、解码输出差异和 Attention/Linear 输出差异；
   只有重选后五字段全部与父逐位相同，或最终解码输出全部逐位相同，才记 `ABSORBED_BY_HIERARCHY`；
   mantissa changed count 单独归零不能作为关闭理由（对照活动总计划 §4.5）。
7. 不因局部分片或角色结果修改 rank、邻码范围、处理轮数、每块修改数量或宽层名单。

### 4.5 结果处理

- R0 判为 `NEIGHBORHOOD_CLOSED`、校准期精确核对未过门、或六 shard 最终与父逐位相同：
  关闭 PLA1，不生成版本号，不改成 per-row/per-block 阈值或其它粒度重试。
- 有合法非等价输出：把源码、固定配置和结果交给活动总计划；由总计划在最新完整根上重放后决定
  正式归档与官方提交。本地正负只写诊断，不作为提交门。
- 官方超时：只关闭这次运行时纠码实现，不把 rank4 改成 rank2 或减少覆盖重试。

## 5. PLW1：C70 的当前根单轮整层静态 A@W 重构（受 R0 门控）

### 5.1 要解决的问题

C70 在旧 GPT-2 上正向、OPT/Qwen 负向，L5d 已判 joint X/W residual 路线关闭；旧实现用三轮
逐块 Gauss–Seidel，对模型、父坐标和校准窗口高度敏感。PLW1 保留"冻结真实 Q(A)，按最终输出
残差重构 Q(W)"的核心，去掉多轮更新和局部立即接受，只做一次整层接受。

### 5.2 固定算法

1. 完整执行当前父的 Linear 校准，冻结最终 transform、activation state 和 `W_hat`。
2. 对全部 4B 校准 case 生成父动态激活 `X_hat`，教师输出 `Y = X Wᵀ`（X 与 W 必须与父校准路径
   同源，写进实现注释并在结果中记录来源）；父残差 `R = X_hat W_hatᵀ − Y`；case 按父口径
   等权（`ω_f = 1/(F·n_f·o)`），不得改成"按元素数归一后等权"以外的其它口径。
3. 对每个自然 64 通道权重块只生成一个 C70 型候选集合：保持父 sign 语义，固定使用
   `{-2, −1, +1, +2, +3}` 的 E6M2 offset 集，每个 offset 完整重解合法 lv2/lv3/mantissa。
4. 用固定 `X_hat` 下的精确二次型计算每个块候选；每块只保留损失最低且优于父的一个状态。
5. 把所有保留块一次性组装成唯一整层候选，再计算一次完整整层 A@W 损失。整层严格改善才写回，
   否则整层全部恢复父状态。
6. 只做这一轮，不按写回结果更新残差，不扫 offset、块比例、fold、阻尼或更新顺序。
   动态 Linear API 完全不增加计算。

**与活动编码器的关系（v2 新增）**：上述 offset 集与现役 `_WEIGHT_OFFSETS`（`(-1,1,2,3)`）
及默认关闭的 `_weight_e2e_refine`（`(-3,-2,-1,1,2,3,4)`）同处一个 offset 维度，目标函数
（输出残差二次型）也已是父 group-gram 路径的形态。本卡唯一可声明的增量是"整层一次性精确接受"，
其是否构成新机制由 §3.2 的 R0-B 判定；未通过则本卡关闭。

### 5.3 执行

1. 通过 §3.2 的 R0-B 后实现；先做等价性检查：在相同候选集下与活动编码器输出逐位比对，
   若所有提案与活动编码器等价，记 `SUBSUMED_BY_ACTIVE_ENCODER`，不运行模型评测。
2. 记录 attempted blocks、locally improved blocks、layer accepted、五字段 changed count 和
   精确整层 A@W 损失变化。
3. **校准侧成本预注册（v2 新增）**：先只对单层/单 role 记录候选/父 `api_total` 比值与绝对值；
   超过父 1.5× 记 `TIME_RISK` 并停在 shard0，不进入六 shard（只提示，不设本地时间门；
   官方 300s 是唯一硬约束，当前余量 19s）。
4. 运行 Linear shard0 排除接口和不可达问题；**shard0 零翻码或与父逐位相同即关闭，不跑六 shard**。
5. 有真实改动时固定运行六 shard一次。
6. 不根据 4B 结果增加第二轮、改变 offset 集合或拆分 role/layer 专属配置。

### 5.4 结果处理

- R0 判为 `NEIGHBORHOOD_CLOSED`、被活动编码器完全覆盖、shard0 零翻码或最终整层全部回退：
  关闭 PLW1，不生成版本号。
- 形成合法非等价候选：交给活动总计划在最新完整根上重放并安排正式归档、官方提交。
- 官方负向或超时后关闭本实现，不从旧 C70 邻域继续搜索。

## 6. 并行安排、资源与交付

PLA1 与 PLW1 代码和缓存独立，但**单卡 8GB 上必须串行占用 GPU**；两条线不相互叠加，也不等待
对方结果形成正式候选：

| 工作 | 目录 | 日志 | 输出目录 | 输出 |
|---|---|---|---|---|
| PLA1 低秩运行时纠码 | `pla1-lowrank-output-correction/` | `…-pla1.md` | `…/pla1-<run-id>/` | candidate、control、shard0、六 shard 结果 |
| PLW1 静态整层重构 | `plw1-c70-layer-update/` | `…-plw1.md` | `…/plw1-<run-id>/` | candidate、等价性记录、shard0、六 shard 结果 |

- 两条线**各自独立**的执行日志与 `--output-dir` 子目录，不共用文件、不并发追加。
- 缓存清理只清理本线候选：`python workbench/cache_cleanup/prune_calibration_cache.py
  --keep <当前根前缀> --keep <本线候选前缀> --min-age-hours 2 [--apply]`；
  不得删除 `qwen3.5-4b-proxy-v2.pt` 主缓存，不得按年龄误删另一线两小时内产生的缓存
  （`AGENTS.md §6`，2026-09-09 实测并发清理删掉活候选缓存 11.21 GB）。
- 每条线完成后在**自己的**执行日志中写清算法是否可达、实际改变了哪些字段、4B 诊断、API 时间
  与候选/父 `api_total` 比值。

## 7. 记录契约（v2 新增）

每条候选无论成功、失败、超时或未提交，都必须记录：源码 SHA256、固定配置、attempted/accepted、
changed 五字段计数、代理与精确判据的一致率、候选/父 `api_total` 比值、六 shard 配对 delta、
结论标签（`REJECTED` / `NO_REACHABILITY` / `ABSORBED_BY_HIERARCHY` / `SUBSUMED_BY_ACTIVE_ENCODER` /
`NEIGHBORHOOD_CLOSED` / `TIME_RISK` / `TIMEOUT` / `unregistered/NA`）。缺源码/SHA/配置的结果标
`non-reproducible`。归档与 git 提交由活动总计划统一执行，本附录不自行提交根文件。

## 8. 完成条件

本计划完成条件是 PLA1、PLW1 各得到一次明确裁决并把非等价代表交回活动总计划。若两条线都被 R0
或早停关闭（`NEIGHBORHOOD_CLOSED` / `SUBSUMED_BY_ACTIVE_ENCODER` / `NO_REACHABILITY`），整条
"历史过拟合 Linear 机制恢复"方向记 `CLOSED`，归档前不再开新线。结束后将本文件移入
`docs/superpowers/archive/plans/`；当前唯一活动总计划在整个过程中保持不变。
