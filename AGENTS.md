# HiF4 竞赛工作约定

本文件是本仓库的持久化工作记忆。每次开始任务、上下文恢复或切换算法前，先读取本文件，
再读取活动计划、当前状态、归因记录、归档索引和评测器。用户明确回传的官方结果优先于
本地代理结果和旧文档；不能依赖未写入仓库的隐式记忆。

> **读旧文档前先看 [`过期信息清单`](docs/stale-information-inventory-2026-09-02.md)**：
> 其中登记了已失效但仍留在历史文档中的信息（旧权重分数、420s 时间口径、36000 目标、
> official-shape-v1 协议等）。**v74 在当前官方评测集仅 `14561 / 188.9s`**（旧权重
> `22750` 已失效），不再是安全基线或归档冠军。

## 1. 代码架构

### 1.1 活动提交

- 根目录 [`solution.py`](solution.py) 是当前活动提交文件，只实现评测器要求的六个公共 API：
  - `hif4_calibration_and_quantize_weight`
  - `hif4_dynamic_quantize_activation`
  - `hif4_calibration_attention`
  - `hif4_dynamic_quantize_q`
  - `hif4_dynamic_quantize_k`
  - `hif4_dynamic_quantize_v`
- **正式版本的 `solution.py` 必须是完整、单文件、自包含的提交代码**：评测时不能通过
  `importlib`、相对路径、绝对路径、归档目录、其他 `.py` 文件或本地实验脚本加载任何实现。
  复制/合并后的文件必须在脱离仓库其他源码时仍能独立导入并提供六个 API。
- 仅用于本地归因的组合控制可以加载 immutable archive，但必须明确标记为
  `LOCAL ATTRIBUTION CONTROL`，不得作为正式版本、晋级候选或官方提交包。
- 编码器、解码器、E6M2 scale、lv2/lv3、mantissa/sign 等底层格式逻辑与六个 API 保持在同一
  提交模块内，所有输出必须满足 `evaluator/reference_hif4.py` 的合法状态检查。

### 1.2 Linear 路径

- 校准阶段负责反量化校准激活、收集统计量、选择等价变换、量化权重并编译 activation state。
- 动态阶段只执行校准阶段确定的变换和已编译的量化规则；禁止把校准搜索、完整矩阵求逆或
  未限制的 Python 候选循环带入在线路径。
- 研究算法的统一目标是实际输出误差，而非孤立 operand MSE：

  `XW^T - Q(XR) Q(WR^{-T})^T`

- 允许研究的结构包括 SmoothQuant/Permutation/block-Hadamard 等价变换、完整合法 HiF4
  block 的 Weight GPTQ、部署权重 Gram 驱动的 Activation GPTQ、层级 scale/lv2/lv3/mantissa
  联合选择，以及双侧 Weight–Activation 残差优化。
- 变换必须保持连续域乘积不变；Hessian/Gram 必须在最终变换和部署权重坐标系中计算，不能
  混用旧坐标统计量。
- 宽层、窄层和 q/k/v/o、gate/up、down/proj 等 role 可以使用不同求解器，但差异必须由
  矩阵形状、谱结构或输出误差解释，不能仅凭参数试出。

### 1.3 Attention 路径

- Attention 与 Linear 分开维护、分开归因，不能在同一次精度实验中同时修改两侧。
- 在 Linear 研究期间，默认冻结已验证的 v86 Attention；除非计划明确开启独立 Attention
  实验，否则不得使用 v138–v145 的缩减 Attention 替代参照。
- Attention 的动态复杂度必须显式受限：固定候选数量、token 视图和 block 结构；不把未限制的
  per-sequence/per-token 搜索作为“精度优化”带入官方候选。

### 1.4 评测与数据

- 唯一本地主评测器是 [`evaluator/official_eval.py`](evaluator/official_eval.py)，协议标签为
  `proxy-v2`；`official-shape-v1` 只保留为 immutable 历史诊断，不得继续生成或与新结果混排。
- 固定本地结构假设为 Qwen2.5-0.5B、同一只读 CUDA cache、Attention calibration lengths
  `[10,128,512,1024,1024]` 和独立 HiF4 validation。说明书没有公开指定 Qwen、层数、GQA 或
  RoPE，因此本地模型只用于同机机制诊断，不代表官方隐藏结构。
- 默认复核 panel 为 168 Linear（24 层×7 role）+ 120 Attention（24 层×五个长度）；
  `--full-cases` 的 2016+288 只作 stress。算法快速迭代使用 `--effect-panel`：Linear 固定选择
  layer `0/3/7/10/13/16/20/23` 并保留每层全部 7 role，共 56 cases；Attention 使用五个覆盖
  深度与长度的哨兵。`--linear-cases/--attention-cases` 是顺序前缀 smoke，尤其 14/56 不是
  纵深采样，禁止用于判断算法是否有效。
- effect panel 只缩减动态评分与 evaluator-only 分解；校准仍保持完整 168 Weight state +
  24 Attention state 的调用图，禁止为了提速改成按选中 case 校准或 per-case oracle。
- 主要本地字段是 `linear_mean`、`attention_mean`、逐 case 分数、六个 API 的
  `api_total_seconds` 和 `wall_seconds`。本地等权显示值只用于公开 panel 诊断，不拟合官方总分。
- 机制实验必须使用父子版本逐 case 配对：父版本先保存 immutable JSON；候选使用同一 cache、
  同一 panel 和 `--baseline-json`，按 `(layer, role, test_window, split, length)` 精确匹配。
  已有相同 panel 的结果用 `--candidate-json` 零 API 重放。case identity、`mse_standard` 或
  `reference_energy` 不一致时比较必须失败，不能手工对齐不同 panel 的均值。
- 每次配对先声明实际修改的 `--focus-linear-roles`（可用具体 role 或 `fc/qkv` family），并按
  以下顺序读取 `paired_effect`：focus 的 mean/median signed delta 与正负 case；未修改 control
  是否 no-effect；Linear 的 W-only/A-only/Both/interaction 或 Attention 的 Q/K/V/QK/QKV；
  最坏 role/layer/shape/split/length；同机 API 时间差。符号标签 `consistent_improvement`、
  `consistent_regression`、`mixed`、`no_effect` 只描述结果，不是新的人为阈值。
- 官方总分、官方时间和本地指标分开记录。官方历史提交中的“通过/超时”案例（包括已知的
  v86 通过样本和 v128/v129/v131 等超时样本）是时间风险判断的主要参考证据；先比较候选与这些
  案例的实际算法结构、API 调用和复杂度变化，再决定是否值得提交。
- 本地秒数不能换算成官方平台时间，也不能把本地 `300s` 当作通过/超时门槛。没有新的官方回传时，
  候选的官方时间状态必须写 `unknown`；本地计时只用于同机 A/B、算子热点和明显回归诊断。
- 官方 `<300s` 只能由官方回传确认；本地结果不得覆盖已有官方通过/超时事实。

## 2. 实验原则

### 2.1 启动顺序与证据优先级

开始任何实现或评测前，按顺序读取：

1. `AGENTS.md`；
2. `docs/superpowers/plans/README.md` 与当前唯一活动计划；
3. `docs/current-solution-status.md`；
4. 相关归因/研究记录和 `solutions/README.md`；
5. `evaluator/official_eval.py`；
6. 作为父版本的源码、`result.md`、JSON 和执行报告。

证据优先级为：当前用户明确官方结果 > 活动计划中的已确认事实 > 归档 result/log > 本地
JSON/report > 未验证推测。发生冲突时保留原始证据并更新状态文档，不覆盖历史数据。

### 2.2 单变量与理论算法

- 先固定场景再改另一场景：Linear 实验冻结 Attention，Attention 实验冻结 Linear。
- 每个正式版本只引入一个可解释的数学机制；用简短说明交代目标函数、不变量、误差传播路径
  和复杂度变化即可，不要求为探索版本编写过度设计文档。
- 不进行无理论依据的 alpha、offset、seed、rank、block 数、sweep 或 damping 逐个试探。
  参数网格只能作为同一算法内部的未编号 workbench，并在一份汇总日志中记录。
- 优先实现活动计划中的结构算法（例如 block-Schur GPTQ、低秩+块对角 Hessian、双侧联合
  残差和相同部署复杂度的结构变换），不能把“小修补”包装成新方向。
- 需要判断一个算法是否有效时，优先记录能回答当前问题的逐 role/逐 case 输出误差和 API
  分解；不要求每个小实验都生成完整诊断矩阵。

### 2.3 评测与决策

- 已有结果足以回答的问题不重复跑全量评测；只有代码发生实质变化、需要复核异常或用户明确
  要求时才重新评测。小改动可以先做针对性 smoke/单层测试，不强制跑完整测试套件。
- **固定评测流水线（2026-09-02 起）**：同一 parent、cache、设备和 evaluator 只建立一次
  immutable parent JSON；后续候选一律复用该 JSON，不重复运行 parent。每个新机制最多按
  `smoke → effect-panel → default-panel` 顺序推进：`smoke` 只检查六 API、合法状态和目标
  layer/role，不能作为效果证据；`effect-panel` 使用 `--effect-panel --baseline-json`，保留
  完整 168 Weight + 24 Attention calibration，只减少动态评分到 56 Linear + 5 Attention，
  用于一次父子逐 case 归因；只有 focus 方向、control 无泄漏、最坏 case 可解释且没有需要
  立即拒绝的回归时，才运行一次 `default-panel`（168 + 120）复核。default 未通过即停止并
  记录 `REJECTED`，不因计时波动或小数变化重跑。
- 已生成的同 panel JSON 必须使用 `--candidate-json` 做零 API 的配对重放；不得为了重新输出
  W/A、Q/K/V 分解而再次调用候选 API。只有以下情况允许重跑同一阶段：源码/评测器/cache/
  device 实质变化、进程或环境明确失败、或用户明确要求复核；重跑原因必须写入日志。
- 评测阶段的失败要区分 `ERROR`（接口/环境失败，修复后才可重跑）、`REJECTED`（机制证据
  已足以否定）和 `TIMEOUT`（超时事实）。不把一次失败重跑产生的数字与原始 JSON 覆盖合并。
- 新机制的默认顺序是：接口/合法性 smoke → 与固定 parent JSON 的 effect-panel 配对 → 只有
  focus 方向、control、误差源和最坏 case 都可解释时才跑默认 168+120 panel。不得用 aggregate
  `linear_mean` 的微小变化代替配对证据，也不得因少数浅层收益掩盖 median 或深层回归。
- 完整评测使用同一 cache、协议、设备和命令，并保存 JSON 与 Markdown report；探索阶段不因
  缺少无关的附加报告而阻塞实现。
- 本地结果用于检查合法性、同一 Attention 下的回归、必要的 role 误差和同机成本；跨 Attention
  家族的本地排序不能代替官方排序。
- 失败结果必须如实记录为 `REJECTED` 或 `TIMEOUT`；官方通过但分数低于已知基线也属于
  `REJECTED`，不能因为时间通过就保留为父版本。
- 没有明显精度、复杂度或理论变化的实验不分配版本号，也不单独归档；不为满足形式流程而
  制造额外版本或测试。

### 2.4 目标

工作目标只有两个：

1. Linear 精度继续提升，围绕 `linear_mean=0.8` 做可达性验证；
2. 官方端到端时间严格小于 `300s`。

不增加与这两个目标无关的门禁、审批或人为阈值。**禁止过度工程化**：算法规划和测试只做
能改变决策的最小检查，不设置“先通过一长串门禁才能继续”的流程。必要的检查仅限于接口/格式
合法、结果可复现、明显回归和时间记录；发现问题直接记录并调整，不把门禁本身当成目标。

## 3. 归档与提交规则

### 3.1 版本目录

- `solutions/` 只保存 immutable `solution.py` 快照；版本号全局唯一，正式版本目录使用：

  `YYYYMMDD_vNNN_<description>_<outcome>`

- `<outcome>` 必须明确标记：`retained`、`rejected` 或 `timeout`。**凡未晋级为后续父版本的
  代码，目录名必须包含 `_rejected`**；包括官方通过但分数低于基线、只有本地提升、时间不满足、
  回归或实验无效的版本。官方结果尚未回传但实验有归因价值时可以先使用 `scoreNA_timeNA`，
  但只要判定不晋级就必须改为带 `_rejected` 的目录名；明确超时的版本使用 `_timeout`，没有
  归因价值的快照直接删除。
- 微参数 sweep 不逐项建立版本目录；一个算法族最多保留一个完整实现和一个代表性失败样例，
  其余只保留汇总 JSON/日志。
- 目录名和 `result.md` 的状态必须一致。官方通过但低于当前基线的版本仍使用 `_rejected`；
  `RETAINED` 只能用于已明确晋级或被明确保留为后续父版本的代码。

### 3.2 `result.md` 必填内容

每个保留的快照必须记录：

- `Status`：`RETAINED`、`REJECTED`、`TIMEOUT` 或 `ERROR`；
- 父版本、唯一算法变化和是否固定另一场景；
- 评测协议、模型/数据 revision、cache、设备和完整命令；
- Local Linear/Attention mean、逐 role 结果（如有）、六 API 时间和 wall time；机制实验还要
  记录 parent JSON、focus/control、正负 case、主要 W/A 或 Q/K/V 来源和最坏 case；
- 源码 SHA256；
- 官方分数、官方时间、官方状态；未知字段写 `unregistered`/`NA`，不能用本地值填充；
- 与父版本的差分和下一步决定。

### 3.3 原始证据与状态更新

- `artifacts/official_eval/*.json`、`logs/official_eval/*.md` 和 `logs/execution/*.md` 是审计
  证据；原始 JSON/report 不因后续官方结果而改写。官方回传使用独立 correction log、
  `result.md` 和状态文档补记。
- 用户确认官方分数或时间后，立即同步 `docs/current-solution-status.md`、`solutions/README.md`
  和对应执行日志；失败必须同时在目录名和 `result.md` 标注。
- 产生实质版本或状态更新后立即 `git add/commit/push`，提交前运行 `git diff --check`，提交后
  确认 `git status` 干净并记录 commit；不得声称“已提交/已归档/已推送”而未核验。
- 根 `solution.py` 只有在明确切换活动父版本时才改变；失败实验不能留在根文件污染下一轮。
- 正式版本归档前必须做一次“脱离仓库依赖检查”：将归档 `solution.py` 单独复制到临时目录，
  在没有兄弟源码和归档目录可解析的条件下导入六个 API。依赖检查失败的版本不得标记
  `RETAINED`，应标记为 `_rejected` 或删除；该检查只做一次，不设置额外工程化门禁。

## 4. 当前工作锚点

- 官方最高已知分数：用户确认的 **17816**，但其源码、版本号、官方时间和 Attention 配置
  尚未同步，不能伪造归档或时间结论。
- 已验证官方基线：v86，**16744 / 222.7s**；其 Attention 是 Linear 后续实验的冻结参照。
- v140：官方 **15838 / 207s**，时间通过但精度低于 v86，已标记 `REJECTED`。
- v147：官方 **16579 / 211s**，时间通过但精度低于 v86，已标记 `REJECTED`；其官方提交 SHA
  未确认。pre-A3 本地归因控制为 Linear `0.5073546371`、Attention `0.7196960689`、API
  `222.227s`，不能把这份本地源码 SHA 冒充官方提交 SHA。
- v152 的 fc CAT-off 配对结果为 mixed；v153/v154 的直接 decoupled activation/scale 路径明确
  回归，均已拒绝。L3-D0 teacher/oracle 已完成：规范结果为
  `margin_exists_but_not_compile_safe`，same-fold joint margin 虽为正，但 layer 3 / fold 128
  的 exact output margin 对 `fc_gate/fc_up` 分别为 `-0.094751/-0.112680`，不能编译成稳定
  student/v155。规范 teacher 约 `597.7s` 只算研究成本；日常迭代只能用 layer-3 batched
  Jacobi fast probe（约 `10.35s`）定位最坏层，不能把它当候选评分。
- L3 cross-fold feature/decision stability 快探针已完成但未找到可压缩的固定规则，L3 表示族
  关闭；L2 的首个 2×2 analytic pair-balance local-only probe 也已拒绝：fc focus 配对均值
  `-0.314079`（16/16 回归），说明朴素矩阵平衡破坏静态 Weight code，不再重复同类无约束
  变换。L5a quartile-interleave permutation 的原始 effect panel 虽为 mixed（5 个回归），
  稳定性门控版正式单文件 v155 的 Qwen default paired 仅 `+0.000116536`（4/0/164），严格
  GPT-2 配对为 `-0.000153`，所以 v155 只保留为 Qwen-local diagnostic control，不作为优化
  parent。L4-WD 的正式单文件 v156 已完成隔离依赖检查，Qwen effect `+0.000107624`、GPT-2
  `+0.000029454`，本地增益很小但按用户要求保留为待提交官方候选；官方回传前 root 不变，
  Attention 继续冻结 v86。
- 评测结果必须先看 `evaluation_scope`：只有同 cache 的 `default-panel` 可做本地 proxy 排名；
  `effect-panel`/`paired-json-replay` 只做父子诊断，`full-stress` 只做压力，`smoke-prefix`
  只做接口，GPT-2/hif4 与旧 `official-shape-v1` 只做跨结构/历史审计。任何 scope 都不是
  官方分数等价物，详见 `artifacts/official_eval/README.md`。
