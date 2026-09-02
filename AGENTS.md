# HiF4 竞赛工作记忆

> 最后整理：2026-09-02。这里只保留当前可执行规则和不可误读的状态；旧实验细节、失败版本
> 和历史分数只在 `docs/current-solution-status.md`、`solutions/README.md`、`artifacts/`、
> `logs/` 中保留，不复制到本文件。读历史文档前先看
> [`过期信息清单`](docs/stale-information-inventory-2026-09-02.md)。

## 1. 当前状态

- v159 原始 SHA `0508045A...4242` 的官方分数为 **17532**、时间未知；当前根目录和 v159 归档
  SHA 为 `13C9CF0B...5EC79`，只增加数学等价的 GPU device 修复与中间量复用，尚未官方复测。v158
  **16861 / 223s** 仍是时间与源码均完整的安全父版本，其 Attention Matrix-Smooth 继续冻结。

- 用户确认的官方最高分是 **17816**，但源码、版本号、官方时间和 Attention 配置尚未同步；
  它只能作为外部官方锚点，不能伪造归档或替代本地实验结果。

- 根目录 `solution.py` 与现有 v159 归档同步；设备修复和同算法复杂度优化直接更新该归档，
  不创建新版本。必须同时保留官方计分 SHA 与当前归档 SHA，不能把未复测 SHA 写成官方结果。

- 本地 proxy 只用于同机机制诊断和时间记录，不能换算官方分数或官方 `<300s`；已知历史中
  存在本地排序与官方排序反转，任何本地正向都必须等待官方回传确认。

- GPU 修复和 CUDA Linear-only compact/default 已完成；transformed samples 与 Weight Gram
  等价复用也已完成。下一步先分解校准热点，再做单项复杂度消融；完成前不增加新算法。
  17816 源码无法提供，不再作为等待项。

## 2. 提交代码约束

- 正式提交是根目录 [`solution.py`](solution.py)，必须单文件、自包含，只提供评测器要求的六个
  API：
  `hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation`、
  `hif4_calibration_attention`、`hif4_dynamic_quantize_q`、
  `hif4_dynamic_quantize_k`、`hif4_dynamic_quantize_v`。

- 正式代码不得通过 `importlib`、相对/绝对路径、归档目录或其他 Python 文件加载实现；脱离
  仓库仍必须能导入六个 API。

- 所有编码器、解码器、E6M2 scale、层级 scale/lv2/lv3、mantissa/sign 和状态逻辑必须在同一
  提交模块内，并通过 `evaluator/reference_hif4.py` 的合法状态检查。

- 在线动态 API 只执行校准阶段编译的规则；禁止把校准搜索、完整矩阵求逆或未限制的 Python
  候选循环带入在线路径。

- Linear 研究目标是实际输出误差
  `XW^T - Q(XR) Q(WR^{-T})^T`，变换必须保持连续域乘积不变；Hessian/Gram 必须在最终
  变换和部署权重坐标系中计算。

- Attention 与 Linear 分开改、分开归因。Linear 实验冻结 v158 Attention；Attention 实验
  冻结 Linear。只有明确的端到端审计才同时运行两侧。

### 2.1 编码原则

- 默认只处理用户明确提出的问题，先给结论并保持简短；不得自行扩展为额外评测、实现、文档整理
  或长篇分析。只有完成当前问题确实需要时才运行工具或展开细节，用户要求深入时再补充。

- 不追求过度防御：只写能解决当前问题的最小代码，不做猜测性的扩展设计。

- 只修改必须改的地方；不顺手改动与任务无关的代码、格式或注释。

- 只清理自己产生的问题；不重构或清理他人遗留、无关的问题。

- 禁止为不可能发生的场景添加错误处理、回退、空值检查或校验；只信任框架保证，仅在系统
  边界（用户输入、外部 API）进行必要校验。

- 禁止为一次性操作创建辅助函数、工具类或抽象；三行相似代码优于提前抽象，不为假想的未来
  需求设计参数、标志或兼容层。

## 3. 唯一评测口径

- 唯一本地主评测器是 [`evaluator/official_eval.py`](evaluator/official_eval.py)，协议是
  `proxy-v2`。`official-shape-v1`、GPT-2 和外部 hif4 只作历史/跨结构诊断，不能与当前结果
  混排。

- 本地固定结构假设：Qwen2.5-0.5B、24 层、WikiText-2 raw v1、Attention calibration
  lengths `[10,128,512,1024,1024]`，以及独立的 HiF4 validation。说明书没有公开这些隐藏
  结构，因此它们不是官方模型证据。

- 默认 panel 是 168 Linear（24 层 × 7 role）+ 120 Attention（24 层 × 5 长度）。
  `--full-cases` 的 2016 + 288 只作 stress；`--linear-cases/--attention-cases` 是顺序前缀
  smoke，不能用来判断泛化或晋级。

- `--compact-panel` 是低成本机制筛选：Linear 选 layer `0/8/15/23`、7 role、两组
  validation/test 同长度 holdout，共 56 cases，只建立 28 个 Weight state；Attention
  compact 只保留四个深度/长度哨兵。它只做父子机制和跨 holdout 泛化诊断，不能冒充 default
  panel 或官方调用图。

- 单侧场景必须隔离：`--linear-only` 不调用 Attention API，`--attention-only` 不调用 Linear
  API；本侧校准仍按共享 state 调用图执行，不按 case 制造 oracle。

### 3.1 Local proxy 的定义

每个 case 的本地分数是：

```text
gain = (MSE_STD - MSE_PLAYER) / MSE_STD
     = 1 - MSE_PLAYER / MSE_STD
```

`STD` 是标准 HiF4 对同一 NVFP4 解码输入的输出，`PLAYER` 是候选 API 的输出；它不是模型
准确率、不是官方总分，也不是官方时间。`overall_mean` 是实际 case 的等权平均，不拟合
Linear/Attention 权重。只有同一 `proxy-v2` cache、同一 panel、同一 device 的
`default-panel` 才能做本地 proxy 横向比较。

### 3.2 防止过拟合（强制）

- 本地 proxy 只用于否定机制、定位误差和比较同机成本，不得凭本地均值正向直接晋级；官方结果
  只验证预先声明的单一假设，失败后不得围绕 threshold、seed、alpha、offset 或候选数量做邻域调参。
- 校准、候选选择和验证必须分离。A@W/GPTQ 的参数只用 calibration folds 学习，晋级读取独立
  holdout；不得用同一 fold 同时选规则和证明收益。多折选择使用 median、worst-fold 或固定 robust
  聚合，禁止只取第一折或最好一折。
- 每个版本只改变一个可解释机制，候选数量固定且与数据结果无关。优先使用低自由度解析结构、
  block-Schur/块对角/低秩补偿和预先固定的正则；不得通过扩大 permutation、Hadamard seed、搜索
  网格或多机制叠加换取本地分数。
- Linear 必须评估最终部署目标 `Q(A)Q(W)^T`，并在最终变换坐标系计算 Hessian/Gram；operand MSE、
  对角 importance 和 aggregate mean 只能用于诊断，不能代替输出误差与跨折证据。
- 晋级至少同时检查 focus 的 median、worst-quartile、负 case、跨 holdout 同号率和未修改 control。
  收益若集中在少数 layer/role/fold、依赖单一模型形状，或 control 发生变化，按过拟合处理。
- 官方 mini 用例只做接口、合法性和真实形状复杂度 smoke，不用于选算法或参数；Qwen/GPT-2 等
  本地结构只作机制压力测试。发生本地与官方排序反转后，立即停止用该 proxy 为同一路线晋级。

## 4. NVFP4 输入缓存

- `--nvfp4-cache-mode auto`（默认）按 scenario/panel/case profile 持久化已编码的 NVFP4
  carrier/scale；缓存只含 evaluator 输入，不含候选 state 或候选输出。

- profile、协议、codec/mode、dense source identity、数据 hashes 或 panel 不一致时，`read`
  拒绝命中，`auto` 重建。`write` 强制重建，`off` 禁用 NVFP4 持久化。

- `--cache-mode auto` 先读已有 dense cache，只有 dense cache 不存在时才重新做模型前向；
  `--cache-mode read` 只读，不允许隐式重新捕获。

- cache 命中只减少输入准备/量化时间，不改变候选 API 数量、输出误差或本地分数。缓存文件是
  本机生成的 ignored artifact，不作为正式源码证据。

## 5. 当前评测步骤（固定）

1. **启动读取**：先读本文件，再读 `docs/superpowers/plans/README.md`、唯一活动计划、
   `docs/current-solution-status.md`、`solutions/README.md`、目标父版本和
   `evaluator/official_eval.py`。历史文档先过 stale inventory。
2. **固定父版本**：父版本只运行一次并保存 immutable JSON/report；后续候选使用同一 cache、
   panel、device 和 evaluator，不重复运行父版本。
3. **接口 smoke**：选定一个目标场景，运行目标侧最小 smoke，检查六 API/状态合法性和
   `--nvfp4-cache-mode auto` 是否可用；smoke 只判接口，不判效果。
4. **compact 配对**：

   - Linear：`--linear-only --compact-panel`；

   - Attention：`--attention-only --compact-panel`。
     候选使用 `--baseline-json`，先看 focus 的 mean/median signed delta、正负 case、未修改
     control、W/A 或 Q/K/V 来源、最坏 layer/role/shape/split/length 和 API 时间。必须精确匹配
     `(layer, role, test_window, split, length)`、`mse_standard`、`reference_energy`；已有同 panel
     JSON 用 `--candidate-json` 零 API 重放。
5. **泛化判断**：Linear 至少检查 median、q25/q75、worst-quartile、negative cases、
   validation/test 同号率和 interaction；Attention 至少检查 Q/K/V、QK/QKV interaction、
   logits/probability 误差和最坏长度/层。任何一个方向不能用 aggregate mean 单独晋级。
6. **单侧 default audit**：compact 方向、control、尾部和复杂度均可解释后，才运行目标侧
   default panel（Linear 168 或 Attention 120）。旧 `--effect-panel` 只在需要“完整校准图 +
   缩减动态 case”的专项审计时使用，不是默认必经步骤。
7. **完整端到端审计**：只有明确需要检查集成调用图时，才省略 `--linear-only/--attention-only`
   跑完整 168 + 120 panel，六 API 全部执行；`--full-cases` 仍只作压力测试。完整测试必须
   保存 JSON 和 Markdown report，并把 local proxy、API total、wall time、official 状态分开写。
8. **决策与归档**：接口/环境失败记 `ERROR`；机制证据否定记 `REJECTED`；官方明确超时记
   `TIMEOUT`；官方未知写 `unregistered/NA`，不能用本地秒数填充。没有实质算法/复杂度变化的
   运行不分配版本号。正式版本归档前只做一次脱离仓库单文件导入检查。

推荐命令模板：

```powershell
# Linear compact / cached input（必须使用 CUDA venv；系统 Python 是 CPU-only）
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Attention compact / cached input
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Complete default-panel integration audit
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda
```

## 6. 证据、比较和 Git

- 原始 `artifacts/official_eval/*.json`、`logs/official_eval/*.md`、`logs/execution/*.md` 不覆盖；
  修正使用独立日志和状态更新。结果先看 `evaluation_scope`：compact/effect/replay/smoke/stress
  都不是官方分数等价物。

- 官方结果优先级最高，其次是活动计划已确认事实，再次是归档 result/log 和本地 JSON/report；
  未验证推测不得写成结论。当前官方事实集中维护在
  [`docs/current-solution-status.md`](docs/current-solution-status.md) 和
  [`solutions/README.md`](solutions/README.md)。

- 每次实质代码或状态更新后运行 `git diff --check`，提交、push，并核验 `git status`；不要把
  ignored 的大 cache、`.codegraph/` 或临时目录加入提交。

<!-- End of current memory. Historical details stay in the linked evidence files. -->
