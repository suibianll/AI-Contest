# HiF4 量化竞赛工程

> 数据快照日期：2026-08-31；当前事实以本文件、最新评测日志和 `solution.py` SHA 为准。

华为 2026 算法竞赛 NVFP4 → HiF4 赛道的开发工作区。根目录
`solution.py` 是唯一活跃、可提交的算法文件；历史候选保存在
`solutions/`，不会被运行时引用。

英文版：[README_EN.md](README_EN.md)

## 当前状态

- 官方评测集已更新为 **250 个 Linear case + 200 个 Attention case**；逐
  case 求和的分数和端到端时间都会高于旧口径，旧分数不能与新分数直接比较。
- 当前归档候选中，修订评测集下的合规官方冠军为 v066 / C66，
  `22557 / 217.2s`；此前 v051 / C47b 为 `22451 / 234s`，v031 / C39-FW
  与 v034 / C41b 均为 `21864`，时间分别为 `161.3s` 与 `159.4s`。
- 外部参考：[`youxilee/hif4`](https://github.com/youxilee/hif4) 当前公开代码据
  用户提供的同口径官方结果为 `24153 / 239s`；未导入本仓库。未修改的 v2.7
  源码在本地 CPU 代理上复测后，最高单模型是 Qwen native `369.527269`，同口径
  Qwen panel 为 `250.327102`（应作为本地最高比较基准）；五模型相加的
  `1085.743597` 仅是诊断量，不能排名或换算官方分数。CUDA 路径还存在外部代码
  的设备混用问题。完整表格见 [`当前主版本算法效果与评测状态`](docs/current-solution-status.md)。
- 历史 v024 得分为 `16043 / 173.8s`，但其 Linear 输出监督路径把输出信息
  用于激活侧选择；这类 `A@W -> Q(A)` 用法仍不合规，因此不作为后续合规父版本。
- 当前根 `solution.py` 为 v125 C1c structured rank-8 / max-blocks-8 + C1b structured
  gradient refresh（sweep2）精度 parent；在 v118 L6d 的最多 8 个 `64×64` kernel
  proposal 上批量化独立 row/block 的 15-level 候选评估。Qwen 全 24 层本地实测
  native `423.394380`、shaped panel `295.847849`、Linear mean `0.5097598050`、
  Attention mean `0.8420394885`、正式 API `2653.580314s`。v125 较 v124 panel
  `+0.027620`，但 runtime 无效；v106 仍是最近的 `<420s` 时间 parent。当前探索阶段
  只按精度排序，时间暂作记录；最终冻结时再压缩到 420s 内。逐项结果、归档实现
  审计和复现实验配置见 [`当前主版本算法效果与评测状态`](docs/current-solution-status.md)、
  [`算法全景`](docs/algorithm-inventory-and-directions.md)、
  [`归档实现审计`](docs/archive-implementation-audit.md) 与 [`solutions/README.md`](solutions/README.md)。
  L5d 外部组件审计与 L5e 可达性 checkpoint 已完成；L6a–L6e 已完成并归档；C1a 已完成，
  C1b 的 block-refresh screen（v120）被拒绝，两轮 refresh（v121）已通过，C1c rank-2/block-2
  screen 均被拒绝，rank-8/max-blocks-8（v125）已在 full-layer 取得当前最高精度；由于
  API 时间远超 420s，下一步转入唯一活跃计划的 C2 低成本跨模型 guardrail 和 C3 state/runtime
  压缩，不再扩大 block budget。C3 已具体化为部署权重因子 exact gate、selected-block
  稀疏增量 exact gate 和 structured gradient 增量刷新；C3 完成并归档后，才启动共享
  正交 butterfly/Givens frame + 冻结 activation state 后完整离散 JDRQ-weight 的新计划。
  36000/Linear 门槛推导见
  [`当前实验结果与可达性 checkpoint`](logs/execution/2026-08-31-current-results-target-feasibility.md)。
- 当前根源码 SHA256：
  `C9B419717E38BCEC69D907D1CAB6638409F1FA9A3072892DDE9494EF9DA3CC8E`（规范 LF）。
- L1 full-hierarchy Weight-LRH 已完成合成测试与五层×七 role screen，但 screen
  `both_player=0.523019429222563` 与 L0 逐条持平；候选 v105 已归档。L2 固定
  `α=0.25` 的 expansive-FFN CAT balance 已通过 full-layer，v106 成为时间 parent；
  L3 Gram-gated Global Activation-LRH 在只看精度的 full-layer 得到 v107；L4a final
  deployed-Gram row gate 得到 v109；L4b final-Gram GALS 得到 v110；L5a block-local
  permutation 得到 v111。L5b/v112、L5c/v113、L5d/v114 均已按 screen 归档拒绝，
  L5e 记录固定表示/接口的可达性证据；L6a rank-16、L6b 宽 rank-4 与 L6c 完整
  `G_64` hierarchy 与 L6d 结构化跨 block factor 均已完成，L6 计划已归档；C1c
  `max_blocks=8` 已在 v125 完成并停止，下一步执行唯一活跃计划的 C2/C3。
  证据见 [`v111 execution log`](logs/execution/2026-08-31-v111-l5a-joint-permutation-qwen-full.md)。
- 旧版本地评测器（单模型 dev 与 frozen holdout）曾因 calibration/test
  文本重叠不能可靠排序合规候选，相关代码（`real_data_eval.py`、
  `holdout_eval.py`、`cap_oracle.py`）已于 2026-08-28 移除；诊断结论见
  [C40 官方结果与评测器诊断](logs/candidates/C40-official-evaluator-diagnosis.md)，
  历史代码可从 git 历史恢复。
- 当前唯一活跃评测器为 `real_model_suite.py`：默认用 Qwen2.5-0.5B 作主模型，
  将冻结语料上的 Linear/Attention 平均 case gain 投影到官方的 250/200 面板；
  其他模型只作软 guardrail。`official_flow_total` 原始逐 case 求和仍保留作诊断，
  但不再按模型层数直接累加主排序。评测仍不能模拟官方隐藏数据分布，只用于
  A/B 排序。
- v107 官方 `Attention / wrong answer` 已按同一 Qwen cache、同一 NVFP4 codec 与同一
  API 和 v31、v51、归档外部 v002 做逐输出对照；四版本均无 state/shape/finite/五字段
  契约失败，v107 的 Attention MSE mean `0.00169248` 反而低于 v31/v51 的 `0.00382519`
  和外部 v002 的 `0.00529873`。详细数字见
  [`v107 Attention 输出差分日志`](logs/execution/2026-08-31-v107-v31-v51-external-attention-output-diff.md)。
  该外部数字仅代表本地归档 v002，不等同于最新 v2.7 源码；官方隐藏输入仍需同包复测。
- 官方复测候选已收敛为 **v100**：full Qwen panel `293.797301`、API `392.42s`，无
  v107 的完整 `deployment_gram` 状态；相对 v106 仅损失 `0.475332` panel，但增加
  `20.23s` 时间余量。v66/v100 同缓存 layer-0 复核均通过，v100 panel/API 为
  `336.037091 / 18.559s`，c66 为 `314.731294 / 25.196s`。裁决见
  [`v107 WA 安全提交候选选择`](logs/execution/2026-08-31-v107-wa-safe-submission-selection.md)。
  根 `solution.py` 仍为 v125 precision-only 研究版本，本次未替换 active 算法。

本地时间和本地分数仅用于候选比较，不冒充官方结果。任何官方结果都应与
实际提交 SHA、分数和时间一起归档。

## 数据与计划治理（必须遵守）

### 数据及时更新

README 顶部的“当前状态”、[`solutions/README.md`](solutions/README.md)、
[`docs/current-solution-status.md`](docs/current-solution-status.md) 和最新执行日志
共同组成当前事实快照。每次本地评测、官方回传或 active `solution.py` 变更，都必须在
同一提交中更新：

1. 数据日期、模型/数据 revision、完整命令和缓存模式；
2. Linear/Attention 分项、panel、API 时间和 source SHA256；
3. `solutions/README.md` 比较表、当前状态报告和对应 execution log；
4. 若是官方结果，追加官方提交 SHA、分数、时间和日期；若未知，保留 `NA`，不得用本地值代填。

若文档数字冲突，按“根 `solution.py` + 最新可复现评测 JSON/日志 →
`solutions/README.md` → 当前状态报告 → 其他研究文档”的顺序裁决；归档计划和旧日志
只用于历史追溯。每次数据更新还要刷新文档的 `更新日期/数据快照日期`，不能继续沿用旧快照描述。

### 计划写入与执行

仓库同时只能有一份活跃优化计划，位置是
[`docs/superpowers/plans/`](docs/superpowers/plans/)，当前文件见
[`2026-08-31-hif4-active-c1-structured-linear-plan.md`](docs/superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md)。
执行任何优化时**只参考这份 active 计划**、当前根代码、最新评测数据和官方规则；
`docs/superpowers/archive/plans/` 中的文件一律是只读历史，不得作为下一步指令。

计划生命周期规则：

1. 新建计划前先确认 `plans/` 除 README 外没有第二个 `.md`；需要换主线时，在同一提交中把旧计划移入 `archive/plans/`、创建新 active 计划并更新两个 README。
2. 每个步骤必须写清假设、代码入口、数据集/模型、验收指标、预计产物和失败处理；执行后立即填入实际结果、source SHA、日志链接和 `done/rejected/blocked` 状态。
3. 一次实验无论成功、失败、超时或未提交，都先按候选归档流程保存；没有完整源码、SHA 或配置的结果只能标记为不可复现。
4. 计划完成、被替换、明确停止或连续阻塞后，立即归档；不得在旧文件中继续追加新的“下一步”，也不得保留多份“current/active”文字。
5. 归档文件不改写历史结论；若发现实现 bug 或数据错误，新增审计说明或新 active 计划修复，并在索引中标注影响范围。

执行前后至少检查：

```powershell
Get-ChildItem docs\superpowers\plans -File -Filter *.md |
  Where-Object Name -ne README.md
git diff --check
```

第一条命令必须只返回一个 active 计划；如果返回 0 或多个，先整理计划目录，不能开始算法实验。

## 本地评测是否能反映官方方向

使用已冻结的五模型结果，新的 Qwen 主面板与官方锚点给出相同的相对顺序：

| 候选 | 官方分数 | Qwen panel（本地相对分） |
| --- | ---: | ---: |
| C39 | 21864 | 230.096230 |
| C41b | 21864 | 230.096230 |
| C47b | 22451 | 237.541351 |
| C66 | 22557 | 238.282409 |

官方与本地均为 `C39 = C41b < C47b < C66`；Qwen 主面板的 Spearman 为
`1.0000`，五模型 raw sum 为 `0.9487`。这只证明相对排序方向，不证明本地
分数不可以线性换算成官方分数。外部 `youxilee/hif4` 的最高同口径 Qwen panel 为
`250.327102`，当前根为 `295.847849`，领先 `45.520747`（`18.19%`）；外部
Qwen native `369.527269` 仍作为第二诊断线，五模型合计不作为基准。

## 修订版官方评测锚点（2026-08-29）

以下结果按新版 `250 Linear + 200 Attention` 样例统计；前三项为用户确认的本
地归档提交结果，最后一项是外部仓库参考，不属于本仓库提交：

| 方案 | 分数 | 时间 | 备注 |
| --- | ---: | ---: | --- |
| v031 / C39-FW | 21864 | 161.3s | 合规归档锚点 |
| v034 / C41b | 21864 | 159.4s | 合规归档锚点 |
| v051 / C47b | 22451 | 234s | 此前本地官方冠军 |
| v066 / C66 | **22557** | **217.2s** | 当前本地官方冠军 |
| `youxilee/hif4` | **24153** | **239s** | 外部官方参考；本地最高 Qwen native `369.527269`、panel `250.327102`；五模型 `1085.743597` 仅诊断 |

新版官方时间限制为 **7 分钟（420 秒）**。历史 `14613 / 159.2s`、
`14437 / 166.6s` 等数值属于旧评测集，仍保留作历史记录，不与上表直接混算。

## 官方硬约束

1. **离线校准可以使用 `A@W` 优化离线量化器，尤其是 `Q(W)`。** 但不得把
   `A@W`、其量化输出或输出残差用于拟合、选择或反推在线激活量化器 `Q(A)`，
   也不得将这类信息写入 `activation_state`。因此规则禁止的是输出监督驱动
   `Q(A)`，不是一律禁止离线权重量化目标中的 `A@W`。
2. 输出必须是合法 HiF4 五字段，API、state、shape、dtype 和设备必须符合要求。
3. 最终官方评测总时间必须严格小于 `420s`（7 分钟）。
4. 不使用 holdout 或官方分数反向调参。

除上述规则外，不设置固定的增益、coverage、beam、单组件非退化或中间时间门槛。
开发阶段允许完整扫描和超过 420 秒的诊断实验；发现精度信号后，再通过算法和实现
优化压入最终时间限制。

## 当前算法

当前根是重写后的 clean Gram-hierarchy + B1/B2 + L5a + L6a + L6b + L6c + L6d + C1a + C1b + C1c rank-8 版本；v086/C86 仍是不可变历史归档。
评测和优化优先级如下：

| 优先级 | 组件 | 当前机制 | 作用/状态 |
| --- | --- | --- | --- |
| 1 | Linear | BOAT：RMS 对角平衡 + 4/8/16/64 signed-Hadamard | 先压低两侧 operand-local 误差；不构造 Linear 输出 |
| 2 | Linear | Cross-fold Weight-HSDQ：`AᵀA` 二阶增量、15 levels、top-2 block、1 sweep | 只更新离线 `weight_params`；跨 fold 验证后才接纳 |
| 3 | Linear | Gram-hierarchy Activation-HSDQ：静态 `WᵀW`、offset/hierarchy 选择、最多 128 block、2 sweeps | 在线 state 仅含合法静态统计；v106 基线 Linear mean `0.503459` |
| 4 | Linear | Expansive-FFN CAT balance：`rows > channels`、固定 α=0.25 | v106 仅改善 fc_gate；不增加 state 字段 |
| 5 | Linear | Global Activation-LRH：窄输入 rank-16、宽输入 rank-4 off-block proposal，逐行 exact deployed-Gram gate | v125 Linear mean `0.509760`；L6a–L6d 均通过 |
| 6 | Linear | L6d + C1a + C1b/C1c structured factor：8 个 `64×64` kernel，批量 proposal，block refresh×2，完整 `G_q` gate | v125 panel `295.847849`；较 v124 `+0.027620`，API `2653.58s`（runtime invalid） |
| 7 | Attention | reciprocal RMS、K-centering、GQA 对齐、GQRB、PAWV diag-only | 使用真实 non-causal Attention 输出排序；当前 mean `0.842039` |
| 8 | 下一步 | L0–L5e、L6a–L6e、C1a、C1b、C1c（v125 `max_blocks=8`）已完成 → **C2 跨模型 guardrail → C3 state/time 压缩** | 只参考唯一活跃计划；Attention PAWV 独立延后 |

优化决策只看同一冻结缓存上的相对增量：Qwen `primary_panel_score_total` 是主
指标，其他模型用于发现结构性回退。不得用官方分数反向调参，也不设置固定的
增益、coverage 或“每个模型必须正向”门槛；只有合规、合法性、非 finite 和
主模型精度方向是当前硬条件。accuracy-first 阶段只记录时间、不因超过 420 秒拒绝；
进入最终提交压缩阶段后，`<420s` 才恢复为硬约束。

### Linear

1. 按 NVFP4 scale 和 E2M1 载荷重建浮点参考。
2. BOAT 用激活/权重 RMS 搜索对角平衡和 4/8/16/64 维 signed-Hadamard；
   候选只使用两侧 operand-local 误差，不构造 Linear 输出。
3. Cross-fold Weight-HSDQ 对宽度满足条件的权重使用 `AᵀA` block Hessian，
   在 15 个 signed levels 上做 top-2 block、1 轮坐标增量；fold 间不泛化的候选
   不进入最终参数。
4. 在线 Activation-HSDQ 使用静态变换后权重 `WᵀW` 的 Gram block 选择
   E6M2 offset/hierarchy，最多处理 128 个 block、每块 2 轮；state 不含输出监督
   或测试样本。

### Attention

当前搜索 reciprocal RMS 平衡、K-centering、GQA 对齐和 16/32/64 维共享
signed-Hadamard。便宜代理扫描后只对前 4 个候选执行完整部署量化和真实
Attention 输出复评；Q/K state 只保存 CPU 静态 Gram、重要性、整数 block/seed
和符号，V 保持独立合法 HiF4 编码。旧 C86 的实验开关、Segment-CVaR 和无收益
V importance 分支不再位于根文件。

## 开发原则

- 真实部署路径的配对分数是候选裁判；oracle 和局部损失只用于诊断、排序和解释。
- 不用任意百分比阈值在实现前否决候选，除非存在严格数学不可能证明。
- 允许同时删除冗余计算并重新分配预算；完成后再做消融归因。
- 保留完整的精度—时间 Pareto 曲线，不因单次负结果宣称整个赛事空间不可达。
- 小而稳定的正增量可以累计，不要求每个候选达到固定百分点。
- 失败实验照常归档，但失败结论只约束被实际测试的实现和配置。

## 工程结构

```text
solution.py                         唯一活跃提交文件
evaluator/real_model_suite.py       多模型真实语料评测、前向缓存与 Qwen 主面板排序
evaluator/reference_hif4.py         独立官方评分协议、标准基线与合法性校验
evaluator/nvfp4_sim.py              NVFP4 编码模拟
evaluator/real_data_eval.py         共享的候选加载/计时/评分工具与旧版单模型评测入口
evaluator/synthetic_attention_eval.py
                                    576-case Attention 安全矩阵（性质诊断，不参与排名）
evaluator/linear_compliance_guard.py
                                    Linear 合规静态/运行时检查
evaluator/linear_error_decomposition.py
                                    Linear 误差归因诊断
tests/                              发布、格式、合规和算法测试
solutions/                          不可变候选归档
artifacts/real_model_suite/         评测 JSON 结果；cache/ 为本地模型快照，不入库
logs/evaluations/                   评测运行报告（每次运行显式指定路径）
logs/candidates/                    候选官方结果与诊断报告
logs/execution/                     执行日志与校准记录
docs/current-solution-status.md    当前根算法、全层实测和分数归因快照
docs/real-model-evaluator.md        评估器使用说明
docs/research/                      文献调研
docs/superpowers/plans/             唯一活跃实施计划
docs/superpowers/specs/             设计与规范
docs/superpowers/archive/plans/     已失效优化计划，仅供历史查阅
```

## 运行评测

使用工程虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

### 推荐：Qwen 主评测（已有缓存）

这是日常比较候选的最短命令。它只使用 Qwen2.5-0.5B，主分固定投影为
250 Linear + 200 Attention；`--cache-mode read` 要求对应快照已经存在。
没有 CUDA 时使用下面的 CPU 命令，结果仍可用于相对排序：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --candidates c39 c41b c47b c66 `
  --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\qwen-panel-YYYYMMDD.json `
  --report logs\evaluations\qwen-panel-YYYYMMDD.md
```

有可用 CUDA 时，将上面两项改为 `--device cuda --algorithm-device cuda`。
若缓存不存在，先执行下方“采集缓存”命令；`read` 模式不会偷偷下载模型或
改用其他配置。

结果字段按下面方式读取：

| 字段 | 用途 |
| --- | --- |
| `results[*].panel_score.total` | 单模型固定面板分；Qwen 主模型使用它参与排序 |
| `official_ranking_audit.primary_panel_score_total` | 候选主排序特征 |
| `official_ranking_audit.guardrail_panel_mean_total` | 其他模型的软稳定性诊断 |
| `results[*].official_flow_score.total` | 旧版 native 逐 case 和，仅用于回溯 |
| `timing.official_api_total_seconds` | 单个模型六 API 代理耗时；主模型必须 `<420s` |

带 `--solution` 的命令在主模型非法、非 finite 或超时会返回退出码 `2`，但仍会
写出 JSON 和 Markdown，便于定位问题；只做锚点比较时不带 `--solution`。

CPU 全量 Qwen 评测可能接近或超过 420 秒，适合诊断；正式时间判断应使用 CUDA
或先用 `--layers 1 --calib 1 --test 1` 做接口冒烟，再运行完整配置。

先做不加载模型的环境检查：

```powershell
.\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
.\.venv\Scripts\python evaluator\real_model_suite.py --help
```

单模型快速评测（gpt2-small，优先读缓存）：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model gpt2-small `
  --device cpu --algorithm-device cpu --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-YYYYMMDD.json `
  --report logs\evaluations\quick-YYYYMMDD.md
```

GQA 示例（Qwen2.5-0.5B 自带 14Q/2KV 与 RoPE 适配）：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-qwen-YYYYMMDD.json `
  --report logs\evaluations\quick-qwen-YYYYMMDD.md
```

Attention 合成矩阵：

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

当前 clean 根版本的发布回归（格式、合规、reference codec、真实模型套件）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
  tests/test_real_model_suite.py --basetemp=.tmp_pytest\clean-root
```

当前环境结果为 **34 passed**。`test_jdrq.py`、`test_weight_cross64.py`、
`test_weight_full64.py` 和 `test_release_candidate.py` 中仍有针对已删除 C86/JDRQ
私有 helper、实验开关或旧 state schema 的历史断言；它们不再是 clean 根版本的发布
门禁，不能用整库 `pytest -q` 的失败数评价当前算法效果。若重启这些方向，先按
当前 API/state 重新编写测试，再把测试加入发布命令。

如需单独检查真实模型评测器（使用仓库内被忽略的临时目录）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_real_model_suite.py --basetemp=.tmp_pytest\readme-verify
```

### 候选测试顺序与结果保存

每次实验只修改根目录 `solution.py`。先完成语法、合规和单模型真实路径测试，再进行多模型比较；不要直接修改 `solutions/` 中的历史源码。

1. **提交前的快速检查**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
   .\.venv\Scripts\python.exe -m pytest -q `
     tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
     tests/test_real_model_suite.py --basetemp=.tmp_pytest\clean-root
   ```

2. **冒烟测试当前根 `solution.py`**

   下面命令显式只跑 `gpt2-small`，用于快速确认输出格式、Linear、Attention
   和本地计时；它不是官方方向的主排序。需要比较候选时，请使用上面的 Qwen
   主评测命令，或把本命令的模型和 `--primary-model` 一并改成
   `qwen2.5-0.5b`：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --models gpt2-small --candidates c39 `
     --solution solution.py --candidate-name active `
     --panel-profile qwen-official --primary-model gpt2-small `
     --device cpu --algorithm-device cpu --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   完整候选比较的默认主排序使用 Qwen shaped panel：
   `panel_score.total = 250 * Linear_mean + 200 * Attention_mean`。推荐始终配对
   `--candidates c39 c41b c47b c66`；本地分数只用于 A/B 排序，不填入 Official
   Score；同时记录完整命令、源 case 数量、目标 250/200 面板、API 总时间和
   source SHA256。`official_flow_total` 仍写入 JSON，便于与旧报告回溯。

3. **一次性采集多模型真实前向数据**

   `real_model_suite.py` 默认覆盖 GPT-2 small/medium、OPT-125M、Pythia-160M、Qwen2.5-0.5B，并对已登记的当前修订面板锚点 C39/C41b/C47b/C66 进行比较。Qwen 是主模型，其余模型用于软 guardrail；先采集模型数据，避免每个候选重复执行模型前向：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --device cpu --algorithm-device cpu --cache-mode write --capture-only `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\cache-capture-YYYYMMDD.json `
     --report logs\evaluations\cache-capture-YYYYMMDD.md
   ```

   命令中的 `YYYYMMDD` 应替换为实际运行日期。机器有 CUDA 时可将两项 device
   同时改为 `cuda` 以缩短采集时间。快照保存在
   `artifacts/real_model_suite/cache/`，不提交到 Git；它包含真实模型权重、
   Linear 输入、真实 Q/K/V、token ids、模型/data revision 和窗口校验信息。

4. **只从缓存评测**

   缓存生成后，候选测试不再加载 tokenizer/model、不执行模型 forward，也不访问网络：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --candidates c39 c41b c47b c66 --solution solution.py --candidate-name active `
     --panel-profile qwen-official --primary-model qwen2.5-0.5b `
     --device cpu --algorithm-device cpu --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   `read` 模式遇到缺失、版本不一致、配置不一致、窗口泄漏或张量形状错误会直接失败，不会偷偷重新加载模型。`auto` 适合日常使用：有效缓存直接读取，缺失或过期时重新采集；`write` 强制刷新；`off` 不保存缓存。更换 seq、calib、test、层数、模型或固定数据集 revision 后，必须生成对应的新缓存。

5. **检查本地排列是否复现官方排列**

   评测器只用官方锚点做 Spearman 和 pairwise 排序审计，不拟合官方绝对分数。默认候选晋级比较同一冻结语料上的 Qwen `primary_panel_score_total`；其他模型的 panel 均值只用于发现结构性回退，不能覆盖主排序。`official_flow_total` 是兼容诊断字段。推荐用 `--candidates c39 c41b c47b c66` 与当前修订官方锚点配对运行。

   官方流程代理分为：

   ```text
   score(case) = (MSE_STD - MSE_PLAYER) / MSE_STD
   native official_flow_total = sum(all native Linear case scores) + sum(all native Attention case scores)
   qwen panel_score.total = 250 * mean(Linear case scores) + 200 * mean(Attention case scores)
   ```

   标准 NVFP4/HiF4 反量化、HiF4 参数校验和 state 校验全部由评测器独立完成；候选只需实现赛事规定的六个 API。评分器中的 `A@W` 只在候选返回量化结果后用于计算参考误差，不会作为输出传回候选；候选在离线 `hif4_calibration_and_quantize_weight` 中可以自行使用 `A@W` 优化 `Q(W)`，但不能让它进入 `activation_state` 或在线 `Q(A)` 选择。

   赛事说明未附官方“标准 HiF4 量化函数”源码；当前独立标准 codec 使用历史已审计实现并在每份报告记录 SHA256。取得官方函数后必须逐位替换并升级评分协议版本。

6. **确认时间约束**

   主模型代理的一次完整六 API 评测，其 `official_api_total_seconds` 必须严格小于官方硬限制 `420s`；等于 420 秒也判失败。多模型套件的时间只用于检查各代理，不把多个代理的时间相加冒充一次官方提交；软 guardrail 缺失不会否决 Qwen 主排序。缓存读取只省去模型前向时间，不能掩盖候选算法自身的超时；最终仍需以官方端到端评测确认。

### 候选归档步骤

一次实验无论成功、失败、未提交或官方超时都要归档，不能只保留“提升”的版本。归档前先固定根 `solution.py` 的字节和测试结果：

1. 分配下一个版本号，目录格式为 `solutions/YYYYMMDD_vNNN_topic_scoreSCORE_timeTIME/`。不知道官方结果时使用 `scoreNA_timeNA`；不要把本地分数或本地时间写进 Official Score/Time，也不要事后覆盖原始记录。
2. 将根文件复制为归档快照，根文件继续作为唯一活跃提交文件：

   ```powershell
   New-Item -ItemType Directory -Path solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA
   Copy-Item -LiteralPath solution.py `
     -Destination solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   Get-FileHash -Algorithm SHA256 solution.py
   Get-FileHash -Algorithm SHA256 solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   ```

   两个 SHA256 必须完全相同；归档后的 `solution.py` 不再修改。
3. 在同一目录创建 `result.md`，至少记录：日期、版本/父版本、唯一算法变化、假设、完整测试命令和配置、各 Linear/Attention/时间结果、缓存文件名与 dataset/model revision、active source SHA256、官方分数/时间、delta、状态、结论和下一步。缓存未入库时，`result.md` 还要注明“缓存需按 README 重新采集”。

   推荐使用以下最小模板，并把 `NA` 保留为未知值：

   ```markdown
   # vNNN — topic

   - Date: YYYY-MM-DD
   - Parent: vNNN / commit
   - Change: one primary algorithm change
   - Hypothesis: why this change may improve accuracy
   - Test command: `完整命令`
   - Test config: model/data/cache/mode/layers/algorithm-device
   - Local official-flow Linear sum / cases: ...
   - Local official-flow Attention sum / cases: ...
   - Local official-flow total and paired ordering: ...
   - Local official API total runtime: ...
   - Cache: filename, schema, dataset revision, model revision
   - Source SHA256: `...`
   - Official score: NA
   - Official runtime: NA
   - Status: `local-rejected` / `local-accepted` / `official-compliant-champion`
   - Conclusion: evidence-based decision
   - Next direction: next falsifiable experiment
   ```

4. 更新 `solutions/README.md` 的比较表和必要的执行日志；官方结果返回后只追加官方 SHA、分数、时间和日期，不覆盖已有本地证据。官方提交文件必须与归档 SHA256 一致。
5. 检查归档和测试后提交：

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   .\.venv\Scripts\python.exe -m pytest -q `
     tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
     tests/test_real_model_suite.py --basetemp=.tmp_pytest\archive-check
   git add solution.py solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py `
     solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\result.md solutions\README.md `
     logs\execution\YYYYMMDD-experiment.md
   git commit -m "archive vNNN candidate"
   git push origin master
   ```

   若本次只更新评测器或文档，也要在提交说明中明确“不改变 active `solution.py`”。

## 记录位置

- 当前优化事实以根 `solution.py`、最新执行日志和可复现评测输出为准。
- 历史版本及其结论见 [solutions/README.md](solutions/README.md)。
- 最新执行记录见
  [2026-08-26-optimization-execution-log.md](logs/execution/2026-08-26-optimization-execution-log.md)。
- 候选归档流程见
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/archive/plans/2026-08-26-solution-archive-workflow.md)。
- 归档实现问题与不可复现候选见
  [archive-implementation-audit.md](docs/archive-implementation-audit.md)。
- 多模型真实语料、缓存模式和合规边界见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 官方流程逐 case 求和、独立 codec/校验和排序审计见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 旧优化计划已移入 `docs/superpowers/archive/plans/`，不再作为后续执行依据。
