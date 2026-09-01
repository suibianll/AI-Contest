# 当前状态：官方最高分更新为 17816，Linear 新框架成立

更新：2026-09-01。

## 0. 最新官方进展

用户确认：一套新的 Linear 算法已经在官方评测上产生明确提升，当前最高总分达到
**17816**，比旧官方最高 v86 的 16744 高 **1072 分**。

用户同时确认 v147 的官方结果为 **16579 / 211s**。它通过 `<300s` 时间限制，但比 v86
低 165 分，因此按归档规则标记为 `REJECTED`，不作为后续父版本。v147 目录曾被原地替换，
当前无法确认官方提交对应的准确源码 SHA；官方结果只绑定版本号，不伪造 SHA 归因。

新算法不是 v138–v145 的局部调参延续，而是完整的：

- SmoothQuant + Permutation + block Hadamard 等价变换搜索；
- 变换后完整协方差驱动的 Weight GPTQ；
- 部署权重输出 Gram 驱动的 Activation GPTQ；
- quadratic AdaRound、E6M2 offset、data-driven refinement 和 edge extension；
- proxy 与 e2e 混合选择，窄层执行联合变换搜索。

新提交的版本号、源码 SHA、官方运行时间以及对应 Attention 配置尚未提供，因此当前只把
**17816 记为用户确认的官方精度锚点**，不伪造时间结论，也暂不建立源码归档。完整理论分析和
后续执行顺序见
[`活动计划`](superpowers/plans/2026-09-01-hif4-hierarchy-encoder-and-analytic-attention-plan.md)。

## 1. 版本结论

- **旧仓库内官方基线：v86，16744 分 / 222.7s。** 新的用户确认最高分为 17816，但源码与
  官方时间尚未同步到仓库。
- 根目录 [`solution.py`](../solution.py) 当前是 v140 Linear + v86 Attention + 一轮额外 A3
  的单文件组合，SHA256 `44E37709A02B962CDAEDFC57E3AD999B2C9A2C0606B8B9DB7E4E81DC4DC92672`。
  最近完整同行为结果为 Linear `0.5100503237`、Attention `0.7196960689`、API
  `300.3507s`；官方分数和时间未登记。它不是 17816 源码，也不能沿用 v140 的官方
  `15838 / 207s` 结论。
- v138/v139 虽在官方 `<300s` 内通过，但只有 `15715/15716`，比 v86 低约 1029 分；
  v138–v145 这条“压缩 Attention 后继续叠 Linear 局部模块”的路线已经失败并关闭。
- 下一阶段不从 v140/A3 继续调参，也不等待 17816 源码才开始。先恢复可信 pre-A3 对照并完成
  role 归因；第一正式机制是 Activation-only Decoupled HiF4 Encoder，Attention 固定 v86。
  17816 源码若后续到位，再作为独立官方快照归档和对照，不倒推或覆盖当前证据。

## 2. 评测口径

本地统一使用 [`evaluator/official_eval.py`](../evaluator/official_eval.py) 的
`proxy-v2`：Qwen2.5-0.5B、24 层、默认枚举全部已捕获的真实 W/A（Linear 为每层每 role 每个
holdout 窗口，Attention 为每层每个 holdout 窗口；窗口本身是固定、可复现的 holdout capture）、Attention calibration 长度
`[10,128,512,1024,1024]`，validation/test 交替 holdout。主字段是
`linear_mean`、`attention_mean`、`overall_mean`、六 API `api_total_seconds` 和
`wall_seconds`。旧 `official-shape-v1` 只保留为不可迁移的历史证据。

该协议使用固定公开模型、跨文档 WikiText holdout 和全量 case 枚举，而官方使用隐藏数据与未公开
的新权重。因此它只用于算法诊断和同机耗时记录，不能继续作为官方排序器，也不能把本地时间换算为
鲲鹏时间。校准状态按官方调用图共享：168 个 layer/role Weight state、24 个 Attention state；
`trend_diagnostics` 仅对同一官方权重 cohort 做顺序一致性检查，发现反转时必须停止用本地分数晋级。

E0 修复记录：旧 v1 的 E4M3 scale 忽略 subnormal、窗口集中在少数文档，且曾误把 calibration
放进每个 case。proxy-v2 已修正这三点；共享校准版本将在下一次开发复测中核对
`168/25/24/20/20/20` 的 API 调用图。此前 per-case 校准复测中，v138 相对 v86 的本地顺序仍为
反转；因此该问题被记录为 `inversion_detected`，不得再把 proxy 分数当晋级依据；现已取消比例与
case 上限，下一次按全量真实 W/A 重新评测，再判断是否仍需 stress panel。

## 3. 历史 v1 结果表（不可与 proxy-v2 混用）

下表保留旧 `official-shape-v1` 的同机数字，仅用于审计此前的失真；proxy-v2 全真实 W/A
复测不覆盖这些历史值。

| 版本 | Linear mean | Attention mean | API(s) | Wall(s) | 官方结果 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| v84 | 0.406668 | 0.718107 | 279.191 | 300.848 | 16517 / 252.563s | 官方通过 |
| **v86** | **0.406668** | **0.719696** | **299.302** | 321.996 | **16744 / 222.7s** | **官方基线** |
| v128 | 0.465655 | 0.837789 | 310.732 | 332.557 | timeout | 失败 |
| v129 | 0.465655 | 0.836579 | 248.363 | 270.606 | timeout | 失败 |
| v130 | 0.471837 | 0.836579 | 295.437 | 317.607 | timeout | 失败 |
| v131 | 0.473131 | 0.836579 | 294.835 | 317.708 | timeout | 失败 |
| v134 | 0.507320 | 0.834256 | 289.042/289.832 | 312.315/313.181 | 未提交 | 本地研究父版本 |
| v135 | 0.500812 | 0.834256 | 290.823 | 313.365 | 未提交 | REJECTED |
| v136 | 0.500132 | 0.834256 | 287.816 | 310.472 | 未提交 | REJECTED |
| v137 | 0.507163 | 0.834256 | 296.755 | 319.306 | 未提交 | REJECTED |
| v138 | 0.507320 | 0.715942 | 187.935–192.996 | 210.855–216.324 | **15715 / 208s** | 官方通过但路线失败 |
| v139 | 0.507278 | 0.715942 | 193.389 | 217.196 | **15716 / 202s** | 官方通过但路线失败 |
| v140 | 0.507355 | 0.715942 | 205.365 | 229.337 | **15838 / 207s** | **REJECTED；官方通过但低于 v86** |
| v141–v145 | 0.281760–0.506256 | 0.715942 | 204.681–211.460 | 228.127–234.842 | 未提交 | REJECTED；源码已清理 |
| **v147** | **0.507355†** | **0.719696** | **222.227†** | **245.038†** | **16579 / 211s** | **REJECTED；时间通过但低于 v86** |
| v148 | **0.509729** | 0.719696 | **369.038** | 391.615 | 未提交 | **REJECTED；A3 提升 Linear 但校准超时** |

完整原始数据见 [`artifacts/official_eval/`](../artifacts/official_eval/)，官方回传记录见
[`logs/execution/`](../logs/execution/)。

## 3.1 官方分数归因（已完成）

固定场景对照、线性权重检验和本地评测边界已经整理在
[`官方分数归因记录`](evaluation-attribution-2026-09-01.md)。要点如下：

- v138/v139/v140 的 Attention mean 完全相同；v140 相对 v138 的官方分数为 `+123`，证明
  固定 Attention 修改 Linear 会改变官方结果；
- v84/v86 的 Linear mean 完全相同；v86 官方分数相对 v84 增加 `+227`，证明 v86 Attention
  必须作为后续 Linear 实验的冻结基线；
- v138 本地等权显示比 v86 高 `+2441`，官方却低 `−1029`，所以本地均值/等权总分不能用来
  推断官方 Linear/Attention 权重，也不能做跨 Attention 家族排序；
- 250/200 只是公开 case 数比例（55.56%/44.44%），不是实际官方分数权重。

v147 是按该结论生成的组合控制：原始 local JSON 的 Linear mean 与 v140 完全相同，Attention
mean 恢复到 v86 空闲复测值 `0.7196960689`；本地 API `222.227s`、墙钟 `245.038s`。用户
确认其官方结果为 `16579 / 211s`，时间通过但分数低于 v86，已经拒绝。† v147 归档后来被替换
为带 A3 的单文件源码，另一个本地 JSON 为 Linear `0.510050`、API `300.351s`；官方提交 SHA
未确认，因此两份本地结果均不冒充官方提交源码证据。

## 4. v86 与 v138 的关键反转

当前本地协议下：

| 对比 | Linear 差值 | Attention 差值 | 本地 API 差值 | 官方时间差值 | 官方分数差值 |
|---|---:|---:|---:|---:|---:|
| v138 − v86 | +0.100651 | **−0.003754** | −111.37s | −14.7s | **−1029** |

v138 的本地等权显示为 `27001.827`，明显高于 v86 的 `24560.627`，但官方排序完全相反。
这证明当前 Linear 本地增益没有在隐藏评测上转化，而 v86 的 Attention 表示对官方分数非常重要。

v138 也不是原样 v86 Attention：它缩小统计 token、候选 block/seed 和输出终选范围，并删除了
v86 的部分 scale-aware/output-aware 机制。此前把它描述为“v86 级静态 Attention”是不准确的。

## 5. 已关闭的算法路线

以下方向不再通过调参数继续：

- v138 的缩减 Attention shortlist；
- v139 连续 output-aware gain；
- v140 局部 reciprocal pair/ROAB-P2；
- v141–v145 非对称选列 BDLR、锚点冻结和阻尼变体；
- v128–v131 动态 Q/K Gram、PAWV 和随序列放大的 Attention 搜索；
- 增加 alpha、offset、sweep、block 数、阻尼、角度或候选槽位的局部扫描。

这些路线要么官方超时，要么官方分数低于 v86，要么只有固定本地 panel 上的 `10^-5–10^-4`
级差值，不能支撑继续投入。

## 6. 新的理论算法主线

新的唯一活动计划是
[`2026-09-01-hif4-hierarchy-encoder-and-analytic-attention-plan.md`](superpowers/plans/2026-09-01-hif4-hierarchy-encoder-and-analytic-attention-plan.md)。核心顺序为：

1. 先恢复可信的 pre-A3 单文件父版本，修复 v147 源码/JSON 混淆，并按 role 归因当前第二次
   `_crossfold_weight_output` 的收益与约 `78.1s` 新增代价。
2. Linear 第一正式机制改为 Activation-only Decoupled HiF4 Encoder：编码尺度只决定 code，
   最终仍保存合法 E6M2/lv2/lv3/mantissa/sign。
3. 用解析式 2×2 Hierarchical Matrix Balance 替换候选式 Smooth/Permutation/Hadamard，保持
   `XR(WR^{-T})^T=XW^T` 且不叠加动态算子。
4. 把 calibration 合法输出 oracle 编译成阈值/LUT 型固定复杂度 Activation encoder，删除在线
   candidate/coordinate loop。
5. 只有 Activation 路线验证有效后，才把同一 decoupled encoder 应用到 Weight；block-Schur
   和双侧残差降为一次性后期残差步骤。
6. Linear 稳定后才独立研究 Attention：解析 Matrix-Smooth Q/K、量化感知 K 公共平移、
   Q/K/V decoupled encoder 和静态 Fisher importance；禁止 Gram dynamic sweep、PAWV 和随序列
   增长的候选搜索。

## 7. 归档现状与待整理项

已完成：

- v128–v131 的 `result.md` 和目录名均标记 `TIMEOUT`；
- v135–v137 的 `result.md` 和目录名均标记 `REJECTED`；
- v132/v133 已补齐 `RETAINED / LOCAL HISTORICAL PARENT` 结果文件；
- v134 标记为 `RETAINED / LOCAL RESEARCH PARENT`，不代表官方可提交；
- v140 ROAB-P2 改为 `REJECTED / LOCAL-ONLY`，归档目录标记 `_rejected`；
- 空的重复 v140 curvature 目录已删除；
- v141–v145 失败源码目录删除，逐次 JSON/日志保留。

2026-09-01 归档整理（区分新旧评测分数体系）：

- 旧协议产物移入 `*/archive/legacy-*-20260901/`：`artifacts/oracle_dashboard/` →
  `artifacts/archive/legacy-oracle-dashboard-20260901/`；`artifacts/jdrq/` →
  `artifacts/archive/legacy-jdrq-diagnostics-20260901/`；`logs/candidates/` →
  `logs/archive/legacy-candidates-20260901/`；根目录 `solution_b0_tmp.py` →
  `logs/archive/legacy-root-files-20260901/`；
- v031 目录名由旧面板 `official14613` 更正为官方 `21864`（同步更新
  `evaluator/official_eval.py` ARCHIVE_MANIFEST 与 `solutions/README.md`）；
- v125 screen 记录目录更名为 `20260831_v125b_c1c-block8-screen-positive`，版本号恢复全局唯一；
- 清理 12 个 pytest 临时目录残留（`.pytest-tmp-*`、`.tmp-pytest`、`.tmp_pytest`、
  `artifacts/pytest_tmp*`，均为 gitignore 覆盖的本地临时目录）；
- 根 README 新增「分数体系与归档对照」章节，三套分数体系（官方旧权重 / 官方新权重 /
  本地协议分）明确隔离，`solutions/` 目录名中的数字字段口径已在索引中说明。

以后微参数实验不分配版本号；只有新数学算法、官方提交或一个代表性失败实现进入
`solutions/`，目录名和 `result.md` 必须同时标注 `retained/rejected/timeout`。
