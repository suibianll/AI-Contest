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

本地主评测仍使用 [`evaluator/official_eval.py`](../evaluator/official_eval.py) 的
`proxy-v2`：Qwen2.5-0.5B、24 层、默认固定分层真实 W/A panel（Linear 为每层每 role 一个真实
窗口，共 168 cases；Attention 为每层五个官方长度，共 120 cases；窗口是固定、可复现的 holdout
capture），Attention calibration 长度 `[10,128,512,1024,1024]`，validation/test 交替 holdout。
但必须明确：[`赛事说明书.txt`](../赛事说明书.txt) 没有公开指定 Qwen、层数、hidden size、GQA 或
RoPE；Qwen 只是当前最完整的本地结构假设，不是官方模型证据。主字段是 `linear_mean`、
`attention_mean`、`overall_mean`、六 API `api_total_seconds` 和 `wall_seconds`。旧
`official-shape-v1` 只保留为不可迁移的历史证据。

因此 `proxy-v2` 只用于同机算法诊断和耗时记录，不能作为官方排序器，也不能把本地时间换算为
鲲鹏时间。校准状态按公开调用图共享；`trend_diagnostics` 仅对同一官方权重 cohort 做顺序
一致性检查，发现反转时必须停止用本地分数晋级。为检查结构假设，新增
[`evaluator/cross_model_eval.py`](../evaluator/cross_model_eval.py)：它用本地 GPT-2 的真实 fused
QKV、MHA、绝对位置编码和单一 GELU FFN 建立独立 `cross-model-probe-v1`，不改写 Qwen cache，
不参与官方 proxy 排名。
`--full-cases` 可额外展开 2016 Linear + 288 Attention 做 stress，但不能与默认 panel 混排。

当前评测还默认输出 evaluator-only 的误差源分解：Linear 的 `E00/E10/E01/E11` 四臂（标准、
W-only、A-only、W+A）以及 role/layer/shape/length/split 聚合；Attention 的 Q-only、K-only、
V-only、QK-only、QKV 五个控制臂，以及 logits MSE、softmax probability MSE、KL 和
layer/length 聚合。细分结果位于每个 JSON 的 `decomposition` 和 `case_scores`，不会改变主
`score` 或候选 API 调用数；只有快速 smoke 才使用 `--no-decomposition`。

E0 修复记录：旧 v1 的 E4M3 scale 忽略 subnormal、窗口集中在少数文档，且曾误把 calibration
放进每个 case。proxy-v2 已修正这三点；共享校准版本将在下一次开发复测中核对
`168/25/24/20/20/20` 的 API 调用图。此前 per-case 校准复测中，v138 相对 v86 的本地顺序仍为
反转；因此该问题被记录为 `inversion_detected`，不得再把 proxy 分数当晋级依据；现已取消比例与
case 上限，下一次按固定分层真实 W/A panel 重新评测；完整笛卡尔集只作为显式 stress，不用于快速迭代。

2026-09-01 的四版本分层 panel 复测（v84/v86/v140/v147）记录在
[`proxy-v2-stratified-trend-v84-v86-v140-v147.md`](../logs/execution/2026-09-01-proxy-v2-stratified-trend-v84-v86-v140-v147.md)：
v84→v86 与官方同向，但四个官方锚点 pairwise 只有 3/6 同向、3/6 反转，整体状态仍为
`inversion_detected`。v140/v147 的本地 Linear 分数比 v86 高约 `0.122`，而官方分数反而低
`906/165`；这项反转集中在 Linear 耦合坐标变换，不能用本地 overall_mean 晋级。v84/v86 的
Attention 变化与官方方向一致；Q/K 单侧控制恶化而 QK 配对改善，V 基本中性，说明 Attention
应继续按 Q/K 配对和 logits/softmax 归因。

## 2.1 跨模型 GPT-2 结构探针（已完成）

由于官方说明书没有模型结构，不能只凭 Qwen proxy 断言“官方就是 Qwen”。使用同一 WikiText
窗口、同一 NVFP4/HiF4 codec、同一 API 调用图，在本地 `gpt2`（12 层、hidden 768、12×64
MHA、绝对位置编码、fused `c_attn`、单一 GELU `c_fc`）上重新捕获并评测 v86、v140、v147。
GPT-2 的 `c_attn` 已按真实权重切成 Q/K/V，`ffn_in` 不复制成 gated 两个 role；结果单独存于
`artifacts/official_eval/gpt2-*-panel.json`，执行摘要见
`logs/official_eval/gpt2-*-panel.md`。

| 候选 | 官方分数 | GPT-2 Linear | GPT-2 Attention | GPT-2 overall |
|---|---:|---:|---:|---:|
| v86 | 16744 | 0.375010 | 0.411100 | 0.391414 |
| v147 | 16579 | 0.518011 | 0.411100 | 0.469415 |
| v140 | 15838 | 0.519794 | 0.497247 | 0.509545 |

官方顺序为 `v86 > v147 > v140`，GPT-2 顺序为 `v140 > v147 > v86`，三对全部反转；Qwen
分层 proxy 也把 v147/v140 排在 v86 之上。因此“换成 GPT 后仍反转”说明失真不只来自 Qwen
结构，隐藏数据/权重分布、官方 case 构成或实现细节也与公开本地 proxy 不同；同时 Qwen 的
绝对 Attention 分数和 role 权重不能再当作官方结构证据。GPT-2 运行只作为跨结构稳健性探针，
不能被解释为官方得分或官方 runtime。

GPT-2 v147 的误差源分解仍显示同一可优化结构：Linear W-only `-240.979`、A-only `-138.412`、
Both `0.518`、interaction `379.909`，属于 `paired_coordinate_coupling_likely`；Attention
Q-only `-25.810`、K-only `-28.547`、V-only `0.016`、QK-only `0.389`、Both `0.411`，属于
`paired_qk_coupling_likely`。这支持下一步继续拆分 Linear 的坐标/双侧编码和 Attention 的
Q/K 配对，而不是围绕 Qwen 特有的 GQA/RoPE 参数继续调参。

## 2.2 hif4 外部评测器复测（已完成）

按用户要求，使用 [youxilee/hif4](https://github.com/youxilee/hif4) 的
`real_data_eval.py` 对本地 v84/v86/v140/v147 快照做了统一的 GPT-2 12 层全层复测：
`amax6 / seq128 / calib2 / test2 / config=current`。结果为：

| 候选 | Linear q/k/v/o/fc/proj | Linear mean | Attention |
|---|---|---:|---:|
| v84 | 0.6221 / 0.6341 / 0.6133 / 0.5578 / 0.5558 / 0.5373 | 0.586733 | 0.4477 |
| v86 | 0.6221 / 0.6341 / 0.6133 / 0.5578 / 0.5558 / 0.5373 | 0.586733 | 0.4727 |
| v140 | 0.6630 / 0.7241 / 0.6218 / 0.5560 / 0.5107 / 0.5221 | 0.599617 | 0.4661 |
| v147† | 0.6630 / 0.7241 / 0.6218 / 0.5560 / 0.5107 / 0.5221 | 0.599617 | 0.4713 |

该脚本使用内置 synthetic text、重复不足 token，且标准基线来自候选私有 codec；因此它是
跨模型结构探针，不是官方复刻。外部等权排序为 `v147 > v140 > v86 > v84`，仍与官方
`v86 > v147 > v140` 不同；但 v84→v86 的“Linear 不变、Attention 上升”及 v140/v147
的逐 role 分化提供了有用的结构归因。完整命令、SHA 和外部 `test_solution.py` 私有接口
不兼容记录见 [`hif4 外部复测日志`](../logs/execution/2026-09-01-hif4-external-gpt2-v84-v86-v140-v147.md)。

† v147 为当前归档含 A3 源码，官方提交 SHA 尚未确认。

## 2.3 外部 hif4 的 Linear role 结论（已完成）

为回答“问题到底在 qkv 还是 o/fc/proj”，直接使用上游
[youxilee/hif4](https://github.com/youxilee/hif4) 的 `real_data_eval.py` 做了同配置逐 role
复测。这里的 q/k/v 是静态 fused `c_attn` 投影；Attention 动态 Q/K/V 是另一组控制臂，不能
混称。外部 GPT-2 causal 结果的 v140−v86 为：

| role | Δ mean | 12 层符号 | 结论 |
|---|---:|---:|---|
| q | +0.0409 | 12 正 / 0 负 | 不像主要问题 |
| k | +0.0900 | 12 正 / 0 负 | 不像主要问题 |
| v | +0.0085 | 11 正 / 1 负 | 基本中性 |
| o | −0.0018 | 5 正 / 7 负 | 局部异常，均值近中性 |
| fc | **−0.0452** | **0 正 / 12 负** | **最明确的系统性问题** |
| proj | **−0.0153** | 6 正 / 6 负 | 次要但有严重层级异常 |

临时单变量消融给出同一方向：关闭 proj（`rows < channels`）ROAB 后分数从 `.5221` 到
`.5430`；关闭 fc（`rows > channels`）ROAB 无变化；关闭 fc-CAT 仅到 `.5144`；关闭 fc-BOAT
则降到 `.4599`。所以行动顺序是：冻结 q/k/v 和 o；proj 先做 ROAB-off，再做解耦 encoder；
fc 保留 BOAT、重做 expansive 编码/scale；o 暂不重写。完整逐层数组、命令、外部脚本限制和
置信度说明见 [`外部 role 归因日志`](../logs/execution/2026-09-01-hif4-external-role-attribution-v140-v86.md)。

这只能作为角色退化探针，不能覆盖官方总分趋势：外部脚本的 synthetic text、候选私有 codec
以及与 Qwen/WikiText 评测的协议差异会造成整体排序冲突。因此当前结论是“fc 高置信、proj
中高置信、o 低置信；q/k/v 不是第一修复目标”，而不是声称已经定位了官方隐藏 case 的精确
权重。下一轮本地主评测必须同时报告静态 q/k/v/o/fc/proj 与动态 Attention Q/K/V，逐层显示
误差源。

## 2.4 评测器分解缺口与修复

此前 `official_eval.py` 已有候选内部的 Linear `E00/E10/E01/E11` 和 `by_role`，但它只回答
“该候选相对独立标准 codec 的 W/A 增益”，没有把同一 cache 中的 v140 与 v086 按
`layer/role/window` 配对后计算 signed delta；报告也没有把 `fc_gate/fc_up` 合并为 `fc`，或
自动列出最差层。因此它无法像外部 hif4 一样直接回答“qkv、o、fc、proj 哪一组退化”。

这不是额外的官方评测调用：已在 [`evaluator/official_eval.py`](../evaluator/official_eval.py)
中加入 `role_family`（`qkv/o/fc/proj`）、跨候选静态 Linear role 差分和 worst-case layer
列表。运行 `--archive` 后，JSON 顶层的 `linear_candidate_role_diagnostics` 默认优先以
`v086`（兼容手工运行名 `v86`）为基线，报告同时输出 family/role 的平均 Δ、正负层数和最差
case；单候选 JSON 仍保留原有 `decomposition`，不会改变主分数、调用次数或时间。

评测器仍不能自动关闭候选私有的 ROAB/BOAT/CAT，因为六个公开 API 没有这些开关；这类机制
归因必须用明确的 local-only 变体或外部脚本临时副本，且不能作为官方候选。现在的职责分层是：
主评测器负责同 cache 的 role 差分，hif4 外部脚本负责跨结构/私有机制探针，官方回传负责最终
排序。

## 2.5 v151 proj ROAB-off 控制（已归档）

v151 按 E0.7 的顺序只关闭 `rows < channels` 的 proj/down ROAB，并移除当前根中额外的 A3
残差 pass，以便与 pre-A3 v147 父版本处于同一时间边界。它没有修改 v86 Attention，也没有
改动 q/k/v/o 或 fc 的路由。

在同一只读 Qwen `proxy-v2` cache 上，14 个 Linear case（两层覆盖七个静态 role）和 1 个
Attention case 的结果完全不变：父版本 `Linear=0.582528216 / Attention=0.942927486 / API
=201.258s`，v151 为 `0.582528216 / 0.942927486 / 193.213s`；q/k/v/o/fc_gate/fc_up/proj
逐 role 均相同，约 8 秒差异不作为计时结论。外部 hif4 GPT-2 四层因果 smoke 只在 proj 上
从 `.5029` 到 `.5658`，其余 role 与 Attention 不变，因此 v151 标记 `REJECTED`，不能晋级
为 root 或后续父版本。证据见 [`v151 result`](../solutions/20260902_v151_proj-roab-off_rejected/result.md)
和 [`v151 execution log`](../logs/execution/2026-09-02-v151-proj-roab-off.md)。

这次 no-op 反而缩小了搜索空间：proj 的外部收益不转移到 Qwen，下一步不再继续调整 proj
ROAB；按计划转向 `fc_gate/fc_up` 的 expansive 编码/scale 机制，保留 BOAT，并继续冻结
q/k/v/o 与 v86 Attention。所有新候选仍必须先报告静态 role/family、动态 Attention Q/K/V、
逐层最差 case 和六 API 时间，再决定是否跑完整 panel。

## 2.6 v152/v153 fc 递进实验（已归档）

v152 只关闭 expansive fc 的 CAT、保留 BOAT。Qwen 14-case targeted panel 的 Linear 从
`0.582528216` 到 `0.583139209`，但同口径 56-case 配对 panel 只有
`0.542366307→0.542552798`；fc family 的 signed delta 仅 `+0.000653` 且层间正负混合，
所以不能把短 panel 的小增益当成晋级信号。外部 GPT-2 四层 fc 为 `.5658→.5709`，仅作方向
证据。v152 已标记 `REJECTED`。

v153 按 L1 首版尝试只对 fc Activation 使用 BF16 `s_q=a_max/7` 做层级/码字分配，存储仍为
合法 E6M2，但没有在固定 code 后重新拟合 `s_d`。Qwen 14-case 中 Linear
`0.582528216→0.568753650`，fc_gate `0.396959→0.350049`、fc_up
`0.368327→0.318816`，q/k/v/o/proj 和 Attention 均不变，明确拒绝。这个失败精确指出当前
实现缺的是“固定 code 后的部署尺度闭式回归”，而不是继续切换 CAT/ROAB。

两个快照、命令、SHA 和完整 role 结果见 [`fc follow-up log`](../logs/execution/2026-09-02-v152-v153-fc-followups.md)。
v154 已按该要求在同一 `s_q` code 下用最终 `Q(W)` Gram/输出度量求 `s_d` 闭式解并投影到
合法 E6M2，但 Qwen role 均与 v153 完全相同（fc family `0.334432`），没有恢复 v153 的
回归，因此也标记 `REJECTED`。当前结论是：直接 s_q/s_d 变体先暂停，下一步改做有明确
recoverable margin 输出的 L3 teacher/oracle 诊断，不再试 CAT/ROAB 或 scale 参数。

## 3. 历史 v1 结果表（不可与 proxy-v2 混用）

下表保留旧 `official-shape-v1` 的同机数字，仅用于审计此前的失真；当前 proxy-v2 分层 panel
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
