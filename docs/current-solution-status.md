# 当前状态：两侧比重校准实验就绪（v162/v163/v164 待官方提交），v161 官方 timeout

更新：2026-09-03。

## 0. 最新官方进展

**当前活动实验：官方两侧分数比重校准（2026-09-03，v162 官方已回传 `1001/146s`，v163/v164 待提交）。**
为解耦官方总分中 Linear 与 Attention 的贡献，构造了三个候选（预注册判读表见
[`活动计划`](superpowers/plans/2026-09-03-official-side-weight-calibration-plan.md)）：

| 版本 | 构造 | 本地 default | API | 官方角色 |
|---|---|---|---|---|
| **v162** | 独立最小实现，六 API 全部标准 HiF4 codec（NVFP4→BF16 中间解码→标准编码） | linear/attention mean **均为 0.0**（288 case gain 全 0，与 STD 逐位一致） | 2.6s | 官方 **1001/146s**；锚点已测，S(v162)>0 触发预注册 §3 行 |
| **v163** | v160 归档零改动 + 末尾追加标准 Attention 四 API 重定义 | Linear 168 case 与 v160 **逐位一致**（0.633526）；Attention mean 0.0 | 228s | Δ_L = S(v163)−1001 |
| **v164** | v160 归档零改动 + 末尾追加标准 Linear 两 API 重定义 | Attention 120 case 与 v160 **逐位一致**（0.742354）；Linear mean 0.0 | 70s | Δ_A = S(v164)−1001 |

判读：`S(v162)=1001>0` 触发预注册解释表"官方存在非零基础分或 STD 定义不同"行，
不反推公式；后续判读改为 Δ_L = `S(v163)−1001`、Δ_A = `S(v164)−1001`、可加性
`S(v163)+S(v164)−1001 ≈ 17532`（即 S(v163)+S(v164) ≈ 18533）。**时间下界发现**：
本地 API 2.6s → 官方 146s，官方时间含 ~140s 评测器开销；有效算法预算 ≈ 154s，
v160 算法份额 ≈ 86s（232−146），v161 timeout 回溯解释成立（算子类成本比 ≥ 2.4×），
v163/v164 时间上界 ≈ 232s，风险可忽略。三个 SHA 行为互不相同，不属于被禁止的
相同 SHA 确定性验证。构建方式（复制 v160 + 模块级追加重定义）保证保留侧输出与 v160 逐位一致，
本地已用 case 级对比确认（max Δgain/Δmse = 0.0）。证据：
[`v162 result`](../solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/result.md)、
[`v163 result`](../solutions/20260903_v163_v160-linear_standard-attn_scoreNA_timeNA/result.md)、
[`v164 result`](../solutions/20260903_v164_standard-linear_v160-attn_scoreNA_timeNA/result.md)、
`artifacts/official_eval/sidecal-v16{2,3,4}-*.json`。

**v161 官方回传（2026-09-03）：`timeout（>300s，无分数）`。** v161 = v160 + Attention
Q/K 交叉算子 Gram64 per-call 精化（v128 机制移植，V/Linear 冻结）。本地全漏斗通过
（Qwen default 120 paired `+0.052502`、`106+/14−`、touch 88.3%；GPT-2 `+0.0678` 同号；
D1 本地满足；attention API `+28.0s` CUDA 在 +40s 门禁内），但官方机（鲲鹏）上动态
per-call 小张量算子成本远超本地 CUDA 外推，v160 的 68s 官方余量被耗尽。归档目录已
更名 `_timeout`，per-call 动态自适应族正式关闭；S2（校准搜索解析化）前置条件不满足，
不启动。**修正时间核算结论：v128 家族超时元凶不只是校准期候选搜索（199.8s/24 calls），
动态精化本身（本地 `0.092s/call` CUDA）在官方硬件上即超预算**（v138 无 dyn refine 官方
208s 通过，v128/v129/v130/v131/v161 含 dyn refine 全部官方 timeout）。D1 维持 3/3
（v161 无官方分数，不计入）；P9 检验无法记录。证据：
[`v161 官方超时日志`](../logs/execution/2026-09-03-v161-official-timeout.md)、
[`v161 result`](../solutions/20260903_v161_v160-attn-s1-qk-gram-refine_scoreNA_timeout/result.md)。
**当前无活动计划**：本地已知机制族全部闭环（Linear 结构 full64/Householder、Attention
解析静态族、Attention per-call 动态族），下一步为外部材料搜索或用户指定新机制。

**v160 官方回传（2026-09-03）：`17532 / 232s`，与 v159 完全相同。** 232s 通过
`<300s`，时间风险解除。v160 = v159 Linear（L1 逐位等价编码）+ A2/A1（Attention）：
- L1/A1 等价性在官方 panel 上得到验证——分数与 v159 逐分相同（17532），bit-exact
  编码没有破坏官方任何 case；
- A2（GQA mode-4 重启用，本地 attention default mean +0.0066）在官方为 **no-op**：
  隐藏数据上 mode-4 未被采用（与 GPT-2 60/60 zero-delta 的行为一致）；
- **规律提取**：① 本地 <0.01 级 Attention mean 微调不迁移官方（官方整数分对
  门控小改进不敏感，与历史"本地增益未转化"教训一致）；② v160 官方 232s vs 本地
  API 290.7s，官方机约快 1.25×，68s 余量可用于后续机制（勿花完）；③ 突破 17532
  必须改变官方 panel 候选行为且 effect 足够大——Attention 门控通道已证无效，当前只验证
  固定 Householder Linear 机制；17816 源码不再作为等待项。

用户确认：根目录同 SHA 的 v159 合并版本官方分数为 **17532**，比 v158 的 16861 高
**671 分**；官方时间未提供。另一个 17816 结果仍是更高的外部锚点，比 v159 高 284 分，
但完整源码、Attention 配置和官方时间尚未同步。

用户同时确认 v147 的官方结果为 **16579 / 211s**。它通过 `<300s` 时间限制，但比 v86
低 165 分，因此按归档规则标记为 `REJECTED`，不作为后续父版本。v147 目录曾被原地替换，
当前无法确认官方提交对应的准确源码 SHA；官方结果只绑定版本号，不伪造 SHA 归因。

新算法不是 v138–v145 的局部调参延续，而是完整的：

- SmoothQuant + Permutation + block Hadamard 等价变换搜索；
- 变换后完整协方差驱动的 Weight GPTQ；
- 部署权重输出 Gram 驱动的 Activation GPTQ；
- quadratic AdaRound、E6M2 offset、data-driven refinement 和 edge extension；
- proxy 与 e2e 混合选择，窄层执行联合变换搜索。

用户提供的 `linear.txt`/`linear_dep.txt` 已合成为 v159，并已获得 17532 官方分数；17816 的
完整提交仍未同步，不能把两者视为同一源码。完整执行顺序见
[`活动计划`](superpowers/plans/2026-09-03-official-pattern-and-linear-structure-experiments.md)。

## 0.1 v160 本地集成（2026-09-03，官方 = 17532/232s no-op）

v160 = v159 Linear（L1 逐位等价编码）+ A2/A1（Attention）。官方回传 `17532/232s`
与 v159 完全相同（见 §0 规律提取）；归档 `solutions/20260903_v160_v159-linear-l1batch_v158-attn-a2_scoreNA_timeNA/`
（SHA `33B1D061…`，仅六 API、单文件），本地证据：

- **Linear**：v159（用户 17816 实现 + 官方 17532 基底）+ L1 逐位等价批编码
  （ec18a88）；default 168 linear_mean `0.633526`（与 L1 前逐位一致），
  calib API `166.6s`。
- **Attention**：v158 + A2（`_ATTN_SCALE_AWARE_CENTER_GQA=True`，mode 4 进入 GQA
  竞争，`_candidate_is_safe` 兜底）+ A1 等价清理（K 居中/旋转 signs 候选共享，
  120/120 + GPT-2 60/60 逐位一致）。default 120 attention_mean `0.742354`
  （v158 的 0.735752 → +0.0066，17+/3−/100z，全长度正向；已知尾部 layer11 len10
  −0.17、layer14 len128/512 ≈ −0.02）。
- **L2 消融 4 项全 REJECTED**：seeds/sizes/RMS-smooth/wide-alphas 均承重，无安全消融。
- **完整 default 集成审计**：overall `0.678871`，六 API `290.7s`
  （calib_w 166.6 / dyn_a 60.7 / calib_a 60.0 / dyn Q/K/V 3.4），wall `318.4s`；
  本地时间不能换算官方 `<300s`。
- **GPT-2 完整集成**：linear `0.603115` / attention `0.389583`（与各自 parent 逐位
  一致，无跨模型回归），API `113.8s`。

L3 首次 full64 探针因死分支判为 **INVALID EXPERIMENT**；修正 reachability 后已按用户要求仅
重跑一次 compact：24 次 refine attempted `659456` row-blocks、accepted `657540`，但 Linear
`0.705508→0.687588`，paired `6+/42-/8=`、mean delta `-0.017920`。W-only delta `+0.107169`
被 interaction `-0.118818` 反转，说明块内 full-H 目标与最终 `Q(A)Q(W)^T` 不一致。该实验
正式 `REJECTED`，不再调参或扩大测试。E3 首个固定 64-block Householder 候选已完成 Qwen
Linear compact：`0.705508→0.699190`，paired `8+/48-/0=`，API `46.052→47.387s`，违反
compact 门禁；五个 C 源构造变体（amax/rms/xrms/x-only/w-only）全部低于基线
（`0.699719–0.703344`），机制全族否定，正式根已保持该研究臂默认关闭。Linear 侧两个正交
结构假设（同坐标码字、坐标几何）均无本地可迁移余量，Linear 结构实验闭环；下一算法实验
为活动计划 Attention 解析式宽域机制（A1 Matrix-Smooth 组内扩展首选）。

## 1. 版本结论

- **当前仓库内最高已绑定源码的官方分数：v159/v160，17532 分；v160 时间为 232s。** v160
  归档 SHA `33B1D061...680D` 是当前 score/time 均完整的实验父版本；v158 `16861/223s` 是
  更低复杂度的安全基线，17816 仍只作为外部锚点。
- 根 `solution.py` 在 v160 后只增加默认关闭的 L3 gate，行为不变但 SHA 已不同；规律实验必须从
  v160 归档分支。v159 的 17532 仍绑定原始 SHA `0508045A...4242`，其官方时间未知。
- v159 与同 cache 的 v158 compact 配对 mean Δ 为 `+0.149191`（56 改善、0 回归）；`proj`
  八个 case 的配对 Δ 也为正（`+0.124209`）。这仍只是 compact Qwen proxy 证据，不能外推为
  官方泛化结论。候选说明见
  [`v159 result`](../solutions/20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA/result.md)。
- CUDA device 错误已经修复；L1 batching 已把 v160 Linear calibration 降至约 `166.6s`，
  输出逐位不变。官方 2×2 与时间探针停止；Householder 全族 REJECTED 后 Linear 结构实验
  闭环，转向 Attention 解析式宽域机制。
- v138/v139 虽在官方 `<300s` 内通过，但只有 `15715/15716`，比 v86 低约 1029 分；
  v138–v145 这条“压缩 Attention 后继续叠 Linear 局部模块”的路线已经失败并关闭。
- v155 官方 `16581 / 208.5s`、v156 官方 `16580 / 204.3s`，两者时间均通过但分别低于 v86
  `163/164` 分，已正式拒绝。它们本地 `10^-4` 级正向没有迁移。
- v157 exact-v86 + ROAB-only 官方 `16729 / 218.96s`，时间通过但低于 v86 `15` 分，已拒绝。
  这说明 `v138→v140 +123` 不能作为可移植 ROAB 主效应。
- v158 从 exact v86 只增加解析式 GQA 组内 2×2 Attention Matrix-Smooth；Linear 与 V
  逐字段冻结。effect 配对 Attention `1/0/4`、mean delta `+0.007195`；default 配对
  `49/16/55`、mean delta `+0.011018`，Linear control `0/0/168`。官方为 **`16861 / 223s`**，
  相对 v86 **`+117 / +0.3s`**，正式晋级；本地 mixed 不能覆盖官方正向事实。
- 评测流程立即改为场景隔离：Linear 优化只运行 Linear，Attention 优化只运行 Attention；
  单侧实验不再重复另一侧完整校准与计分。17816 源码若后续到位，再独立归档。

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

## 2.7 配对机制评测（新迭代默认）

此前 14/56-case 运行是默认 case 序列的前缀：14 只覆盖 layer 0–1，56 只覆盖 layer 0–7，
并不是模型纵深采样。它们适合查接口，却会把浅层偶然收益当作机制趋势；同时父候选比较依赖
手工相减 aggregate mean，不能稳定区分目标 role、路由泄漏和 W/A 来源。现已把算法迭代方式
改为 `paired-effect-panel-v1`：

- `--effect-panel` 固定选择 layer `0/3/7/10/13/16/20/23`，每层保留全部七个静态 Linear
  role，共 56 cases；Attention 选择五个覆盖模型深度与五个公开长度的哨兵；
- 校准仍按完整调用图产生 168 个 Weight state 和 24 个 Attention state，只减少动态评分与
  evaluator-only 分解，因此不会因为缩短 panel 而改变 calibration 生命周期；
- 父版本只运行一次并保存 immutable JSON。候选通过 `--baseline-json` 在完全相同的
  `layer/role/window/split/length` 上配对；已有结果也可用 `--candidate-json` 零 API 重放；
- `--focus-linear-roles fc` 将 fc_gate/fc_up 作为目标组，并把 q/k/v/o/proj 作为 control。
  报告同时输出 mean/median Δgain、改善/回归/不变 case、MSE ratio、逐 role/family/layer、
  W-only/A-only/Both/interaction、最坏 case、Attention Q/K/V 和六 API 时间差；
- case identity、标准臂 MSE 或 reference energy 不一致时比较直接失败。符号标签只描述结果，
  不增加人为门槛，也不把 proxy delta 换算成官方分数。

用新逻辑重放已有证据后，v152 相对父版本的 Linear overall 为 `+0.000186`，但只有
`3 改善 / 3 回归 / 50 不变`；focus fc 为 `+0.000653`、`3/3/10`，结论为 `mixed`，control
40 cases 与 Attention 均完全不变。进一步拆分显示 fc_gate `+0.001871`，fc_up
`−0.000565`；最坏 case 是 layer 7 fc_gate `−0.007283`，其次 layer 0/1 fc_up
`−0.003087/−0.001435`。所以旧的 `+0.000187` 不是“弱但稳定提升”，而是 gate/up 与层间
正负抵消，拒绝结论得到更具体的原因。证据见
[`v152 paired report`](../logs/official_eval/v152-fc-cat-off-paired-effect.md)。

v153 重放则更清晰：focus fc mean `−0.048211`、median `−0.049511`，`0 改善 / 4 回归 /
0 不变`，median player-MSE ratio `1.078387`，为 `consistent_regression`；10 个 control 和
Attention 均不变。证据见
[`v153 paired report`](../logs/official_eval/v153-fc-decoupled-paired-effect.md)。后续 L3 的
每个候选都先按同一父 JSON、effect panel 和目标 role 生成 paired effect；只有 focus 方向、
control、误差源和最坏层都可解释，才运行默认 168+120 panel。完整 panel 仍用于复核，不再是
每次小迭代的第一步。

## 2.8 L3-D0 fc 合法码字余量诊断（DONE / 不可直接编译）

以 SHA `800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D` 的 pre-A3
local parent 为固定对照，规范 teacher 覆盖 layer `0/3/7/10/13/16/20/23`、`fc_gate/fc_up`、
fold `[10,128]` 和五类合法邻域。完整证据见
[`l3-fc-legal-oracle.json`](../artifacts/official_eval/l3-fc-legal-oracle.json) 与
[`执行报告`](../logs/execution/2026-09-02-l3-fc-legal-oracle.md)。

| edit | mean margin | median margin | 正/负 case | 结论 |
|---|---:|---:|---:|---|
| mantissa | `+0.000287` | `+0.000226` | `19/13` | mixed |
| lv3 | `+0.000138` | `+0.001518` | `27/5` | mixed |
| lv2 | `+0.000133` | `+0.000343` | `25/6` | mixed |
| E6M2 scale | `−0.000231` | `+0.003588` | `29/3` | mixed |
| joint | `+0.002084` | `+0.007144` | `30/2` | mixed |

layer 3 / fold 128 是决定性反例：joint exact output margin 为 `−0.094751`（`fc_gate`）和
`−0.112680`（`fc_up`），而 fold 10 仍为 `+0.001447/+0.009264`。所以 D0 结论是
`margin_exists_but_not_compile_safe`：same-fold quadratic teacher 有局部余量，但跨 fold
的真实输出方向不稳定，不能据此创建 v155。规范 teacher 约 `597.7s` 是研究成本，不是候选
API 时间。

为缩短迭代，新增 [`l3_fc_fast_probe.py`](../workbench/l3_fc_fast_probe.py) 只跑 layer 3、
joint 和一次 batched Jacobi，约 `10.35s`；它再次得到 fold 128 两个 fc role 的负方向。
该 probe 仅用于定位最坏层，不能替代规范 D0、proxy 排名或官方时间。

因此当前下一步不是继续调 `s_q/s_d`、CAT、ROAB 或 offset，而是做一个只覆盖 layer 3 最坏
fold 的 cross-fold feature/decision stability 快探针；若不能得到固定 threshold/LUT 规则，
立即关闭 L3 表示族，转 L2 解析式层级矩阵平衡。根 `solution.py` 和 v86 Attention 保持冻结。

## 2.9 L2 解析 pair-balance 探针（REJECTED）

在同一 pre-A3 parent 上做了 local-only 的 2×2 analytic pair-balance（只替换 expansive
fc 的 BOAT/CAT/ROAB 选择，其他 role/Attention 不变）。结果见
[`l2-pair-probe-effect.json`](../artifacts/official_eval/l2-pair-probe-effect.json) 与
[`报告`](../logs/official_eval/l2-pair-probe-effect.md)：Linear `0.588023229→0.498286314`，
focus fc `−0.314079`（`0 改善 / 16 回归 / 0 不变`），MSE ratio 中位数 `1.636`；40 个 control
和 Attention 完全 no-op。朴素 pair balance 破坏静态 Weight code，已拒绝，不分配版本号，也
不替换 root。后续 L2 必须以部署输出 metric 修正，而不是再试同类无约束矩阵平衡。

## 2.10 评测 scope 清理（DONE）

评测器曾把历史 `official-shape-v1`、当前 `proxy-v2`、14/56 前缀 smoke、effect panel、
full stress、GPT-2/hif4 外部探针放在同一目录并按均值阅读，造成“系统越来越乱”的表象。现在
`evaluator/official_eval.py` 给每个结果写入 `evaluation_scope`：

- `default-panel`：同一 proxy-v2 cache 内唯一可做本地 proxy 排名的 168+120 panel；
- `effect-panel` / `paired-json-replay`：只做父子逐 case 机制诊断；
- `full-stress`：只做压力回归；
- `smoke-prefix`：只做接口/合法性检查；
- `official-shape-v1`、GPT-2、外部 hif4：历史/跨结构诊断，永不与 proxy 排名混用。

报告和 archive JSON 现在同时写 scope、intent 和 `official_score_equivalent=false`；没有任何
本地 scope 可以替代官方分数或官方 `<300s`。具体字段契约见
[`artifacts/official_eval/README.md`](../artifacts/official_eval/README.md)。

## 2.11 v155 L5a permutation-stability（REJECTED / official 16581, 208.5s）

在 L3-D0 的 teacher margin 跨 fold 不稳定后，按计划只做了一次参数无关的 stability probe，
没有继续调 `s_q/s_d`、CAT、ROAB 或阈值。结果是 fold-0 生成的 fc 特征/决策在 fold-1 上
符号不稳定（`fc_gate` 仅 `8/14` 符号一致、`fc_up` `6/14`，相关系数约 `0.32–0.44`），
held-out 正向 precision 为零；因此直接 activation teacher-to-student 编译族关闭。

probe 同时保留了一个不同层级的低自由度坐标机制：在既有 BOAT 后计算 64-channel pressure，
固定四分位 low/high interleave，并要求两折 product loss 均下降且最小折收益不小于折间分歧。
它不改变连续乘积、HiF4 codec 或六 API 调用图。正式单文件快照为
[`v155 solution`](../solutions/20260902_v155_l5a-permutation-stability_rejected/solution.py)，
SHA256 `816ECBF5E253745C5EBFD04233BD04A2B772CF1510641393C7900CDAFA0EB4CC`，完整说明见其
[`result.md`](../solutions/20260902_v155_l5a-permutation-stability_rejected/result.md)。

正式 effect panel（56 Linear + 5 Attention）为 Linear `0.588162284`、Attention `0.757433277`，
相对 pre-A3 parent 的 Linear `+0.000139055`（2 改善/0 回归/54 不变），focus fc `+0.000486693`
（2/0/14），静态 control 和 Attention 全 no-op。默认 168+120 的等价 paired replay 为
Linear `0.570998953`、Attention `0.724734669`，相对 parent `+0.000116536`（4/0/164），
focus fc `+0.000407876`（4/0/44）；本地 API `248.121s`、wall `280.763s` 仅作同机诊断。

该收益的 W/A 分解不是 operand 独立改善：W-only、A-only 仍为负而 Both 略正，属于双侧坐标
耦合；且只命中 4 个默认 case，召回率很低。进一步的严格 GPT-2 配对（同一 cache、同一
pre-A3 parent）使 Linear `0.519793773→0.519641076`（`−0.000153`），`ffn_in/fc`
`−0.000916`，Attention 完全不变。用户随后回传官方 `16581 / 208.5s`：时间通过，但比 v86
低 `163` 分，正式判为 `REJECTED`。这证明本地低召回微增益没有迁移；不替换根
`solution.py`，也不再扩大 permutation 的层列表/分位阈值。GPT-2 证据见
[`v155 cross-model report`](../logs/official_eval/gpt2-v155-l5a-perm-stability.md)。

## 2.12 v156 L4-WD（REJECTED / official 16580, 204.3s）

v156 从 pre-A3 单文件父版本编译了一个只改 Weight 的闭式 stored-scale 机制：固定 sign、
mantissa、lv2/lv3 与坐标，按变换后校准 Gram 为每个 expansive row/block 求尺度，再投影到
合法 E6M2 并用两折真实部署输出 loss gate。Qwen effect panel 的 Linear 为 `0.588130853`
（相对 parent `+0.000107624`，5/0/51），focus fc `+0.000376686`（5/0/11），controls
和 Attention no-op；GPT-2 同结构配对为 Linear `+0.000029454`、fc `+0.000176722`，
Attention no-op。该增益很小，不能替代官方验证，但源码已完成单文件隔离导入检查，SHA256
为 `594EF2FBB70AE54E06BF2D896E11E637E4BA9AF67AD54C01F10D57136EB8DF85`。

按用户要求保留目录
[`v156 solution`](../solutions/20260902_v156_l4-weight-decoupled_rejected/solution.py)。用户回传
官方 `16580 / 204.3s`：时间比 v86 少 `18.4s`，但分数低 `164`，且比 v155 还低 `1` 分，
正式判为 `REJECTED`。本地 Qwen/GPT-2 的 `10^-5–10^-4` 正向没有迁移，停止该 stored-scale
路线；根 `solution.py` 不切换。

## 2.13 v157 exact-v86 + ROAB-only（REJECTED / official 16729, 218.96s）

用户指出 v86 官方结果已经存在，继续提交 v86 不能产生新信息；同时当前本地评测无法预测
官方排序。重新按固定场景读取官方历史后，唯一干净的正向 Linear 增量是 v138 到 v140：两者
使用同一 reduced Attention，v140 只增加 ROAB-P2，官方 `15715→15838`（`+123`），时间
`208s→207s`。v140 的绝对分数低于 v86 说明其组合父路径失败，不等于这个独立增量失败。

因此 v157 从 exact v86 单文件（SHA256
`E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`）分支，只在 v86
全部 Linear 变换冻结后加入一次解析 2×2 reciprocal pair 变换：`X→XU`、`W→WU^{-T}`，
用 bounded plain-HiF4 输出误差在 parent 与 proposal 间二选一。ROAB 被拒绝时 Linear 输出和
state 与 exact v86 字段级一致；Attention calibration 与 Q/K/V dynamic 也字段级一致。

没有运行任何本地排名 panel。已通过 `35` 个仓库测试、六 API 合法性、selected branch、连续
乘积/协方差不变量和脱离仓库单文件导入检查。正式源码 SHA256 为
`984BF752156187B8892894060A99FE52027E2457F37FC23C11657041B29B86E1`。用户回传官方
`16729 / 218.96s`：时间比 v86 快 `3.74s`，但分数低 `15`，正式判为 `REJECTED`；根
`solution.py` 不切换。`v138→v140 +123` 是组合上下文交互而非可移植 ROAB 主效应，ROAB
路线关闭，不再调 pair size、threshold 或 role gate。

## 2.14 Compact generalization panel（DONE）

为同时解决“本地不能泛化、effect panel 仍然很慢、Linear 只有 aggregate mean”三个问题，
`evaluator/official_eval.py` 新增 `--compact-panel`：

- Linear 只建立 layer `0/8/15/23` × 7 role 的 28 个 Weight state；每个 state 使用
  validation/test 两个同长度独立 holdout，共 56 个动态 case；calibration 使用训练集不同文档的
  128/512 两折；
- Attention compact 只建立四个纵深 state；单侧运行在 prepare 前即裁掉禁用侧；NVFP4 pair
  只为选中 state/case 编码，不再每次重建完整 dense cache；
- JSON `analysis.linear_generalization` 新增 median、q25/q75、worst-quartile mean、min/max、
  正负 case、player/standard MSE ratio，按 role/family/layer/shape/split/length 汇总；同时对同
  layer/role/length 的 validation/test 做 sign consistency、gain gap 和 paired minimum-gain；
- decomposition 开启时，上述分析同时保存 W-only/A-only/Both/interaction 的分布，而非只看
  Linear mean。

根 `solution.py` 的一次 Linear-only 验证产物为
[`JSON`](../artifacts/official_eval/root-compact-generalization-linear-v2.json) 与
[`report`](../logs/official_eval/root-compact-generalization-linear-v2.md)：28 次 Weight calibration、
56 次 Activation dynamic，API `40.408s`、candidate wall `45.438s`、cache load+prepare
`8.203s`，总周转约 `53.64s`。28/28 个 validation/test pair gain 同号，gap median
`0.009432`、max `0.063375`；这些是基线的本地稳健性描述，不是官方预测。旧 v86 default 为
prepare `47.904s` + wall `322.895s`；scope
不同，不能比较分数，但可确认评测复杂度和周转时间显著下降。compact 明确标记
`official_score_equivalent=false`、`comparable_for_proxy_ranking=false`；日常只做父子机制与
跨 holdout 泛化诊断，候选提交前仍运行一次目标侧 default audit。

## 2.15 NVFP4 输入持久化缓存（DONE）

此前 compact panel 虽然只量化实际使用的 28 个 Weight state 和 56 个 Linear 动态输入，但每次
父子评测仍会重新加载 `10,984,305,646` 字节 dense cache 并执行相同的 `nvfp4_encode`。现在
`official_eval.py` 默认以 `--nvfp4-cache-mode auto` 按 scenario/panel/profile 保存已经量化的
carrier/scale `PreparedPack`；缓存不包含候选 state 或输出，因此同一份输入可安全复用于不同
算法。schema、协议、codec/mode、dense 源文件 identity、数据 SHA 和完整 profile 不一致时拒绝
只读命中，auto 模式则重建。

真实 compact-linear capture-only 复核：首次从 dense cache 构建为 `9.278575s`，生成
`476,399,887` 字节 NVFP4 输入缓存；第二次强制 `--nvfp4-cache-mode read` 命中为
`0.202392s`，准备阶段减少约 `97.8%`，且 `data_source=nvfp4_cache`。证据位于
[`build JSON`](../artifacts/official_eval/nvfp4-cache-build-check.json) 和
[`hit JSON`](../artifacts/official_eval/nvfp4-cache-hit-check.json)。原始 dense cache 仍作为首次构建
和失效重建来源；`--cache-mode auto` 也已修正为存在时读取，而不是无条件重新模型前向。

Attention 侧也完成了独立缓存复核：`attention-only + compact-panel` 首次构建
`7.381841s`、缓存大小 `59,184,287` 字节，强制只读命中 `0.053160s`；Linear 与 Attention
不会共享错误的 scenario/profile。随后用 `both-default` 缓存做了一次完整 default-panel
端到端审计（168 Linear + 120 Attention，168 Weight calibration + 24 Attention calibration，
六个公开 API 均调用）：default NVFP4 输入缓存首次构建 `19.321546s`、大小 `2,872,472,567`
字节；完整测试命中缓存的准备阶段为 `1.185913s`，API 总计 `617.842032s`，candidate wall
`669.348815s`。本地 proxy 为 Linear `0.570268537`、Attention `0.724718506`、overall
`0.634622690`；Linear 168 case 中 `166/2/0`（正/负/零），median `0.572989`，worst-quartile
mean `0.309062`，最差 `-0.562535`。该结果的 official score/time 为 `unregistered/NA`，
本地秒数不能换算官方 `<300s`，因此只作为完整调用图和缓存有效性的审计证据，不改变当前
v158 官方基线。证据见 [`Attention cache build`](../artifacts/official_eval/nvfp4-cache-attention-build-check.json)、
[`Attention cache hit`](../artifacts/official_eval/nvfp4-cache-attention-hit-check.json) 和
[`完整 default audit`](../artifacts/official_eval/root-nvfp4-full-20260902.json)。

## 3. 历史 v1 结果表（不可与 proxy-v2 混用）

下表保留旧 `official-shape-v1` 的同机数字，仅用于审计此前的失真；当前 proxy-v2 分层 panel
复测不覆盖这些历史值。

| 版本 | Linear mean | Attention mean | API(s) | Wall(s) | 官方结果 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| v84 | 0.406668 | 0.718107 | 279.191 | 300.848 | 16517 / 252.563s | 官方通过 |
| v86 | 0.406668 | 0.719696 | 299.302 | 321.996 | 16744 / 222.7s | 上一代基线 |
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
| v155 | **0.570999** | **0.724735** | **248.121** | **280.763** | **16581 / 208.5s** | **REJECTED；时间通过但低于 v86 163 分** |
| v156 | 0.588131（effect） | 0.757433（effect） | 203.994（effect） | 216.749（effect） | **16580 / 204.3s** | **REJECTED；时间通过但低于 v86 164 分** |
| v157 | NA（仅合法性） | NA（Attention 字段级一致） | NA | NA | **16729 / 218.96s** | **REJECTED；时间通过但低于 v86 15 分** |
| **v158** | **0.448180（default；冻结）** | **0.735752（default）** | **295.069** | **325.896** | **16861 / 223s** | **RETAINED；相对 v86 +117 / +0.3s** |

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
- ROAB-P2：虽然固定 reduced Attention 的 `v138→v140 = +123`，但 exact-v86 单变量 v157
  官方为 `16729 / 218.96s`、低于 v86 `15` 分，证明收益不可迁移，整个路线关闭；
- v141–v145 非对称选列 BDLR、锚点冻结和阻尼变体；
- v128–v131 动态 Q/K Gram、PAWV 和随序列放大的 Attention 搜索；v161（S1 交叉算子
  Gram64 per-call 精化，v160 坐标系干净移植）官方再次 timeout，证明该族超时根因是
  动态 per-call 精化在官方硬件的成本而非校准搜索，全族结构性关闭；
- 增加 alpha、offset、sweep、block 数、阻尼、角度或候选槽位的局部扫描。

这些路线要么官方超时，要么官方分数低于 v86，要么只有固定本地 panel 上的 `10^-5–10^-4`
级差值，不能支撑继续投入。

## 6. 当前活动计划

唯一活动计划是
[`2026-09-03-official-side-weight-calibration-plan.md`](superpowers/plans/2026-09-03-official-side-weight-calibration-plan.md)：
官方两侧分数比重校准实验（v161 timeout 后本地已知机制族全部闭环，本实验用官方回传
确定下一优化方向的边际依据）。要点：

1. **v162 全标准基线**：六 API 镜像 reference codec，本地两侧 mean 精确 0.0，官方回传
   `S(v162)` 为标准行为锚点；
2. **v163**（v160 Linear + 标准 Attention）测 Δ_L，**v164**（标准 Linear + v160
   Attention）测 Δ_A；保留侧与 v160 逐位一致（构建零改动 + case 级验证）；
3. 预注册可加性检验：`S(v163)+S(v164)−S(v162) ≈ 17532`；判读表见计划 §3；
4. 三个版本各一次官方提交（用户执行，顺序 v162 → v163 → v164），回传只记录判读，
   不围绕结果调参。

当日已归档：Attention per-call 序列自适应精化计划（v161 官方 timeout，per-call 动态族
关闭）、Attention 解析式宽域计划（A1a 4×4 REJECTED、A2 无病因、A3 未启动）、
Householder 快速验证计划（全族 REJECTED）。Linear 侧 T<d 秩亏伪增益通道结构性封闭。
实验结束后本计划归档，比重结论写入本文件。

## 7. 归档现状与待整理项

已完成：

- v128–v131 的 `result.md` 和目录名均标记 `TIMEOUT`；
- v135–v137 的 `result.md` 和目录名均标记 `REJECTED`；
- v132/v133 已补齐 `RETAINED / LOCAL HISTORICAL PARENT` 结果文件；
- v134 标记为 `RETAINED / LOCAL RESEARCH PARENT`，不代表官方可提交；
- v140 ROAB-P2 改为 `REJECTED / LOCAL-ONLY`，归档目录标记 `_rejected`；
- 空的重复 v140 curvature 目录已删除；
- v141–v145 失败源码目录删除，逐次 JSON/日志保留。
- v155 L5a permutation-stability 已按官方 `16581 / 208.5s` 保存为 `REJECTED` 并将目录改为
  `_rejected`；它不改变 root。
- v156 L4-WD 已按官方 `16580 / 204.3s` 保存为 `REJECTED` 并将目录改为 `_rejected`；本地
  微增益未迁移，停止 stored-scale 路线。
- v157 exact-v86 + ROAB-only 已按官方 `16729 / 218.96s` 保存为 `REJECTED` 并将目录改为
  `_rejected`；六 API 与不变量虽通过，但精度低于 v86，ROAB 路线关闭。
- v158 exact-v86 + Attention Matrix-Smooth 已按官方 `16861 / 223s` 保存为 `RETAINED` 并将
  目录改为 `_retained`；它是当前仓库内官方可复现基线。

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
