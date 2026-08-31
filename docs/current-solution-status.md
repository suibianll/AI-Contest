# 当前主版本算法效果与评测状态

> 更新日期：2026-08-31
> 适用文件：根目录 [`solution.py`](../solution.py)
> 文档性质：本地可复现实测记录，不是官方成绩承诺。

> **评测协议 v5（2026-08-31 修订）**：本文的当前主结果只读取
> `sampled-means-v1` 的 `Linear mean` / `Attention mean`。旧 full-layer、native
> sum、shaped panel 和本地 `<300s` 判断均为 legacy，不能与 v5 主表混用。统一
> 口径、官方锚点拟合和时间校准见 [`local metric calibration`](../logs/execution/2026-08-31-local-metric-calibration.md)。
> 官方评测（2026-08-31 再次修订）不再限制任何 `A@W` 拟合用法，只限制端到端
> 运行时间（`<300s`）。

## 一句话结论

根目录当前为 v127：v106 Linear 路径 + PAWV 变长修复。固定 v4 sample plan（Qwen、
seed `20260831`、层 `[0,1,5,10,13,15,22,23]`、全部 role、4 windows）下，
Linear mean `0.509408`、Attention mean `0.828395`、Local API `151.136s`、
Wall `161.840s`；同一计划 v74 为 `0.440305 / 0.671106 / 218.619s / 229.485s`。
以下 v125 等数字是 legacy precision parent，不覆盖 v4 主结果。算法链为 BOAT +
expansive-FFN CAT balance + cross-fold HSDQ + Gram-hierarchy Activation-HSDQ，
保留 Attention B1 GQRB 与 B2 PAWV diag-only（v127 变长修复）；L3–L6/C1 的实验机制
已于 2026-08-31 从根文件裁剪，只保留在归档与历史日志中。v125 的全量
`295.847849 / 2653.580s` 仅作为 legacy precision 证据；v127 的 v4 sampled
均值才是当前比较口径。v126/v127 的 PAWV 变长修复已通过公开 shape smoke，
但 v127 尚未官方提交，不能把本地均值写成官方成绩。官方 300s 只由官方平台确认，
本地 CPU/CUDA 时间不作为硬门。

> **2026-08-31 归档修复与 v5 复评**：v099–v125 共 28 个归档 `solution.py` 携带着
> B2 PAWV 变长 calibration bug（v100/v107 官方 WA 的直接根因），已按 v127 逻辑批量改为
> 按长度分组的 keyed diagonal，全部通过官方长度 `[10,128,512,1024,1024]` 形状复现。
> 修复后按 v5 `sampled-means-v1` 复评：v100-pawv-fixed `0.506715 / 0.828395 / 150.25s`、
> v107-pawv-fixed `0.512967 / 0.828395 / 241.51s`、v121-pawv-fixed
> `0.516685 / 0.828395 / 832.92s`；v107/v121 精度高于 v127 但时间不可行。官方分类不改变
> （v100/v107 官方 WA、v121 官方 timeout）。完整见
> [`pawv 归档修复与 v5 复评`](../logs/execution/2026-08-31-pawv-archive-fix-and-v5-reeval.md)。

当前唯一活跃计划是 [`2026-08-31-hif4-active-c1-structured-linear-plan.md`](superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md)；L6、C1a、C1b、C1c 已完成，下一步为 C2 低成本跨模型 guardrail 与 C3 state/time 压缩；归档候选的写回、目标错位和源码缺失审计见 [`archive-implementation-audit.md`](archive-implementation-audit.md)，v107 Attention 合约对照见 [`2026-08-31-v107-attention-contract-audit.md`](../logs/execution/2026-08-31-v107-attention-contract-audit.md)。

针对官方 v107 `Attention / wrong answer` 的同输入对照已完成：[`v107-v31-v51-external-attention-output-diff.md`](../logs/execution/2026-08-31-v107-v31-v51-external-attention-output-diff.md)。在 24 层 Qwen cache、2 calibration、4 test windows、同一 NVFP4 codec 下，v31/v51/归档外部 v002/v107 的 state、五字段 API、shape、CPU/finite 检查均为 0 failures（每版本 72 states、96 batches、288 个 Q/K/V 输出）；Attention MSE mean 分别为 `0.00382519 / 0.00382519 / 0.00529873 / 0.00169248`，v107 反而最低。v31 与 v51 24/24 层逐位相同，v31 与外部 v002 12/24 层相同，v107 因有意新增 Linear 状态而与 v31 0/24 层相同；目前没有证据表明 v107 Attention 输出契约损坏。外部逐输出数字代表本地归档 v002，不等同于最新 v2.7 源码；官方隐藏输入仍需 v106/v107 同包复测。

后续官方反馈已推翻上述候选裁决：用户确认 v100 同样为 Attention `wrong answer`，且
不是 timeout。新的 [`v100 官方 WA 边界审计`](../logs/execution/2026-08-31-v100-official-wa-boundary-audit.md)
表明 v72 的四个 Attention API、45 个递归可达 helper 和相关常量均与官方通过的 v66
语义一致；v72 本地 Qwen native `356.605602`、Attention `63.119717`、CUDA API
`163.41s`。用户随后确认 v72 官方运行成功，成绩 **`22662 / 226s`（旧权重口径）**，相对 v66
提升 `105` 分并增加 `8.8s`，因此 v72 当时从“增强候选”升级为官方通过基线，
v66 仍为绝对控制组。v74 虽已改变 Attention 共用 helper，但用户随后确认其官方
**`22750 / 239.387s`（旧权重口径）** 正常通过，相对 v72 `+88` 分、`+13.387s`，因此当时安全边界
进一步前移到 v74。v75 起直接改变 Q/K 路径，尚无官方通过证据；v100 的 PAWV
旧路径仍为官方 WA。官方结果记录见
[`v74 official pass`](../logs/execution/2026-08-31-v74-official-pass.md)。
注：v72/v74 官方分数均为**旧评分权重**；2026-08-31 晚新权重官方锚点为 v84
`16517 / 252.563s`（见 [`v84 官方结果`](../logs/execution/2026-08-31-v84-official-result.md)）。

**最新官方裁决（2026-08-31 再次修订）**：官方端到端超时限制收紧为 300s，且不再
限制任何 `A@W` 拟合用法。用户确认 **v98 在最新限制下官方判为超时**（本地 API
`406.24s` > 300s），官方结果更新为 `timeout`，v98 不再作为提交候选。此前误记的
v107 timeout 已纠错：**v107 官方结果保持 Attention `wrong answer`（非 timeout，
与 v100 同类）**，其本地 API `481.04s` > 300s 仅作历史风险提示。更新详情见
[`v98 官方 timeout`](../logs/execution/2026-08-31-v98-official-timeout.md)。

**2026-08-31 晚第三次修订（评分权重变更）**：官方**减少了 Linear 样例的评分权重**，
官方总分据此大幅下降；新权重下用户确认 **v84 官方通过：`16517 / 252.563s`
（< 300s）**，是新评分权重的当前官方锚点。v72/v74 等旧权重官方分数（2 万+）仅作
历史参考，与新权重不可互相换算；官方未提供两项权重系数，本地不复制 case 拟合
官方绝对分。详见
[`v84 官方结果`](../logs/execution/2026-08-31-v84-official-result.md)。

进一步按任务书复核并构造变长 calibration list 后，已稳定复现 v100/v107 的直接异常：
B2 PAWV 的 `_build_pawv_metric` 按首个样本长度建立固定 `tokens×tokens` 矩阵，却直接
累加不同长度样本的 `P^TP`。`seq=32/48` 时 v72/v98 通过，v100/v107 均报 shape
mismatch；任务书没有 calibration sample 等长约束，且规定任一运行异常即提交失败。
官方自测进一步给出 calibration 长度 `[10,128,512,1024,1024]` 和精确异常
`size 10 must match size 128 at dimension 1`，因此该根因已被确认，而不再只是高置信度推测。见
[`v100/v107 Attention WA 根因`](../logs/execution/2026-08-31-v100-v107-attention-wa-root-cause.md)。
修复实现与验证见 [`v126 PAWV 变长修复`](../logs/execution/2026-08-31-v126-pawv-variable-length-fix.md)。
v4 sampled 的可比结果见 [`v127 sampled`](../logs/execution/2026-08-31-v127-sampled-means-qwen.md)
和 [`v74 sampled`](../logs/execution/2026-08-31-v74-sampled-means-qwen.md)；统一统计、
官方锚点拟合和时间校准见 [`local metric calibration`](../logs/execution/2026-08-31-local-metric-calibration.md)。

2026-08-31 已按执行计划完成 E0-C、E1→A6、B1、B2、L1、L2、L3 和 L4a。B1 GQRB margin
先把 panel 提升到 `293.793700`，B2 PAWV diag-only 再提升到 `293.797301`，L2
expansive-FFN CAT balance 将 panel 提升到 `294.272633`、Linear mean 提升到
`0.503458942243`；L3 修复 v095 gate 后，4-block precision parent 的 panel 为
`295.157057`、Linear mean `0.5069966356`，较 v106 分别 `+0.884423` 和 `+0.003538`；
L4a 精确 final-Gram 行级 gate 再提升到 panel `295.239309`、Linear mean
`0.5073256468`，较 v107 分别 `+0.082253` 和 `+0.000329`。官方评测不可用
期间，所有新候选仍以固定 Qwen panel 为门禁。L1 已完成真正的 scale/lv2/lv3/mantissa
原子写回与合成测试，但五层×七 role 预筛与 L0 逐条持平（`0.523019429222563`），
因此候选 v105 已归档；v106 是时间 parent，v107/v109/v110 是前一精度 parent，v111
是前一精度 parent；v115 L6a、v116 L6b、v117 L6c、v118 L6d、v119 C1a、v121 C1b、v124 C1c 与 v125 C1c/max-blocks=8 均通过 full-layer，v125 成为当前 precision-only parent；其 API 超过 300s（按最新官方限制），下一步转入 C2/C3，不再增加 C1c block budget。

## 当前实现

`solution.py`（v127）只保留六个正式 API 和必要的 codec/优化原语：

1. **BOAT**：用激活/权重各自的 RMS 构造对角平衡 `D`，再搜索 4/8/16/64 维
   signed-Hadamard 块和两个确定性 seed。连续乘积保持不变：

   $$X'=XD^{-1}R,\qquad W'=W D R,\qquad X'W'^T=XW^T.$$

   不构造 Linear 输出，因此固定参数可以安全写入 `activation_state`。
2. **Cross-fold Weight-HSDQ**：对满足宽度/形状条件的权重块使用校准激活
   `A_f^T A_f` 的低秩 Hessian，对 HiF4 的 15 个 signed levels 做精确二次增量
   搜索；fold 1 生成的候选必须改善 fold 2，最终只改变离线 `weight_params`。
3. **Gram-hierarchy Activation-HSDQ**：从静态变换后权重计算 64 维 Gram block，
   先按二次型选择层级和 E6M2 offset，再做最多 128 个 block、2 轮坐标扫描。状态
   只保存 CPU 上的静态 `gram64`、BOAT 逆缩放和整数/符号配置。
4. **Expansive-FFN CAT balance**：只对 `weight_rows > weight_channels` 的结构形状，
   使用固定 `α=0.25` 的 RMS 对角 balance；不依赖 role-id/模型名，不增加 state 字段，
   operand-local proxy 不优于 BOAT 时回退 parent。
5. **Attention 输出感知 shortlist**：搜索 reciprocal RMS 平衡、K-centering、
   16/32/64 维共享 signed-Hadamard，以及 B1 GQRB 的 2×2/4×4 group-local
   orthogonal mixing；保留 parent 的原始四候选，并要求 mixing exact loss 至少
   改善 0.1% 才能替换。B2 PAWV 用 attention probability 的 token-row 对角
   Hessian 做 V 的离散坐标 refinement；V 仍保持独立合法 HiF4 编码。B2 PAWV
   diag-only 的旧实现已确认存在变长崩溃。v126/v127 改为逐样本直接计算 diagonal、
   按 `seq_len` 分组平均，校准/在线 V 精确查找当前长度，无匹配则回退；并删除无用的
   full `P^TP` 与 `eigh`。该修复解决正确性与复杂度问题，完整精度仍待重测。

> **已裁剪机制（2026-08-31）**：L3 Global Activation-LRH、L4a final deployed-Gram
> row gate、L4b GALS、L5a block-local permutation、L6a–L6d（rank-16 / wide rank-4 /
> `G_64` hierarchy / structured factor）与 C1a–C1c（向量化、refresh×2、rank-8）已全部
> 从根 `solution.py` 移除，仅保留在 `solutions/` 归档与下文历史记录中；它们作为
> C2/C3 压缩阶段的精度上界证据，不代表当前根文件行为。

## 最新全层实测（legacy precision parent v125，非当前根行为）

报告文件：[`2026-08-31-v125-c1c-block8-qwen-full.md`](../logs/execution/2026-08-31-v125-c1c-block8-qwen-full.md)；
原始 JSON：[`v125-c1c-block8-qwen-full.json`](../artifacts/real_model_suite/v125-c1c-block8-qwen-full.json)。
v106 时间 parent 对照：[`v106-l2-cat-qwen-full.md`](../logs/execution/2026-08-30-v106-l2-cat-qwen-full.md)。
上一 parent 的对照报告：[`b2-pawv-diagonly-qwen-full.md`](../logs/evaluations/b2-pawv-diagonly-qwen-full.md)。

> 下表为 v125（含已裁剪的 L4a/L4b/L5a/L6/C1 机制）在旧 full-layer 口径下的历史数据，
> 仅作精度上界证据；当前根 v127 的主口径是 v4 sampled-means 的两个均值。

固定输入为 Qwen2.5-0.5B（24 层、hidden 896、14 Q heads、2 KV heads、head dim 64），
calibration 使用 train 的 2 个窗口，test 使用 validation 的 4 个不重叠窗口。

| 指标 | 当前 v125 | v124 | 相对 v124 |
|---|---:|---:|---:|
| Linear native mean | **0.509760** | 0.509649 | **+0.000110** |
| Attention native mean | 0.842039 | 0.842039 | 0 |
| Qwen panel Linear | **127.439951** | 127.412331 | **+0.027620** |
| Qwen panel Attention | 168.407898 | 168.407898 | 0 |
| Qwen panel total | **295.847849** | 295.820229 | **+0.027620** |
| official-flow native total | **423.394380** | 423.320136 | **+0.074251** |
| six-API time | 2653.580314 s | 2323.911178 s | +329.669136 s |
| wall time | 2686.541758 s | 2356.200547 s | +330.341211 s |

五模型 C0 确认报告：[`2026-08-30-c0-b2-pawv-five-model.md`](../logs/evaluations/2026-08-30-c0-b2-pawv-five-model.md)。
v125 Qwen panel `295.847849`、Linear `0.509760`、Attention `0.842039`、
API `2653.580s`；该 precision-only parent 暂不满足最终 300s 冻结条件。相对 v124 的增益来自
C1c `max_blocks=8`，Attention 与 v100/v106/v107 逐位不变；v98 的官方结果已在最新
300s 限制下确认为 timeout，v107 官方保持 Attention WA（用户确认非 timeout），合约
审计未发现数值/字段差异的分析保留为历史解释。
gpt2-small/OPT/Pythia 的旧 parent API 分别为
`196.975s/192.776s/193.423s`，gpt2-medium 为 `492.641s`（仅软 guardrail 时间
超限，未影响 Qwen 主门禁）。五模型 aggregate panel `263.604453` 仅作泛化诊断。

## 测试分层与速度结论

当前 Qwen screen（5 层 × 7 Linear role，固定两折 calibration）约 **529.3s wall**；
Qwen 全 24 层约 **2686.5s wall / 2653.6s API**，也就是一次完整候选约 44.8 分钟。
因此后续候选固定采用：先做合成/合规与 Qwen screen，只有 screen 正向且无 role 回退才做
Qwen full；Qwen full 是本地精度排序的唯一硬门。OPT/Pythia 不再对每个候选跑五模型全量，
只在 precision parent 变更后或每 2–3 个候选做 3–5 层软 guardrail，这能把大多数无效候选
的成本压到一次 screen。

“只用 Qwen 全量测试”可以作为本地 A/B 排序规则，但不能作为官方发布充分条件：发布前仍需
Attention API smoke、state/shape/compliance 检查，以及至少一次跨模型回归。v107 的官方
Attention `wrong answer` 已通过本地函数体/状态/输出差分排除明显数值改动，且用户确认非
timeout（官方 v98 才是最新 300s 限制下的 timeout，二者是两类失败）；WA 直接根因随后由
官方自测确认是 B2 PAWV 变长 calibration shape mismatch。

## 外部代码本地复测与最高基准

外部 [`youxilee/hif4`](https://github.com/youxilee/hif4) 的 v2.7 提交
`dd5ee6515323169dbd4133b3d4fd1ff1cb7be646` 已在本地用未修改源码复测。直接走
CUDA 会在外部代码的 CPU state / CUDA activation 混用处触发 device mismatch，因而
以下结果采用 `device=cpu, algorithm_device=cpu`，固定
`amax6 / seq=128 / calib=2 / test=4 / cache=read`。这是一组代理诊断，不是官方
评测复现；逐模型证据见
[`外部 v2.7 本地差距审计`](../logs/candidates/2026-08-29-external-hif4-gap-analysis.md)。

| 外部本地模型 | Linear | Attention | native total | API(s) | 基准含义 |
|---|---:|---:|---:|---:|---|
| gpt2-small | 150.431816 | 24.250501 | 174.682317 | 108.66 | 软 guardrail |
| gpt2-medium | 258.627694 | 48.719212 | 307.346907 | 320.94 | 软 guardrail |
| opt-125m | 28.245796 | 19.574717 | 47.820513 | 106.06 | 软 guardrail |
| pythia-160m | 145.543961 | 40.822630 | 186.366591 | 108.08 | 软 guardrail |
| **qwen2.5-0.5b** | **303.581186** | **65.946083** | **369.527269** | **357.67** | **最高单模型 native 基准** |
| 五模型相加 | 886.430453 | 199.313144 | 1085.743597 | — | **仅诊断，禁止作排名分** |

本地排名必须使用最高的**同模型、同面板口径**，而不是把不同模型层数直接相加：

1. **最高单模型 native**：外部 Qwen `369.527269`；它是五个代理中最高值，且
   结构上最接近新增的 GQA/RoPE/SwiGLU 压力用例。
2. **最高同口径 panel（主比较基准）**：外部 Qwen `250.327102`，由
   `250 × (303.581186 / 672) + 200 × (65.946083 / 96)` 得到；五模型合计不参与
   这个投影。
3. **最高官方基准**：用户确认的外部官方 `24153 / 239s`。本地 native/panel
   没有官方缩放因子，不能把 `369.527269` 线性换算成 `24153`。

当前根与外部最高本地基准的差值如下：

| 比较口径 | 当前根 | 外部最高基准 | 当前根领先 |
|---|---:|---:|---:|
| Qwen native total | **423.394380** | 369.527269 | **+53.867111（+14.58%）** |
| Qwen shaped panel | **295.847849** | 250.327102 | **+45.520747（+18.19%）** |
| panel Linear | **127.439951** | 112.939429 | **+14.500522** |
| panel Attention | 168.407898 | 137.387673 | **+31.020225** |

因此，后续本地算法 A/B 应以外部 Qwen `250.327102` 作为第一比较线，
以外部 Qwen native `369.527269` 作为第二诊断线；`1085.743597` 只能用于检查
跨模型结构性回退，不能作为“外部最高分”或与官方 `24153` 做差值。

当前版本的官方流程诊断由 672 个 Linear case 和 96 个 Attention case 求和得到：

| 组件 | case 数 | gain sum | gain mean | global gain |
|---|---:|---:|---:|---:|
| Linear | 672 | 342.558589 | 0.509760 | 0.446685 |
| Attention | 96 | 80.835791 | 0.842039 | 0.857899 |
| 合计 | 768 | **419.160200** | — | — |

`panel_score` 不是把 768 个 case 复制成 450 个，而是保留组件均值后投影：

$$P_L=250\times0.5097598050=127.439951,$$

$$P_A=200\times0.8420394885=168.407898,$$

$$P_{total}=P_L+P_A=295.847849.$$

因此 `official_flow_total` 与 `panel_score.total` 同时出现是设计结果，不是计算冲突。

## Linear 角色归因

| 角色 | native mean |
|---|---:|
| q | 0.616758 |
| k | 0.629137 |
| v | 0.571384 |
| o | 0.498290 |
| fc_gate | **0.395579** |
| fc_up | 0.433860 |
| proj | 0.423311 |

`v` 与 `fc_gate` 是当前最弱角色；L2 CAT balance 已改善 fc_gate，v125 的 `max_blocks=8`
进一步改善了 `proj`，但从总体收益看，扩张 FFN 和输出投影仍受跨
64-block 相关性、校准 fold 数量和运行时约束共同限制，不能仅靠增加 offset 或 sweep
解决。

## 2026-08-30—31 执行计划结果

以下均为同一 Qwen2.5-0.5B、24 层、`cache=read` 的本地 shaped panel；parent
永远保留，候选失败后已恢复。详细 fold 与角色数据见各 execution log。

| 实验 | panel | Linear mean | API time | 裁决 |
|---|---:|---:|---:|---|
| E0/D0 多模型 scale-lattice oracle | — | — | 13.76–14.84s/模型 | 完成诊断；scale gap 亚百分比，无跨模型统一增益 |
| E0-C GALS-C 稀疏 activation（layer-1） | 335.988995 | 0.602878 | 57.41s | 拒绝；解析召回 oracle `1.0`，部署版回退 `0.048096` |
| A7 量化后权重 Gram `WqᵀWq`（layer-1/full） | 336.562922 / 290.226694 | 0.605174 / 0.487275 | 24.89s / 470.58s | 拒绝；单层正向不迁移且全层超时 |
| L1 full-hierarchy cross-block Weight-LRH（v105） | 0.523019（五层×七 role screen） | — | 265.87s screen | 拒绝；70 个 fold 候选仅 1 个 cross-fold admitted，最终 0/35 case 改变 parent；未触发 full-layer |
| L2 expansive-FFN CAT balance（v106） | **294.272633** | **0.503459** | **412.65s** | **采纳；fc_gate +0.013309，较 v100 panel +0.475332，API 按当时 420s 限制达标** |
| E1 progressive full-hierarchy | 290.923906 | 0.490233 | 693.21s | 拒绝，跨层回退且超时 |
| A2 expansive sparse-row | 292.831952 | 0.497865 | 385.48s | 拒绝 |
| A3 rowwise block-leverage | 293.250467 | 0.499539 | 384.83s | 拒绝 |
| A4 blockwise BOAT-2 | 292.978009 | 0.498449 | 368.23s | 拒绝 |
| A5 joint-fold offline A@W | 284.595177 | 0.464918 | 358.24s | 拒绝 |
| A3 true cross-block LRH-r8 | 292.426982 | 0.496245 | 381.84s | 暂不采纳；已由 v105 正确 hierarchy 写回复验，screen 无增益 |
| A4 full CAT-inspired BOAT-2 | 283.159693 | 0.459176 | 600.61s | 拒绝，超时 |
| A5 frozen-Q(A) ridge/Qronos | 293.755106 | 0.501558 | 455.73s | 持平但超时 |
| A6 Global Activation-LRH | 282.616646 | 0.457010 | 373.97s | 暂不采纳；全局 Gram 目标与最终 MSE gate 错位，需修复复验 |
| B1 GQRB margin | 293.793700 | 0.501558 | 406.24s | archived baseline |
| B2 PAWV diag-only + B1 GQRB（v100） | 293.797301 | 0.501558 | 392.42s | previous parent |
| C0 五模型确认（无代码变更） | **293.797301** | **0.501558** | **401.13s（Qwen）** | **confirmed** |
| **L2 expansive-FFN CAT balance（v106 当前根）** | **294.272633** | **0.503459** | **412.65s** | **active** |
| **L3 Global Activation-LRH Gram gate（v107 前一 parent）** | **295.157057** | **0.506997** | **481.04s** | 前一精度 parent |
| **L4a final deployed-Gram row gate（v109 当前 parent）** | **295.239309** | **0.507326** | **517.29s** | **精度采纳；L4b 继续探索，时间暂不作为探索门禁** |
| **L4b final-Gram GALS（v110 前一 parent）** | **295.242780** | **0.507340** | **701.90s** | **精度采纳；已被 L5a 超越** |
| **L5a block-local permutation（v111 前一 parent）** | **295.482473** | **0.508298** | **726.09s** | **已被 v115 超越** |
| **L6a rank-16 global LRH（v115 前一 parent）** | **295.680651** | **0.509091** | **716.48s** | **精度采纳；已被 L6b 超越** |
| **L6b wide rank-4 cross-block factor（v116 前一 parent）** | **295.734045** | **0.509305** | **739.42s** | **精度采纳；已被 L6c 超越** |
| **L6c full `G_64` hierarchy coordinate sweep（v117 前一 parent）** | **295.785829** | **0.509512** | **2019.48s** | **精度采纳；已被 v118 超越** |
| **L6d structured block-circulant factor（v118 前一 parent）** | **295.808212** | **0.509601** | **2249.75s** | **精度采纳；已被 C1a 语义等价版本超越** |
| **C1a structured proposal vectorization（v119）** | **295.808212** | **0.509601** | **2040.50s** | **精度逐位等价 v118；API −9.30%；已被 v121 超越** |
| C1b block refresh（v120 screen） | 0.533373 screen | 0.533373 screen | 419.63s screen | rejected；低于 v118 screen `0.5333753185` |
| **C1b structured refresh×2（v121）** | **295.811281** | **0.509614** | **2180.45s** | **full-layer 正向；用户确认官方 runtime timeout；仅保留精度证据** |
| C1c rank-2（v122 screen） | 0.533363 screen | 0.533363 screen | 425.70s screen | rejected；低于 v118 screen |
| C1c max-blocks-2（v123 screen） | 0.533352 screen | 0.533352 screen | 429.95s screen | rejected；低于 v118 screen |
| **C1c rank-8 / max-blocks-8（v125 当前 precision-only）** | **295.847849** | **0.509760** | **2653.58s** | **full-layer 正向；较 v124 panel `+0.027620`；runtime invalid** |
| L3 1-block 对照（v107b1） | 294.483738 | 0.504303 | 446.29s | 低于 v107，不作为 parent |
| stable parent | 293.755106 | 0.501558 | 382.15s | baseline |

归档目录：`solutions/20260830_v087...` 至 `solutions/20260830_v106...`；
执行日志：`logs/execution/2026-08-30-e1-progressive-hsdq.md`、
`2026-08-30-a2-expansive-sparse-hsdq.md`、
`2026-08-30-a3-rowwise-block-hsdq.md`、
`2026-08-30-a4-blockwise-boat.md`、
`2026-08-30-a5-joint-aw.md`、
`2026-08-30-a3-lrh-r8.md`、
`2026-08-30-a4-cat-boat2.md`、
`2026-08-30-a5-frozen-qronos.md`、
`2026-08-30-a6-global-activation-lrh.md`、`2026-08-30-b1-gqrb.md`、
`2026-08-30-b2-pawv.md`、`2026-08-30-c0-five-model.md`、
`2026-08-30-e0c-gals-candidate.md`、`2026-08-30-e0g-multimodel-dashboard.md`、
`2026-08-30-a7-quant-weight-gram.md`、`2026-08-30-l1-lrh-stratified.md`、
`2026-08-30-l1-full-hierarchy-lrh.md`、`2026-08-30-l2-cat-stratified.md`、
`2026-08-30-l2-expansive-cat.md`、`2026-08-30-v106-l2-cat-qwen-full.md`、
`2026-08-31-l4a-final-gram-corrected-stratified.md`、
`2026-08-31-l4a-final-gram-gated-stratified.md`、
`2026-08-31-l4b-gals-final-stratified.md`、
`2026-08-31-l4b-gals-final-gated-stratified.md`、
`2026-08-31-v109-l4a-final-gram-gated-qwen-full.md`、
`2026-08-31-v110-l4b-gals-final-gated-qwen-full.md`。
L3 证据：`2026-08-30-l3-global-lrh-stratified.md`、
`2026-08-30-l3-global-lrh-diagnostic.md`、
`2026-08-30-v107-l3-global-lrh-qwen-full.md`、
`2026-08-30-l3-global-lrh-b1-stratified.md`、
`2026-08-30-l3-global-lrh-b1-diagnostic.md`、
`2026-08-30-v107b1-l3-global-lrh-qwen-full.md`。

## 当前目标与本地时间推断（2026-08-31 更新）

> **目标变更**：官方第三次修订（减少 Linear 样例权重）后，旧权重口径的 `36000`
> 绝对分目标已废弃。当前唯一目标组合是：
>
> 1. **Linear 场景本地 `linear_mean` 达到 `0.8`**（v4 sampled 口径；当前 v127
>    `0.509408`，研究链最高 v121-pawv-fixed `0.516685`）；
> 2. **Attention 场景尽可能高**（pawv-fixed 系当前上限 `0.828395`，继续寻找无损
>    或低时间成本的 Attention 增益）；
> 3. **官方端到端时间 `< 300s`**——本地时间预算按下表由已有官方评测结果推断。

### 本地时间 → 官方时间推断表

同一批有官方记录的候选已在 v4 sampled 口径（224L+32A、CPU、`cache=read`）复评；
"比值"= 官方端到端时间 ÷ 本地 sampled API 时间。官方 450 case 与本地 256 case 的
构成差异、鲲鹏 920B 负载波动都会造成比值漂移，因此只给出区间与预算红线：

| 候选 | 官方时间 (s) | 本地 sampled API (s) | 比值 |
|---|---:|---:|---:|
| v031 / c39 | 161.3 | 80.500 | 2.00 |
| v034 / c41b | 159.4 | 79.094 | 2.02 |
| v051 / c47b | 234 | 116.557 | 2.01 |
| v066 / c66 | 217.2 | 187.353 | 1.16 |
| v072 / C74 | 226 | 228.777 | 0.99 |
| v074 / C75 | 239.387 | 218.619 | 1.10 |
| **v84 / C84** | **252.563** | **422.615** | **0.60** |
| v098 | timeout（>300s） | 219.039 | — |

**推断规则（用于预算规划，不替代官方判定）**：

- 比值观测区间为 **`[0.60, 2.02]`**（v84 证明本地很慢的候选在官方硬件上可能
  远快于本地；c39 系早期候选官方相对最慢）。按最保守上界 `2.02` 反推，
  **本地 sampled API `≤ 150s` 是官方 `<300s` 的安全预算红线**（150 × 2.02 ≈ 303s）。
- 本地 `150–450s` 为灰区：官方通过（v84 `422.6s → 252.6s`）与 timeout
  （v098 sampled `219s` 仍官方 timeout）都出现过，必须提交官方实测。
- 本地 `>450s` 基本不可行（v121-pawv-fixed 本地 `832.9s`，官方 timeout）。
- 当前 v127 sampled API `151.136s` 恰在安全红线边缘；Linear 0.8 目标允许精度
  候选先超预算探索，但进入提交冻结前应把 sampled API 压回 `≤150s`，或以最接近的
  官方已测版本（如 v84）做结构对比后再提交实测。
- 该推断线只约束提交冻结阶段；探索阶段仍按 accuracy-first 只记录时间。

### 官方记录候选的 v5 复评覆盖矩阵

全部 11 个有官方记录的本仓库候选均已在 v4 sampled 口径（seed `20260831`，
224 Linear + 32 Attention、CPU、`cache=read`）完成复评：

| 候选 | 官方结果 | v5 Linear mean | v5 Attention mean | 本地 API (s) | 复评日志 |
|---|---|---:|---:|---:|---|
| v031 / c39 | 21864（旧权重） | 0.439775 | 0.667092 | 80.500 | [`official-anchors-sampled`](../logs/execution/2026-08-31-official-anchors-sampled.md) |
| v034 / c41b | 21864（旧权重） | 0.439775 | 0.667092 | 79.094 | 同上 |
| v051 / c47b | 22451（旧权重） | 0.433744 | 0.667092 | 116.557 | 同上 |
| v066 / c66 | 22557（旧权重） | 0.432060 | 0.671106 | 187.353 | 同上 |
| v072 / C74 | 22662 / 226s（旧权重） | 0.432117 | 0.671106 | 228.777 | [`v072-sampled`](../logs/execution/2026-08-31-v072-sampled.md) |
| v074 / C75 | 22750 / 239.387s（旧权重） | 0.440305 | 0.671106 | 218.619 | [`v74-sampled`](../logs/execution/2026-08-31-v74-sampled-means-qwen.md) |
| **v84 / C84** | **16517 / 252.563s（新权重）** | **0.477266** | **0.709020** | **422.615** | [`v84-sampled`](../logs/execution/2026-08-31-v84-sampled-means-qwen.md) |
| v098 | timeout | 0.506715 | 0.828323 | 219.039 | [`v098-sampled`](../logs/execution/2026-08-31-v098-sampled.md) |
| v100 | Attention WA | 0.506715 | 0.828395 | 150.25 | [`v100-pawv-fixed-sampled`](../logs/execution/2026-08-31-v100-pawv-fixed-sampled.md) |
| v107 | Attention WA | 0.512967 | 0.828395 | 241.51 | [`v107-pawv-fixed-sampled`](../logs/execution/2026-08-31-v107-pawv-fixed-sampled.md) |
| v121 | timeout | 0.516685 | 0.828395 | 832.92 | [`v121-pawv-fixed-sampled`](../logs/execution/2026-08-31-v121-pawv-fixed-sampled.md) |

说明：v100/v107/v121 的复评使用 PAWV 变长修复版归档（`-pawv-fixed`），原始归档
携带变长 bug 不再可运行；v024（`16043/173.8s`，当时规则下不合规）为历史版本，
不参与复评；外部 `youxilee/hif4` 不属于本仓库提交。**v84 是唯一的新权重官方
通过锚点，其 v5 基线（`0.477266 / 0.709020`）是新权重口径下最有价值的本地对照**；
当前 v127（`0.509408 / 0.828395`）两项均值均高于 v84，但尚无官方验证。

历史方法记录（旧权重口径的 36000 推导）见
[`当前实验结果与可达性 checkpoint`](../logs/execution/2026-08-31-current-results-target-feasibility.md)
与 [`36000 潜力研究`](2026-08-30-hif4-36000-potential-and-algorithms.md)；两者均不再
作为当前目标依据。

按新目标 `linear_mean=0.8` 计算诊断距离：当前 v127 的 `0.509408` 还需消除

$$\Delta g_L=0.8-0.509408=0.290592,\qquad \frac{\Delta g_L}{1-g_L}=\frac{0.290592}{0.490592}\approx 59.2\%$$

的剩余归一化 Linear 误差；v121-pawv-fixed（`0.516685`）需消除约 `57.6%`。这是
本地数轴，不是官方绝对分数。旧权重口径的 36000/0.9 推导已整体移入上文引用的
历史文档，不再在此重复。

## 当前唯一后续计划

现阶段只执行 [`2026-08-31-hif4-active-c1-structured-linear-plan.md`](superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md)。顺序为：

1. L0：已完成五个分层层位、全 role 的 Linear 单侧误差、合法 oracle 和放宽上限诊断；
2. L1：已完成原子写回完整 hierarchy 与正确二次型复验；预筛拒绝并归档 v105；
3. L2：已完成只按合法 expansive shape 路由的低自由度 FFN CAT balance，v106 已采纳；
4. L3：已完成部署 Gram 二次型修复 v095 Activation-LRH gate，v107 成为前一精度 parent；
5. L4a：已完成最终部署 Gram 的双候选 + 完整行级 gate，v109 成为当前精度 parent；
6. L4b：已完成最终 Gram GALS 小预算验证，v110 成为前一精度 parent；
7. L5a：已完成 block-local permutation，v111 成为当前精度 parent；
8. L5b/v112、L5c/v113、L5d/v114：均已完成 screen 并归档拒绝；L5e 已完成当前表示/接口可达性 checkpoint；
9. L6a：已完成 rank-16 global LRH，v115 成为前一 precision parent；
10. L6b：已完成宽输入 rank-4 factor，v116 成为前一 precision parent；
11. L6c：已完成 full `G_64` hierarchy coordinate sweep，v117 成为前一 precision parent；
12. L6d：已完成 structured block-circulant factor，v118 成为当前 precision parent；
13. L6e：已完成 cross-block recall/`J_64`/state checkpoint，L6 计划已归档；
14. C1a：已完成 proposal 向量化等价实现并采纳 v119；
15. C1b：已测试 block refresh（v120，screen 拒绝）与两轮 refresh（v121，full-layer 采纳）；
16. C1c：rank-2（v122）与 max-blocks-2（v123）screen 拒绝，rank-8（v124）与
    `max_blocks=8`（v125）full-layer 均正向；v125 作为 precision-only 证据，不再增加
    block budget；
17. 下一步执行 C2 低成本跨模型 guardrail；再执行 C3 的部署权重因子 exact gate、
    selected-block 稀疏增量 exact gate 与 structured gradient 增量刷新，最后恢复
    提交冻结（本地 sampled API `≤150s` 预算红线，官方 `<300s` 硬门）。C3 完成后才新建
    表示级计划，验证共享正交 butterfly/Givens frame 与冻结 activation state 后的完整
    离散 JDRQ-weight。PAWV rank 属于独立 Attention 队列，不插入 Linear 主线。
    当前目标与时间推断见上文「当前目标与本地时间推断」章节。

L4b 的正式产物为 [`v110-l4b-gals-final-gated-qwen-full.json`](../artifacts/real_model_suite/v110-l4b-gals-final-gated-qwen-full.json)
和 [`v110 L4b archive`](../solutions/20260831_v110_l4b-gals-final-gated_score295.242780_time702s/)。
v110 Linear mean `0.5073395278`、panel `295.242780`，较 v109 分别提升
`+0.0000138810`、`+0.0034702542`；GALS 只在两折完整部署 Gram 均正向时启用，
在线再次用完整 Gram 行级 gate，避免 block-only 回退。API `701.900553s`，探索阶段
只记录，不否决精度 parent。

L4a 的正式产物为 [`v109-l4a-final-gram-gated-qwen-full.json`](../artifacts/real_model_suite/v109-l4a-final-gram-gated-qwen-full.json)
和 [`v109 L4a archive`](../solutions/20260831_v109_l4a-final-gram-gated_score295.239309_time517s/)。
v109 Linear mean `0.5073256468`、panel `295.239309`，较 v107 分别提升
`+0.0003290112`、`+0.0822528`；收益来自 expansive FFN 的 final deployed-Gram
候选，并由完整 Gram 行级门控防止回退。此前 v108 的首次 L4a screen 因 dynamic
shape 路由误判实际是 no-op，已在 v108 archive 中更正，不能当作实验否定证据。

L5a 的正式产物为 [`v111-l5a-joint-permutation-qwen-full.json`](../artifacts/real_model_suite/v111-l5a-joint-permutation-qwen-full.json)
和 [`v111 L5a archive`](../solutions/20260831_v111_l5a-joint-permutation_scoreNA_timeNA/)。
候选在每个 64 维层级块内对 identity、压力排序和低/高交错排列做 operand-local
两折门控；选中的单一排列与 `D`、signed-Hadamard 同步作用于 W/A。screen Linear
mean `0.5318869457`，full-layer Linear mean `0.5082983001`、panel `295.482473`，
较 v110 分别 `+0.0009587723`、`+0.2396930806`；Attention 未变化。API
`726.094116s`，远超最新 300s 限制，仅作为精度 parent 记录，后续统一 C1 压缩。

L6a 的正式产物为 [`v115-l6a-rank16-qwen-full.json`](../artifacts/real_model_suite/v115-l6a-rank16-qwen-full.json)
和 [`v115 L6a archive`](../solutions/20260831_v115_l6a-rank16-accepted_score295.680651_time716s/)。
候选只将窄输入 global Activation-LRH rank 从 8 提到 16，保留原有两折 operand-local
proposal、完整部署 `G_q` gate、state/device 和 block budget；30 项定向回归通过，
静态/运行时 compliance 均为 0 violations。screen Linear mean `0.53284175`，
full-layer Linear mean `0.5090910148`、panel `295.6806514001`，较 v111 分别
`+0.0007927147`、`+0.1981786718`；Attention `0.8420394885` 不变。API
`716.482861s`、wall `748.372825s`，仍只作精度探索记录；随后已执行 L6b 宽输入
rank-4 factor。

L6b 的正式产物为 [`v116-l6b-wide-rank4-qwen-full.json`](../artifacts/real_model_suite/v116-l6b-wide-rank4-qwen-full.json)
和 [`v116 L6b archive`](../solutions/20260831_v116_l6b-wide-rank4-accepted_score295.734045_time739s/)。
候选只对 `d>1024,d<=8192` 增加 rank-4 off-block range factor，窄输入 rank-16
保持不变；32 项定向回归、静态/运行时合规检查通过。screen Linear mean
`0.5330906465`，full-layer Linear mean `0.5093045894`、panel `295.7340450430`，
较 v115 分别提升 `+0.0002135746`、`+0.0533936429`；唯一正向角色为宽 `proj`
（`0.4200260922→0.4215211142`），Attention `0.8420394885` 不变。API
`739.424609s`、wall `771.865345s`，仍只作精度探索记录，随后执行 L6c 完整
`G_64` hierarchy coordinate solver。

L6c 的正式产物为 [`v117-l6c-g64-hierarchy-qwen-full.json`](../artifacts/real_model_suite/v117-l6c-g64-hierarchy-qwen-full.json)
和 [`v117 L6c archive`](../solutions/20260831_v117_l6c-g64-hierarchy-accepted_score295.785829_time2019s/)。
候选固定 E6M2 scale，对每行最多 4 个高损 block 做一轮 `lv2/lv3` 坐标更新；每个
候选按完整 `G_64` 二次型增量重编码，并由部署 `G_q` 逐行 gate。33 项定向测试和
静态/运行时合规检查通过；screen Linear mean `0.5332946034`，较 v116
`+0.0002039570`，7 个 role 均不降；full-layer Linear mean `0.5095117268`、
panel `295.7858293956`，较 v116 分别 `+0.0002071374`、`+0.0517843527`，
Attention `0.8420394885` 不变。API `2019.475204s`、wall `2051.884441s`，仅作
精度探索记录；v117 成为前一 precision parent，随后执行 L6d。

L6d 的正式产物为 [`v118-l6d-structured-factor-qwen-full.json`](../artifacts/real_model_suite/v118-l6d-structured-factor-qwen-full.json)
和 [`v118 L6d archive`](../solutions/20260831_v118_l6d-structured-factor-accepted_score295.808212_time2249s/)。
候选只对宽输入 `1024<d<=8192` 生成最多 4 个 `64×64` block-circulant kernel，
用距离系数产生 proposal，再由完整部署 `G_q` 逐行 exact gate；结构化合成最大重构误差
`2.68e-7`，36 项定向回归与合规检查通过。screen Linear mean `0.53337532`，full-layer
Linear mean `0.5096012555`、panel `295.8082115559`，较 v117 分别 `+0.0000895286`、
`+0.0223821603`；唯一新增正向 role 为 `proj`（`0.4215743858→0.4222010863`），
Attention `0.8420394885` 不变。API `2249.746436s`、wall `2282.625213s`，只作
精度探索记录，随后执行 C1a 向量化。

C1a 的正式产物为 [`v119-c1a-structured-vectorized-qwen-full.json`](../artifacts/real_model_suite/v119-c1a-structured-vectorized-qwen-full.json)
和 [`v119 C1a archive`](../solutions/20260831_v119_c1a-structured-vectorized-accepted_score295.808212_time2040s/)。
实现保留 v118 reference helper，将选中的 row/block proposal 和 15-level 候选评估批量化，
但仍按 coordinate 升序串行更新；37 项定向测试、reference/vectorized `atol=1e-6` 对照和
静态/运行时合规检查通过。Qwen screen Linear mean `0.5333753185`，full-layer 的
Linear `0.5096012555`、Attention `0.8420394885`、panel `295.8082115559`、native
`423.2878345580` 与 v118 全部逐位相同。API 从 `2249.7464359s` 降至 `2040.5046895s`
（`−209.2417464s`, `−9.30%`），dynamic 从 `1832.8779521s` 降至 `1633.3390318s`
（`−10.88%`），wall 从 `2282.6252131s` 降至 `2072.6976340s`（`−9.19%`）。
因此 v119 是 C1b 的 precision/time parent。

C1b 的正式产物为 [`v121-c1b-structured-refresh2-qwen-full.json`](../artifacts/real_model_suite/v121-c1b-structured-refresh2-qwen-full.json)
和 [`v121 C1b archive`](../solutions/20260831_v121_c1b-structured-refresh2-accepted_score295.811281_time2180s/)。
实现沿用 v119 的 4-kernel structured proposal 和完整 `G_q` row gate，在每个 selected
block 后刷新 proposal gradient，并对同一 block rank list 做两轮 sweep；38 项定向测试、
合成目标单调性以及静态/运行时 compliance 检查通过。一次 refresh 的 v120 screen 为
`0.5333730058`，低于 v118 screen `0.5333753185`，已拒绝；两轮 refresh 的 v121 screen
为 `0.5333964596`，full-layer Linear `0.5096135327`、panel `295.8112808759`，较 v119
分别 `+0.0000122773`、`+0.0030693200`。除 `proj` 从 `0.4222010863` 到 `0.4222870273`
外，其余 Linear role 和 Attention 均保持不变。API `2180.450151s`、wall
`2212.661980s`，远超最新 300s 限制；用户随后确认 v121 官方显示运行超时，见
[`v121 官方 timeout`](../logs/execution/2026-08-31-v121-official-timeout.md)。该结果不改变
本地精度消融，但把 v121 明确排除出提交候选；它只在 accuracy-first 研究链中接替过
精度 parent。

C1c rank-8 的正式产物为 [`v124-c1c-rank8-qwen-full.json`](../artifacts/real_model_suite/v124-c1c-rank8-qwen-full.json)
和 [`v124 C1c archive`](../solutions/20260831_v124_c1c-rank8-accepted_score295.820229_time2324s/)。
仅将 structured kernel rank 从 4 提升到 8，`max_blocks=4` 与两轮 refresh 保持不变；
v122 rank-2 与 v123 max-blocks-2 screen 分别为 `0.53336284`、`0.53335171`，均被拒绝。
v124 screen `0.53343639` 后进入 full，Linear `0.5096493233`、panel `295.8202285103`，
较 v121 分别 `+0.0000357905`、`+0.0089476344`；7 个 Linear role 均不降，`proj`
为唯一变化明显的 role（`0.4222870273→0.4225375610`），Attention 保持 `0.8420394885`。
API `2323.911178s`、wall `2356.200547s`，仍仅作 accuracy-first 精度 parent；随后 v125
完成最后的 `max_blocks=8` 验证。

C1c `max_blocks=8` 的正式产物为 [`v125-c1c-block8-qwen-full.json`](../artifacts/real_model_suite/v125-c1c-block8-qwen-full.json)
和 [`v125 C1c archive`](../solutions/20260831_v125_c1c-block8-precision-only_score295.847849_time2654s/)。
v125 screen `0.53358298`，full-layer Linear `0.5097598050`、panel `295.8478489516`，
较 v124 分别 `+0.0001104818`、`+0.0276204413`；7 个 Linear role 均不降，Attention
`0.8420394885` 逐位不变。API `2653.580314s`、wall `2686.541758s`，远超 300s，
因此只接受为 precision-only 证据，C1c 队列停止，不再增加 block budget。

v107/v106 Attention 合约审计见 [`2026-08-31-v107-attention-contract-audit.md`](../logs/execution/2026-08-31-v107-attention-contract-audit.md)：
v100/v106/v107/v107b1 的 Attention 函数体 SHA 和 Qwen Attention case 输出逐位相同，
独立 5 场景×3 模式合约矩阵与 30 个 Q/K/V 状态校验均为 0 failures；用户随后确认
不含完整 `deployment_gram` 的 v100 也得到官方 Attention WA，且不是 timeout；v107 同为
Attention WA（用户确认非 timeout）。这些发现证明本地矩阵缺少官方失败
输入；v72、v74 的连续官方通过把安全边界前移到 v74，
不再推荐 v100/v106/v107 后代提交。

L6e checkpoint 见 [`2026-08-31-l6e-crossblock-checkpoint.md`](../logs/execution/2026-08-31-l6e-crossblock-checkpoint.md)：
在真实 layer-23 `proj(d=4864)` 的 4 个 test window 中，结构化 proposal 共 2048 个
block proposals，完整 `G_q` gate 接受 71 个（recall `3.4668%`），`J_64` 相对下降
`0.0991%`；结构化增量 state 为 `66,752` bytes，而该 shape 总 activation state 为
`96,043,200` bytes。该 checkpoint 支持继续做语义等价向量化，但不把压缩 proposal
误称为对 dense deployed Gram 的高 recall。

L5b–L5d 的拒绝证据分别为 [`v112 archive`](../solutions/20260831_v112_l5b-sparse-schur_rejected-screen_score0.530855_time140s/)、
[`v113 archive`](../solutions/20260831_v113_l5c-meta-router_rejected-screen_score0.531887_time169s/)
和 [`v114 archive`](../solutions/20260831_v114_l5d-external-sampling_rejected-screen_score0.527311_time125s/)。
v112 screen Linear `0.5308551016`（较 v111 `-0.0010318441`），v113 与 v111 逐 case
完全相同（`0.5318869457`，no-op），v114 外部 stride sampling 为 `0.5273114999`
（`-0.0045754462`），均未跑 full-layer。L5d 外部逐组件审计见
[`l5d external audit`](../logs/execution/2026-08-31-l5d-external-component-audit.md)。

L5e 的完整诊断见 [`l5e JSON`](../artifacts/oracle_dashboard/l5e-linear-ceiling-v111-qwen.json)
和 [`l5e log`](../logs/execution/2026-08-31-l5e-linear-ceiling-v111.md)：固定 frame
screen `0.5318869457`，weight-perfect `0.7140714612`，activation-perfect `0.8188904986`；
255-code oracle 的加权下降为 weight plain `0.04746%`、weight Gram `4.56979%`、
activation Gram `0.11279%`。当前固定 HiF4 hierarchy/state 接口若要到 `0.9`，还需
减少约 `78.64%` 剩余误差；下一方向转为压缩跨 block 表达，不再重复 offset、sampler
或 joint residual。

L3 的正式产物为 [`v107-l3-global-lrh-qwen-full.json`](../artifacts/real_model_suite/v107-l3-global-lrh-qwen-full.json)
和 [`v107 L3 archive`](../solutions/20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s/)。
v107 Linear mean `0.5069966356`、panel `295.1570566`，较 v106 分别提升
`+0.0035376934`、`+0.8844233`；收益来自窄输入 q/k/v/o 的 global-LRH 候选，
fc_gate/fc_up/proj 不依赖 global state。Gram/MSE 冲突率为 `0.567475`，说明原
v095 MSE gate 确实会做出不同裁决。1-block 对照为 `0.5043033601`，低于 4-block，
因此保留 4-block 作为精度 parent。

L2 的正式产物为 [`v106-l2-cat-qwen-full.json`](../artifacts/real_model_suite/v106-l2-cat-qwen-full.json)
和 [`2026-08-30-l2-expansive-cat.md`](../logs/execution/2026-08-30-l2-expansive-cat.md)。
L2 screen 的 selected-layer `both_player=0.525228958652` 触发 full-layer；v106 的
Linear mean `0.503458942243`、panel `294.272633253`，API `412.654599s`，较 v100
分别提升 `+0.001901329745`、`+0.475332436`，并成为当前 parent。收益只来自
`fc_gate`，q/k/v/o/fc_up/proj 不变。

L1 的正式产物为 [`l1-lrh-stratified-qwen.json`](../artifacts/real_model_suite/l1-lrh-stratified-qwen.json)
和 [`2026-08-30-l1-full-hierarchy-lrh.md`](../logs/execution/2026-08-30-l1-full-hierarchy-lrh.md)。
L1 candidate screen 的 selected-layer `both_player=0.523019429222563` 与 L0 逐条一致；
70 个 fold 候选只有 1 个 cross-fold admitted，最终没有任何 case 改变 stable parent，
所以没有运行 full-layer。候选完整源码在 `solutions/20260830_v105_l1-full-hierarchy-lrh-rejected_screen523019_time266s/`。

L0 的正式产物为 [`l0-linear-ceiling-qwen.json`](../artifacts/oracle_dashboard/l0-linear-ceiling-qwen.json)
和 [`2026-08-30-l0-linear-ceiling.md`](../logs/execution/2026-08-30-l0-linear-ceiling.md)。五层×七 role
的 `both_player=0.52301943`、`weight_perfect=0.70417026`、
`activation_perfect=0.82035698`；整体 activation-side headroom (`0.29733755`)
大于 weight-side (`0.18115083`)，但 q/k 权重侧更突出，FFN/proj 激活侧更突出。
这只调整 L1–L4 的优先级，不产生部署 parent，也不替代 v125 当前 24 层 precision-only parent `0.5097598050`。

旧版 active 计划已经归档；官方提交改为接口恢复时触发的外部事件，不阻塞当前本地执行。

## 时间与合规（v5）

- v127 sampled（224/32）六 API 累计 `151.136s`，本地 wall `161.840s`；同一
  sample plan 的 v74 为 `218.619s / 229.485s`。v74 当前 CPU full 复测为
  `658.877s / 690.600s`，但官方实际 `239.387s`，证明本地 `<300s` 不能作为
  官方判定。v125 full `2653.580s` 保留为 legacy。
- 调用次数：weight calibration 168；attention calibration 24；dynamic activation
  672；Q/K/V 各 96。
- `activation_state` 只包含 BOAT 逆缩放、静态 Gram、Attention GQRB 正交 mixing、
  PAWV 的静态 token-row diagonal 与旋转整数配置/符号；官方 2026-08-31 修订已放开
  `A@W` 信息源限制，候选可按需使用输出或残差优化 `Q(W)` / `Q(A)`。
- 当前源码 SHA256（规范 LF 内容）：
  `75F21B7BE3630FFEFEAF2883BB699CE4901DF1BF6C0B39DD6E40F253561E32C0`（与
  `solutions/20260831_v127_v106-pawv-variable-length-safe_scoreNA_timeNA/` 归档一致）。
- 发布前检查：C1c synthetic/reference-equivalence、合规/精度门控和 v107 Attention
  contract audit 均通过；`guard_solution_file` 为 `violations=[]`、`static_violations=[]`。
  v127 的 `local_result_valid=true` 仅表示本地结果完整、finite、接口合法，不表示官方
  通过；官方 v100/v107 均确认 Attention `wrong answer`（非 timeout），官方 v98 已在最新
  300s 限制下确认为 timeout。后续官方候选以 v74 的完整 Attention 可达闭包为回归基线，
  且把 end-to-end `<300s` 作为提交硬门。
- 当前根定向回归命令（`test_expansive_cat.py`、`test_linear_compliance_guard.py`、
  `test_linear_error_decomposition.py`）为 **21 passed / 3.55s**（2026-08-31 裁剪后实测）。
  `test_global_activation_lrh.py`、`test_l5a_joint_transform.py`、`test_release_candidate.py`
  等针对已裁剪 L4a/L4b/L5a/L6/C1 机制与旧 release flag 的历史断言共 **17 failed**；它们不是
  当前 v127 根文件的发布门禁，不能计入当前算法分数。若重启这些方向，先按当前 API/state
  重新编写测试，再把测试加入发布命令。

历史 C86 源码和每次实验结果仍在 `solutions/`、`logs/` 与 Git 历史中，本文只描述
当前根目录实现；历史记录不应被当作当前代码行为。
