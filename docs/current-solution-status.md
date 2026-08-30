# 当前主版本算法效果与评测状态

> 更新日期：2026-08-31
> 适用文件：根目录 [`solution.py`](../solution.py)
> 文档性质：本地可复现实测记录，不是官方成绩承诺。

## 一句话结论

根目录当前为 v111 L5a block-local permutation + BOAT + expansive-FFN CAT balance +
cross-fold HSDQ + Global Activation-LRH Gram gate + L4a final deployed-Gram row gate +
L4b GALS，并保留 Attention B1 GQRB 与 B2 PAWV diag-only。固定 Qwen2.5-0.5B 缓存、
`seq=128`、`calib=2`、`test=4`、全 24 层、CPU 的完整运行中，当前精度 parent 的
Qwen shaped panel 为 **295.482473**，Linear mean **0.5082983001**，正式 API 累计
**726.094s**；探索阶段只记录时间，不以
`420 s` 否决精度候选，最终冻结时再压缩。该数值用于本地 A/B 排序，不能线性换算为
官方排行榜分数。

当前唯一活跃计划是 [`2026-08-31-hif4-active-l6-compressed-crossblock-plan.md`](superpowers/plans/2026-08-31-hif4-active-l6-compressed-crossblock-plan.md)；L5 完成计划已归档，归档候选的写回、目标错位和源码缺失审计见 [`archive-implementation-audit.md`](archive-implementation-audit.md)。

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
是当前精度 parent，L0–L5e 已完成，v112/v113/v114 已归档拒绝，下一步执行 L6a。

## 当前实现

`solution.py` 只保留六个正式 API 和必要的 codec/优化原语：

1. **BOAT + L5a block-local permutation**：用激活/权重各自的 RMS 构造对角平衡
   `D`，在每个 64 维层级块内以独立 `amax/rms` pressure 搜索至多一个固定排列，
   再搜索 4/8/16/64 维 signed-Hadamard 块和两个确定性 seed。连续
   乘积保持不变：

   $$X'=XD^{-1}PR,\qquad W'=W D P R,\qquad X'W'^T=XW^T.$$

   排列只有在两折 operand-local HiF4 重建误差均不变差时才写入 state；不构造
   Linear 输出，因此固定参数可以安全写入 `activation_state`。
2. **Cross-fold Weight-HSDQ**：对满足宽度/形状条件的权重块使用校准激活
   `A_f^T A_f` 的低秩 Hessian，对 HiF4 的 15 个 signed levels 做精确二次增量
   搜索；fold 1 生成的候选必须改善 fold 2，最终只改变离线 `weight_params`。
3. **Gram-hierarchy Activation-HSDQ**：从静态变换后权重计算 64 维 Gram block，
   先按二次型选择层级和 E6M2 offset，再做最多 128 个 block、2 轮坐标扫描。状态
   只保存 CPU 上的静态 `gram64`、BOAT 逆缩放和整数/符号配置。
4. **Expansive-FFN CAT balance**：只对 `weight_rows > weight_channels` 的结构形状，
   使用固定 `α=0.25` 的 RMS 对角 balance；不依赖 role-id/模型名，不增加 state 字段，
   operand-local proxy 不优于 BOAT 时回退 parent。
5. **L4a final deployed-Gram row gate（当前 Linear parent）**：在
   `rows > channels` 且 `channels <= 1024` 的 expansive FFN 上，同时生成 v107
   parent 与最终 `G_q=W_q^T W_q` 候选；用完整 `G_q` 逐行比较二次型，只写回不变差
   的候选行。它不增加公开 API 字段，state 只保存静态部署 Gram。
6. **Attention 输出感知 shortlist**：搜索 reciprocal RMS 平衡、K-centering、
   16/32/64 维共享 signed-Hadamard，以及 B1 GQRB 的 2×2/4×4 group-local
   orthogonal mixing；保留 parent 的原始四候选，并要求 mixing exact loss 至少
   改善 0.1% 才能替换。B2 PAWV 用 attention probability 的 token-row 对角
   Hessian 做 V 的离散坐标 refinement；V 仍保持独立合法 HiF4 编码。

## 最新全层实测（当前精度 parent）

报告文件：[`2026-08-31-v111-l5a-joint-permutation-qwen-full.md`](../logs/execution/2026-08-31-v111-l5a-joint-permutation-qwen-full.md)；
原始 JSON：[`v111-l5a-joint-permutation-qwen-full.json`](../artifacts/real_model_suite/v111-l5a-joint-permutation-qwen-full.json)。
v106 时间 parent 对照：[`v106-l2-cat-qwen-full.md`](../logs/execution/2026-08-30-v106-l2-cat-qwen-full.md)。
上一 parent 的对照报告：[`b2-pawv-diagonly-qwen-full.md`](../logs/evaluations/b2-pawv-diagonly-qwen-full.md)。

固定输入为 Qwen2.5-0.5B（24 层、hidden 896、14 Q heads、2 KV heads、head dim 64），
calibration 使用 train 的 2 个窗口，test 使用 validation 的 4 个不重叠窗口。

| 指标 | 当前 v111 | v110 | 相对 v110 |
|---|---:|---:|---:|
| Linear native mean | **0.508298** | 0.507340 | **+0.000959** |
| Attention native mean | 0.842039 | 0.842039 | 0 |
| Qwen panel Linear | **127.074575** | 126.834882 | **+0.239693** |
| Qwen panel Attention | 168.407898 | 168.407898 | 0 |
| Qwen panel total | **295.482473** | 295.242780 | **+0.239693** |
| official-flow native total | **422.412249** | 421.767954 | **+0.644295** |
| six-API time | 726.094116 s | 701.900553 s | +24.193563 s |
| wall time | 758.279099 s | 734.220364 s | exploratory timing |

五模型 C0 确认报告：[`2026-08-30-c0-b2-pawv-five-model.md`](../logs/evaluations/2026-08-30-c0-b2-pawv-five-model.md)。
v111 Qwen panel `295.482473`、Linear `0.508298`、Attention `0.842039`、
API `726.094s`；该精度 parent 暂不满足最终 420s 冻结条件。gpt2-small/OPT/Pythia 的旧 parent API 分别为
`196.975s/192.776s/193.423s`，gpt2-medium 为 `492.641s`（仅软 guardrail 时间
超限，未影响 Qwen 主门禁）。五模型 aggregate panel `263.604453` 仅作泛化诊断。

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
| Qwen native total | **421.537530** | 369.527269 | **+52.010261（+14.07%）** |
| Qwen shaped panel | **295.482473** | 250.327102 | **+45.155371（+18.04%）** |
| panel Linear | **126.749159** | 112.939429 | **+13.809730** |
| panel Attention | 168.407898 | 137.387673 | **+31.020225** |

因此，后续本地算法 A/B 应以外部 Qwen `250.327102` 作为第一比较线，
以外部 Qwen native `369.527269` 作为第二诊断线；`1085.743597` 只能用于检查
跨模型结构性回退，不能作为“外部最高分”或与官方 `24153` 做差值。

当前版本的官方流程诊断由 672 个 Linear case 和 96 个 Attention case 求和得到：

| 组件 | case 数 | gain sum | gain mean | global gain |
|---|---:|---:|---:|---:|
| Linear | 672 | 340.701739 | 0.506997 | 0.442719 |
| Attention | 96 | 80.835791 | 0.842039 | 0.857899 |
| 合计 | 768 | **419.160200** | — | — |

`panel_score` 不是把 768 个 case 复制成 450 个，而是保留组件均值后投影：

$$P_L=250\times0.5073256468=126.831412,$$

$$P_A=200\times0.8420394885=168.407898,$$

$$P_{total}=P_L+P_A=295.239309.$$

因此 `official_flow_total` 与 `panel_score.total` 同时出现是设计结果，不是计算冲突。

## Linear 角色归因

| 角色 | native mean |
|---|---:|
| q | 0.619369 |
| k | 0.626984 |
| v | 0.571105 |
| o | 0.487739 |
| fc_gate | **0.392694** |
| fc_up | 0.432110 |
| proj | 0.421376 |

`v` 与 `fc_gate` 是当前最弱角色；L2 CAT balance 已改善 fc_gate，但从总体收益看，扩张 FFN 和输出投影仍受跨
64-block 相关性、校准 fold 数量和运行时约束共同限制，不能仅靠增加 offset 或 sweep
解决。

## 2026-08-30 执行计划结果

以下均为同一 Qwen2.5-0.5B、24 层、`cache=read` 的本地 shaped panel；parent
永远保留，候选失败后已恢复。详细 fold 与角色数据见各 execution log。

| 实验 | panel | Linear mean | API time | 裁决 |
|---|---:|---:|---:|---|
| E0/D0 多模型 scale-lattice oracle | — | — | 13.76–14.84s/模型 | 完成诊断；scale gap 亚百分比，无跨模型统一增益 |
| E0-C GALS-C 稀疏 activation（layer-1） | 335.988995 | 0.602878 | 57.41s | 拒绝；解析召回 oracle `1.0`，部署版回退 `0.048096` |
| A7 量化后权重 Gram `WqᵀWq`（layer-1/full） | 336.562922 / 290.226694 | 0.605174 / 0.487275 | 24.89s / 470.58s | 拒绝；单层正向不迁移且全层超时 |
| L1 full-hierarchy cross-block Weight-LRH（v105） | 0.523019（五层×七 role screen） | — | 265.87s screen | 拒绝；70 个 fold 候选仅 1 个 cross-fold admitted，最终 0/35 case 改变 parent；未触发 full-layer |
| L2 expansive-FFN CAT balance（v106） | **294.272633** | **0.503459** | **412.65s** | **采纳；fc_gate +0.013309，较 v100 panel +0.475332，API 仍 <420s** |
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
| **L5a block-local permutation（v111 当前 parent）** | **295.482473** | **0.508298** | **726.09s** | **精度采纳；下一步 L6a 压缩跨 block** |
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

## 距离 Linear 0.9 与 36,000

当前精度 parent v111 的 Linear native mean 为 `0.5082983001`。若以本地 panel 的 mean 作为诊断目标：

$$\Delta g_L=0.9-0.5082983001=0.3917016999,$$

$$\frac{\Delta g_L}{1-g_L}=\frac{0.3917016999}{0.4917016999}=79.66\%.$$

也就是还要消除当前 Linear 剩余归一化误差的约 **79.66%**，对应 250-case panel
仍差 **97.925425** 分。这个数轴不是官方排行榜的绝对分数。

官方历史合规锚点为 C66：`22557 / 217.2s`；外部参考 `youxilee/hif4` 为用户提供的
`24153 / 239s`。从 C66 到 `36000` 的官方分差是 **13443**，但当前本地 panel
不做官方绝对分回归，因此不能声称当前版本对应某个官方分数或已经接近 `36000`。

## 当前唯一后续计划

现阶段只执行 [`2026-08-31-hif4-active-l6-compressed-crossblock-plan.md`](superpowers/plans/2026-08-31-hif4-active-l6-compressed-crossblock-plan.md)。顺序为：

1. L0：已完成五个分层层位、全 role 的 Linear 单侧误差、合法 oracle 和放宽上限诊断；
2. L1：已完成原子写回完整 hierarchy 与正确二次型复验；预筛拒绝并归档 v105；
3. L2：已完成只按合法 expansive shape 路由的低自由度 FFN CAT balance，v106 已采纳；
4. L3：已完成部署 Gram 二次型修复 v095 Activation-LRH gate，v107 成为前一精度 parent；
5. L4a：已完成最终部署 Gram 的双候选 + 完整行级 gate，v109 成为当前精度 parent；
6. L4b：已完成最终 Gram GALS 小预算验证，v110 成为前一精度 parent；
7. L5a：已完成 block-local permutation，v111 成为当前精度 parent；
8. L5b/v112、L5c/v113、L5d/v114：均已完成 screen 并归档拒绝；L5e 已完成当前表示/接口可达性 checkpoint；
9. L6a–L6e：当前活跃计划，依次尝试窄/宽 rank 压缩跨 block factor、完整 `G_64` 层级求解、结构化 factor 和 checkpoint；
10. 所有精度方向完成后再做 `<420s` 压缩。PAWV rank 属于独立 Attention 队列，不插入 Linear 主线。

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
`726.094116s`，仍超 420s，仅作为精度 parent 记录，后续统一 C1 压缩。

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
这只调整 L1–L4 的优先级，不产生部署 parent，也不替代 v111 当前 24 层精度 parent `0.5082983001`。

旧版 active 计划已经归档；官方提交改为接口恢复时触发的外部事件，不阻塞当前本地执行。

## 时间与合规

- 六个 API 累计：v109 calibration+dynamic `517.285773 s`；本地 wall
  `549.506262 s`。探索阶段只记录时间；C1 再以 `<420s` 作为提交门禁。
- 调用次数：weight calibration 168；attention calibration 24；dynamic activation
  672；Q/K/V 各 96。
- `activation_state` 只包含 BOAT 逆缩放、静态 Gram、Attention GQRB 正交 mixing、
  PAWV 的静态 token-row diagonal 与旋转整数配置/符号；输出监督只用于离线
  Attention 候选和权重侧选择，不进入在线 `Q(A)`。
- 当前源码 SHA256（规范 LF 内容）：
  `6b229081121c4a7edd69575c93dc01488be8f8b5e1479007522421e93e1adc57`。
- 发布前检查：L3/L4a 合成/合规/精度门控测试 `19 passed`，另有既有 Linear 合规
  `15 passed`；v109 full-layer `valid_submission=true`，但 `under_official_runtime_limit=false`。

历史 C86 源码和每次实验结果仍在 `solutions/`、`logs/` 与 Git 历史中，本文只描述
当前根目录实现；历史记录不应被当作当前代码行为。
