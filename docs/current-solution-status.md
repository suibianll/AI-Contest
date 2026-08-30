# 当前主版本算法效果与评测状态

> 更新日期：2026-08-30
> 适用文件：根目录 [`solution.py`](../solution.py)
> 文档性质：本地可复现实测记录，不是官方成绩承诺。

## 一句话结论

根目录当前为 BOAT + expansive-FFN CAT balance + cross-fold HSDQ，并加入已通过本地门禁的
Attention B1 GQRB 与 B2 PAWV diag-only 候选。固定 Qwen2.5-0.5B 缓存、`seq=128`、
`calib=2`、`test=4`、全 24 层、CPU 的完整运行中，当前主版本 Qwen shaped panel 为
**294.272633**，正式 API 累计 **412.654599 s**，低于
`420 s` 限制；该数值用于本地 A/B 排序，不能线性换算为官方排行榜分数。

当前唯一活跃计划是 [`2026-08-30-hif4-active-optimization-plan.md`](superpowers/plans/2026-08-30-hif4-active-optimization-plan.md)；归档候选的写回、目标错位和源码缺失审计见 [`archive-implementation-audit.md`](archive-implementation-audit.md)。

2026-08-30 已按执行计划完成 E0-C、E1→A6、B1、B2、L1 和 L2。Linear 的 E1/A2/A3/A4/A5/A6
当前运行配置均未超过 stable parent；但归档审计发现 v095 最终 gate 存在问题，仍待后续修复验证。B1 GQRB margin 先把 panel 提升到 `293.793700`，B2
PAWV diag-only 再提升到 `293.797301`，L2 expansive-FFN CAT balance 将 panel 提升到
`294.272633`、Linear mean 提升到 `0.503458942243`，API 为 `412.654599 s`，因此当前根
切换到 v106；随后 C0 五模型确认完成，Qwen 主模型仍通过门禁。官方评测不可用
期间，所有新候选仍以固定 Qwen panel 为门禁。L1 已完成真正的 scale/lv2/lv3/mantissa
原子写回与合成测试，但五层×七 role 预筛与 L0 逐条持平（`0.523019429222563`），
因此候选 v105 已归档；v106 已通过 full-layer 并成为当前 parent，下一步转入 L3。

## 当前实现

`solution.py` 只保留六个正式 API 和必要的 codec/优化原语：

1. **BOAT（Block Output-Alignment Transform）**：用激活/权重各自的 RMS 构造
   对角平衡 `D`，再搜索 4/8/16/64 维 signed-Hadamard 块和两个确定性 seed。连续
   乘积保持不变：

   $$X'=XD^{-1}R,\qquad W'=W D R,\qquad X'W'^T=XW^T.$$

   候选只依赖两侧 operand-local HiF4 重建误差，不构造 Linear 输出，因此固定参数
   可以安全写入 `activation_state`。
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
   Hessian 做 V 的离散坐标 refinement；V 仍保持独立合法 HiF4 编码。

## 最新全层实测

报告文件：[`v106-l2-cat-qwen-full.md`](../logs/execution/2026-08-30-v106-l2-cat-qwen-full.md)；
原始 JSON：[`v106-l2-cat-qwen-full.json`](../artifacts/real_model_suite/v106-l2-cat-qwen-full.json)。
上一 parent 的对照报告：[`b2-pawv-diagonly-qwen-full.md`](../logs/evaluations/b2-pawv-diagonly-qwen-full.md)。

固定输入为 Qwen2.5-0.5B（24 层、hidden 896、14 Q heads、2 KV heads、head dim 64），
calibration 使用 train 的 2 个窗口，test 使用 validation 的 4 个不重叠窗口。

| 指标 | 当前 v106 | 旧 C86 | 变化 |
|---|---:|---:|---:|
| Linear native mean | **0.503459** | 0.477821 | +0.025638 |
| Attention native mean | 0.842039 | 0.739264 | +0.102775 |
| Qwen panel Linear | **125.864736** | 119.455153 | +6.409583 |
| Qwen panel Attention | 168.407898 | 147.852757 | +20.555141 |
| Qwen panel total | **294.272633** | 267.307909 | **+26.964724（+10.09%）** |
| official-flow native total | **419.160200** | 392.064774 | +27.095426 |
| six-API time | **412.654599 s** | 313.577669 s | +99.076930 s |
| wall time | 446.069189 s | — | `API <420 s` |

五模型 C0 确认报告：[`2026-08-30-c0-b2-pawv-five-model.md`](../logs/evaluations/2026-08-30-c0-b2-pawv-five-model.md)。
v106 Qwen panel `294.272633`、Linear `0.503459`、Attention `0.842039`、
API `412.654599s`，仍低于 `420s`；gpt2-small/OPT/Pythia 的旧 parent API 分别为
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
| Qwen native total | **419.160200** | 369.527269 | **+49.632931（+13.43%）** |
| Qwen shaped panel | **294.272633** | 250.327102 | **+43.945531（+17.56%）** |
| panel Linear | **125.864736** | 112.939429 | **+12.925306** |
| panel Attention | 168.407898 | 137.387673 | **+31.020225** |

因此，后续本地算法 A/B 应以外部 Qwen `250.327102` 作为第一比较线，
以外部 Qwen native `369.527269` 作为第二诊断线；`1085.743597` 只能用于检查
跨模型结构性回退，不能作为“外部最高分”或与官方 `24153` 做差值。

当前版本的官方流程诊断由 672 个 Linear case 和 96 个 Attention case 求和得到：

| 组件 | case 数 | gain sum | gain mean | global gain |
|---|---:|---:|---:|---:|
| Linear | 672 | 338.324409 | 0.503459 | 0.440481 |
| Attention | 96 | 80.835791 | 0.842039 | 0.857899 |
| 合计 | 768 | **419.160200** | — | — |

`panel_score` 不是把 768 个 case 复制成 450 个，而是保留组件均值后投影：

$$P_L=250\times0.5034589422=125.864736,$$

$$P_A=200\times0.8420394885=168.407898,$$

$$P_{total}=P_L+P_A=294.272633.$$

因此 `official_flow_total` 与 `panel_score.total` 同时出现是设计结果，不是计算冲突。

## Linear 角色归因

| 角色 | native mean |
|---|---:|
| q | 0.616561 |
| k | 0.620526 |
| v | 0.563596 |
| o | 0.483463 |
| fc_gate | **0.388435** |
| fc_up | 0.430255 |
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
`2026-08-30-l2-expansive-cat.md`、`2026-08-30-v106-l2-cat-qwen-full.md`。

## 距离 Linear 0.9 与 36,000

当前 Linear native mean 为 `0.5034589422`。若以本地 panel 的 mean 作为诊断目标：

$$\Delta g_L=0.9-0.5034589422=0.3965410578,$$

$$\frac{\Delta g_L}{1-g_L}=\frac{0.3965410578}{0.4965410578}=79.86\%.$$

也就是还要消除当前 Linear 剩余归一化误差的约 **79.86%**，对应 250-case panel
仍差 **99.1353** 分。这个数轴不是官方排行榜的绝对分数。

官方历史合规锚点为 C66：`22557 / 217.2s`；外部参考 `youxilee/hif4` 为用户提供的
`24153 / 239s`。从 C66 到 `36000` 的官方分差是 **13443**，但当前本地 panel
不做官方绝对分回归，因此不能声称当前版本对应某个官方分数或已经接近 `36000`。

## 当前唯一后续计划

现阶段只执行 [`2026-08-30-hif4-active-optimization-plan.md`](superpowers/plans/2026-08-30-hif4-active-optimization-plan.md)。顺序为：

1. L0：已完成五个分层层位、全 role 的 Linear 单侧误差、合法 oracle 和放宽上限诊断；
2. L1：已完成原子写回完整 hierarchy 与正确二次型复验；预筛拒绝并归档 v105；
3. L2：已完成只按合法 expansive shape 路由的低自由度 FFN CAT balance，v106 已采纳；
4. L3：用部署 Gram 二次型修复 v095 Activation-LRH gate（当前下一步）；
5. L4：把 final-weight Gram 与 GALS 拆成两个有 oracle 依据的小预算实验；
6. L5：若前述方向没有结构增益，进入外部逐组件审计和联合坐标—层级离散新路线；
7. 出现新精度 parent 后再做 `<420s` 压缩。PAWV rank 属于独立 Attention 队列，不插入 Linear 主线。

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
这只调整 L1–L4 的优先级，不产生部署 parent，也不替代 v106 当前 24 层 parent `0.5034589422`。

旧版 active 计划已经归档；官方提交改为接口恢复时触发的外部事件，不阻塞当前本地执行。

## 时间与合规

- 六个 API 累计：v106 calibration+dynamic `412.654599 s < 420 s`；本地 wall
  `446.069189 s` 仅作诊断。
- 调用次数：weight calibration 168；attention calibration 24；dynamic activation
  672；Q/K/V 各 96。
- `activation_state` 只包含 BOAT 逆缩放、静态 Gram、Attention GQRB 正交 mixing、
  PAWV 的静态 token-row diagonal 与旋转整数配置/符号；输出监督只用于离线
  Attention 候选和权重侧选择，不进入在线 `Q(A)`。
- 当前源码 SHA256（规范 LF 内容）：
  `708081b5281e02da0c2a6e21881027b2e8d31eed423fd3c70e4572424667dd77`。
- 发布前检查：`30 passed`，包含 reference HiF4、Linear 合规、ceiling dashboard 和
  L2 CAT transform 测试；L2 full-layer `valid_submission=true`。

历史 C86 源码和每次实验结果仍在 `solutions/`、`logs/` 与 Git 历史中，本文只描述
当前根目录实现；历史记录不应被当作当前代码行为。
