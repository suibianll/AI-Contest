# HiF4 算法全景：已实现、已验证效果与未实现方向

> 整理日期：2026-08-31
> 数据来源：`solutions/README.md`（v000–v125）、`docs/current-solution-status.md`、`docs/archive-implementation-audit.md`、`logs/execution/2026-08-30-e0g-scale-oracle.md`、`logs/execution/2026-08-30-e0g-multimodel-dashboard.md`、`logs/execution/2026-08-30-a7-quant-weight-gram.md`、`logs/execution/2026-08-30-l1-full-hierarchy-lrh.md`、`logs/execution/2026-08-31-v110-l4b-gals-final-gated-qwen-full.md`、`logs/execution/2026-08-31-v111-l5a-joint-permutation-qwen-full.md`、`logs/execution/2026-08-31-l5d-external-component-audit.md`、`logs/execution/2026-08-31-l5e-linear-ceiling-v111.md`、`logs/execution/2026-08-31-v115-l6a-rank16-qwen-full.md`、`logs/execution/2026-08-31-v116-l6b-wide-rank4-qwen-full.md`、`logs/execution/2026-08-31-v117-l6c-g64-hierarchy-qwen-full.md`、`logs/execution/2026-08-31-v118-l6d-structured-factor-qwen-full.md`、`logs/execution/2026-08-31-l6e-crossblock-checkpoint.md`、`logs/execution/2026-08-31-v119-c1a-structured-vectorized-qwen-full.md`、`logs/execution/2026-08-31-c1b-structured-refresh-stratified.md`、`logs/execution/2026-08-31-c1b-structured-refresh2-stratified.md`、`logs/execution/2026-08-31-v121-c1b-structured-refresh2-qwen-full.md`、`logs/execution/2026-08-31-v124-c1c-rank8-screen.md`、`logs/execution/2026-08-31-v124-c1c-rank8-qwen-full.md`、`logs/execution/2026-08-31-v125-c1c-block8-qwen-full.md`、`logs/execution/2026-08-31-v107-attention-contract-audit.md`、`logs/execution/2026-08-31-c1b-structured-refresh-synthetic.md`。
> 口径纪律：本地只能比 **Qwen 同口径 panel**（`250·g_L + 200·g_A`）；五模型合计 `1085.743597` 只用于检查跨模型结构性回退，**禁止**与官方分数做差值。官方评测集为 250 Linear + 200 Attention case，时间上限 **420s**。

---

## 1. 当前根：算法构成与效果

根 `solution.py`（规范 LF SHA `47e2e3ab76c6deaac8de47bbcbd8f689cf5989dc8ff9e9081a887ec89e819b08`）为 v126：在 v125 clean 单一路径上修复 B2 PAWV 变长 calibration，按 `seq_len` 分组 diagonal，并删除未使用的 full `P^TP/eigh`。当前完整精度最高的已测版本仍为 v125（precision-only）；v126 通过变长合成/API 回归但尚未重跑 full-layer。其余 C1/L6/L5/Linear 组件保持不变，详见 [`归档实现审计`](archive-implementation-audit.md)。

| 组件 | 内容 |
|---|---|
| **BOAT** | 全层统一对角 alpha + 固定 signed-Hadamard；搜索在合法域内完成，只改变离线 `weight_params` |
| **cross-fold Weight-HSDQ** | fold 1 生成的候选必须改善 fold 2，最终只改变离线 `weight_params` |
| **Gram-hierarchy Activation-HSDQ** | 从静态变换后权重计算 64 维 Gram block，先按二次型选层级与 E6M2 offset，再做最多 128 个 block、2 轮坐标扫描；state 只含 CPU 静态 `gram64`、BOAT 逆缩放与整数/符号配置 |
| **Expansive-FFN CAT balance** | 仅对 `rows > channels` 路由，固定 α=0.25 RMS 对角 balance；v106 仅 fc_gate 正向 |
| **Global Activation-LRH Gram gate** | 输入宽度 ≤1024 的窄形状生成 rank-16、宽形状 `1024<d<=8192` 生成 rank-4 off-block proposal；最终离散候选用部署量化权重 `G_q=W_qᵀW_q` 逐行精确二次型门控；v125 precision-only parent |
| **L6c full `G_64` hierarchy** | 固定 E6M2 scale，对每行最多 4 个高损 block 做 `lv2/lv3` 坐标更新，完整 `G_64` 增量和部署 `G_q` gate；v117 精度正向 |
| **L6d/C1a/C1b/C1c structured factor** | 宽输入最多 8 个 `64×64` kernel + 距离系数生成跨 block proposal；C1b 每个 selected block 后刷新梯度并扫两轮；C1c rank=8、`max_blocks=8`；最终完整 `G_q` 行级 gate；v125 `proj(d=4864)` 正向 |
| **C1 proposal path** | C1a 批量独立 row/block 的 15-level proposal；C1b block refresh×2；C1c rank 4→8、`max_blocks=4→8`；coordinate 顺序与 exact `G_q` gate 不变；v125 panel `295.847849` |
| **L4a final deployed-Gram row gate** | 仅 expansive `rows > channels` 且 `channels <=1024`；v107 parent 与 final-Gram 候选用完整 `G_q` 逐行比较，v109 精度正向 |
| **Attention 输出感知 shortlist** | reciprocal RMS/K-centering/共享 Hadamard + B1 GQRB 2×2/4×4 group-local mixing；B2 PAWV 用 attention probability 的 token-row 对角 Hessian 做 V refinement；V 保持独立合法 HiF4 编码 |

**实测**（Qwen2.5-0.5B 全 24 层，`seq=128/calib=2/test=4/amax6`，缓存只读）：

```text
Linear mean       0.509760      Attention mean  0.842039
Qwen panel        295.847849    native total    423.394380
six-API time      2653.580314 s wall time       2686.541758 s   （探索阶段只记录时间）
```

分角色 Linear gain：`q 0.616758 / k 0.629137 / v 0.571384 / o 0.498290 / fc_gate 0.395579 / fc_up 0.433860 / proj 0.423311`。

**官方分数：无（从未提交）。** 这是全局最大的信息缺口。

---

## 2. 效果演进：正向链与官方锚点

### 2.1 近期正向链（Qwen panel）

| 版本 | 机制 | Qwen panel | 增量 |
|---|---|---:|---:|
| v075 / C76.4 | GQA head-local signed Hadamard H16/H32/H64 | 258.840 | — |
| v076 / C77 | all-shape gram64 激活精修 + GQA rotation | 260.060 | +1.22 |
| v080 / C80 | gram64 全覆盖（ratio 1.0, max 128） | 265.373 | +5.31 |
| v084 / C84 | gram64 全覆盖 + 5 轮坐标扫描 | 267.290 | +1.92 |
| v086 / C86 | Attention Q/K 共享 block-Hadamard + v084 | 267.308 | +0.02 |
| v100/v101 | clean + B1 GQRB + B2 PAWV diag-only | 293.797 | +26.49（+9.91%） |
| **v106 时间 parent** | **+ expansive-FFN CAT balance** | **294.273** | **+0.475（+0.16%）** |
| **v107 前一精度 parent** | **+ Global Activation-LRH exact Gram gate** | **295.157** | **+0.884（+0.30%）** |
| **v109 前一精度 parent** | **+ final deployed-Gram row gate** | **295.239** | **+0.082（+0.03%）** |
| **v110 前一精度 parent** | **+ final-Gram GALS 小预算** | **295.243** | **+0.003（+0.001%）** |
| **v111 前一精度 parent** | **+ L5a block-local permutation** | **295.482** | **+0.240（+0.081%）** |
| **v115 前一精度 parent** | **+ L6a rank-16 global LRH** | **295.681** | **+0.198（+0.067%）** |
| v116 前一精度 parent | + L6b wide rank-4 cross-block factor | 295.734 | +0.053（+0.018%） |
| v117 前一精度 parent | + L6c full `G_64` hierarchy coordinate sweep | 295.786 | +0.052（+0.018%） |
| v118 前一精度 parent | + L6d structured block-circulant factor | 295.808 | +0.022（+0.008%） |
| v119 C1a precision/time parent | + structured proposal vectorization | 295.808 | ±0（API −9.30%） |
| v121 C1b precision parent | + structured gradient refresh×2 | 295.811 | +0.003（Linear +0.000012） |
| **v124 precision parent** | **+ C1c structured rank-8** | **295.820** | **+0.009（Linear +0.000036）** |
| **v125 当前 precision-only parent** | **+ C1c `max_blocks=8`** | **295.848** | **+0.028（Linear +0.000110）** |

**关键观察**：本轮最大的一次跃迁（+9.89%）不是来自新算法，而是**把实验集合重写为 clean 单一路径**——删掉 dormant branch、统一路径后反而大幅变好。

### 2.2 官方锚点（新版 250/200 面板）

| 版本 | 官方分数 | 官方时间 |
|---|---:|---:|
| v031 / C39-FW | 21864 | 161.3s |
| v034 / C41b | 21864 | 159.4s |
| v051 / C47b | 22451 | 234s |
| v066 / C66 | 22557 | 217.2s |
| **v072 / C74** | **22662** | **226s** |
| 外部 youxilee/hif4 v2.7 | **24153** | 239s |

旧口径（不可直接比较）：v024/C21 `16043`、v025/C21-C `14437`、v030/C38 `14092`、v032/C40 `14432`。

外部本地基准：Qwen panel `250.327102`、Qwen native `369.527269`。当前 v125 本地领先外部 panel **+18.19%**，但官方尚未验证且 API 超时。L5d 逐组件审计表明 codec/dequant 没有差异，外部 joint residual/H32-H64 不能合规迁移；L5e 记录固定表示/接口到 `0.9` 的证据性不可达。

---

## 3. 已实现算法全景（按技术类别）

### 3.1 Attention

| 版本 | 机制 | 结果 |
|---|---|---|
| v003 / C1 | A1 真实 Attention 输出选择器 | **采纳**，Attention 0.3786 → 0.4497 |
| v004 / C2、v005 / C2a | 独立段 / query 段 CVaR | 拒绝（causal 回退） |
| v033 / C41 | scale-aware K 公共平移（全模型） | 拒绝：MHA +0.72%/+0.75%，但 GQA（Qwen）−0.88% |
| **v034 / C41b** | **同机制，仅 MHA 启用** | **采纳**，五模型无一负向，官方 21864 |
| v075 / C76.4 | GQA head-local signed Hadamard | 采纳，Qwen Attention 63.12 → 70.96 |
| v086 / C86 | Q/K 共享 block-Hadamard（4/8/16）+ 终选器 | 采纳，GPT-2 Attention 明显提升，OPT 小幅回退 |

现状：Attention mean `0.842039`，贡献 panel `168.408`（超过 Linear 的 `125.389`）。

### 3.2 Linear — 权重量化

| 版本 | 机制 | 结果 |
|---|---|---|
| v006–v011 | top-K 8×8/16×16/32×32/64×64 二次型 | 逐级小幅正向（+0.0006~+0.0110） |
| v013 / C10 | wide activation quadratic | **官方 15799**（旧口径），+486 |
| v023 / C20、v024 / C21 | exact discrete cross-gain / gated | 官方 16081 / 16043，**但输出监督喂给 Q(A)，不合规** |
| v025 / C21-C | 合规基线 | 官方 14437（旧口径锚点） |
| v027 / C23 | FULL64 Weight | 当时拒绝，机制后被推广 |
| v031 / C39-FW | wide-layer FULL64 calibration | **官方 21864**（新版） |
| v032 / C40 | robust Block-LDLQ 128 | **失败**：本地 +0.36pp，官方 −181 分 |
| v054–v056 | Weight headroom 覆盖 50%/75%/100% | 逐级正向（+4.73/+1.82/+0.59） |
| v058 / C58 | headroom E6M2 offsets ±6 | 拒绝（持平，无有效候选） |
| v063 / C63 | Linear 候选 `weight_sample` 512 行 | **采纳**，五模型 +17.84 |
| v064 / C64、v067 / C67 | 1024 行 / 640 行 | 拒绝（512 行更稳） |

### 3.3 Linear — 激活量化

| 版本 | 机制 | 结果 |
|---|---|---|
| v014–v020 | wide / all-width / gated activation 8×8、coverage 4%/8% | 逐级小幅正向 |
| v021 / C18、v022 / C19 | 激活/权重误差交叉项、cross-aware gain | 小幅正向 |
| v069 / C69 | 激活二次项 Gram-8 覆盖上限 12% | 采纳（+0.0038） |
| v076 / C77 | all-shape gram64 激活精修 | 采纳（+1.22） |
| v080 / C80、v084 / C84 | gram64 全覆盖、5 轮坐标扫描 | 采纳（+5.31、+1.92） |
| v066 / C66 | 动态激活损失覆盖目标 1.0 | 官方 22557（前一控制组） |
| **v072 / C74** | **JDRQ fixed-Q(A) hierarchy residual** | **官方 22662（当前本地归档冠军）** |
| v104 / A7 | 用量化后权重 `WqᵀWq` 替换浮点 `WᵀW` | **拒绝**：layer-1 `+0.525831`，full `−3.570607` 且 API `470.58s` |

### 3.4 坐标变换 / CAT 系

| 版本 | 机制 | 结果 |
|---|---|---|
| v026 / C22 | Linear R64 incoherence | 拒绝（所有分量回退） |
| v036 / C43 | analytic CAT-64 | 拒绝（Linear −0.901） |
| **v037 / C43b** | **CAT-64 β=0.25** | **采纳**（Linear +1.596，GPT-2 small 正向） |
| v038 / C43c | CAT-64 full-H selector | 拒绝（OPT −158，模型敏感） |
| v044 / C46a | CAT β 网格 {0.125,0.25,0.375} | 拒绝（OPT 结构性回退 −826） |
| v046 / C46b | CAT β 窄细化 {0.20,0.25,0.30} | 拒绝（OPT −12.06） |
| v049 / C49 | CAT block-Hessian operand metric | 拒绝（持平，仅增开销） |
| v050 / C47、v051 / C47b | CAT-aware 4→64 通道分组 + 0.5% 软门 | **采纳，官方 22451** |
| v052 / C47c | 1% 软门 | 拒绝（低于 v051） |
| v053 / C48 | CAT + 16/32 通道 micro-Hadamard | 拒绝（持平，未接受任何组合） |
| v061 / C61、v062 / C62 | CAT `WᵀW` 统计 1024 行 / 宽度分流 | v061 拒绝（Qwen −20.2），**v062 采纳**（+0.81） |

**CAT 的总体结论**：β=0.25 + 宽度分流 + 通道分组有效；β 网格、full-H selector、block-Hessian 度量均不稳定或无效。

### 3.5 输出目标 / A@W 系（合规用法）

| 版本 | 机制 | 结果 |
|---|---|---|
| v039 / C45b | fixed-Q(A) A@W 静态 Q(W) 选择器 | 拒绝（固定 Q(A) 产品目标过拟合） |
| v040 / C45c | 原始 A@W + max-dim 4096 | 采纳（+0.54） |
| v041 / C44 | MR-GPTQ parent full-H 覆盖 97% | 拒绝（−4.72，误差扩散） |
| v042 / C45e | 多折 A@W + max-dim 4096 | 采纳（+5.19） |
| v043 / C45f | adaptive headroom {−4..4} + 多折 A@W | 采纳（+15.89，OPT +10.70） |
| v045 / C45g | 放开 Qwen 4864-wide headroom | 拒绝（持平，仅增开销） |
| v047 / C45h | 全宽多折 A@W，预算 8192 | 拒绝（−0.47） |
| v048 / C45i | 按输出行数限制静态 A@W | 采纳（+0.09） |
| v057 / C57 | 产品候选比例 25% | 拒绝（三模型均回退） |
| v059 / C59 | 逐 64-block A@W headroom 混合 | 拒绝（OPT −14.67，严重过拟合） |
| v060 / C60 | A@W 产品条件步长网格 | 拒绝（−1.43，自由度过高） |
| v065 / C65 | A@W 折间软混合 0.50 | 拒绝（0.25 更稳） |
| v068 / C68 | A@W 静态块预算 15% | 拒绝（12.5% 更稳） |
| v072 / C74 | JDRQ fixed-Q(A) hierarchy residual（down-proj） | 采纳，Qwen +6.45，无 C71 式崩溃 |
| v073 / C75 | source-aware activation + project-only gram64 + fixed-Q(A) JDRQ | 采纳（Qwen 297.54） |
| v074 / C75 | rowwise JDRQ + wide gram64 hierarchy + H32/H64 候选池 | 采纳（Qwen 298.38；**H32/H64 output reranker 因合规审计禁用**） |
| v070 / C70 | **外部 v2.6 X/W 联合残差补偿移植**（3 轮 GS） | **拒绝**：GPT-2 +6.27，但 OPT −0.62、Qwen −6.99 |
| v071 / C71 | proj H32/H64 + 终量化器候选排序 | **拒绝**：GPT-2 +8.33、Qwen +30.74，但 OPT **−139.32 灾难回退** |

**A@W 系的总体结论**：有效但极度过拟合敏感。稳定配方是"多折 + 尺寸上限 + 自适应 headroom"；全宽、逐块、大步长网格、高混合比例都会过拟合。**外部 v2.6 的核心机制（C70）在本工程上未能复现其收益。**

### 3.6 尺度 / 码域搜索

| 实验 | 机制 | 结果 |
|---|---|---|
| v028 | activation scale-code oracle | 仅诊断，无提交源 |
| v029 / C29 | HAES probe | 拒绝（无行为变化） |
| **E0-G** | **完整 255 个 E6M2 code 尺度 oracle** | **除 `v` 的 activation Gram（0.6302%）外，各 role 总 gap 均 <0.1%** → 顶层 scale 搜索空间已耗尽 |
| **D0** | **五模型 layer-1/2/3 全 role dashboard** | 少数层/role 的 activation-Gram 局部 gap 达 2–6%，但跨模型/跨层不稳定；只保留诊断，不部署全局搜索 |

### 3.7 L0 Linear ceiling / error decomposition（2026-08-30）

新增 evaluator-side 五层分层诊断，固定 Qwen `layers={0,5,11,17,23}`、七个 Linear role、
两折校准和 255 个合法 E6M2 scale code。输出四个 evaluator arms：当前双侧量化、
权重无损、激活无损和双侧无损；同时记录 weight-plain、weight-Gram、activation-Gram
的独立 block oracle。完整结果见 [`l0-linear-ceiling-qwen.json`](../artifacts/oracle_dashboard/l0-linear-ceiling-qwen.json)。

| arm / oracle | 五层×七 role mean |
|---|---:|
| both player | 0.52301943 |
| weight perfect | 0.70417026 |
| activation perfect | 0.82035698 |
| both perfect | 1.00000000 |
| weight plain 255-code gap | 0.0229% |
| weight Gram 255-code gap | 0.6065% |
| activation Gram 255-code gap | 0.6410% |

L0 的直接结论是：整体 activation-side headroom 大于 weight-side，但 q/k 为
weight-dominant，v 为 transform-coupled，`fc_gate/fc_up/proj` 的 activation headroom
最大；scale-code 搜索本身不可能填补 `linear_mean` 到 `0.9` 的大缺口。L1 已完成
完整 hierarchy 写回复验但 cross-fold 泛化不足，下一步进入低自由度 FFN 结构候选；
L0 oracle 只作诊断，不进入部署路径。

E0-G 明细：

| role | 侧别 | 改善 blocks | 总相对 gap | 最大单 block |
|---|---|---:|---:|---:|
| fc_gate | weight / Gram | 5 / 19（/448） | 0.0572% / 0.0836% | 13.76% / 17.33% |
| fc_up | weight / Gram | 1 / 13（/448） | 0.0709% / 0.0470% | 22.20% / 12.82% |
| v | weight / **Gram** | 4 / **60**（/448） | 0.0313% / **0.6302%** | 7.47% / **19.32%** |
| proj | weight / Gram | 18 / 50（/2432） | 0.0390% / 0.0475% | 9.94% / 20.53% |

D0 三层均值（weight / activation-Gram）如下，原始 JSON 保存在
`artifacts/oracle_dashboard/e0g-*-layer{1,2,3}.json`：

| 模型 | layer 1 | layer 2 | layer 3 |
|---|---:|---:|---:|
| gpt2-small | 0.0744% / 0.3161% | 0.0673% / 0.6096% | 0.0667% / 2.7787% |
| gpt2-medium | 0.0533% / 0.3912% | 0.0642% / 0.8577% | 0.0569% / 0.5605% |
| opt-125m | 0.0366% / 0.3317% | 0.0304% / 6.6520% | 0.0596% / 1.6992% |
| pythia-160m | 0.0934% / 0.2091% | 0.0591% / 0.2362% | 0.0840% / 0.1941% |
| qwen2.5-0.5b | 0.0677% / 2.0133% | 0.0544% / 0.3222% | 0.0642% / 0.3667% |

---

## 4. 2026-08-30 的结构性回退与审计修正

融合方案中的 E1–E6、以及 36000 计划的 A1–A6，已经有 v087–v095 的实验记录；v095 的最终 gate 仍存在实现问题，v092 的旧层级写回问题则已由 v105 重新实现并完成预筛，因此下表的“失败”只对已运行配置成立：

| 版本 | 机制（对应计划项） | panel | 相对 stable parent | 时间 | 失败原因 |
|---|---|---:|---:|---:|---|
| v087 | E1 渐进 full-hierarchy HSDQ | 290.924 | **−2.831** | 693s | q/v/proj 回退，超 420s |
| v088 | A2 扩张 FFN 稀疏 row-block（1/2/5%） | 292.832 | −0.923 | 385s | fc_gate/fc_up 回退 |
| v089 | A3 扩张 FFN rowwise block-leverage（0.5/1/2%） | 293.250 | −0.505 | 385s | fc_gate/fc_up 仍回退 |
| v090 | A4 blockwise BOAT-2 指数调度 | 292.978 | −0.777 | 368s | q/k/v/o 回退 |
| v091 | A5 joint-fold 离线 A@W HSDQ | 284.595 | **−9.160** | 358s | q/k/v/o/proj 严重回退 |
| v092 | A3 真正跨 block LRH（rank-8, max 4） | 292.427 | −1.328 | 382s | 保存源码沿用 parent denominator；旧写回审计描述已更正 |
| **v105** | **L1 full-hierarchy LRH 原子写回** | **0.523019 screen** | **0** | **265.87s screen** | **70 fold candidates，1 cross-fold admitted，最终 0/35 case 改变 parent；不跑 full-layer** |
| **v105** | **L1 full-hierarchy cross-block LRH（scale/lv2/lv3/mantissa 原子写回）** | **0.523019 screen** | **0 vs L0** | **265.87s screen** | **正确实现；70 候选仅 1 cross-fold admitted，0/35 final change，未触发 full-layer** |
| v093 | A4 完整 CAT 式 BOAT-2（balance + 层级置换 + Householder） | 283.160 | **−10.595** | 601s | 单层正向未迁移，超 420s |
| v094 | A5 frozen-Q(A) ridge / Qronos（η=1/8, λ=1e-4） | 293.755 | **持平** | 456s | 无精度增益且超时 35.73s |
| v095 | A6 全局 Activation-LRH（rank-8, 10% 能量） | 282.617 | **−11.138** | 374s | 旧 MSE gate；不能直接否定 Gram-gated 修复 |

**这个模式必须被正视，但不能过度外推**：

1. 失败幅度从 −0.5 到 −11.1，**不是噪声**（E0-G 已证明单点噪声 <0.1%）。
2. 多数结构性改动确实撞墙，说明当前根是一个**很强的局部最优**；v105 已验证正确的 full-hierarchy 写回仍无跨 fold 增益，但 v107/v109/v110 证明修正部署 Gram gate 后仍有约 `+0.970` panel 的累计精度空间。
3. 多数实现都接近或超过 420s，说明当前根的精度是靠**大量校准计算**换来的，再加机制就超时。
4. 唯一接近成功的是 v094（精度持平），但它也超时。
5. 历史经验一致：v071（OPT −139）、v044（OPT −826）、v061（Qwen −20.2）都说明**跨模型不一致是主要失败模式**。

---

## 5. 已执行方向与后续工作

### 5.1 计划中方向的最新裁决

| 方向 | 说明 | 评估 |
|---|---|---|
| **B1 GQRB / FASA 扩展** | Attention 侧的 group-wise Q/K rotation / flash-attention 敏感对齐 | **已执行并采纳**：v098 margin，panel `293.793700` |
| **B2 PAWV（V 路径优化）** | attention probability token-row Hessian 的 V refinement | **已执行并采纳 diag-only**：v100，panel `293.797301`；rank-8 跨 token 变体回退 |
| **C0 全模型/全角色/宽形状综合选择** | 五模型固定缓存确认，Qwen 主模型排序、其他模型软 guardrail | **已执行并确认**：v101；Qwen panel `293.797301` |
| **v 的稀疏 GALS-C 插件** | E0-G 唯一正向信号（activation Gram 0.6302%，60/448 blocks） | **已执行但拒绝部署**：v102 layer-1 `335.988995`，Linear 回退 |
| **role-aware GALS-C** | 只在 attention-shaped role 开启候选 | **已执行但拒绝部署**：v103 layer-1 `335.978356`，q/k/v 回退 |
| **量化后权重 Gram 激活 Hessian** | 用 `WqᵀWq` 贴近部署权重 | **已执行但拒绝部署**：v104 full `290.226694`，API 超时 |
| **L1 full-hierarchy Weight-LRH** | 跨 block rank-8；scale/lv2/lv3/mantissa 原子写回 | **已执行但拒绝部署**：v105 screen `0.523019429222563` 与 L0 逐条相同；不再扩大 rank/block/sweep |
| **L3 Global Activation-LRH Gram gate** | rank-8 off-block proposal；最终用部署 `WqᵀWq` 精确 gate | **已执行并采纳为精度 parent**：v107 full `295.157057`，Linear `0.5069966356`；4-block 较 v106 `+0.884423` panel；时间 `481.04s` 只作记录 |
| **L4a final deployed-Gram row gate** | expansive shape 双候选；完整 `G_q` 逐行选择 v107 parent 与 final-Gram proposal | **已执行并采纳为精度 parent**：v109 full `295.239309`，Linear `0.5073256468`；较 v107 `+0.082253` panel；时间 `517.29s` 只作记录 |
| **L4b final-Gram GALS 小预算** | 最多 4 个高损失 block，解析 E6M2 offset 候选，校准增益后才启用，并用完整 `G_q` 行级 gate | **已执行并采纳为前一 precision parent**：v110 full `295.242780`、Linear `0.5073395278`，较 v109 `+0.003470` panel；时间 `701.90s` 只作记录 |
| **L5a block-local permutation** | 每个 64 维 hierarchy block 的压力排序/低高交错排列；与 `D`、signed-Hadamard 组成等价 `T=DPR`，两折 operand-local gate | **已执行并采纳为前一 parent**：v111 full `295.482473`、Linear `0.5082983001`，较 v110 `+0.239693` panel；时间 `726.094s` 只作记录 |
| **L6a rank-16 global LRH** | 窄输入 off-block residual 的 rank-16 factor；proposal 后完整部署 `G_q` 逐行 gate | **已执行并采纳为前一 parent**：v115 full `295.680651`、Linear `0.5090910148`，较 v111 `+0.198179` panel；时间 `716.483s` 只作记录 |
| **L6b wide rank-4 factor** | `d>1024,d<=8192` 的压缩 off-block range factor；proposal 后完整部署 `G_q` 逐行 gate | **已执行并采纳为前一 parent**：v116 full `295.734045`、Linear `0.5093045894`，较 v115 `+0.053394` panel；唯一正向角色为 `proj(d=4864)`，时间 `739.425s` 只作记录 |
| **L6c full `G_64` hierarchy** | 固定 scale、最多 4 个高损 block 的 `lv2/lv3` 有界坐标 sweep；完整 `G_64` 增量与部署 `G_q` gate | **已执行并采纳为前一 parent**：v117 full `295.785829`、Linear `0.5095117268`，较 v116 `+0.051784` panel；7 role 均不降，时间 `2019.475s` 只作记录 |
| **L6d structured block-circulant factor** | 宽输入最多 4 个 `64×64` kernel + circular-distance coefficient 生成跨 block proposal，完整部署 `G_q` gate | **已执行并采纳为前一 parent**：v118 full `295.808212`、Linear `0.5096012555`，较 v117 `+0.022382` panel；`proj(d=4864)` 正向，时间 `2249.746s` 只作记录 |
| **C1b structured gradient refresh** | 每个 selected block 后刷新 structured proposal gradient，并对 block rank list 扫两轮；候选仍由完整部署 `G_q` 逐行 gate | **已执行并采纳为精度证据 v121**：screen `0.5333964596`；full `295.811281`、Linear `0.5096135327`，较 v119 `+0.003069` panel；API `2180.450s`，用户确认官方 runtime timeout，禁止提交 |
| **C1c structured rank/budget** | 逐变量扫描 kernel rank 与 selected block 数；rank 2、block 2 screen 回退，rank 8 与 block 8 full-layer 正向 | **已执行并采纳 v125 precision-only**：v124 rank-8 full `295.820229`，v125 `max_blocks=8` full `295.847849`、Linear `0.5097598050`，较 v124 `+0.027620` panel；API `2653.580s`，runtime invalid |
| **L5b sparse Schur** | 最多两对跨 block PSD Schur proposal，完整部署 Gram 逐行 gate | **已执行并拒绝**：v112 screen Linear `0.5308551016`，较 v111 `-0.0010318441` |
| **L5c operand-local meta-router** | 八维静态特征、两折一层 stump，在既有子路径中选择 | **已执行但 no-op**：v113 screen `0.5318869457`，逐 case 等于 v111 |
| **L5d 外部组件审计 / sampling** | 对比 youxilee/hif4 v2.7 的 codec、层级、采样、transform、state；单变量 stride sampling | **已执行并拒绝**：v114 screen `0.5273114999`；codec/dequant parity 为 0，joint residual/H32-H64 标记不可行动 |
| **L5e 表示族可达性** | 255-code scale oracle、单侧理想臂、跨 block coupling | **已完成诊断**：screen `0.5318869457`，到 `0.9` 需减少 `78.64%` 剩余误差；固定 frame/state 接口记录为证据性不可达 |

### 5.2 已判定不适用 / 排除

| 方向 | 排除依据 |
|---|---|
| Four-over-six（M=4/M=6 自适应） | HiF4 的 mantissa 是**均匀**网格 `{0,0.25,…,1.75}`，不存在 NVFP4 E2M1 的 4→6 断层 |
| 跨层误差累积纠正（Qronos 跨层、RDQ 级联漂移） | 评测器每层提供**未级联量化的真实激活**，不存在跨层传播 |
| SpinQuant / FlatQuant 的 task-output loss | 会把输出监督引入 `Q(A)`，不合规 |
| 全局 E6M2 / 顶层 scale 扩张 | E0-G：各 role 总 gap <0.1% |

### 5.3 L6 已完成与 C1 方向状态（唯一活跃计划）

| 方向 | 说明 |
|---|---|
| **L6a rank-16 global LRH** | 窄输入 rank-8 off-block factor 的单变量扩展；完整 `G_q` gate |
| **L6b 宽输入 rank-4 factor** | 已完成并采纳 v116：将压缩 off-block factor 扩展到 `d>1024`（重点 4864 proj），最多 4 个高损 block，state 一个 CPU tensor |
| **L6c 完整 `G_64` hierarchy** | 已完成并采纳 v117：固定 scale 的层级坐标更新，禁止 group-only objective |
| **L6d block-circulant / DCT factor** | 已完成并采纳 v118：用少量结构化 block kernel 近似跨 block Gram，避免逐通道 dense state |
| **L6e checkpoint** | 已完成：真实 `proj(d=4864)` 共 2048 proposals、exact gate 接受 71（3.4668%），`J_64` 下降 0.0991%，结构化增量 state 66,752 bytes；L6 已归档并转 C1 |

| C1a proposal vectorization | 已完成并采纳 v119：逐 coordinate 的 15-level Python loop 改为批量 tensor 运算；与 v118 `atol=1e-6` 等价，full API `-9.30%`，exact `G_q` gate 保持 |
| C1b structured gradient refresh | 已完成：v120 一次 refresh screen `0.5333730058` 被拒绝；v121 两轮 refresh screen `0.5333964596`、full panel `295.811281` 被采纳 |
| C1c rank/budget scan | **已完成**：v122 rank-2、v123 max-blocks-2 已拒绝，v124 rank-8 与 v125 `max_blocks=8` 均 full 正向；不再增加 block budget |
| C2 cross-model audit | 待执行：OPT/Pythia 低成本复筛，检查 `proj` 之外的结构性回退 |
| C3 state/time checkpoint | 待执行：在精度队列完成后再恢复 `<420s` 最终门禁 |

---

## 6. 结论与建议（基于全景）

1. **当前根是强局部最优但仍有精度空间**：多数结构性方向失败且失败幅度远超噪声；v105 已验证正确的 full-hierarchy 写回在两折上缺乏泛化，而 v107、v109、v110 与 v111 的部署 Gram/等价坐标 gate 修复带来可复现正向增益。
2. **历史最大跃迁来自"重写为 clean 单一路径"（+9.89%）**，而不是新算法。这提示下一轮收益可能来自**路径精简与计算重分配**，而非新增机制。
3. **P0 仍是提交当前根**：本地 +18.19% 的领先从未兑现为官方分；当前根已超 420s，真实官方反馈和 C3 压缩仍是必要条件。
4. **本轮新增十一个精度 parent及一个等价时间 parent**：v107 的 Gram-gated Global Activation-LRH 将 panel 提升到 `295.157057`，v109 的 final deployed-Gram row gate 提升到 `295.239309`，v110 的 final-Gram GALS 再提升到 `295.242780`，v111 的 L5a block-local permutation 提升到 `295.482473`、Linear mean `0.5082983001`，v115 的 L6a rank-16 global LRH 再提升到 `295.680651`、Linear mean `0.5090910148`，v116 的 L6b wide rank-4 factor 再提升到 `295.734045`、Linear mean `0.5093045894`，v117 的 L6c full `G_64` hierarchy 再提升到 `295.785829`、Linear mean `0.5095117268`，v118 的 L6d structured factor 再提升到 `295.808212`、Linear mean `0.5096012555`，v119 C1a 保持这些分数逐位不变并将 API 降至 `2040.504690s`，v121 C1b 两轮 refresh 再提升到 `295.811281`、Linear mean `0.5096135327`，v124 C1c rank-8 再提升到 `295.820229`、Linear mean `0.5096493233`，v125 `max_blocks=8` 再提升到 `295.847849`、Linear mean `0.5097598050`；v106 保留为历史时间 parent。L5b/v112、L5c/v113、L5d/v114、v120、v122、v123 已拒绝，L5e、L6 与 C1 已完成；当前唯一 active 计划转入 C2/C3。
5. **L5e/L6e/C1b/C1c 的证据约束下一轮搜索**：固定 frame 的单侧理想臂最高 `0.8188905`，scale oracle 不能填补 0.9 缺口，896 输入宽度 `ρ_off` 平均高达 weight `0.76125`、activation `0.88382`；L6e 只观察到 3.47% proposal recall，C1b/C1c 累计取得 `+0.039637` panel，因此 C1c 已停止，下一步转入跨模型审计与状态/时间压缩，不重复局部 offset 或输出残差。
6. **警惕跨模型不一致**：这是本工程最主要的失败模式（v044、v061、v071、v091 等）。L6 screen 若出现正向，full-layer 前应增加 OPT/Pythia 低成本复筛，而不是全层跑完再判。
