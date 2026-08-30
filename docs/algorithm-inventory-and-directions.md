# HiF4 算法全景：已实现、已验证效果与未实现方向

> 整理日期：2026-08-31
> 数据来源：`solutions/README.md`（v000–v110）、`docs/current-solution-status.md`、`docs/archive-implementation-audit.md`、`logs/execution/2026-08-30-e0g-scale-oracle.md`、`logs/execution/2026-08-30-e0g-multimodel-dashboard.md`、`logs/execution/2026-08-30-a7-quant-weight-gram.md`、`logs/execution/2026-08-30-l1-full-hierarchy-lrh.md`、`logs/execution/2026-08-31-v110-l4b-gals-final-gated-qwen-full.md`。
> 口径纪律：本地只能比 **Qwen 同口径 panel**（`250·g_L + 200·g_A`）；五模型合计 `1085.743597` 只用于检查跨模型结构性回退，**禁止**与官方分数做差值。官方评测集为 250 Linear + 200 Attention case，时间上限 **420s**。

---

## 1. 当前根：算法构成与效果

根 `solution.py`（规范 LF SHA `3abf9beb7ba50285b65344ce94773350eca16a24ce36a296db1401b9bafeb1ec`）为 clean 单一路径，加入 v106 expansive-FFN CAT balance、B1 GQRB、B2 PAWV diag-only、v107 Gram-gated Global Activation-LRH、v109 L4a final deployed-Gram row gate 和 v110 L4b final-Gram GALS。当前精度最高版本为 v110；v106 仍是时间 parent。E0-C 的两个 GALS 稀疏变体（v102/v103）、A7 量化后权重 Gram（v104）和 L1 v105 full-hierarchy LRH 均已归档，详见 [`归档实现审计`](archive-implementation-audit.md)。

| 组件 | 内容 |
|---|---|
| **BOAT** | 全层统一对角 alpha + 固定 signed-Hadamard；搜索在合法域内完成，只改变离线 `weight_params` |
| **cross-fold Weight-HSDQ** | fold 1 生成的候选必须改善 fold 2，最终只改变离线 `weight_params` |
| **Gram-hierarchy Activation-HSDQ** | 从静态变换后权重计算 64 维 Gram block，先按二次型选层级与 E6M2 offset，再做最多 128 个 block、2 轮坐标扫描；state 只含 CPU 静态 `gram64`、BOAT 逆缩放与整数/符号配置 |
| **Expansive-FFN CAT balance** | 仅对 `rows > channels` 路由，固定 α=0.25 RMS 对角 balance；v106 仅 fc_gate 正向 |
| **Global Activation-LRH Gram gate** | 只在输入宽度 ≤1024 的窄形状生成 rank-8 off-block proposal；最终离散候选用部署量化权重 `G_q=W_qᵀW_q` 逐行精确二次型门控；v107 精度正向 |
| **L4a final deployed-Gram row gate** | 仅 expansive `rows > channels` 且 `channels <=1024`；v107 parent 与 final-Gram 候选用完整 `G_q` 逐行比较，v109 精度正向 |
| **Attention 输出感知 shortlist** | reciprocal RMS/K-centering/共享 Hadamard + B1 GQRB 2×2/4×4 group-local mixing；B2 PAWV 用 attention probability 的 token-row 对角 Hessian 做 V refinement；V 保持独立合法 HiF4 编码 |

**实测**（Qwen2.5-0.5B 全 24 层，`seq=128/calib=2/test=4/amax6`，缓存只读）：

```text
Linear mean       0.507340      Attention mean  0.842039
Qwen panel        295.242780    native total    421.767954
six-API time      701.900553 s  wall time       734.220364 s   （探索阶段只记录时间）
```

分角色 Linear gain：`q 0.6194 / k 0.6270 / v 0.5711 / o 0.4877 / fc_gate 0.3927 / fc_up 0.4321 / proj 0.4214`。

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
| **当前根 v106** | **+ expansive-FFN CAT balance** | **294.273** | **+0.475（+0.16%）** |
| **v107 前一精度 parent** | **+ Global Activation-LRH exact Gram gate** | **295.157** | **+0.884（+0.30%）** |
| **v109 前一精度 parent** | **+ final deployed-Gram row gate** | **295.239** | **+0.082（+0.03%）** |
| **v110 当前精度 parent** | **+ final-Gram GALS 小预算** | **295.243** | **+0.003（+0.001%）** |

**关键观察**：本轮最大的一次跃迁（+9.89%）不是来自新算法，而是**把实验集合重写为 clean 单一路径**——删掉 dormant branch、统一路径后反而大幅变好。

### 2.2 官方锚点（新版 250/200 面板）

| 版本 | 官方分数 | 官方时间 |
|---|---:|---:|
| v031 / C39-FW | 21864 | 161.3s |
| v034 / C41b | 21864 | 159.4s |
| v051 / C47b | 22451 | 234s |
| **v066 / C66** | **22557** | **217.2s** |
| 外部 youxilee/hif4 v2.7 | **24153** | 239s |

旧口径（不可直接比较）：v024/C21 `16043`、v025/C21-C `14437`、v030/C38 `14092`、v032/C40 `14432`。

外部本地基准：Qwen panel `250.327102`、Qwen native `369.527269`。当前 v110 本地领先外部 panel **+17.94%**，但官方尚未验证。

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
| v066 / C66 | 动态激活损失覆盖目标 1.0 | **官方 22557**（本地归档冠军） |
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
| **L4b final-Gram GALS 小预算** | 最多 4 个高损失 block，解析 E6M2 offset 候选，校准增益后才启用，并用完整 `G_q` 行级 gate | **已执行并采纳为当前 parent**：v110 full `295.242780`、Linear `0.5073395278`，较 v109 `+0.003470` panel；时间 `701.90s` 只作记录 |

### 5.2 已判定不适用 / 排除

| 方向 | 排除依据 |
|---|---|
| Four-over-six（M=4/M=6 自适应） | HiF4 的 mantissa 是**均匀**网格 `{0,0.25,…,1.75}`，不存在 NVFP4 E2M1 的 4→6 断层 |
| 跨层误差累积纠正（Qronos 跨层、RDQ 级联漂移） | 评测器每层提供**未级联量化的真实激活**，不存在跨层传播 |
| SpinQuant / FlatQuant 的 task-output loss | 会把输出监督引入 `Q(A)`，不合规 |
| 全局 E6M2 / 顶层 scale 扩张 | E0-G：各 role 总 gap <0.1% |

### 5.3 尚未尝试的方法论方向（非算法）

| 方向 | 说明 |
|---|---|
| **提交当前精度 parent 验证兑换率** | 本地领先外部约 +17.91%，但从未提交官方；当前 precision parent 仍待最终时间压缩 |
| **重构 / 简化主路径** | 历史上唯一的大跃迁（+9.89%）来自"重写为 clean 单一路径"，而非新算法；v106 CAT 已保持单一路径，后续只做有证据的精简 |
| **外部实现的差异审计** | C70 移植失败（Qwen −6.99），但外部能到 24153。需要逐组件比对实现差异（不只是机制名） |
| **跨模型一致性作为硬门禁** | 九连败中多数是"单模型正向、他模型回退"。应在单层/单模型阶段就用异构模型（OPT 尤其敏感）复筛 |

---

## 6. 结论与建议（基于全景）

1. **当前根是强局部最优但仍有精度空间**：多数结构性方向失败且失败幅度远超噪声；v105 已验证正确的 full-hierarchy 写回在两折上缺乏泛化，而 v107、v109 与 v110 的部署 Gram gate/GALS 修复带来可复现正向增益。
2. **历史最大跃迁来自"重写为 clean 单一路径"（+9.89%）**，而不是新算法。这提示下一轮收益可能来自**路径精简与计算重分配**，而非新增机制。
3. **P0 仍是提交当前根**：本地 +17.37% 的领先从未兑现为官方分；九连败也说明本地 panel 已进入饱和区，更需要真实官方反馈来校准方向。
4. **本轮新增三个精度 parent**：v107 的 Gram-gated Global Activation-LRH 将 panel 提升到 `295.157057`，v109 的 final deployed-Gram row gate 提升到 `295.239309`，v110 的 final-Gram GALS 再提升到 `295.242780`、Linear mean `0.5073395278`；v106 保留为时间 parent。L0–L4 已完成，已归档 v2 计划并转入唯一活跃的 v3 L5 计划。
5. **警惕跨模型不一致**：这是本工程最主要的失败模式（v044、v061、v071、v091 等）。任何新机制都应在单层阶段就用 OPT / Pythia 复筛，而不是全层跑完再判。
