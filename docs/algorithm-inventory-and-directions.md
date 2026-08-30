# HiF4 算法全景：已实现、已验证效果与未实现方向

> 整理日期：2026-08-30
> 数据来源：`solutions/README.md`（v000–v104）、`docs/current-solution-status.md`、`docs/archive-implementation-audit.md`、`logs/execution/2026-08-30-e0g-scale-oracle.md`、`logs/execution/2026-08-30-e0g-multimodel-dashboard.md`、`logs/execution/2026-08-30-a7-quant-weight-gram.md`。
> 口径纪律：本地只能比 **Qwen 同口径 panel**（`250·g_L + 200·g_A`）；五模型合计 `1085.743597` 只用于检查跨模型结构性回退，**禁止**与官方分数做差值。官方评测集为 250 Linear + 200 Attention case，时间上限 **420s**。

---

## 1. 当前根：算法构成与效果

根 `solution.py`（规范 LF SHA `617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`）为 clean 单一路径，并在其上加入 B1 GQRB 与 B2 PAWV diag-only。当前最高版本由 C0 五模型确认（v101）复核；E0-C 的两个 GALS 稀疏变体（v102/v103）和 A7 量化后权重 Gram（v104）均已归档，未进入主线。v092/v095/v099 的负结果存在实现审计保留，详见 [`归档实现审计`](archive-implementation-audit.md)。

| 组件 | 内容 |
|---|---|
| **BOAT** | 全层统一对角 alpha + 固定 signed-Hadamard；搜索在合法域内完成，只改变离线 `weight_params` |
| **cross-fold Weight-HSDQ** | fold 1 生成的候选必须改善 fold 2，最终只改变离线 `weight_params` |
| **Gram-hierarchy Activation-HSDQ** | 从静态变换后权重计算 64 维 Gram block，先按二次型选层级与 E6M2 offset，再做最多 128 个 block、2 轮坐标扫描；state 只含 CPU 静态 `gram64`、BOAT 逆缩放与整数/符号配置 |
| **Attention 输出感知 shortlist** | reciprocal RMS/K-centering/共享 Hadamard + B1 GQRB 2×2/4×4 group-local mixing；B2 PAWV 用 attention probability 的 token-row 对角 Hessian 做 V refinement；V 保持独立合法 HiF4 编码 |

**实测**（Qwen2.5-0.5B 全 24 层，`seq=128/calib=2/test=4/amax6`，缓存只读）：

```text
Linear mean       0.501558      Attention mean  0.842039
Qwen panel        293.797301    native total    417.882506
six-API time      392.423565 s  wall time       424.693400 s   （API 在 420s 内）
```

分角色 Linear gain：`q 0.6166 / k 0.6205 / v 0.5636 / o 0.4835 / fc_gate 0.3751 / fc_up 0.4303 / proj 0.4214`。

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
| **当前根 v100/v101** | **clean + B1 GQRB + B2 PAWV diag-only** | **293.797** | **+26.49（+9.91%）** |

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

外部本地基准：Qwen panel `250.327102`、Qwen native `369.527269`。当前根本土领先外部 panel **+17.37%**，但官方尚未验证。

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

融合方案中的 E1–E6、以及 36000 计划的 A1–A6，已经有 v087–v095 的实验记录；但其中 v092 的层级写回、v095 的最终 gate 存在实现问题，因此下表的“失败”只对已运行配置成立，不等于这些算法的理论上限：

| 版本 | 机制（对应计划项） | panel | 相对 stable parent | 时间 | 失败原因 |
|---|---|---:|---:|---:|---|
| v087 | E1 渐进 full-hierarchy HSDQ | 290.924 | **−2.831** | 693s | q/v/proj 回退，超 420s |
| v088 | A2 扩张 FFN 稀疏 row-block（1/2/5%） | 292.832 | −0.923 | 385s | fc_gate/fc_up 回退 |
| v089 | A3 扩张 FFN rowwise block-leverage（0.5/1/2%） | 293.250 | −0.505 | 385s | fc_gate/fc_up 仍回退 |
| v090 | A4 blockwise BOAT-2 指数调度 | 292.978 | −0.777 | 368s | q/k/v/o 回退 |
| v091 | A5 joint-fold 离线 A@W HSDQ | 284.595 | **−9.160** | 358s | q/k/v/o/proj 严重回退 |
| v092 | A3 真正跨 block LRH（rank-8, max 4） | 292.427 | −1.328 | 382s | 首次实现真跨块 Hessian，但 hierarchy 写回存在问题，需修复复验 |
| v093 | A4 完整 CAT 式 BOAT-2（balance + 层级置换 + Householder） | 283.160 | **−10.595** | 601s | 单层正向未迁移，超 420s |
| v094 | A5 frozen-Q(A) ridge / Qronos（η=1/8, λ=1e-4） | 293.755 | **持平** | 456s | 无精度增益且超时 35.73s |
| v095 | A6 全局 Activation-LRH（rank-8, 10% 能量） | 282.617 | **−11.138** | 374s | 单层门禁失败，全层仍回退；最终 gate 与 Gram 目标错位，需修复复验 |

**这个模式必须被正视，但不能过度外推**：

1. 失败幅度从 −0.5 到 −11.1，**不是噪声**（E0-G 已证明单点噪声 <0.1%）。
2. 多数结构性改动确实撞墙，说明当前根是一个**很强的局部最优**；但 v092/v095 需修复后再决定是否加入停止清单。
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
| **提交当前根验证兑换率** | 本地领先外部约 +17.37%，但从未提交官方；先校准刻度仍是最高优先级 |
| **重构 / 简化主路径** | 历史上唯一的大跃迁（+9.89%）来自"重写为 clean 单一路径"，而非新算法。可考虑对当前根再做一轮路径精简与 dead-branch 清理 |
| **外部实现的差异审计** | C70 移植失败（Qwen −6.99），但外部能到 24153。需要逐组件比对实现差异（不只是机制名） |
| **跨模型一致性作为硬门禁** | 九连败中多数是"单模型正向、他模型回退"。应在单层/单模型阶段就用异构模型（OPT 尤其敏感）复筛 |

---

## 6. 结论与建议（基于全景）

1. **当前根是强局部最优**：多数结构性方向失败且失败幅度远超噪声；但 v092/v095 的修复版尚未验证，不能把当前实现写成理论上限。
2. **历史最大跃迁来自"重写为 clean 单一路径"（+9.89%）**，而不是新算法。这提示下一轮收益可能来自**路径精简与计算重分配**，而非新增机制。
3. **P0 仍是提交当前根**：本地 +17.37% 的领先从未兑现为官方分；九连败也说明本地 panel 已进入饱和区，更需要真实官方反馈来校准方向。
4. **本轮未再有已验证的部署空白**：PAWV diag-only 已采纳；GALS-C 两个稀疏变体与 A7 量化 Gram 均已实测并回退，但 final-weight Gram + role-id GALS 仍未验证。后续按唯一活跃计划先修复 LRH，再做小预算 GALS。
5. **警惕跨模型不一致**：这是本工程最主要的失败模式（v044、v061、v071、v091 等）。任何新机制都应在单层阶段就用 OPT / Pythia 复筛，而不是全层跑完再判。
