# HiF4 方向探索全景与 A@W 拟合经验教训

> 日期：2026-09-02
> 状态：策略总结（非执行计划；执行以 `docs/superpowers/plans/` 唯一活动计划为准）
> 本地协议：`proxy-v2` + `paired-effect-panel-v1`（旧 `official-shape-v1` 仅作历史诊断，不可迁移比较）

本文档回答两个问题：

1. **v000–v154 已经探索了哪些方向**，哪些闭环、哪些未闭环、哪些完全没碰；
2. **A@W 输出域拟合为什么理论上应该大幅提升 Linear 精度，却在官方评测中系统性失败**。

所有官方分数/时间来自用户回传，本地数字来自 `artifacts/official_eval/`。证据优先级：用户明确官方结果 > 活动计划已确认事实 > 归档 result/log > 本地 JSON > 未验证推测。

---

## 1. 官方事实锚点

| 版本 | 官方分数 | 官方时间 | 裁决 | 备注 |
|---|---:|---:|---|---|
| v74 | 22750（旧权重）→ **14561**（当前评测集回传） | 239.387 s → **188.9 s** | pass | **旧权重最高；当前评测集远低于 v84/v86，非安全基线** |
| v84 | 16517 | 252.563 s | pass | 新权重 |
| **v86** | **16744** | **222.7 s** | **pass** | **当前仓库内官方基线；Attention 为后续 Linear 实验冻结参照** |
| v98 / v100 / v107 / v121 | — | >300 s | timeout / WA | v98、v121 timeout；v100、v107 Attention WA |
| v128–v131 | — | >300 s | timeout | 全部使用高复杂度动态 Attention 路径 |
| v138 | 15715 | 208 s | pass | 官方通过但低于 v86，路线关闭 |
| v139 | 15716 | 202 s | pass | 同上 |
| v140 | 15838 | 207 s | pass | REJECTED：精度低于 v86 906 分 |
| v147 | 16579 | 211 s | pass | REJECTED：精度低于 v86 165 分 |
| 17816 | 17816 | 未提供 | — | 用户确认最高分；源码/SHA/时间/配置未同步，不作代码父版本 |

**关键对照（v147 vs v86 是最干净的单变量实验）**：v147 与 v86 的本地 Attention mean 完全相同（0.7196960689），只有 Linear 不同（v140 Linear vs v86 Linear）。官方差距 −165 分全部来自 Linear。

---

## 2. 本地评测系统的"错配"（结论先行）

本地评测**不能**预测官方分数排序。这不是评测器 bug，而是结构性错配，已被三个独立本地代理交叉验证。

| 对照 | 本地信号 | 官方信号 | 一致性 |
|---|---:|---:|---|
| v86 vs v147（A 相同，纯 Linear） | proxy-v2 Linear 0.4482 → 0.5705（**+0.122**） | 16744 → 16579（**−165**） | ❌ 反转 |
| v84 vs v86（L 相同，纯 Attention） | proxy-v2 overall +0.0064 | +227 | ✅ 一致 |
| v147 vs v140（同 Linear 家族） | proxy-v2 overall 0.6348 > 0.6340 | 16579 > 15838 | ✅ 一致 |
| v138 vs v86（本地 v1） | 等权 +2441 | −1029 | ❌ 反转 |

**交叉验证**（本地代理全部把 v140/v147 排在 v86 之上，官方相反）：

| 本地代理 | 本地顺序 | 官方顺序 |
|---|---|---|
| Qwen proxy-v2 面板（168 Linear + 120 Attention） | v140 > v147 > v86 | v86 > v147 > v140 |
| GPT-2 跨模型探针 | v140 > v147 > v86（三对全反转） | 同上 |
| 外部 youxilee/hif4 复测 | v147 > v140 > v86 > v84 | 同上 |

**可外推的边界**：

- ✅ **可用**：合法性检查、同机 A/B 回归检测、时间相对量级（Spearman ρ=0.874）、机制诊断（误差源分解、role 差分、teacher oracle）。
- ❌ **不可用**：`linear_mean` / 等权显示的**方向与幅度** → 官方分数；本地 <300 s → 官方 pass（v129 本地 248 s 官方仍 timeout）。
- ⚠️ **准绳**：本地只做诊断与筛选，**晋级唯一依据是官方回传**。

评测器已修复的三处历史失真（`official-shape-v1` → `proxy-v2`）：E4M3 subnormal 钳制（`e4m3-subnormal-ceil-v1`）、per-case 校准泄漏（改为官方调用图 168 次 Weight + 24 次 Attention 共享校准）、等权显示与人为比例误用（改为全量枚举、unweighted `overall_mean`）。

---

## 3. 已探索并闭环的方向

| # | 方向 | 关键版本 | 官方证据 | 结论 |
|---|---|---|---|---|
| 1 | E6M2 scale 搜索 | v033/034、v054–056 | v034 = **21864** pass | ✅ 有效（scale-aware k-center，C41b +0.476） |
| 2 | E1 层级 coordinate 优化 | v004–024、v086、v125 | v024 = 16043、v86 = 16744 | ✅ 有效 |
| 3 | SmoothQuant + permutation + block-Hadamard（BOAT） | v073–084、v138 | v084 = 16517 pass | ✅ 有效 |
| 4 | GPTQ/HSDQ output-aware 权重优化 | v138–140、v141–145 | 15715/15716/15838 | ❌ **关闭**（见第 6 节） |
| 5 | K centering | v034、v086 | 21864 / 16744 | ✅ 有效 |
| 6 | Q/K 联合缩放 | v086 | 16744 | ✅ 有效 |
| 7 | Q/K/V 非对称量化 | v066/072/074、v094/095 | v074 旧权重 22750（**当前评测集仅 14561**） | ⚠️ 旧权重有效；动态版 v127–131 timeout；v74 机制在当前评测集未证明优于 v86 |
| 8 | robust V / activation ratio | v066 | 22557（旧权重） | ✅ 有效 |
| 9 | 偏置舍入 / AdaRound | v013–024、v136/137 | v024 = 16043 pass | ✅ 有效 |
| 10 | Attention 真实输出 rerank | v086 | 16744 | ✅ 有效 |

---

## 4. 探索但未闭环的方向（下一步主战场）

| # | 方向 | 尝试证据 | 未闭环原因 | 优先级 |
|---|---|---|---|---|
| 1 | block-Schur HiF4-GPTQ（一次结构更新） | v148（API 369 s 超时）、v149/150 wrapper、workbench `l4_fc_output_block_probe.py` | 重复完整 oracle 超预算；**单次可复用增量未实现**（失败在复杂度，非方向） | **P0** |
| 2 | 低维 activation compensation（固定 code 后闭式 `s_d` 回归） | v153（focus fc −0.048，0 改善/4 回归）、v154（no-op） | v153 精确指出缺"固定 code 后的部署尺度闭式回归"；**v154 疑似实现未生效，需审计而非放弃** | **P0** |
| 3 | 双侧联合残差（Joint W-A） | v148、v070 | 完整版本超时；轻量两轮版未做 | P1（保留为后期一次性残差步骤） |
| 4 | head sensitivity / softmax-Fisher importance | v086 雏形、计划 A4 | 未独立实验 | P1 |
| 5 | decoupled HiF4 encoder（编码/解码尺度解耦） | v153/v154 首试 | 首版失败，正确实现未完成 | P1（计划 L1） |
| 6 | L3 teacher → student 编译 | workbench `l3_fc_legal_oracle.py` / `l3_fc_stability_probe.py`、`l3-fc-legal-oracle.json` | fc 合法码字余量诊断进行中 | P1（进行中） |
| 7 | L2 解析式层级矩阵平衡 | workbench `l2_pair_balance_probe.py` / `l2_output_metric_probe.py` | 探针阶段，未进正式版本 | P2 |
| 8 | Attention 动态路径时间优化 | v128–131（全 timeout） | 时间未解；当前冻结 v86 静态 Attention | P2（Linear 稳定后重启） |

---

## 5. 未探索的方向（多数不推荐）

| # | 方向 | 来源 | 建议 |
|---|---|---|---|
| 1 | case-type classifier（按数据分布选算法） | 文档 V3 | ❌ 不推荐：本地评测错配背景下必然过拟合本地分布 |
| 2 | per-group low-dimensional learned scaling | 文档 V3 | ❌ 不推荐：同上，收益不可预测 |
| 3 | non-invariant `D_A`/`D_W` 低维补偿 | 文档 V3 | ⚠️ 只做过 v148 重版本（超时）；低维版与第 4 节第 2 项重叠，可并入 |

---

## 6. A@W 输出域拟合：理论有效，为什么官方失败

### 6.1 现象

| 版本 | 本地 Linear（v1 / proxy-v2） | 官方分差（相对 v86） |
|---|---:|---:|
| v138 | 0.5073 / 0.6164（dev 样本） | **−1029** |
| v140 | 0.5074 / 0.5709 | **−906** |
| v147 | 0.5074 / 0.5705 | **−165** |

本地 +0.10 量级的"巨大"增益，在官方全部为负。

### 6.2 先排除评测器因素

- proxy-v2 修复 per-case 校准泄漏与 E4M3 bug 后，v138 相对 v86 的本地增益**几乎不缩水**（0.6164 vs 0.5197）→ 不是泄漏造成的虚高；
- GPT-2 探针与外部 hif4 复测同样把 v140/v147 排在 v86 之上 → 不是单一评测器/模型特有现象；
- 结论：**本地增益是真实的（在本地分布上真实有效），但无法泛化到官方隐藏数据**。

### 6.3 理论为何有效（前提）

A@W 拟合目标是 `min ‖AW^T − Q(A)Q(W)^T‖²`，允许 A、W 的量化误差在输出域相互抵消，可行域大于逐张量 MSE。GPTQ/AdaRound 的成功依赖两个前提：

1. 校准分布 ≈ 评估分布；
2. 足够校准样本（128–256 sample，Hessian 稳定）。

### 6.4 本赛题两个前提同时不成立（5 条根因）

1. **校准样本太少**：本地校准仅两折（10/128），二阶统计 `H = AᵀA` 估计方差大，高自由度拟合直接过拟合噪声。
2. **分布漂移结构性**：本地固定 WikiText holdout vs 官方隐藏数据——**拟合对象 ≠ 评估对象**。v140 的 +0.1 恰说明变换、scale、码字都调到了本地窗口统计上。
3. **同时拟合 Q(A) 与 Q(W) 使自由度翻倍且无正则**：离散 HiF4 码域目标高度非凸多峰；不约束 scale 范围/变换复杂度，解会"病态"——本地完美、换数据崩（ROAB-P2、crossfold HSDQ 均属此类）。
4. **相对评分放大选择误差**：`score = (MSE_STD − MSE_PLAYER)/MSE_STD`；本地选"本地分母下改善最大"的方案，官方隐藏 case 的分母与误差结构不同，该选择在官方分母下不再最优。
5. **v86 为什么反而更高**：低自由度（静态层级阈值 + 有限 scale 搜索）≈ 内建强正则，分布偏移下最稳健。官方评的是隐藏数据上的相对改善——**稳健性 > 校准集峰值**。

### 6.5 让 A@W 生效的改进路径

| 措施 | 针对根因 |
|---|---|
| 只拟合静态 `Q(W)`，不拟合动态 `Q(A)`（GPTQ 经典用法） | 3 |
| 限制参数维度：per-64-group 标量 / 固定 α / scale 限幅 [0.5, 2] + Ridge | 3 |
| 跨文档 holdout 验证选择（validation/test 交替窗口），不以校准窗口为裁判 | 2 |
| **A@W 当 teacher 编译规则（计划 L3），不直接部署逐 case 拟合结果** | 2、4 |
| 目标用多 fold 的 worst-case / median，而非 mean | 1 |
| 增加本地校准 token 量以稳定 Hessian | 1 |

---

## 7. 决策纪律（从经验提炼）

1. **本地 `linear_mean` 的方向与幅度不作为官方预测依据**（v147 反例：+0.122 → −165）。
2. **A@W 直接拟合的本地增益默认视为"本地分布过拟合"**，须通过跨文档 holdout + paired panel + 官方回传三层验证。
3. **晋级唯一依据为官方回传**；本地 `<300 s` 不是官方 pass 的保证（提交前建议本地 API ≤ 240 s 且动态路径静态化）。
4. **每个正式版本只引入一个数学机制**；alpha / seed / rank / offset / block size 等内部比较写入 workbench，不分配版本号。
5. **未晋级即标记 `_rejected`，超时标记 `_timeout`**；版本目录名与 `result.md` 状态必须一致。
6. **迭代顺序**：paired effect panel（56 + 5）筛选 → 全量 panel（168 + 120）复核 → 官方回传定晋级。
7. **不再做**：case-type classifier、per-group learned scaling、A@W 直接拟合进候选。

---

## 8. 下一步优先级

| 优先级 | 动作 | 验收 |
|---|---|---|
| **P0** | 审计并修复 v154 的"固定 code 后闭式 `s_d` 回归" | paired panel focus fc 由 `consistent_regression` 转为可解释改善 |
| **P0** | 实现 block-Schur HiF4-GPTQ 一次结构更新（先 fc，后 proj） | 单次增量、按 role 预算 ≤50 s API、向量化共享 Hessian |
| **P1** | 收尾 L3 teacher → student 编译 + A4 head sensitivity | teacher 余量可复现 → 阈值/LUT student 编译 |
| **P1** | 保持 v86 Attention 冻结，Linear 与 Attention 不同时变更 | 单变量纪律 |
| **P2** | L2 解析层级矩阵平衡（探针转正） | 连续域不变量 + 两折量化 MSE |

---

## 9. 证据索引

| 内容 | 路径 |
|---|---|
| 当前状态与官方锚点 | `docs/current-solution-status.md` |
| 官方分数归因与权重不可求证明 | `docs/evaluation-attribution-2026-09-01.md` |
| 唯一活动计划 | `docs/superpowers/plans/2026-09-01-hif4-hierarchy-encoder-and-analytic-attention-plan.md` |
| proxy-v2 面板结果 | `artifacts/official_eval/{v084,v086,v140,v147}-proxy-v2-panel.json` |
| 历史 v1 面板（不可迁移，已隔离） | `artifacts/official_eval/legacy-v1/archive-official-shape-v1.json` |
| L3 fc 合法码字 oracle | `artifacts/official_eval/l3-fc-legal-oracle.json` |
| v151–v154 配对实验 | `artifacts/official_eval/v15{1,2,3,4}-*-targeted.json` |
| 赛题深度分析（含 420s 过时口径，见下注） | `华为2026_NVFP4到HiF4_高精度量化赛题完整分析与优化方案.md` |

> 注：根目录赛题分析文档第 4 行与第 22 章仍按 **420 s** 设计时间预算，官方已于 2026-08-31 收紧为 **300 s**；其面板口径 250 + 200 亦为旧值（官方已降低 Linear 权重但未公开）。阅读时以本文与官方回传为准。
