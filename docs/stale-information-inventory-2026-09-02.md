# 过期信息清单（Stale Information Inventory）

> 日期：2026-09-02
> 目的：集中登记仓库中**已失效但尚未从历史文档中移除**的信息，避免它们干扰后续优化决策。
> 处置原则（AGENTS.md 3.3）：原始 JSON / report / 归档计划 **不改写**，历史数字保留为审计证据；
> 失效描述通过本清单与状态文档标注。**阅读旧文档前先看本文件。**

## 使用方式

执行优化时只读三处活动来源：① 本清单（判断哪些旧描述已失效）→ ②
`docs/superpowers/plans/` 唯一活动计划 → ③ `docs/current-solution-status.md`。
归档计划、历史执行日志、旧研究文档**只提供背景，不提供下一步指令**。

---

## 1. 旧权重分数（最危险的过期信息）

官方在 2026-08-31 晚调整了评分权重，**旧权重分数与新权重分数不可换算**。

| 条目 | 旧记录 | 当前事实 | 处置 |
|---|---:|---|---|
| **v74** | `22750 / 239.387s` | **当前评测集 `14561 / 188.9s`**（2026-09-02 回传，差 −8189） | ❌ **不再是安全基线 / 归档冠军**；不得作为出发版本 |
| v72 | `22662 / 226s` | 新评测集未回传，视为旧权重 | ⚠️ 历史证据 |
| v66 | `22557 / 217.2s` | 同上 | ⚠️ 历史证据 |
| v51 | `22451 / 234s` | 同上 | ⚠️ 历史证据 |
| v31 / v34 | `21864` | 同上 | ⚠️ 历史证据 |
| 外部 youxilee/hif4 v2.7 | `24153 / 239s` | 旧权重外部结果 | ⚠️ 不可作为目标 |

**当前有效锚点**：v84 `16517 / 252.563s`、v86 **16744 / 222.7s**（基线）、v140 `15838 / 207s`
（rejected）、v147 `16579 / 211s`（rejected）、17816（用户确认最高，源码未同步）。

**受影响的表述**（不得再作为依据）：

- “从 v74 出发并冻结其 Attention 完整闭包”——`docs/superpowers/archive/plans/2026-08-31-hif4-active-c1-structured-linear-plan-superseded.md`
- “以 v74 Attention 完整可达闭包为安全基线”——同上
- “当前本地归档冠军 = v74 22750”——`docs/algorithm-inventory-and-directions.md`（已就地更正）
- 以 `22750` 为起点的 36000 潜力推算——`docs/research/2026-08-30-hif4-36000-potential-and-algorithms.md`

**2026-09-02 补充登记（本次已就地修复）**：

- `docs/archive-implementation-audit.md` 3.6 节 v074 条目、第 4 节“安全边界推进到 v74”——已加 v74 新权重失效标注
- `solutions/README.md` v074 行——已改为 `22750（旧权重）→ 14561（当前评测集回传）`
- `solutions/20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA/result.md`——已补记 2026-09-02 新权重回传
- `华为2026_NVFP4到HiF4_高精度量化赛题完整分析与优化方案.md` 第 4 行——已加 420s / 250+200 口径失效标注

---

## 2. 时间口径：420s → **300s**

官方 2026-08-31 把端到端限制从 420s 收紧为 **300s**。任何按 420s 做的预算、可行性判断均失效。

| 位置 | 过期内容 |
|---|---|
| `华为2026_NVFP4到HiF4_高精度量化赛题完整分析与优化方案.md` 第 4 行、第 22 章 | 通篇按 420 s / 7 分钟设计预算 |
| `docs/algorithm-inventory-and-directions.md` 第 294/302/310/375 行 | “超 420s”判断、P0 结论 |
| `docs/archived-algorithm-summary.md` 第 6/159 行 | “上限 420s”、“仍低于 420s” |
| `docs/algorithm-implementation-audit.md` 第 128 行 | “超过 420s” |
| `docs/research/2026-08-28-hif4-algorithm-literature/report.md` 第 7 行 | “当前约束为 <420s” |
| `docs/research/2026-08-30-hif4-36000-potential-and-algorithms.md` 第 4/13 行 | 已自行标注失效（保留为示例做法） |

---

## 3. 目标分数：36000 / 45000 失效

新权重下最高已知分为 17816（旧权重最高 22750）。以 36000、45000、22000 为目标的研究与计划
均基于旧分体系，**目标数值本身已不成立**。

- `docs/research/2026-08-30-hif4-36000-potential-and-algorithms.md`（已自标失效）
- `docs/superpowers/archive/plans/` 中 36000 / 22000 系列计划（历史决策记录）
- 当前有效目标：**Linear `linear_mean` 向 0.8 做可达性验证** + **官方 <300s**（见活动计划）

## 4. 面板与协议：250+200 / official-shape-v1 已退役

| 过期 | 当前 |
|---|---|
| `official-shape-v1`（250 Linear + 200 Attention 采样、含 per-case 校准泄漏与 E4M3 subnormal 钳制） | `proxy-v2`：共享校准调用图（168 Weight + 24 Attention state）、全量真实 W/A、unweighted `overall_mean`、误差源分解 |
| 等权显示 / 5:4 比例 / `45000` 满分 | 已取消；`overall_mean` 为未加权均值 |
| 单次运行即排名 | `paired-effect-panel-v1`：56 + 5 配对筛选 → 全量 panel 复核 |
| `real_model_suite.py`、`sampled-means-v1/v2`、`cap_oracle` 等 | 已退役，归档于 `evaluator/archive/legacy-20260901/` |

**历史 v1 本地分数不可与 proxy-v2 混用**（例如 v86 本地 v1 `0.406668` vs proxy-v2 `0.448180`）。

## 5. A@W 合规边界（已于 2026-09-01 更正）

官方 2026-08-31 起**不再限制任何 `A@W` 拟合用法**。以下旧表述已就地更正：

- `HiF4量化算法实施方案手册.md` 第 5 行（合规前提）、Step 6（审核可解释性）
- `华为算法大赛-HiF4量化赛题完整解析与算法方案.md` 第 2 节、2.2、2.3、5.2、5.3

**仍然有效的约束**：`<300s` 端到端时间；`activation_state` 格式（合法五字段、CPU tensor、
state 深度/节点数上限）。
**但 A@W 合法 ≠ 有效**：v138/v140/v147 的 A@W 输出监督在官方分别 −1029 / −906 / −165，
详见 `docs/direction-map-and-aw-fitting-lessons-2026-09-02.md` 第 6 节。

## 6. 已关闭路线（不要再投入）

v138–v145（缩减 Attention / output-aware gain / ROAB-P2 / BDLR 变体）、v128–v131（动态
Q/K Gram、PAWV、随序列放大搜索）、alpha/offset/sweep/block 数/阻尼/角度/候选槽位的局部扫描。
**proj ROAB-off（v151）、fc CAT-off（v152）、fc decoupled（v153/v154）** 也已归档 rejected。

## 7. 版本状态速查

| 版本 | 状态 | 备注 |
|---|---|---|
| v140 | rejected | 官方 15838 / 207s，低于 v86 |
| v147 | rejected | 官方 16579 / 211s，低于 v86；**提交 SHA 未确认**（目录源码曾被替换） |
| v148 / v149 | rejected | API 369s 超时 / wrapper 不可上传 |
| v150 | 单文件 candidate | v140 Linear + v86 Attention，pre-A3 对照用途 |
| v151–v154 | rejected | proj / fc 递进实验，见 status 2.5–2.6 |

---

## 8. 维护规则

1. 每次官方回传后：先写独立 correction log（`logs/execution/*-official-result-correction.md`），
   再更新本清单与状态文档，**不改写原始 JSON / report**。
2. 新发现的过期信息登记到本清单（含位置、正确值、处置建议），并在对应活动文档中加简短标注。
3. 归档计划与历史日志中的失效表述**只标注不改写**；如需修正结论，写审计说明或新计划。
