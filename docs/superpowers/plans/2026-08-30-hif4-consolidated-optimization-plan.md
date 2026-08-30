# HiF4 完整优化方案（目标差距定量分解 + 融合路线）

> 日期：2026-08-30
> 性质：将 [`36,000 Accuracy-First 详细方案`](2026-08-30-hif4-accuracy-first-36000-plan.md)（1065 行，含 E0-G 实测）与 [`网格对齐补充方案`](2026-08-30-hif4-grid-aligned-complement-plan.md) 融合为单一执行文件。
> 父版本：根 `solution.py`，clean BOAT + cross-fold Weight-HSDQ + Gram-hierarchy Activation-HSDQ + Attention deployed shortlist，SHA `5d1128cc79fef58154da2f600ec4b472ff95030e1f1e61b96593d06fd9aac94f`。
> 合规边界：校准输出 `A@W` 只能优化离线 `Q(W)`；不得用于拟合、选择或反推在线 `Q(A)`，不得写入 `activation_state`。
> 时间上限：官方 420s；当前根 six-API `382.15s`、wall `414.03s`。

---

## 1. 目标差距的定量分解

### 1.1 当前基线与目标刻度

固定 Qwen2.5-0.5B 全 24 层、`seq=128`、`calib=2`、`test=4`、`amax6`：

```text
Linear mean     0.5015576125
Attention mean  0.8418285164
panel = 250 × g_L + 200 × g_A = 125.389403 + 168.365703 = 293.755106
native total    417.862253
```

若以 `panel = 360` 作为 36,000 的本地诊断刻度、且 Attention 不回退：

```text
g_L^360 = (360 − 200 × 0.8418285164) / 250 = 0.7665371869
Δg_L   = 0.7665371869 − 0.5015576125 = 0.2649795744
ρ_L    = 0.2649795744 / (1 − 0.5015576125) = 53.16%
```

**即：必须捕获当前 Linear 剩余误差的 53.16%。**

### 1.2 分角色差距（关键约束）

| role | 当前 gain | 剩余误差 | 均匀捕获 53.16% 后 |
|---|---:|---:|---:|
| q | 0.616561 | 0.383439 | 0.8204 |
| k | 0.620526 | 0.379474 | 0.8223 |
| v | 0.563596 | 0.436404 | 0.7956 |
| o | 0.483463 | 0.516537 | 0.7581 |
| fc_gate | 0.375126 | 0.624874 | 0.7073 |
| fc_up | 0.430255 | 0.569745 | 0.7332 |
| proj | 0.421376 | 0.578624 | 0.7290 |
| **mean** | **0.501558** | **3.489097** | **0.7665** |

两个已在原计划中确立、但值得反复强调的约束：

1. **不能只补弱角色**。即使 `fc_gate/fc_up/proj` 全部达到 `1.0`，其余保持当前值：
   `g_L = (0.616561+0.620526+0.563596+0.483463+3)/7 = 0.754878 < 0.766537`。
2. 若 q/k/v 完全不变，则 `o/gate/up/proj` 四角色平均须达 `0.891269`——对当前最弱的三个角色（`0.375~0.430`）而言不现实。

**结论：36000 要求全角色提升，包括已经相对较强的 q/k/v/o。** 这排除了任何"只优化 MLP"或"只优化 Attention 投影"的局部策略。

### 1.3 更现实的分配与它暴露的难度

按"弱角色提升更多、强角色接近饱和"的现实假设重算：

| role | 假设捕获率 | 结果 gain |
|---|---:|---:|
| q | 30% | 0.7316 |
| k | 30% | 0.7344 |
| v | 45% | 0.7600 |
| o | 55% | 0.7676 |
| fc_gate | 65% | 0.7813 |
| fc_up | 60% | 0.7721 |
| proj | 60% | 0.7686 |
| **mean** | | **0.7594** |

**即使给出相当乐观的分配，仍差 0.0071（约 1.8 个 panel 点）。** 这意味着 36000 不是一个"按当前方法努力就能到"的目标，而要求至少一个方向产生**方法论级**的跃迁。

### 1.4 已证伪的路径（E0-G 实测，务必记住）

E0-G 在 Qwen 第 1 层、BOAT 后前 32 行、完整 255 个 E6M2 code 上跑了尺度 oracle：

| role | 侧别 | 改善 blocks | 总相对 gap | 最大单 block gap |
|---|---|---:|---:|---:|
| fc_gate | weight MSE | 5 / 448 | 0.0572% | 13.764% |
| fc_gate | activation Gram | 19 / 448 | 0.0836% | 17.326% |
| fc_up | weight MSE | 1 / 448 | 0.0709% | 22.199% |
| fc_up | activation Gram | 13 / 448 | 0.0470% | 12.815% |
| v | weight MSE | 4 / 448 | 0.0313% | 7.466% |
| **v** | **activation Gram** | **60 / 448** | **0.6302%** | **19.321%** |
| proj | weight MSE | 18 / 2432 | 0.0390% | 9.943% |
| proj | activation Gram | 50 / 2432 | 0.0475% | 20.534% |

三条硬结论：

1. **顶层 scale 的搜索空间已基本耗尽**：除 `v` 的 activation Gram 外，各 role 总 gap 均 < 0.1%。不要再投入全局 scale 候选扩张。
2. **"网格对齐"设想被证伪**：补充方案曾假设「目标值是 `E2M1×Δ` 离散集、可用均匀 mantissa 网格精确命中」。但当前 BOAT 含对角缩放与 signed-Hadamard，**变换后元素通常不再属于单一 `E2M1×Δ` 集合**，该前提不成立。原始 `denom/Δ` 命中率只能作为 identity/纯置换下的归因参考，**禁止作为主算法依据**。
3. **局部高损 block 仍有空间**：最大单 block gap 达 13%–22%，说明收益集中在少数 block。方向应是"稀疏定位 + 局部插件"，而非全局扩张。

### 1.5 与外部基准的相对位置

| 口径 | 当前根 | 外部 v2.7 | 差 |
|---|---:|---:|---:|
| Qwen panel（本地，唯一可比口径） | **293.755106** | 250.327102 | **+17.35%** |
| 官方 | **未提交** | 24153 / 239s | — |

C66 官方 `22557 / 217.2s` 是现有唯一官方锚点。**当前根本土领先外部基准 17.35%，却从未提交官方**——这是全局最大的信息缺口。

> 口径纪律：五模型合计 `1085.743597` 只能检查跨模型结构性回退，**禁止**与官方 24153 做差值，也禁止跨候选拼接对比。

---

## 2. 融合后的技术路线

把原计划（A1–A6、B1/B2）与补充方案（GALS 修正版、CAT、Qronos、稀疏插件）合并，按"先证明合法求解器能兑现收益，再增加更强连续目标"排序：

```text
P0  提交 stable parent → 取得官方锚点与兑换率       【当前应执行】
  ↓
D0  合法上限仪表盘（E0-G 已完成）
  ↓
E1/A1  渐进全层级 HSDQ              【已拒绝：全层 −2.831 panel，693s】
  ↓
E2/A2  expansive 稀疏 row HSDQ      【已拒绝：全层 −0.923 panel】
  ↓
E3/A3  逐行 block-leverage HSDQ     【已拒绝：全层 −0.505 panel】
  ↓
E4/A4  blockwise BOAT-2             【已拒绝：全层 −0.777 panel】
  ↓
E5/A5  joint-fold 离线 A@W           【已拒绝：全层 −9.160 panel】
  ↓
E3-LRH true cross-block rank-8  【已拒绝：全层 −1.328 panel，381.84s】
  ↓
E4/CAT-BOAT-2 full组合       【已拒绝：全层 −10.595 panel，600.61s】
  ↓
E5/Qronos（已拒绝：panel 持平但 455.73s） + Global Activation-LRH（已拒绝：全层 −11.138 panel）
  ↓
B1 GQRB margin 【已接受：panel 293.793700，406.24s】
  ↓
B2 PAWV diag-only 【已接受：panel 293.797301，392.42s】
  ↓
C0 五模型确认 【下一步】
  ↓
P1  取所有本地候选最高分；官方恢复后再建立兑换率
```

这不是“全部算法已完成”的声明：E0-C（GALS 解析召回）、
冻结-Q(A) ridge/Qronos 与 Global Activation-LRH 已执行并归档；Attention GQRB
与 PAWV diag-only 已通过本地门禁，PAWV rank-8 低秩变体已拒绝；最终五模型综合
（C0）仍待确认。已执行项与未执行项不应混为一谈。

### 2.1 E1/A1：渐进跨 fold 全层级 HSDQ（已拒绝）

- **问题**：`_polish_weight` 在单个 fold 上做完整坐标扫描，只输出最终候选 → 历史 HSDQ-1 在 calibration 改善、独立 test 回退。
- **方法**：把"生成路径"与"选择路径"分离。fold A 生成路径检查点 `{parent, hierarchy-only, 1, 4, 16, 32, 64 accepted moves}`，fold B 只评价；然后 A/B 互换。
- **候选变量**：`q = s1·s2·s3·m·σ`（E6M2 顶层 / 8 个 lv2 / 16 个 lv3 / 64 个 mantissa / 64 个符号），构造分层 beam，而非固定 hierarchy 后只改 mantissa。
- **稳健目标**：`J(c) = mean + β·max + γ·|L_A−L_B| + λ·‖Q_W(c)−Q_W(parent)‖²`
- **稀疏插件**：仅在 E0-G 显示高 gap 的 block（当前已知 `v` 的 activation Gram）生成 GALS-C 候选。
- **裁决**（四个对照：parent、fixed-hierarchy、progressive mantissa-only、progressive full-hierarchy）：
  - full-hierarchy 在 validation 明显胜出 → 继续 E2/E3；
  - mantissa-only 正向而 hierarchy 回退 → 保留 path selector、收缩 scale 自由度；
  - 全部只改善 train → 立即转 E4；
  - **合法 oracle 本身也几乎无增益 → 判定当前坐标系不足，停止扩大 HSDQ。**

实测归档：`logs/execution/2026-08-30-e1-progressive-hsdq.md`。一层 panel
`338.627176` 的收益未迁移到 24 层，完整 panel `290.923906`，因此恢复 parent。

### 2.2 E2/A2：扩张 FFN 稀疏 row-block

`rows > 2·channels` 的扩张 FFN 权重当前**完全跳过** Weight-HSDQ——恰好覆盖最弱的 `fc_gate`（0.375）与 `fc_up`（0.430）。这是"最弱角色 × 完全未优化"的最大一块未开垦地，优先级应高于任何全局 scale 扩张。

实测归档：`logs/execution/2026-08-30-a2-expansive-sparse-hsdq.md`。1%/2%/5% 行
子矩阵 HSDQ 使全层 panel 从 `293.755106` 降至 `292.831952`，停止。

### 2.3 E3/A3：LRH 跨块低秩 Hessian

当前 Activation-HSDQ 只保存 64×64 block-diagonal Gram，缺跨 64-block 相关性。已
实现 rank-8、最多 4 block 的真实 LRH：保留块内 Gram，对 off-block 做 PSD 低秩
近似，再经完整 Gram 的离散验收；但 Qwen 全层 panel `292.426982`，比 parent
`293.755106` 低 `1.328124`，详见 `logs/execution/2026-08-30-a3-lrh-r8.md`，
因此停止扩大该实现。注意 C40 的教训：跨块二阶结构对校准分布敏感，必须跨 fold
和跨模型复验。

当前实现的逐行 block-leverage 试验（`logs/execution/2026-08-30-a3-rowwise-block-hsdq.md`）
并未达到 LRH 目标，全层 panel `293.250467`，停止继续扩大 FFN HSDQ。

### 2.4 E4/A4：BOAT-2（坐标系）

- `T_b = D_b P_b H_b R_b`：blockwise 对角平衡（每个 64-block 独立 α，而非全层统一）+ 层级感知置换（极端值分散到不同 lv2/lv3 子组）+ signed-Hadamard + Householder/低秩可逆修正。
- **CAT 初始化**：`C = A_ε^{1/2} B_ε A_ε^{1/2}`、`T_CAT = C^{1/4} A_ε^{-1/2}`（注意乘法顺序不可交换，行/列向量约定须整体转置）。CAT 不是为 HiF4 层级量化推导的，**只作连续初始化与 alignment oracle**，最终必须投影为合法 BOAT-2 结构。
- role-aware 顺序：`fc_gate/fc_up` → `v/o/proj` → q/k（仅在跨模型一致时扩展）。禁止使用模型名门控。

先行的两种 blockwise exponent schedule 全层 panel `292.978009`，详见
`logs/execution/2026-08-30-a4-blockwise-boat.md`；该变体已回退。

随后执行完整 CAT-inspired BOAT-2（blockwise balance + 五种 hierarchy-aware
permutation + 正则化协方差 Householder），单层 panel `336.716334`，但 24 层
panel 降至 `283.159693`，API `600.61s` 超过 420s，详见
`logs/execution/2026-08-30-a4-cat-boat2.md`；该 full-search 变体已回退。CAT/
Householder 只保留为未来小规模离线 oracle，不再直接写入部署主线。

### 2.5 E5/A5：FS-JDRQ（最可能的大收益，对应外部核心优势）

外部 v2.6 的核心是**校准期 X/W 联合输出残差补偿**——本地复现已证明是有效信号，这是外部能到 24153 的关键。

合规顺序：

```text
仅用 operand-local/静态 W Gram 选 BOAT 与 activation_state
  → 冻结 activation_state → 得到 Z = Q(A)
  → 允许用 Y = A@W 优化离线 Q(W)
  → 输出信息只进入 weight_params 候选选择
```

- 连续目标：`W̃^T = (Z^T Z + λI)^{-1} Z^T Y`，研究 `λ/trace(Z^TZ) ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}`。
- **不直接量化连续解**：只提供低自由度 target `W_η = (1−η)W + ηW̃`，`η ∈ {0, 1/16, 1/8, 1/4, 1/2, 3/4, 1}`，每个都必须经 A1/A3 合法投影后进入跨 fold selector。
- Qronos 块级纠错只作用于**冻结激活状态之后**的权重求解，目标是让 `Z·Q(W)^T` 接近 `AW^T`；**本赛题不采用跨层误差传播**（评测器每层提供未级联量化的真实激活）。

第一轮合并 fold `A@W` 候选已执行但被拒绝：24 层 panel `284.595177`，q/k/v/o/proj
均回退（`logs/execution/2026-08-30-a5-joint-aw.md`）。另外，直接把带 activation
residual 的 frozen `Q(A)` 放进 weight residual 会触发合规守卫的 cross-residual 规则，
不能绕过守卫；后续若重启，必须先提出不含该交叉项的新目标并单独做合规证明。

### 2.6 E7/B：Attention

Attention 当前 `0.842039` 已远高于 Linear，但 200 个 case 的权重使其 panel 贡献
（`168.407898`）超过 Linear（`125.389403`）。B1 GQRB margin 已部署；B2 PAWV
diag-only 使用 attention probability 的 token-row 对角 Hessian 做 V 的合法离散
refinement。PAWV 的 rank-8 跨 token 低秩扩展在 layer-1 panel 看似更高，但未通过
全层前置门禁，已归档；下一步只做五模型 C0 稳健性确认。

### 2.7 已排除（不再重复投入）

| 方向 | 排除依据 |
|---|---|
| 全局 E6M2/顶层 scale 扩张 | E0-G：各 role 总 gap < 0.1% |
| "网格对齐精确命中" | BOAT 变换后元素不再属于单一 `E2M1×Δ` 集合，前提不成立 |
| 跨层误差累积纠正（Qronos 跨层、RDQ 级联漂移） | 评测器提供未级联的真实激活，无跨层传播 |
| SpinQuant/FlatQuant 的 task/output loss | 会把输出监督引入 `Q(A)`，不合规 |
| C40 跨块 Block-LDLQ、C22 R64 | 已验证失败 |
| E1/E2/E3/E4/E5 本轮变体 | 全部低于 stable parent；详见对应 execution log |

---

## 3. P0：先提交当前根（优先级最高）

一次提交同时解决两件不可替代的事：

1. 把 +17.35% 的本地优势兑换成官方分——本地代理**只能判方向、不能报分数**（C38 反例：本地 Linear 0.5695 官方仅 14092，低于 C21-C 的 0.5311→14437）。
2. 建立本地→官方兑换率。目前唯一配对点只有 C66（`22557`），样本太少，无法判断 panel 增量与官方增量的比例。

当前根 six-API `382.15s`、wall `414.03s`，**在 420s 内，具备提交条件**。

---

## 4. 验收与晋级门

沿用并强化（C40 诊断的教训）：

1. **五模型方向一致**：任一模型负向即不晋级（C40 正是在此失手）。
2. **跨 fold**：fold A 生成、fold B 必须同步改善，validation 只终验、不参与选择与调参。
3. **时间**：six-API 累计 < 400s（留 20s 余量）；精度优先阶段不按耗时淘汰方向，但**晋级必须满足**。
4. **合规**：Linear 激活侧调用图不得访问输出评分器；`A@W` 及其残差不得进入 `activation_state`。
5. **单次单变量**：每次只改一个结构变量，保证可归因。

里程碑（本地研究刻度，**非官方分数承诺**）：

| 里程碑 | Linear mean | panel（Attention 不变） | 含义 |
|---|---:|---:|---|
| 基线 | 0.501558 | 293.755 | 当前 |
| M1 | 0.55 | 305.866 | 证明存在跨层结构增益 |
| M2 | 0.65 | 330.866 | 进入双侧联合改善区间 |
| M3 | 0.766537 | 360.000 | 本地 36,000 诊断刻度 |
| M4 | 0.90 | 393.366 | 为官方隐藏分布留安全余量 |

---

## 5. 目标可行性判断（诚实结论）

- **追平/超过外部 24153**：当前根本土已领先 17.35%，若兑换率正常，**可能已经达成或接近**。这必须先提交验证，不能靠推测。
- **36000**：要求捕获 53.16% 的剩余 Linear 误差，且必须全角色提升（含已较强的 q/k/v/o）。1.3 节的现实分配显示即使相当乐观也差约 1.8 个 panel 点。它需要一个**方法论级**突破，最可能来自 E4（坐标系）与 E5（输出目标/FS-JDRQ）的组合，而不是任一单项调参。
- **建议**：以 M1 → M2 → M3 逐级推进，每达到一个里程碑就提交一次官方，用真实兑换率校准下一步的投入方向；**不要在没有官方锚点的情况下连续做多个大改动**。

---

## 6. 风险登记

| 风险 | 说明 | 缓解 |
|---|---|---|
| 本地代理与官方脱节 | C38 反例已证明本地不能报分数 | P0 提交校准；每个里程碑提交一次 |
| 过拟合校准 fold | 历史 HSDQ-1 曾 train 改善、test 回退 | 跨 fold 生成/选择分离；三 fold CVaR |
| 跨块二阶结构不稳定 | C40 官方 −181 分 | 跨 fold + 跨模型双重复验 |
| 更低局部 MSE ≠ 更高任务精度 | Four-over-six 论文明确观察到此现象 | 跨 fold + 五模型复验 |
| 文档并发修改 | 本仓库计划/状态文档存在并发写入（16:25 曾更新，补入 CAT/Qronos/E0-G） | 引用数值标注读取时间；关键决策前重新核对 |
| state 膨胀 | 新机制可能增大 state | 所有 state 必须 CPU、finite、可复现；仅在实际选中时写入新键 |

---

## 7. 参考

- Four Over Six：arXiv 2512.02010（`mit-han-lab/fouroversix`）；NVIDIA Nemotron 3 Ultra 实测中位重构 MSE −16.4%，但该结论针对 NVFP4 的 E2M1 非均匀断层，**HiF4 的 mantissa 是均匀网格，不直接适用**。
- CAT：arXiv 2603.04359。Qronos：arXiv 2505.11695。
- 工程内部：E0-G 报告 `logs/execution/2026-08-30-e0g-scale-oracle.md`；`docs/current-solution-status.md`；`logs/candidates/2026-08-29-external-hif4-gap-analysis.md`；原 `36,000 Accuracy-First 详细方案`；`网格对齐补充方案`（其 G1 全局设想已由 E0-G 证伪，本文件已按实测结果修正为稀疏插件）。
