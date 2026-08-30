# HiF4 唯一活跃优化计划 v2：Linear 结构增益与 0.9 可达性验证

> 状态：**ACTIVE**
> 建立日期：2026-08-30
> 适用根：`D:/工作内容/AI竞赛/solution.py`
> 当前根：v100/v101（代码相同）
> 规范 LF SHA256：`617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`
> 主目标：先提高 Qwen full-layer `linear_mean`，同时判断 `0.9` 在当前合法表示族内是否可达；Attention 单独排队。

## 1. 唯一执行规则

本文件是唯一可执行计划。执行优化时只读取本文件、当前根 `solution.py`、最新可复现评测和官方规则。归档计划、旧日志和研究文档只能提供历史证据，不能绕过本文件产生新的实验顺序。

每完成一个步骤必须先完成四件事，再进入下一步：

1. 保存完整候选源码、配置、结果、parent、规范 LF SHA256 和合规扫描；
2. 写执行日志，记录模型、层、role、fold、Linear/Attention 分项、panel 和耗时；
3. 在本计划的状态表和最高分账本中写入 `done/rejected/blocked` 与证据链接；
4. 若改变主线，先归档本文件，再创建新的唯一 active 计划并同步 README。

官方接口恢复、用户提供新官方成绩或官方规则变化属于外部事件：立即暂停当前候选的后续扩张，先记录当前根的官方结果并重新校准计划，但不删除已经完成的本地证据。

## 2. 基线、目标与数学口径

### 2.1 冻结基线

固定评测为 Qwen2.5-0.5B、24 层、`sequence_length=128`、`calibration_samples=2`、`test_samples=4`、同一只读 cache、`qwen-official` panel。

| 指标 | 当前值 |
|---|---:|
| Linear mean | `0.5015576125` |
| Attention mean | `0.8420394885` |
| Linear panel | `125.389403` |
| Attention panel | `168.407898` |
| panel total | **`293.797301`** |
| native total | `417.882506` |
| API 时间 | `392.423565s`；C0 复测 `401.130873s` |
| 当前根官方成绩 | `NA` |

本地 case gain 定义为

\[
g=\frac{\operatorname{MSE}_{std}-\operatorname{MSE}_{player}}
        {\operatorname{MSE}_{std}}
=1-\frac{\operatorname{MSE}_{player}}{\operatorname{MSE}_{std}}.
\]

当前 Linear 剩余归一化误差为

\[
r_L=1-g_L=0.4984423875.
\]

若目标为 `linear_mean=0.9`，则还差

\[
\Delta g_L=0.3984423875,
\qquad
\frac{\Delta g_L}{1-g_L}=79.9375\%.
\]

等价地，必须把当前 Linear MSE 压到现有值的

\[
\frac{0.1}{0.4984423875}=20.0625\%,
\]

约为五分之一。Attention 不变时，本地 panel 将为

\[
250\times0.9+200\times0.8420394885=393.407898.
\]

该值不是官方 36000 的线性换算。当前根没有官方提交结果，因此计划不得声称已知官方 36000 的精确差距。

### 2.2 正确的二次型约定

权重矩阵按 `output × input` 记为 \(W,Q\in\mathbb R^{m\times d}\)，校准激活按 `token × input` 记为 \(A\in\mathbb R^{n\times d}\)，并定义

\[
H=A^TA\in\mathbb R^{d\times d},\qquad R=Q-W.
\]

Weight 二次型为

\[
J_W(Q)=\operatorname{tr}(RHR^T).
\]

离散更新 \(Q'=Q+\Delta\) 的精确增量为

\[
\Delta J_W
=2\operatorname{tr}(RH\Delta^T)
+\operatorname{tr}(\Delta H\Delta^T).
\]

激活误差 \(E=Q(X)-X\in\mathbb R^{n\times d}\)，部署权重 Gram \(G=W_q^TW_q\) 时，Activation 二次型为

\[
J_A(E)=\operatorname{tr}(EGE^T)
=\operatorname{tr}(E^TEG).
\]

所有后续实现、单元测试和日志统一使用以上行向量方向，禁止重新使用维度不自洽的 `tr(E^T H Q)` 或 `tr(E^T G E)`。

### 2.3 可达性定义

对合法算法族 \(\mathcal C\)，定义其真实最优值

\[
g^*_{\mathcal C}
=1-\frac{\min_{\theta\in\mathcal C}\operatorname{MSE}(\theta)}
        {\operatorname{MSE}_{std}}.
\]

若构造一个放宽约束的超集 \(\widetilde{\mathcal C}\supseteq\mathcal C\)，则

\[
g^*_{\mathcal C}\le g^*_{\widetilde{\mathcal C}}.
\]

因此只有当一个可信的放宽上界仍低于 `0.9` 时，才能否定当前算法族；放宽 oracle 高于 `0.9` 只说明存在数学空间，不证明合法部署可达。

## 3. 硬边界与验收原则

### 3.1 合规边界

1. 离线校准可以用 `A@W` 优化 `Q(W)`，不得用输出、输出残差或测试监督拟合/选择在线 `Q(A)`。
2. 在线 state 只能保存校准产生的合法静态信息；不得保存模型名、测试样本信息或调用顺序推断。
3. Linear 官方 API 没有显式 role 参数。策略只能依赖 weight/activation shape、校准 operand statistics 和由本次校准静态选出的 state；不得在计划中假设可以传入 `q/k/v/o/fc_gate/fc_up/proj`。
4. 输出字段、shape、dtype、device 和 state 节点数必须通过现有合规检查。
5. accuracy-first 阶段只记录耗时；正式候选最终必须严格小于 `420s`。

### 3.2 分层门禁

不再用 layer-1 作为唯一门禁。默认预筛层为 Qwen

\[
\mathcal L_s=\{0,5,11,17,23\},
\]

覆盖早期、中期和后期层，并输出全部七个 Linear role。若模型层数不同，使用对应的 0%、25%、50%、75%、末层位置。

门禁顺序：

1. 合成小矩阵验证离散写回和二次型；
2. Qwen 五层、全 role、cross-fold 预筛；
3. OPT/Pythia 只作结构性回退诊断，不要求每个模型、role 都正向；
4. 预筛总体有正向信号后，最多进行一次 24 层完整评测；
5. 只有同一冻结 cache 上 full-layer `linear_mean > 0.5015576125` 才允许替换 Linear parent。

Attention 不得掩盖 Linear 回退。Linear 候选先比较 `linear_mean`，panel 和 Attention 只作为一致性检查。

### 3.3 路线价值记录

每个 full-layer 候选额外记录目标缺口关闭率：

\[
\rho_{gap}=\frac{g_{candidate}-0.5015576125}{0.3984423875}.
\]

小增益可以成为新 parent，但不能被描述成足以达到 `0.9`。连续两个正确实现、完整门禁的同族候选都没有 full-layer 正增益，且放宽 oracle 也没有实质空间时，归档该算法族，不继续扩大自由度。

## 4. 完整执行队列

执行顺序严格为 `L0 → L1 → L2 → L3 → L4 → L5 → C1`。Attention 队列 `A1` 不插入 Linear 主线；官方提交 `O0` 是随时触发的外部事件。

### L0：Linear 上限与误差分解

**状态：done（2026-08-30）；L1 已成为下一步。**

**假设**：现有记录只能证明局部候选失败，尚未区分 Weight、Activation、交叉项和坐标投影哪个是主要瓶颈，也没有足以判断 `0.9` 的放宽上界。

**实现入口**：新增 evaluator 诊断脚本，不修改根 `solution.py`。

**数据**：Qwen 五个分层层位、七个 role、两个 calibration fold；先复用现有冻结 test 只做最终报告，不用 test 输出选择在线量化器。

**必须计算**：

1. 部署误差
   \[
   J_{both}=\|Q(A)Q(W)^T-AW^T\|_F^2;
   \]
2. 激活单侧误差
   \[
   J_A=\|Q(A)W^T-AW^T\|_F^2;
   \]
3. 权重单侧误差
   \[
   J_W=\|AQ(W)^T-AW^T\|_F^2;
   \]
4. 当前合法 hierarchy/offset/scale 候选的独立 block oracle；
5. 放宽的独立 block/role oracle与合法投影后的 retention；
6. 每层、role、side 的 `current gain / legal oracle / relaxed ceiling / retained fraction`。

**产物**：`artifacts/oracle_dashboard/` JSON、Markdown 汇总、可复现命令和 ceiling 分类：`weight-dominant / activation-dominant / transform-coupled / insufficient-headroom`。

**实际结果**：五个 Qwen 层位 `{0,5,11,17,23}`、七个 role、两折校准完成；固定 cache、`oracle_rows=32`、255 个合法 E6M2 scale code。整体 arms 为：

| arm | 五层×七 role mean |
|---|---:|
| `both_player` | `0.52301943` |
| `weight_perfect` | `0.70417026` |
| `activation_perfect` | `0.82035698` |
| `both_perfect` | `1.00000000` |

权重侧 headroom 为 `0.18115083`，激活侧为 `0.29733755`，因此总体为
`activation-dominant`；但 q/k 是 `weight-dominant`，v 为 `transform-coupled`，
`fc_gate/fc_up/proj` 的 activation headroom 最大。255-code oracle 的平均
weight-plain gap 为 `0.0229%`，weight-Gram gap 为 `0.6065%`，activation-Gram gap
为 `0.6410%`；scale 轴没有接近 `0.3984` 目标缺口的空间。

证据：[`l0-linear-ceiling-qwen.json`](../../../artifacts/oracle_dashboard/l0-linear-ceiling-qwen.json)、
[`2026-08-30-l0-linear-ceiling.md`](../../../logs/execution/2026-08-30-l0-linear-ceiling.md)。
solution LF SHA 为 `617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`；
诊断脚本 LF SHA 为 `c5e20e8f0ae144a9e7593a923123ca64c5ba27c6a18f55c2f3b51f4aef4d63ad`。

**裁决**：L0 不产生部署 parent。单侧无损 arms 仍低于 `0.9`，因此必须联合改善
Weight 与 Activation；继续 L1 的已知 hierarchy 写回修复，但不再扩大无 oracle 依据的全局 scale 网格。

### L1：修复 v092 full-hierarchy cross-block Weight-LRH

**状态：rejected（2026-08-30；正确实现完成，但未通过分层预筛）。**

**审计更正**：保存的 v092 源码实际没有可复现的 scale/lv2/lv3 搜索；它沿用
parent denominator 后调用 `_write_codes`。因此原先“层级字段被写回丢弃”的假设
不能作为 v092 负结果的直接解释。本步骤按原目标重新实现了真正的 full-hierarchy
原子候选，再独立判断是否有跨 fold 泛化。

**实现要求**：

1. 候选对象原子携带并写回 mantissa、sign、E6M2 scale、lv2、lv3；
2. 新 mantissa 必须和生成它的新 denominator 一起解码，禁止和 parent denominator 混用；
3. 使用第 2.2 节的精确 \(\Delta J_W\)，最终离散结果重新计算完整目标；
4. fold-1 生成、fold-2 接受，fold 交换复核；
5. parent fallback 在校准目标上决定，不读取 test Linear 输出；
6. 记录候选数、接受数、实际改动的层级字段、完整目标变化和每层/role 增益。

**测试**：128 维合成矩阵验证写回 round-trip、旧 denominator 不再出现、增量公式与暴力重算一致；然后按第 3.2 节预筛和 full-layer 门禁。

**实际执行**：rank-8、最多 4 block、合法 E6M2 局部 offset、8 个 hierarchy
layout 和 15 个 signed level 已完成合成测试（`29 passed`）。Qwen 层位
`{0,5,11,17,23}` × 七 role 的 35 case 预筛评估了 70 个 fold 候选；仅 1/70
通过交换 fold 的 admission，最终 0/35 case 改变 stable parent。selected-layer
`both_player=0.523019429222563`，与 L0 逐条相同，因此没有触发第 3.2 节的
24 层 full-layer gate。证据：[`l1-lrh-stratified-qwen.json`](../../../artifacts/real_model_suite/l1-lrh-stratified-qwen.json)、
[`2026-08-30-l1-full-hierarchy-lrh.md`](../../../logs/execution/2026-08-30-l1-full-hierarchy-lrh.md)。

**裁决**：实现正确但 cross-fold 泛化不足，标记“corrected LRH rejected”；候选
源码归档在 `v105`，不提升最高分，不再扩大 rank、block 数或 sweep，进入 L2。

### L2：低自由度 expansive-FFN CAT/BOAT-2

**状态：pending。**

**依据**：v093 全局 CAT/BOAT-2 失败，但完整 full-layer role 结果中 `fc_gate/fc_up` 为正向；这只能作为提出结构假设的证据，不能作为逐层 test 回退门。v093 layer-1 日志的 Linear/Attention/panel 三个数字不自洽，执行前先更正或标记该行不可用。

**合法路由**：官方 API 无 role-id，只用静态结构条件 `weight_rows > weight_channels` 定位 expansive FFN 形状；不尝试从调用顺序或模型名区分角色。

**候选**：identity/current BOAT、固定低自由度 CAT balance、最多 1–2 个 Householder 或固定 hierarchy permutation。禁止恢复全 block β 网格和全局逐块 eigensearch。

**选择目标**：operand-local、cross-fold，冻结变换后重新量化 Weight 和 Activation；完整部署结果用于最终评测，不用于在线 selector。

**门禁**：按第 3.2 节。成功则提升 parent；失败则归档该 FFN 结构候选，不恢复 v093 全局搜索。

### L3：修复 v095 Global Activation-LRH 的 Gram gate

**状态：pending。**

**假设**：v095 优化 block/global Gram surrogate，却用元素 MSE 最终 gate，可能错误接受或拒绝候选。

**实现要求**：

1. 使用 \(J_A(E)=\operatorname{tr}(EGE^T)\)；
2. 低秩近似只用于候选生成，最终离散结果必须用实际部署 Gram 重算；
3. 先输出每层/role 的 proposal 数、Gram 接受率、MSE/Gram 冲突率和 fold 稳定性；
4. 若正确 gate 几乎全部回退到 parent，直接结束，不跑无信息的全层扩张；
5. 若五层预筛总体正向，再运行一次 full-layer。

**失败处理**：修复版全层仍不升，则把当前 Global Activation-LRH 加入停止清单；不增加 rank 或 coverage 掩盖无信号。

### L4：final-weight Gram 与 GALS 分拆验证

**状态：pending；低优先级。**

L4 不再把“最终权重 Gram”和“显式 role GALS”捆绑成一次实验。

#### L4a：最终量化权重 Gram 消融

先完成最终 `weight_params`，解码得到 \(W_q\)，再构造

\[
G_q=W_q^TW_q.
\]

基于 L0 只测试显示 activation-side headroom 的形状/层；必须使用合法 shape/statistics selector。v104 已证明全局替换失败，尤其 `proj` 大幅回退，因此不得再次直接全角色启用。

#### L4b：GALS 小预算

只有当 L0 的合法 scale/hierarchy oracle 在对应层/形状上明显超过确定性评测噪声，且理论上能超过 parent 时才运行。候选限制为 1–4 个高 headroom block，cross-fold 选择；若 oracle 本身没有空间，直接记录 `not justified`，不执行部署实验。

**禁止项**：不得声称 API 提供显式 role；不得恢复 shape-proxy 等同 role 的文字；不得进行全局 255-code 部署扫描。

### L5：结构性新路线与外部差异审计

**状态：pending；在 L1–L4 无足够增益时启动。**

L5 的目标是寻找能够同时改善 q/k/v 和 FFN/proj 的方法论变化，而不是继续微调单一弱 role。

按以下顺序：

1. **外部实现逐组件审计**：以外部官方 `24153` 为参考，比较 quantizer decode、scale hierarchy、rounding、sampling、transform 顺序、state 和 runtime；只迁移合法 operand-local/offline-weight 机制，不迁移输出监督到在线 `Q(A)`。
2. **联合坐标—层级离散求解**：交替优化低自由度等价变换 \(T\) 与合法 HiF4 hierarchy codes，每轮都完整重算投影 retention，避免连续 CAT 好、合法投影坏。
3. **跨 block Schur/LDLQ**：根据 L0 的跨块能量只连接少量高耦合 block，使用精确二次型和原子层级写回；不做全宽 dense Hessian。
4. **校准统计元路由**：特征只允许 shape、RMS、kurtosis、condition number、Gram off-diagonal ratio、合法 oracle gap；用 cross-fold 标签训练小决策树，禁止模型名、test 输出和官方分数作为特征。
5. **新表示族 checkpoint**：若可信放宽上界低于 `0.9`，明确记录当前表示族无法达到目标，只有改变等价变换自由度、层级共享结构或官方允许的状态表达后才继续。

每个子方向仍遵循“合成验证 → 五层预筛 → 一次 full-layer”的统一门禁。一个子方向失败后先归档，不并行堆叠多个未经证实的机制。

### C1：计算压缩与最终候选冻结

**状态：pending；只在出现精度 parent 后执行。**

按 profiler 结果依次处理缓存、批量化、候选共享统计、CPU/GPU 边界、重复 decode/Gram/eigh；每次压缩必须与精度 parent 做 bitwise/容差 A/B，禁止为了时间悄悄改变算法选择。

最终候选要求：

1. full-layer Linear 不低于已确认最高分；
2. 官方 API/state/shape/dtype/device 合规；
3. 规范 LF SHA 和源快照完整；
4. 官方同口径总时间严格 `<420s`；
5. 至少一次冷启动和一次复测，无 nonfinite 和偶发 fallback。

## 5. 独立 Attention 队列

### A1：最终 Q/K 坐标后的 PAWV rank/position

**状态：deferred；不阻塞 Linear。**

先固定最终 Q/K 变换 \(T_Q,T_K\)，再构造

\[
P=\operatorname{softmax}((QT_Q)(KT_K)^T/\sqrt d),
\qquad H_V=P^TP.
\]

随后测试 diag + rank-r 或 position bucket，并在每轮更新后重新量化 Q/K/V、用真实 non-causal Attention 输出复评。该方向只改善 Attention，不参与 `linear_mean=0.9` 的可达性结论；只有 Linear 队列进入等待、平台期或用户明确切换总分目标时才执行。

## 6. 官方事件队列

### O0：当前最高分根的真实官方提交

**状态：blocked-external；官方接口恢复即触发。**

提交时不临时改代码，记录：提交源 SHA、官方分数、官方时间、日期、规则版本和失败信息。第一次提交优先使用当时已压入 `<420s` 的最高可复现根；若新精度 parent 尚未压缩，则先提交 v100 作为兑换率锚点。

官方结果只用于校准本地排序可信度和决定是否继续，不允许按隐藏分数逐参数反向调优。

## 7. 当前状态表

| 顺序 | 项目 | 状态 | 进入条件 | 完成产物 |
|---:|---|---|---|---|
| 0 | 基线冻结与审计 | `done` | — | v100/v101、SHA、固定 cache |
| 1 | L0 Linear 上限/误差分解 | `done` | — | ceiling JSON + 报告 |
| 2 | L1 修复 v092 hierarchy LRH | `rejected` | L0 完成 | v105 screen + audit |
| 3 | L2 expansive-FFN CAT/BOAT-2 | `pending` | L1 裁决 | 低自由度结构候选 |
| 4 | L3 修复 v095 Gram gate | `pending` | L2 裁决 | acceptance diagnostic + 候选 |
| 5 | L4 final-Gram/GALS 分拆 | `pending` | L0 显示对应 headroom | 小预算消融 |
| 6 | L5 新结构/外部审计 | `pending` | L1–L4 无足够结构增益 | 新算法族裁决 |
| 7 | C1 压缩/冻结 | `pending` | 出现新精度 parent | `<420s` submission candidate |
| A | A1 Attention PAWV | `deferred` | Linear 等待或切换目标 | Attention candidate |
| O | O0 官方提交 | `blocked-external` | 官方接口恢复 | 官方 score/time 锚点 |

## 8. 最高分账本

执行后只在本表追加经过 full-layer 固定 cache 评测的版本；layer-only、oracle 和不可复现结果不得成为最高分版本。

| 版本 | 源 SHA | Linear | Attention | panel | API 时间 | 官方 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| v100/v101 | `617482ce...1693eba` | **0.5015576125** | **0.8420394885** | **293.797301** | 392.42s / 401.13s | NA | current parent |

## 9. 明确不再执行

以下方向除非新的 L0/L5 证据改变结论，否则不进入主线：

- 全局 E6M2/顶层 scale-code 扩张；
- 只凭 layer-1 正向直接运行高成本 full-layer；
- 未修复写回的 v092、MSE gate 的 v095、旧坐标 PAWV rank-8；
- 全局 quantized-weight Gram、shape-proxy 冒充显式 role；
- full-width/full-H、逐块大自由度 CAT β 网格或无上限依据的 sweep/coverage 扩张；
- 使用 test/holdout/官方输出逐层回退在线 `Q(A)`；
- 同时堆叠多个尚未独立证明正向的算法。

当前唯一下一步是 **L2：低自由度 expansive-FFN CAT/BOAT-2**。L0 已完成，L1
已完成正确实现和预筛但被拒绝；不得回到 L1 扩大 rank、block 数或 sweep，除非
新的 L0/L5 证据改变该裁决。
