# 算法实现审计：当前根与已归档候选

> 审计日期：2026-08-31
> 审计对象：根 [`solution.py`](../solution.py)，规范 LF SHA256
> `c9c45a7911594b4b378d0c5e2769187d76dc587d79b6da9fa5f5a487e4b7cb11`
> 原则：源码、校准目标、部署解码和评测日志必须逐一对应；不能用旧审计文字替代源码证据。

## 0. 结论

根目录当前是 v119 的 precision/time parent：C1a structured proposal vectorization、v118 L6d structured block-circulant factor、L6c full `G_64` hierarchy coordinate sweep、L6b wide rank-4 cross-block factor、L6a rank-16 global-LRH、L5a block-local permutation、BOAT、expansive-FFN CAT balance、cross-fold
Weight-HSDQ、Gram-hierarchy Activation-HSDQ、Gram-gated Global Activation-LRH、L4a
final deployed-Gram row gate、L4b final-Gram GALS、B1 GQRB 和 B2 PAWV diag-only。Qwen
固定 cache 的最高已完成 full-layer panel 为 `295.808212`，Linear mean 为
`0.5096012555`，Attention mean 为 `0.8420394885`，API 时间 `2040.504690s`（探索阶段
记录，最终仍需 C1 压缩）。

L1 v105 已实现真正的 full-hierarchy cross-block Weight-LRH（scale/lv2/lv3/
mantissa 原子写回），并通过 `29 passed` 合成/合规测试；但五层×七 role 的
cross-fold screen 与 L0 逐条持平，70 个 fold 候选仅 1 个被交换 fold 接受，最终
0/35 case 改变 stable parent。候选已归档，不在根主路径。

## 1. 已核实正确

### 1.1 BOAT 的等价变换

若 `D` 是正对角平衡、`R` 是归一化 signed-Hadamard，则

\[
X'=XD^{-1}R,\qquad W'=WDR,\qquad
X'W'^T=XD^{-1}RR^TDW^T=XW^T.
\]

根实现的 FWHT 除以 `sqrt(block_size)`，因此 `R^TR=I`；`activation_state`
保存逆平衡、一个可选的 block-local permutation 和静态统计，没有保存输出监督。

L5a 的排列 `P` 只在校准激活/权重的独立 `amax/rms` pressure 两折均不变差时写入，
并保持

\[
X'=XD^{-1}PR,\qquad W'=WDPR,\qquad X'W'^{T}=XW^{T}.
\]

### 1.2 Weight-HSDQ 的部署边界

权重侧允许用校准激活构造

\[
J_W(Q)=\|A(W-Q)^T\|_F^2
       =\operatorname{tr}\big((W-Q)A^TA(W-Q)^T\big),
\]

根在 15 个 signed levels 上做 block Hessian 增量，cross-fold 后才写回离线
`weight_params`；在线 `Q(A)` 不读取 Linear 输出。

### 1.3 v105 hierarchy 写回复验

v105 候选将每个离散候选表示为

\[
q_{r,j}=\frac14 c_{r,j}\,s_r\,u_{r,g(j)},v_{r,h(j)},
\]

其中 `s` 是 E6M2 scale，`u` 为 lv2，`v` 为 lv3；搜索后先写回全部层级字段，
再按同一 denominator 重算 mantissa/sign。合成测试证明 round-trip 和二次型
增量一致。这个实现与保存的 v092 源码不同，不能把 v092 的实现缺陷继续假定为
已证事实。

## 2. 已确认的实现/实验问题

### 2.1 v092 旧审计结论需要更正

旧文字称 v092 “先搜索 scale/lv2/lv3、再被 `_write_codes` 丢弃”。逐行检查
[`v092 solution.py`](../solutions/20260830_v092_a3-lrh-r8-rejected_score292.426982_time382s/solution.py)
未复现这条路径；保存源码实际沿用 parent denominator。因此 v092 的
`292.426982` 只能约束该保存实现，不能直接证明正确 full-hierarchy LRH 不可行。
v105 已完成该复验并以 cross-fold 泛化不足为由拒绝，详见
[`L1 log`](../logs/execution/2026-08-30-l1-full-hierarchy-lrh.md)。

### 2.2 v095 Global Activation-LRH 的 gate 错位（L3 已修复）

v095 用未加权逐元素 MSE 接受候选，而部署激活误差应使用

\[
J_A(Q)=\|(Q(A)-A)W^T\|_F^2
       =\operatorname{tr}\big((Q(A)-A)W^TW(Q(A)-A)^T\big).
\]

因此 v095 的负结果不能单独否定“用 Gram 二次型 gate 的 Global-LRH”。L3 v107
已将低秩近似限制为候选生成，并用最终量化权重 `G_q` 逐行重算部署 Gram gate；
五层 screen `0.52894931`，full-layer Linear mean `0.5069966356`、panel
`295.157057`。具体门禁冲突率和两折稳定性见
[`L3 diagnostic`](../logs/execution/2026-08-30-l3-global-lrh-diagnostic.md)。

### 2.3 v108 L4a 路由误判与 v109 修复

v108 将 dynamic token 张量的第一维误当成权重 output-row，导致 final-Gram 分支
从未触发；其 `0.5289493081` screen 与 v107 相同，只能标记为 no-op。v109 在
calibration state 写入结构路由，并以 v107 parent 与 final `G_q` 候选做完整二次型
逐行门控。正确复验得到五层 screen `0.5292690913`、full-layer Linear
`0.5073256468`、panel `295.239309`，较 v107 增加 `+0.0003290112` / `+0.0822528`。
源码、测试和全层证据见 [`v109 archive`](../solutions/20260831_v109_l4a-final-gram-gated_score295.239309_time517s/)。

### 2.4 L5a block-local permutation（v111）

在 64 维 hierarchy block 内生成 identity、pressure 排序、低/高交错和四分位交错
候选；排列只改变共享 lv2/lv3 scale 的通道分组，不改变 HiF4 五字段格式。候选
选择完全基于 calibration operand-local proxy，随后由现有合法 hierarchy、部署 Gram
和 GALS 路径重新计算。合成等价性、state 合法性和 runtime guard 共 24 项测试通过；
五层 screen `0.5318869457`，full-layer `0.5082983001` / panel `295.482473`，较
v110 增加 `+0.239693` panel。完整证据见 [`v111 archive`](../solutions/20260831_v111_l5a-joint-permutation_scoreNA_timeNA/)。

### 2.5 仍有待验证的低风险问题

- `_choose_boat` 先选 balance 再选 rotation，尚未与联合网格的部署侧目标比较；
- cross-fold 最终 score 仍含生成 fold，可能有乐观偏差；
- BOAT 的 alpha/seed 网格较稀疏，但扩大前必须有 L0 oracle 证据；
- v099 PAWV rank-8 在最终 Q/K 变换前构造 metric，当前只保留 diag-only。

这些问题不是当前根的已知回归，不得绕过 active plan 直接叠加到主线。

## 3. 当前方向矩阵

| 方向 | 根状态 | 结论/下一步 |
|---|---|---|
| BOAT + Weight-HSDQ | 保留 | stable parent；由 active plan 统一门禁 |
| Gram-hierarchy Activation-HSDQ | 保留 | stable parent；作为 L3 proposal 的局部基座 |
| expansive-FFN CAT balance | 保留（v106） | `rows > channels`、α=0.25；只改善 fc_gate |
| v105 full-hierarchy Weight-LRH | 归档 rejected | 正确写回但 screen 无增益；不扩大 rank/block/sweep |
| expansive-FFN CAT/BOAT-2 进一步变体 | 未执行 | 不恢复全局 block 搜索 |
| v095 Gram-objective Global-LRH | 已修复并采纳（v107 前一精度 parent） | 4-block proposal；最终 Gram gate；时间待 C1 压缩 |
| final-weight Gram row gate | 已修复并采纳（v109 精度 parent） | 仅 expansive `rows > channels`、`channels <=1024`；完整 `G_q` 行级 gate |
| final-weight Gram + GALS | 保留（v110 前一 parent） | 基于 v109 做最多 4 block 的小预算验证；已通过 full-layer |
| L5a block-local permutation | 已采纳（v111 前一 parent） | 两折 operand-local gate；screen/full-layer 均正向；L5b/v112、L5c/v113、L5d/v114 已 screen 拒绝，L5e 已完成 |
| L6a rank-16 global LRH | 已采纳（v115 前一 parent） | 窄输入 off-block rank 8→16；screen/full-layer 均正向，state/compliance 无违规 |
| L6b wide rank-4 cross-block factor | 已采纳（v116 前一 parent） | `d>1024,d<=8192`；screen/full-layer 正向，唯一正向角色为 `proj(d=4864)`，state/compliance 无违规 |
| L6c full `G_64` hierarchy coordinate sweep | 已采纳（v117 当前 parent） | 固定 scale、最多 4 个高损 block 的 `lv2/lv3` 坐标更新；33 项定向测试、screen/full-layer 正向，7 role 均不降；state/compliance 无违规 |
| L6d structured block-circulant factor | 已采纳（v118 当前 parent） | 宽输入最多 4 个 `64×64` kernel 的 proposal，36 项定向测试、screen/full-layer 正向；`proj` 正向；完整部署 `G_q` gate，state/compliance 无违规 |
| C1a structured proposal vectorization | 已采纳（v119 当前 precision/time parent） | 批量独立 row/block 的 15-level proposal；37 项定向测试与 reference `atol=1e-6` 对照；分数逐位等于 v118，API `−9.30%`，exact gate/compliance 无变化 |
| Attention PAWV rank/position | deferred | 不插入 Linear 主线 |

## 4. 计划与证据治理

唯一可执行计划是 [`2026-08-31-hif4-active-c1-structured-linear-plan.md`](superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md)。
每个候选必须保存完整源码、规范 LF SHA、固定 cache/命令、合规扫描和结果日志；
screen/oracle 不能写入最高分账本。L1 v105、v108 no-op 和其余失败候选均按该
规则归档；当前根为 v119，L6 已完成归档，C1a 已完成，下一步是 C1b structured gradient refresh。计划目录不得同时存在第二份 active 计划。

本审计只记录源码与执行证据；它不把本地 panel 线性换算为官方分数，也不改变
历史归档文件内容。
