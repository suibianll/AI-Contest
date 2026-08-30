# HiF4 唯一活跃优化计划 v3：L5 结构性 Linear 精度路线

> 状态：**ACTIVE**
> 建立日期：2026-08-31
> 适用根：`D:/工作内容/AI竞赛/solution.py`
> 当前根：v111 L5a joint-permutation precision parent
> 规范 LF SHA256：`6b229081121c4a7edd69575c93dc01488be8f8b5e1479007522421e93e1adc57`
> 主目标：继续提高 Qwen full-layer `linear_mean`；在合法 API/state 约束内评估
> `0.9` 的剩余可达性。Attention 继续排队，不插入 Linear 主线。

## 1. 唯一执行规则

本文件是唯一可执行计划。执行优化时只读取本文件、根 `solution.py`、最新可复现
JSON/日志和官方规则；`docs/superpowers/archive/plans/` 下的文件只作历史证据，
不产生新的顺序。

每个候选都必须按固定顺序完成：

1. 合成小矩阵验证合法 HiF4 五字段、二次型方向、原子层级写回和非 finite fallback；
2. Qwen 五层 `{0,5,11,17,23}` × 七 role screen，至少两折 calibration；
3. 记录 source LF SHA、完整命令、cache/model/data revision、候选与 parent 差异；
4. screen 总体有正向信号时最多运行一次 Qwen 24 层 full-layer；
5. full-layer 只以 `linear_mean`/固定 Qwen panel 晋级，Attention 只作回归检查；
6. 每轮结果（成功、失败、无效、超时）先归档完整源码和结果，再更新本计划账本；
7. 若 L5 子方向全部完成，归档本计划并新建下一份唯一 active 计划，不在本文件
   追加新的“下一步”。

accuracy-first 阶段不因超过 420 秒拒绝精度候选，但必须记录 API/wall 时间；C1
只有在所有精度子方向完成后才恢复 `<420s` 硬门禁。

## 2. 当前基线与目标

固定配置：Qwen2.5-0.5B、24 层、`seq=128`、`calib=2`、`test=4`、`amax6`、CPU、
只读 cache `artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`。

| 指标 | v111 基线 |
|---|---:|
| Linear mean | `0.5082983001` |
| Attention mean | `0.8420394885` |
| Linear panel | `127.074575` |
| Attention panel | `168.407898` |
| Qwen panel total | **`295.482472728320`** |
| native total | `422.412248589332` |
| API time | `726.094116s` |
| 官方分数/时间 | `NA / NA` |

case gain 仍定义为

\[
g=1-\frac{\mathrm{MSE}_{player}}{\mathrm{MSE}_{std}}.
\]

当前距离 `linear_mean=0.9` 为

\[
\Delta g_L=0.9-0.5082983001=0.3917016999,
\]

剩余归一化误差为 `0.4917016999`，还需消除约 `79.66%`；250-case Linear panel
仍差 `250(0.9-0.5082983001)=97.925425`。这只是本地诊断轴，不能换算官方
`36000`。

## 3. 合规和数学边界

权重 `W,Q∈R^{m×d}`（output × input），激活 `A∈R^{n×d}`。允许的离线权重目标为

\[
H_A=A^TA,\qquad J_W(Q)=\operatorname{tr}[(Q-W)H_A(Q-W)^T].
\]

激活部署目标使用已返回权重 `W_q` 的 Gram

\[
G_q=W_q^TW_q,\qquad J_A(E)=\operatorname{tr}(E G_q E^T),
\quad E=Q(A)-A.
\]

离散更新 `Q'=Q+Δ` 的精确增量为

\[
J(Q')-J(Q)=2\operatorname{tr}(R H Δ^T)+\operatorname{tr}(ΔHΔ^T),
\quad R=Q-W,
\]

其中 `H` 对权重候选取 `H_A`，对激活候选取 `G_q`。任何近似 Hessian/低秩项只能
生成 proposal；写回前必须重新解码合法五字段，并在真实目标上逐行 gate。禁止：
模型名/role-id/调用顺序分支，test/holdout/官方输出进入在线 `Q(A)`，输出残差写入
`activation_state`，以及用 shape proxy 声称显式 role。

## 4. L5 执行队列

### L5a：联合等价坐标与合法 hierarchy 的交替离散优化

**状态：done（2026-08-31；v111 accepted precision parent）。**

假设：v110 已改善 activation-side，但 weight-side 和坐标投影仍有互补空间；先
固定一个低自由度等价变换 `T=D·R`，再在合法 scale/lv2/lv3/mantissa 域里做离散
坐标下降，可能同时改善 `Q(W)` 与 `Q(A)`，而不会引入输出监督。

实现边界：只使用当前已有的 operand-local 校准统计和最终 `G_q/H_A`；`T` 只允许
现有 BOAT 的正对角 balance、已实现 signed-Hadamard block 和至多一个固定 block
符号/置换候选。每个 `T` 候选都完整重算 weight 与 activation hierarchy，不能把
连续 CAT proxy 当成部署结果。使用两折交替：fold-1 生成、fold-2 复核，交换一次。

验收：合成 128/256 维矩阵证明 `T` 的乘积保持、层级字段原子写回、精确增量与暴力
重算一致；Qwen screen 总体必须高于 `0.52929209`（v110 screen），才允许 full-layer。
若连续两个 `T` 候选都无 cross-fold 增益，停止该族，不扩大 rotation/置换自由度。

执行记录：已实现 `solution.py` 的 `_l5a_channel_pressure`、
`_l5a_block_permutation` 和 `_choose_l5a_permutation`，排列只在两折
operand-local proxy 同时不变差时写入 state。24 项合成/合规测试通过；五层×七 role
screen（Qwen2.5-0.5B，cache read）Linear mean 为 `0.5318869456762372`，较 v110
screen `0.52929209` 提升约 `+0.00259486`，因此进入 full-layer。候选归档为
[`v111 L5a`](../../../solutions/20260831_v111_l5a-joint-permutation_scoreNA_timeNA/)，
screen 证据为 [`l5a screen JSON`](../../../artifacts/real_model_suite/l5a-joint-permutation-stratified-qwen.json)
和 [`screen log`](../../../logs/execution/2026-08-31-l5a-joint-permutation-stratified.md)。
full-layer 结果为 Linear mean `0.5082983001444541`、Attention mean
`0.8420394884610322`、native total `422.412249`、Qwen panel `295.482473`，较
v110 panel `295.242779647671` 增加 `+0.239693`；API `726.094s`，仅作探索记录。
原始结果见 [`v111 full JSON`](../../../artifacts/real_model_suite/v111-l5a-joint-permutation-qwen-full.json)
和 [`v111 full log`](../../../logs/execution/2026-08-31-v111-l5a-joint-permutation-qwen-full.md)。
该方向已通过 full-layer，v111 成为新的 precision parent。

### L5b：稀疏跨 block Schur/LDLQ 激活—权重联合 proposal

**状态：done（2026-08-31；v112 rejected at screen）。**

从 `G_q`/`H_A` 的 block off-diagonal ratio 选最多 2 个高耦合 block 对，使用

\[
S_{ij}=H_{ii}-H_{ij}H_{jj}^{-1}H_{ji}
\]

或对应的 2-block Schur complement 生成 proposal；所有 block 的 hierarchy 字段
原子写回，并以完整目标 gate。禁止 full-width dense Hessian、全 block beam 和
无上限 coverage。screen 仍以 cross-fold 和完整部署 Gram 为准。

执行记录：已实现阻尼 PSD Schur block、最多两对互不重叠 block 的 128 维离散坐标
下降、weight 两折 cross-fold gate，以及 activation 侧 `AᵀA` state proposal + 在线
完整 `G_q` 逐行 gate。合成/合规定向测试 29/29 通过；宽度 256、rows 128 的 runtime
compliance 为 0 violations。五层×七 role screen（35 cases）Linear mean 为
`0.5308551015775216`，低于当前 v111 screen `0.5318869456762372`（`-0.0010318441`），
所以没有运行 full-layer，根恢复并保持 v111。候选完整归档于
[`v112 L5b`](../../../solutions/20260831_v112_l5b-sparse-schur_rejected-screen_score0.530855_time140s/)，
screen JSON 为 [`l5b screen`](../../../artifacts/real_model_suite/l5b-sparse-schur-stratified-qwen.json)，
日志为 [`l5b log`](../../../logs/execution/2026-08-31-l5b-sparse-schur-stratified.md)，
候选 source LF SHA 为 `94a06fcce29b3e6639c4dab4d8c96e4e37f4f74947adec6e1f57b87512e0bc9`。

### L5c：统计元路由（只依赖 operand-local 特征）

**状态：done（2026-08-31；v113 rejected/no-op at screen）。**

为每个 calibration call 计算 shape、RMS、kurtosis、condition number、Gram
off-diagonal ratio、合法 scale/hierarchy oracle gap 等静态特征；用两折标签训练
小决策树选择现有 `{v107, v109, v110}` 子路径。禁止模型名、role 名、test 输出、
官方分数和调用顺序。必须在未参与训练的 calibration fold 上复核，再做五层 screen。

执行记录：已实现八维 operand-local 特征（shape ratio、两侧 RMS/kurtosis、block
condition、Gram off-diagonal ratio、hierarchy gap）、两折 leave-one-fold-out 一层
决策树和 `{v107,v109,v110}` route proposal。route 标签探测不使用输出或残差低秩项；
只有两折都不差且累计 MSE gain 超过门槛才写入标量 `meta_route`，否则保持 `-1`。
合成/合规定向测试 34/34 通过，静态与 runtime compliance 均为 0 violations。
五层×七 role screen 为 `0.5318869456762372`，与 v111 screen 逐 case 完全相同，
差值为 `0`，故没有运行 full-layer。候选归档于
[`v113 L5c`](../../../solutions/20260831_v113_l5c-meta-router_rejected-screen_score0.531887_time169s/)，
screen JSON 为 [`l5c screen`](../../../artifacts/real_model_suite/l5c-meta-router-stratified-qwen.json)，
日志为 [`l5c log`](../../../logs/execution/2026-08-31-l5c-meta-router-stratified.md)，
候选 source LF SHA 为 `65e4ad45808e8a4e24bb688f369a0606786344d5470ad6d334cbad436f0b0699`。

### L5d：外部实现逐组件差异审计

**状态：in_progress；L5c screen 无增益后启动。**

对外部 `youxilee/hif4` 做 codec decode、rounding、scale hierarchy、sampling、
transform 顺序和 state/device 的逐项 diff；只迁移 operand-local/offline-weight
机制，并为每一项建立最小合成回归。差异审计若不能构造可验证的单一变化，记录
`not actionable`，不堆叠进根。

### L5e：表示族可达性 checkpoint

**状态：pending；L5a–d 后执行。**

用全合法 scale/hierarchy oracle、weight/activation 单侧上界和跨 block coupling
统计估计当前表示族的可信上界。如果上界仍低于 `0.9`，明确记录“当前 HiF4
表示/接口不可达”，后续只研究允许的新等价变换或状态表达，不再重复扩大已否决
的 offset/rank/coverage。

## 5. 版本账本

只登记固定 cache 的 full-layer 结果；screen/oracle 单独放执行日志。

| 版本 | Linear | Attention | panel | API time | 状态 |
|---|---:|---:|---:|---:|---|
| v106 | 0.5034589422 | 0.8420394885 | 294.272633 | 412.65s | 时间 parent |
| v107 | 0.5069966356 | 0.8420394885 | 295.157057 | 481.04s | 前一精度 parent |
| v109 | 0.5073256468 | 0.8420394885 | 295.239309 | 517.29s | 前一精度 parent |
| **v110** | **0.5073395278** | **0.8420394885** | **295.242780** | **701.90s** | 前一精度 parent |
| **v111** | **0.5082983001** | **0.8420394885** | **295.482473** | **726.09s** | **当前精度 parent；L5a accepted** |

## 6. 完成和换计划条件

L5a–L5e 均完成一次合法实现或有证据的 `not actionable` 裁决后，必须：

1. 把本文件状态改为 `COMPLETED`，写入每个子方向的结果和证据；
2. 移到 `docs/superpowers/archive/plans/`；
3. 更新 `plans/README.md`、根 README、当前状态、算法清单和归档审计；
4. 新建下一份唯一 active 计划，明确新的 precision parent 与下一步；
5. 只有完成上述归档后才允许进入 C1 时间压缩。
