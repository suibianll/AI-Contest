# 当前实验结果、36000 可达性与下一方向

> 日期：2026-08-31  
> 性质：基于已有固定 Qwen 全层结果、官方锚点和 L5e/L6e oracle 的决策 checkpoint；
> 本次没有改动 `solution.py`，也没有新增模型评测。

## 1. 三个必须分开的版本

| 用途 | 版本 | Linear mean | Attention mean | Qwen panel | API time | 结论 |
|---|---|---:|---:|---:|---:|---|
| 已有官方通过控制组 | v66 | NA | NA | `238.282409` | 官方 `217.2s` | 官方 `22557`，用于验证平台与包装 |
| 当前增强复测首选 | v72 | 旧代理 Qwen `293.485885` | `63.119717` | 旧代理 `356.605602` native | CUDA `163.41s` | Attention 完整调用闭包与官方通过 v66 语义一致 |
| 官方失败候选 | v100 | `0.5015576125` | `0.8420394885` | `293.797301` | `392.423565s` | 官方 Attention WA，用户确认不是 timeout |
| 官方超时候选 | v121 | `0.5096135327` | `0.8420394885` | `295.811281` | `2180.450151s` | 用户确认官方 runtime timeout；只保留精度证据 |
| 最近的本地时间 parent | v106 | `0.5034589422` | `0.8420394885` | `294.272633` | `412.654599s` | 分数高于 v100，但仅余 `7.35s` API 余量 |
| 当前研究精度 parent | v125 | `0.5097598050` | `0.8420394885` | `295.847849` | `2653.580314s` | 精度最高，但 runtime invalid，不能提交 |

v100 不含完整 `deployment_gram` 却仍得到同类官方 Attention WA，因此资源/时间不再是
首要解释。v100/v106/v107 及其 clean Attention 后代均没有官方通过证据；正式提交线
改从保持 v66 Attention 完整调用闭包的 v72 开始。

## 2. 36000 对应的数学距离

每个 case 的归一化 gain 为

\[
g_c=1-\frac{\operatorname{MSE}_{player,c}}
              {\operatorname{MSE}_{standard,c}}.
\]

官方 250 个 Linear 和 200 个 Attention case 的 36000 分，对应平均 gain

\[
\bar g=\frac{36000}{100(250+200)}=0.8.
\]

从已通过的 v66 官方 `22557` 出发，平均 gain 是 `0.5012667`。要到 `0.8`，需要把
当前剩余归一化误差再消除

\[
1-\frac{1-0.8}{1-0.5012667}=59.8984\%.
\]

外部 `24153` 的同口径结果也仍需消除 `56.8283%` 的剩余误差。因此 36000 不是
局部参数微调目标，而是表示或求解器层面的跃迁。

本地 Qwen panel 只用于相对排序。若仅在本地诊断轴上固定当前
`g_A=0.8420394885`，要使

\[
250g_L+200g_A=360,
\]

所需 Linear mean 为

\[
g_L=\frac{360-200\times0.8420394885}{250}=0.7663684092.
\]

所以 `linear_mean=0.9` 不是 36000 的必要条件，而是更激进的冗余目标。v125 到
`0.7663684092` 仍差 `0.2566086042`，等价于消除其剩余 Linear 误差的
`52.3434%`；到 `0.9` 则仍需消除 `79.6084%`。由于本地分布不是官方隐藏分布，
这两个数只能用于确定算法量级，不能换算官方成绩。

## 3. 已有 ceiling 对可达性的约束

L5e 五层七 role 诊断给出：

| arm | mean gain |
|---|---:|
| current both-player | `0.53188695` |
| weight-perfect | `0.71407146` |
| activation-perfect | `0.81889050` |
| both-perfect dense reference | `1.0` |

因此：

1. 固定当前 activation 时，即使 weight perfect，`0.7141` 仍低于本地 36000 所需的
   `0.7664`；只优化权重不够。
2. 固定当前 weight 时，activation-perfect 的 `0.8189` 高于 `0.7664`，说明 36000
   在纯数学上没有被该 oracle 排除；但 perfect activation 不是合法 HiF4 算法。
3. 到 `0.9` 时两个单侧理想臂都不够，必须同时改变 activation 表示和离散 weight
   求解。
4. 255-code scale oracle 的 activation Gram 加权下降只有 `0.11279%`；继续扩大
   offset、scale 或局部 hierarchy 不可能填补 52%--80% 的剩余误差。
5. v125 的 C1 增量也已进入明显递减区：v121→v125 只增加约 `0.036568` panel，
   而 API 从 `2180.45s` 增到 `2653.58s`。继续增加 rank/block budget 没有合理性。

结论分级：

- 当前 C1/scale/local-hierarchy 算法族达到 36000：**不可行**；
- 36000 的全局理论可达性：**未被排除，但缺少合法算法证据**；
- 固定当前 frame/state 达到 Linear `0.9`：**证据性不可达**；
- 在 420 秒内同时达到目标：还要完成约 `6.32x` API 压缩和数量级精度跃迁，风险高。

## 4. 下一步的正确顺序

### P0：先取得官方有效反馈

提交/复测 v72；若仍出现 Attention WA，立即用 v66 原文件做平台控制。v100 已确认
Attention WA，不再提交。没有新的官方有效点之前，不使用本地 panel 外推官方绝对分。

### P1：C2 低成本跨模型审计

只对 v125 的新增 `proj`/structured 路径做 OPT/Pythia 五层软 guardrail。C2 不承担
涨分任务，只判断 v125 的 Qwen 增益能否作为后续压缩目标。

### P2：C3 exact-equivalent 时间/状态压缩

按以下单变量顺序执行：

1. **因子化 exact gate**：对部署权重 `W_q`，不要保存和乘完整
   `G_q=W_q^TW_q`，改为精确恒等式

   \[
   e^TG_qe=\lVert eW_q^T\rVert_2^2.
   \]

   对 `out_features < in_features` 的宽投影，保存 `W_q` 因子比保存 `G_q` 更小，
   同时完全保持 gate 语义。先做字段/行选择逐位对照，再测内存与时间。
2. **稀疏增量 exact gate**：候选只改 selected blocks，令
   `delta=q_candidate-q_parent`，比较

   \[
   \Delta J=2e^TG_q\delta+\delta^TG_q\delta,
   \]

   避免 parent/candidate 各做一次完整 dense quadratic form；要求与旧 gate 的 keep mask
   完全一致。
3. **structured gradient 增量刷新**：当前 sweep2 每个 rank 都重新运行完整
   `_structured_gram_matmul`。接受 block 增量 `delta_b` 后，只更新由该 block 引起的
   circular-distance kernel contribution；保持 traversal、tie break 和候选字段不变。
4. **FFT/circular convolution 备选**：只有增量刷新仍不够快时，才把 distance-roll 求和
   改为 block 轴 FFT；先证明与 reference 的数值容差和 keep mask 一致。
5. 完成消融后构建 v100→v106→压缩 precision parent 的精度/时间 Pareto；最终候选必须
   `<420s`，并通过 Attention contract smoke 和峰值 state 检查。

### P3：C3 完成后新建表示级计划

当前 active 计划归档后，下一份计划才启动精度跃迁，优先级是：

1. **operand-local 学习的共享正交 butterfly/Givens frame**：用 activation 自身误差、
   `W^TW` Gram 和 weight 自身误差优化低自由度等价变换；不得用 `A@W` 教师输出选择
   activation state。
2. **冻结 activation state 后的完整离散 JDRQ-weight**：连续 ridge 中心只作初始化，
   真正求解必须联合更新 mantissa、lv3、lv2 和 scale，并用跨 fold 输出残差
   Gauss-Seidel 更新 weight params；教师输出不得写入 activation state。
3. 两者串联：先冻结新 frame/`Q(A)`，再做 weight-only JDRQ。只做 ridge 后调用旧
   量化器已被 v094 证明不足；只做局部 offset/hierarchy 也已被 oracle 排除。

Attention 当前 `0.8420` 不是本地精度瓶颈，但 v100/v107 已被官方 WA 证明其 clean
Attention 路径存在本地未覆盖的风险。后续不扩展 PAWV rank/position bucket；提交线
保持 v66/v72 Attention 闭包不变，避免把 Linear 优化再次带入 Attention 回归。
