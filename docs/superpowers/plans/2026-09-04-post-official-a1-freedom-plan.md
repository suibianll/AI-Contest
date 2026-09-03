# 官方裁决后新一轮算法规划：A1 结构内自由度与可加性延伸（2026-09-04）

> 状态：**COMPLETE — D1 v180 官方 `17597/242s`，相对 v175 `+3/−3s`，RETAINED 为
> 新完整官方父版本；
> D2 已实现为 v181 本地预研并裁决 REJECTED（默认 120 纯 D2 mean −0.002746、
> median −0.000086、54+/66−，确认 A1 group-consistent 结构为承重组件），不占配额；
> D3 的 v175+D1 组合已由 v180 本身完成，不再另耗配额。**
>
> 配额账本（用户约束，2026-09-04 目标设置起 ≤10）：v176 已用 1（官方负，不退还）；
> **v180 为第 2 个（2/10）**。剩余 8。v171/v174/v175 为目标设置前已排期队列（不占配额）。

## 1. 官方裁决后的关键事实（锚点）

- **可加性**：`S(v175) = S(v166) + S(v168) − 1001 = 17594`，interaction=0 精确成立。
  含义：任何单侧（Linear 或 Attention）独立官方正向机制，组合到 v175 上都不会产生
  交互惩罚；组合预测 `S_new = S(v175) + step_gain_new`（若 new 只改一侧且与另一侧
  官方父侧正交）。
- **A1 是唯一官方正向 Attention 机制**（+60 over v164）；Linear 侧 v166 rank-1 官方
  +3。其余已测机制（A2/A3 本地负、A4/C1 官方负、C2/C3 本地负、L2/L3/L4 负）全部关闭。
- **当前完整父**：v180 `17597/242s`，距 300s 硬限余量 58s，距榜首 21765 差 4168。
  D1 无在线新增算子，`−3s` 只记实测，不宣称稳定加速。

## 2. 候选方向（按门禁筛选）

### D1：A1 的 Q/K 非对称折叠分配（首选，A1 结构内解析自由度）

- **现状**：A1 把 per-KV-head 的 `gamma` 以 `sqrt(gamma)` 对称折叠进 Q 和 K multiplier
  （`q_mult *= sqrt(gamma)`、`k_mult *= sqrt(gamma)`）。这保持连续域 `QK^T` 内积不变
  （`Q·K = (Q·sqrt(g))·(K·sqrt(g))`），只重分配量化动态范围。
- **D1 想法**：对称分配未必最优——Q 与 K 的量化误差曲率不同，可用一个 per-KV-head
  标量 `alpha ∈ (0,1)` 非对称分配：`q_mult *= sqrt(g)^(1-alpha)`、`k_mult *=
  sqrt(g)^(1+alpha)`，两个指数之和为 2，乘积仍为 `gamma`（连续域 logits 缩放不变），
  但 Q/K 各自缩放不同。
  校准期解闭式最优点（最小化部署后 Q/K 的加权量化误差或输出误差）。
- **门禁**：六 API 兼容（只改 multiplier 值）；动态零新增；HiF4 兼容；与 A1/C1/C2 族
  不同（非通道细粒度、非 outlier 检测、非旋转，是 A1 折叠方式的解析推广）。
- **风险**：C1 官方负表明 K 侧动态范围调整在官方上可能无收益甚至负；但 D1 与 C1
  不同（C1 是逐通道 outlier 压缩，D1 是 per-head 全局 Q/K 非对称重分配）。本地先裁决。

### D2：per-Q-head logits gain（A1 的 head 维分解，对照）

- A1 是 per-KV-head（GQA 组内 7 个 Q head 共享一个 KV-head gain）。D2 把 gain 分解到
  per-Q-head（14 个独立 gain），同时保持 per-KV-head 的 K 侧共享。这增加了 Q 侧自由度
  但维持 K 侧共享；与 C2（通道维）正交。
- **风险**：GQA 组内 Q 共享 K 的量化误差，per-Q-head gain 会打破 A1 已验证的组内一致
  结构；可能重复 C2 的负交互。仅作对照。
- **裁决（2026-09-04，v181 本地预研 REJECTED）**：纯 D2（D1 折叠关闭、A1 对称父）default
  120 mean −0.002746、median −0.000086、54+/66−、median MSE ratio 1.000333；D1+D2 叠加
  default 120 mean −0.002019（60/60）亦负。两分支均负，确认 A1 的 group-consistent 结构
  是承重组件，per-Q-head 分解重复 C2 式负交互。D2 家族关闭，不提交官方、不占配额。

### D3：组合延伸验证（v175 + 新机制）

- 一旦 D1/D2 单侧本地信号清晰且官方父侧固定，按可加性公式构造组合候选提交官方。
- 候选数量固定：每个机制最多一个预注册配置；失败换机制不扫邻域。

## 3. 门禁结论与排序

- **D1**：v180（alpha=0.3 非对称折叠）官方 `17597/242s`，相对 v175 `+3/−3s`；
  RETAINED 为新完整父。本地 compact +0.000088、default 120 +0.000356（69+/51−）、
  QK interaction +0.01106 与官方同号，但不据此建立分数换算。
- **D2**：对照已裁决 REJECTED（v181，见 §2），确认 A1 group-consistent 结构承重。
- **D3**：组合规则已由 v180（v175 + D1）完成，不需要重复构造或另占配额。
- 候选数量固定：每个机制最多一个预注册配置；失败换机制不扫邻域。
- 最终裁决：D1 RETAINED；D2 REJECTED；D3 fulfilled by v180。禁止围绕 alpha 或 head
  粒度继续做邻域扫描。

## 4. D1 固定数学规则（预注册）

1. 复用 A1 的校准 Q/K pairs 与 `gamma`（per-KV-head 乘性 gain）；
2. 对每 KV head，把对称折叠改成 `sqrt(gamma)^(1-alpha)`（Q 侧）与
   `sqrt(gamma)^(1+alpha)`（K 侧），`alpha ∈ [0,1]`；两边指数和为 2，乘积为 `gamma`；
3. `alpha` 由校准期闭式最小化确定：目标 = 部署后输出误差（用独立 holdout 折的
   logits/输出 MSE），或 Q/K 的加权量化重建误差（若输出不可行）；
4. 不搜索 alpha 网格：每 KV head 独立解 1-D 闭式（沿 alpha 的解析/二分），或固定
   alpha 为 0.5 的对称基线对照；
5. 动态零新增：只改 state 中 Q/K multiplier 值；
6. v165 约束强制（复杂计算只在 calibration）。
