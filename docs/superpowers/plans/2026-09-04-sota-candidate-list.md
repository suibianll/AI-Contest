# 下一阶段算法规划：候选机制清单与可行性注记（2026-09-04）

> 状态：**IMPLEMENTING — C1 已实现并加入官方批测队列（v176），等待官方回传后裁决 C2/C3**
>
> 背景：低复杂度扩展计划已全部裁决/实现（A1-A4 + L1-L4 + 组合 v175），官方批测队列
> v171/v174/v175/v176。用户指示继续算法规划与优化。本清单基于外部 SOTA 搜索
> （KVQuant / KIVI / QuaRot / TurboQuant / LonghornSilicon GQA 实测 / KVLinC /
> VecInfer / ResQ / OTT），按计划 §6.1 的 1-5 项门禁筛选。
>
> 提交配额账本（用户约束：自目标设置起提交通过 ≤15 个）：v171/v174/v175 为目标设置前
> 已排期队列（历史排期，不占本次配额）；**v176 为本目标设置后第 1 个**（1/15）。
> 后续新候选每增加一个从配额扣除 1，官方负向不退还。

## 0. 搜索关键证据（第二轮 2026-09-04）

- **KVLinC（arXiv 2510.05373, 2025-10）**：Hadamard rotation（V 侧）+ 针对量化
  K 引入的 logits 误差做**低秩仿射 linear correction adapters**——logits 误差的
  结构化修正方向与 A1/C2 相同（对 QKᵀ 域的误差建模），但借旋转；旋转侧与 C3 高风险
  同一证据冲突（Longhorn Qwen GQA delocalization）。

- **VecInfer（arXiv 2510.06175, 2025-10）**：smooth + Hadamard 抑制 Key outlier 后
  K-means VQ。smooth 部分是 C1 的幅值插值/等化变体；VQ 码本编码需在线 lookup——
  按 v161/v128 家族证据动态 per-call 小张量算子官方超预算，**排除**。

- **ResQ（ICML 2025）**：PCA 低秩子空间保留 8-bit、其余 4-bit 混合精度——需要
  在线 per-channel 高精度旁路通道表，HiF4 五字段无此存储，同 C4 论证**排除**。

- **OTT（arXiv 2505.10938）**：outlier token 在线追踪并排除——动态逐 token 决策
  分支，官方超时族证据一致，**排除**。

- **第二轮结论**：新 SOTA 的可迁移增量集中在「QKᵀ logits 误差的结构化修正」
  （= A1/C2 数学目标）与「K outlier 处理」（= C1 已覆盖）。无新的零动态、HiF4
  字段兼容、非已闭合家族的独立机制方向；后续以 C2（A1 细粒度仿射/低秩化）为唯一
  已注册新候选，C3 为条件对照。

## 0b. 搜索关键证据（第三轮 2026-09-04，NVFP4 scale 初始化/精化）

- **ScaleSweep（arXiv 2606.07618, 2026-05）**：AbsMax 之外对 FP8 微块 scale 做
  可行候选 sweep 最小化 MSE/WMSE——权重/激活 block-scale 选择空间。与已闭合族
  冲突：v170/A3 已证「标准 scale 已输出最优」（winner 11/12 = 0），且本地
  Linear 结构族（full64/Householder/单折邻域）已闭环；权重侧 scale 精化族即使
  数学上等价于 E6M2 层级 scale 重选，仍属已试域，**不注册为新候选**。

- **H-Scale（arXiv 2608.28113, 2026-08, Qwen Team）**：Hessian 对角加权选 NVFP4
  per-group scale，零在线开销。同一 scale 选择域，且其收益主要落在 weight 侧
  E2M1 微块；本计划 Attention 侧 Q/K/V 的 scale 来自 NVFP4 carrier（非候选可改
  的自由度），权重侧归 Linear 已闭族，**不注册**。

- **NVFP4 微块缩放（E4M3 per-16 + tensor 全局 FP32）**：HiF4 本地参考 codec
  已知；scale 自由度的最优性已被 A3/v170 官方负向与校准实验覆盖。

- **第三轮结论**：scale 初始化/精化类方法（无论 sweep 还是 Hessian 加权）都落入
  Linear 已闭族或 A3 已证最优域；Attention 侧无新增可迁移自由度。排除了第三条
  独立机制路线，C1/C2/C3 排序维持不变。

## 0. 搜索关键证据（第一轮 2026-09-04）

- **KVQuant（NeurIPS 2024）**：per-channel Key 量化 + Pre-RoPE + per-vector
  dense-sparse outlier 隔离——K 激活存在少数固定高幅通道，统一量化的 scale 被其拖累；

- **LonghornSilicon（Qwen2 GQA 实测，同家族）**：KV 量化误差被少数固定高幅 key
  channel 主导；**random rotation 会 delocalize 该误差导致无 per-token 保护**——
  Qwen GQA 上旋转类方法高风险；最终方案 **ChannelQuant = per-channel key + static
  outlier-channel isolation**；

- **v168（A1）官方 +60**：官方 panel 的 Q/K 幅值结构存在可校正的稳定偏置
  （per-KV-head 全局 gamma 捕获整体缩放）。

## 1. 候选机制清单（按门禁筛选）

| #  | 机制                                                                                                      | 来源                   | 1 六API/state | 2 动态 O(TD)     | 3 HiF4 兼容                 | 4 与已关闭族不同                                            | 5 理论预期                                                     |
| -- | ------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | -------------- | ------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| C1 | **K 侧 static outlier-channel 等化**：校准期检测跨 fold 稳定高幅稀疏 K 通道，构造 outlier-aware per-channel 修正折叠进 multiplier | KVQuant/ChannelQuant | ✓            | ✓（零新增算子）       | ✓（只改 multiplier 与 K 编码行为） | 与 SmoothQuant（幅值插值 α）不同：针对固定通道结构、解析检测+固定压缩           | 消除 K scale 被 outlier 拖累；官方对 Q/K 幅值结构敏感（A1 证）               |
| C2 | **A1 细粒度化**：per-(KV-head, 8通道组) logits 增益（A1 的 8 组粒度推广）                                                 | 内部（A1 扩展）            | ✓            | ✓              | ✓                         | 非邻域扫参：同数学目标的分辨率提升                                    | 官方 A1 正证据在线，更细结构或捕获更多                                      |
| C3 | 固定 8×8 随机正交旋转（GQA 组内，固定 seed）                                                                           | QuaRot/TurboQuant    | ✓            | ✓（编码前一次 8×8 乘） | ✓                         | 与 block Hadamard（缩并结构）/Householder（数据感知）不同：非数据感知随机正交 | **高风险**：Longhorn 示 Qwen GQA rotation delocalize error；仅作对照 |
| C4 | V 侧 ChannelQuant per-token——**不可行**：per-token scale 需在线独立 scale，HiF4 五字段无 per-token 表                   | —                    | ✗            | ✗              | ✗                         | —                                                    | 排除                                                         |

## 2. 门禁结论与排序

- **C1 首选**：唯一同时满足「官方可校正幅值结构（A1 证据）+ Qwen GQA 稳健配方
  （ChannelQuant）+ 零动态成本」的机制；实现为 v176（从 P\_A = v168 构造）。

- C2 作为 A1 扩展留档，待 C1 官方结果后决定（避免同侧同线并发）。

- C3 高风险标记；Longhorn 证据指向 Qwen 上旋转有害，仅当 C1/C2 官方均负时考虑
  一次对照。

- 排序冻结：实施顺序 C1 →（官方回传后）C2 →（条件触发）C3。

## 2b. C2 预注册数学模型（细化定稿，C1 回传后直接实施）

C2 = A1 的细粒度化：把 A1 的 per-KV-head 单一 logits 增益推广为
per-(KV-head, 8 通道组) 增益。数学上与 A1 同目标（调整量化后 row-centered
causal logits 相对 float 的乘性偏差），仅提高分辨率。

1. 校准期（final 部署坐标，复用 A1 的校准 Q/K pairs 与对应 Q/K state）把
   head\_dim 划分为 B=8 个连续通道组（每组 head\_dim/8 通道）；

2. 对每 KV head h、每个 case 构建量化后/float 的 row-centered causal logits
   的组内部分和：

   - `logits_q^{(b)} = <q_hat^{(b)}, k_hat^{(b)}>/sqrt(d)`（该组的部分和），

   - center\_q^{(b)} = row-centered causal 化后的组内部分和；

   - center^{(b)} = 同一组 float Q/K 的对应中心化部分和；

3. 每 KV head 独立解 **闭式 8 参数线性最小二乘**
   `min || center − Σ_b g_{h,b} · center_q^{(b)} ||²`（B=8 方阵
   Normal equation，calibration-only，单次）；偶数/奇数 fold 分别拟合，
   取 g 的 log-median 并做与 A1 相同的 log-shrink 与 clamp；

4. 折叠：Q 与 K 在**同一通道组内同乘 sqrt(g\_{h,b})**——保持连续域
   「组内 QᵀK 乘性缩放」结构，即量化后 `Σ_b g_{h,b}·partial_b` 逼近
   float logits，与 A1 的折叠方式逐层一致（A1 是 B=1 特例）；

5. 不搜索 B/shrink/clamp；B=8、shrink/clamp 继承 A1 的预注册常数；
   不做 head/layer 路由；fold 聚合用 median，禁止只取第一/最好 fold；

6. 动态零新增：只改 state 中的 Q/K multiplier。

7. v165 约束（动态 API 无 Gram contraction、无候选循环、复杂计算只在
   calibration）强制。

实施前提（门禁）：仅在 C1 官方回传并裁决后启动；若 C1 官方正 → C2 从 C1
候选继续（C1 语义并入 C2 基础 multiplier）；若 C1 官方负 → C2 从 P\_A=v168
继续。C2 是 A1 的数学扩展而非邻域扫参，仍只允许一个预注册配置。

## 3. C1 固定数学规则（预注册）

1. 校准期（final 部署坐标，复用 a1\_k 前 128 tokens）按每个 KV head 计算
   `peak_j = median_f(amax_t |K_f,t,j|)` 与 `med_j = median_f(median_t |K_f,t,j|)`;
2. 检测 sparse-outlier 通道：`j ∈ O` 当 `peak_j / med_j > rho（固定 4.0）` 且跨折
   符号一致；
3. 构造逐通道修正：`k_eq_j = (peak_j·(1/7) 目标幅值) / peak_j` 对 outlier 通道，
   其余通道 1.0——即把 outlier 通道峰值压缩到普通通道中位量级；
4. 平滑收缩 `k_eq = 1 + beta·(k_eq − 1)`，`beta = 0.25` 固定；乘进既有 K multiplier
   （`k_multiplier *= k_eq`），Q 侧以连续域 `QKᵀ` 不变为约束做对应补偿
   `q_multiplier *= 1/k_eq`（保持内积，仅重分配两侧量化动态范围）；
5. 不搜索 rho/beta/通道数；`rho=4.0`、`beta=0.25` 为公式常数；
6. 动态零新增：只改 state 中的 multiplier 值。

产物：v176（solution.py + result.md），已加入官方批测队列（对 `14005`，即官方父侧
P\_A = v168 而非原始锚点 13945）。

## 4. v176 实现与本地实测（2026-09-04）

- 构造：从 P\_A = v168；Linear 未改动；仅在校准期编译 k\_eq/q\_eq 并折进 Q/K
  multiplier，动态零新增。

- 机制 reachability：outlier 注入测试证伪（outlier 通道 k\_eq < 1 压缩、
  q\_eq = 1/k\_eq 展开到 Q 布局）；GQA 维度已修（q\_eq 由 per-KV-head
  (kv\_heads, head\_dim) 经 repeat\_interleave(group) 展开到 (q\_heads·head\_dim,)）。

- compact 4（配对 v168）：mean Δgain +0.002444、3+/1−；QK-only +0.0029、
  QK interaction +0.358。

- default 120（配对 v168）：mean Δgain −0.004450、56+/64−（win 0.467）；
  QK interaction +50.77 强正；L16 consistent\_improvement（+0.0727）、L4
  consistent\_regression（−0.0039）；len10 最负、len1024 中性微正；V 侧
  v\_only = 0.0 control 干净。

- gpt2 attn compact 4（配对 v160 父）：mean Δgain −0.002753（1+/3−）、
  QK-only +0.002406 为正——标记 `model-specific-risk`，只作封存 holdout，
  不据此调参数/路由，仍由 v176 首次官方结果裁决。

- 时间：attention default 校准 60.15s（v168 基线 68.40s）、动态 Q/K/V 3.36s，
  无时间风险。

- 裁决规则：轻微本地负向不取消首次官方测量；官方负 → C1 关闭，切 C2，不调
  rho/beta/通道数重扫；官方正 → C1 晋级，组合条件维持 `S_pred = 4590 + S_c1 − 1001`。

