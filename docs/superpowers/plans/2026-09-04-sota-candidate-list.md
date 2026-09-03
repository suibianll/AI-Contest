# 下一阶段算法规划：候选机制清单与可行性注记（2026-09-04）

> 状态：**IMPLEMENTING — C1 已实现并加入官方批测队列（v176）；C2/C3 家族本地全部
> 关闭（v177=C2 on v168、v178=C2 on C1、v179=C3 rand8 对照均本地 REJECTED，
> 明确负优化）。计划候选清单 C1/C2/C3 全部裁决完毕，仅 C1 官方批测中**
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

## 0c. 搜索关键证据（第四轮 2026-09-04，FP4/MXFP4 encoder 内自由度与误差分解）

- **MXFP4 误差三分量分解（arXiv 2605.20402, 2026-06）**：scale bias（E8M0
  2 的幂舍入 \~44% 超调）、deadzone truncation（< 块最大 1/24 的条目清零，
  \~9% 权重）、grid noise（E2M1 网格噪声，对 scale 精度不变、不可约）。对应
  修正：Macro-block scaling（= scale 精化域，第三轮已闭）、Outlier Fallback
  （恢复清零条目，需在线逐元素判断/独立存储，HiF4 五字段无表 → 同 C4 排除）、
  AQN（训练侧熵控制，PTQ 不适用）。

- **Hot-Channel Patch / CHON（arXiv 2602.02047, 2026-02, NVFP4 预训练）**：
  post-QK 操作对量化最敏感；训练后期 outlier 收敛为持续 hot channels，HCP
  在线注入残差。post-QK 敏感 = 已由 A1/C2 覆盖（logits 域修正）；hot-channel
  在线残差需逐 token 注入 → 动态排除。

- **ARCQuant（2026）**：outlier channel 残差二次量化并拼接 reduction 维——
  需要在线为残差分配额外通道存储，HiF4 五字段无此表 → 排除。

- **Dissecting Outlier Dynamics**：LA vs SA 的 heavy-tail 差异、FFN SwiGLU
  是 outlier 源——跨结构诊断，无六 API 可注入的自由度。

- **Full-Stack FP4 / Metis（训练侧）**：LoRA-SVD、混合精度训练配方——PTQ
  评测不适用。

- **第四轮结论**：FP4 encoder 内剩自由度可映射为（a）scale 精化（已闭）、
  （b）deadzone/残差恢复（需在线存储，C4 论证排除）、（c）logits/post-QK
  修正（A1/C2 覆盖）。无新可注册机制；C1/C2/C3 排序第三次验证维持。

## 0d. 搜索关键证据（第五轮 2026-09-04，logits 校准/温度匹配域）

- **QuantVLA（arXiv 2602.20309, 2026-02）**：attention temperature matching——
  per-head logits scaling folded into dequant scales。**与 A1（v168，per-KV-head
  logits gain folded into multiplier，官方 +60）数学同构**——证实 A1 方向是
  领域公认的低成本 PTQ 校正，无新独立机制。
- **SageBwd / QK-norm（arXiv 2603.02170, 2026-03）**：R1.0、K-smoothing 必需、
  Q-smoothing 收益有限——与 C1（K 侧 outlier 等化）方向一致；为预训练侧结论，
  无六 API 新注入自由度。
- **Rank-Aware Spectral Bounds（arXiv 2602.18851, 2026-02）**：logits 谱界预测性
  校准——训练/校准 scale 选择域，A3/v170 已证标准 scale 输出最优，不注册。
- **温度标定（temperature scaling）文献**：经典单标量 T 标定——乘性 logits 域，
  已被 A1（per-KV-head、官方正）覆盖的更细结构与 log-shrink 版本胜出；无新方向。
- **第五轮结论**：logits/温度校准域的可迁移增量已全部落在 A1（官方正）与其细粒度
  变体（C2 已本地 REJECTED）；无新可注册机制。C1/C2/C3 裁决与 SOTA 收敛性第五次
  验证维持；官方回传后仅剩 C1 裁决与组合条件。

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
| C2 | **A1 细粒度化**（REJECTED）：per-(KV-head, 8通道组) logits 增益（v177 on v168 + v178 on C1 双基底本地均负） | 内部（A1 扩展） | ✓ | ✓ | ✓ | 已实测：负交互反噬单侧收益 | 家族关闭 |
| C3 | 固定 8×8 随机正交旋转（REJECTED）：GQA 组内固定 seed 正交，compact 4 `−0.229` 大负，与 Longhorn 证据一致 | QuaRot/TurboQuant | ✓ | ✓ | ✓ | 已实测：delocalize 丢失 per-token 保护 | 家族关闭 |
| C4 | V 侧 ChannelQuant per-token——**不可行**：per-token scale 需在线独立 scale，HiF4 五字段无 per-token 表                   | —                    | ✗            | ✗              | ✗                         | —                                                    | 排除                                                         |

## 2. 门禁结论与排序

- **C1 首选**：唯一同时满足「官方可校正幅值结构（A1 证据）+ Qwen GQA 稳健配方
  （ChannelQuant）+ 零动态成本」的机制；实现为 v176（从 P\_A = v168 构造）。

- C2 双分支（v168 基底 / C1 基底）本地均 REJECTED，家族关闭，不提交官方。
- C3（固定 8×8 随机正交旋转对照）本地预研（v179）**明确大负优化**（compact 4
  `mean Δgain −0.229273`、1+/3−），与 Longhorn Qwen GQA rot delocalize 证据
  一致；条件触发（C1 官方负）时直接 REJECTED 不提交，省配额。C3 家族关闭。
- 排序冻结：C1（官方批测中）。计划候选清单 C1/C2/C3 全部裁决完毕。

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

**v177 本地预研（2026-09-04，v168 基底，C1 官方负分支的预演）**：compact 4
`mean Δgain −0.006858`（1+/3−）；default 120 `mean −0.006643`（41+/79−，
win 0.342）、median −0.004289、QK interaction −0.0935（负交互反噬单侧
q-only/k-only 正表面积改善）。结论：A1 单一 per-KV-head 增益已捕获可校正
结构，B=8 组级细粒度化为负优化。C1 负分支 **REJECTED**（v177 归档），不
提交官方、不占配额。

**v178 本地预研（2026-09-04，C1 基底 = v176，C1 官方正分支的预演）**：
compact 4（配对 v176）`mean Δgain −0.009194`（1+/3−）；default 120（配对
v168，同 v177 口径）`mean −0.008610`（47+/73−，win 0.392）。C2 在 C1
基底上同样明确负且整体更负——负交互在 C1 处理后仍存在。C1 正分支
**REJECTED**（v178 归档），不提交官方、不占配额。

**C2 家族裁决：两个预注册分支本地均明确负优化，C2 从候选清单移除**。无论
C1 官方正负，C2 都不应提交官方。后续仅剩 C3 条件对照（C1 官方负时考虑
一次，Longhorn 证据预计亦负）。

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

- opt-125m attn 60（配对 v160 父）：mean Δgain −0.021851（30+/30−）、
  QK-only −0.0246；两架构对 C1 均无正向跨模型信号，风险标记维持；Linear
  侧差异来自父结构不同（v176=standard vs 父=v160）不可归因 C1。

- 时间：attention default 校准 60.15s（v168 基线 68.40s）、动态 Q/K/V 3.36s，
  无时间风险。

- 裁决规则：轻微本地负向不取消首次官方测量；官方负 → C1 关闭，切 C2，不调
  rho/beta/通道数重扫；官方正 → C1 晋级，组合条件维持 `S_pred = 4590 + S_c1 − 1001`。

