# v162 独立 Attention 优化执行日志（A 代理）

> 任务书：[workpackages/v162-attention.md](../../docs/superpowers/plans/workpackages/v162-attention.md)
> 总计划：[2026-09-06-v162-independent-linear-attention-plan.md](../../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> 零点：v162 standard-baseline-both，SHA `56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`，官方 1001/146s。
> 本侧冻结：两个 Linear API、dynamic V、standard codec。工作目录 `workbench/v162_attention/`。

## A0 基线与去重（2026-09-06）

- baseline 复制到 `workbench/v162_attention/baseline/solution.py`，SHA256 复核一致
  （`56101559…C000A`，与总计划唯一零点相同）。
- 环境：`.venv` CUDA 可用（RTX 3060 Ti）。GPU 排队按总计划 §2 使用
  `artifacts/proxy_v3/v162-independent/gpu.lock` 原子创建；锁存在则等待，不删除对方锁。
- 面板事实（来自 cache 与 evaluator 源码）：
  - Qwen2.5-0.5B：hidden 896、q_heads 14、kv_heads 2、head_dim 64（G=2 组、每组 7 个 Q head）；
  - calibration 5 窗口，长度 (10, 128, 512, 1024, 1024)；gate=输入顺序最后一窗（1024），其余 4 窗训练；
  - proxy-v3 每 shard 4 层 × 2 in-dist 窗口 = 8 attention cases，六 shard 共 48；
  - evaluator `_attention` 为 non-causal GQA（repeat_interleave），gain=(MSE_STD−MSE_PLAYER)/MSE_STD。

### novelty 表（任务书 A0.3）

| 维度 | 08-26 learned butterfly（Phase D，仅计划未执行） | v187/v188 importance | 09-06 Jacobian pair-matrix | 本轮机制 |
|---|---|---|---|---|
| 参数空间 | 6 蝶形 stage × 8 共享角度，`[kv_heads,1,6,8]`，无 D×D 密集矩阵 | 对角 importance（无旋转、无新参数） | 固定 2×2 SPD pair transform（闭式拟合，非梯度） | 每 KV group 全密度正交 `R_g∈R^{64×64}`，Cayley(Θ)+Hadamard 初始化 |
| 损失 | MSE ratio + 尾部 relu 惩罚（`_attention_ratio_objective`） | operand/坐标敏感度 | logit→输出 Jacobian Frobenius 加权协方差 | 完整 Attention 输出 MSE / MSE_STD（标准量化 V 进 player 路径） |
| 硬编码器 | STE `_ste_hif4` | 无（解析权重） | 真实部署 gate | v162 标准 HiF4 硬码，STE 仅 backward |
| V 是否量化 | 未明确（沿用旧例程） | n/a | 用 v-state 实际部署输出 | 是：Vh=标准 HiF4，进 O_hat 与 O_std |
| GQA | 未按 group 共享（逐 kv_heads 角度） | n/a | 2×2 pair（非 group 级） | 同 group Q head 与 K head 共享同一 R_g，连续 QK 不变量精确保持 |
| 训练/holdout | 偶数样本训练/奇数验证 | 校准内 | 奇偶窗口隔离 | 4 训练窗 + 第 5 窗 gate + evaluator validation/test 独立 split |
| 可达计数 | 未执行 | 已执行（v187 官方 9167） | 已执行（TIME_REJECTED） | 记录 attempted/accepted 与 gate 决策 |
| 父/耗时 | C21-C（未达实现） | v185/v186 | v189 | v162 零点；32 步/层训练全在 calibration API 内计时 |

结论：三个历史机制均与本轮「group 级全密度正交矩阵 + 完整输出损失梯度学习」不数学等价；
learned butterfly 从未执行，不存在等价的有效失败实验 → 非 DUPLICATE_CLOSED，继续 A1。
另注意：08-26 C22 的 Linear 侧 signed Hadamard R64 被拒（operand-local 指标），与本轮
Attention 输出目标不同面板不同目标，不构成本机制的反证。

### 本地最高精度锚点（提交参照）

- v162 零点本侧 gain=0（构造性）。
- 强对照 v168（`5988AE47…01AC79`）与最新完整父 v189 Attention 侧需在同协议下成对读取，
  不得混用旧 panel 均值；v189 default Attention 侧历史读数 `0.752173407020` 仅作背景，不作直接比较。
- 用户目标：本地 Attention gain 达 0.9；每超过本工作流已测最高即 commit & push 本侧文件。

## A1 最小合法实现（进行中）

- 候选骨架：复制 v162，新增 attention 专用 helper，不改 `_standard_params`/Linear/V 路径。
- state 约定：`q_state`/`k_state` 各含独立 `R`（float32 CPU，`[G,D,D]`）与形状元数据；`v_state={}`。
- 动态路径：NVFP4 解码 → head reshape → group R matmul → v162 标准 HiF4 编码。无候选循环、无跨调用缓存。
- 研究臂开关：`_ROT_ARM ∈ {identity,h,learned,gate}`（研究用副本），部署候选固定 `gate`。

## A1 最小合法实现（2026-09-06 完成）

- 候选：`workbench/v162_attention/candidate/solution.py`（SHA `19159AB9904FB3F1…`，
  完整 SHA 见 git 提交）。v162 标准 codec 原样保留；新增 attention 专用 helper；
  `_standard_params`/Linear/V 路径未动。
- 机制：每 KV group 一个 `R_g∈R^{64×64}`，Cayley(Θ)+规范化 Sylvester Hadamard 初始化；
  同组 7 个 Q head 与 1 个 K head 共享 R_g；动态路径 NVFP4 解码 → group matmul → v162 标准 HiF4 编码；
  state 仅存 CPU float32 `r`（`[G,D,D]`）与审计字段；无候选循环、无跨调用缓存。
- 训练（校准内，arm=learned/gate）：4 训练窗（等间隔确定性采样 ≤128 KV/≤32 Q token，
  Q 行取自 KV 子集）+ 第 5 窗 gate（完整长度、Q 分块 512、K/V 全量）；
  Adam lr=0.01×32 步、clip 1.0、reg `1e-3·mean((C−I)²)`；STE 仅 backward；非有限即 ERROR。
- 部署选择（arm=gate）：先 learned vs H（gate 上严格更好才选 learned），再胜者 vs identity
  （严格更好才部署，平局归 identity）。
- 必要测试 `workbench/v162_attention/tests_a1.py` 全部通过（CPU 模式，不占 GPU）：
  - T1 R=I 六 API 与 v162 逐位对齐（含 state 审计字典合法）✓
  - T2 合成正交 R 的 GQA 连续 QK 误差 2.67e-05 ≤ 4.26e-04 ✓
  - T3 token 数/调用次序互换一致、无跨调用状态污染 ✓
  - T4 五字段过 reference 合法性，decoded == 研究硬前向（逐位）✓
  - T5 STE 前向==decode(encode(x))、梯度有限非零；合成数据训练冒烟 loss 0.9139、正交误差 3.58e-07 ✓
  - T6 Linear 与 V 状态/编码/输出逐位不变 ✓
- 实现期修复：einsum 组维（`gkd`）、gate 评估 K/V 全量（non-causal 不随 Q 分块）、
  state 审计浮点 NaN→-1.0 哨兵、`_cayley_orthogonal` 命名统一。

## A2 固定配置训练实验（2026-09-06 进行中）

- config.json 已在读取训练结果前保存（任务书 A2 表，无邻域扫描）。
- GPU 排队按计划协议执行（`gpu.lock` 原子创建）；期间 Linear 代理多次持锁（l1~l4 run），
  本侧全部等待，未删除对方锁。
- gate 诊断（真实 proxy-v2 cache 校准窗，归一化完整输出损失，identity=1.0，越低越好）：

  | 层 | H | learned | gate 部署 |
  |---|---|---|---|
  | 0 | 6.248150 | 5.970765 | identity |
  | 8 | 1.095325 | 1.125787 | identity |
  | 15 | 0.683861 | **0.619129** | learned |
  | 23 | **0.792165** | 0.795395 | h |

  校准耗时 0.9–3.1 s/层。层 0/8 的 Hadamard/learned 明显劣化，gate 正确回退 identity；
  层 15/23 有材料正信号。数据：`research/a2_gate_diagnostics.json`。
- 四臂 + 强对照 evaluator 运行（shards 0,2,3,5 attention-only，baseline=v162 零点）进行中：
  a2-gate / a2-h / a2-learned / a2-strong-v168 / a2-strong-v189；臂变体 SHA：
  h=`F156D719D561A3C5…`、learned=`8C7C88B7C658501E…`。

## A2 四臂结果（2026-09-06，shards 0,2,3,5，32 case，baseline=v162 零点 gain=0）

| 臂 | mean Δ0 | median | +/0/- | L1_neg | val mean | test mean |
|---|---|---|---|---|---|---|
| **a2-gate（部署）** | **+0.421328** | +0.473353 | 26/6/0 | **0.000000** | +0.433259 | +0.409397 |
| a2-h | +0.376753 | +0.424341 | 26/6/0 | 0.000000 | +0.387913 | +0.365593 |
| a2-learned（未门控） | +0.168222 | +0.489186 | 27/0/5 | 0.260514 | +0.136286 | +0.200157 |
| a2-strong-v168 | +0.749391 | +0.723832 | 32/0/0 | — | +0.754768 | +0.744015 |
| a2-strong-v189 | +0.757064 | +0.728972 | 32/0/0 | — | +0.761906 | +0.752221 |

- **CLEAN_ROOM_PROGRESS**：相对 v162 mean/median 均正、两个 split mean 均正 ✓。
- gate 完全消除 learned 臂的负 case（层 0 灾难性劣化被 identity 回退拦截）；
  gate 优于纯 H 臂（+0.4213 vs +0.3768），学习旋转在 H 之上有材料增量。
- 未超过强对照 v168/v189 → 按任务书记 **RECOVERY_ONLY**，不作为填补榜首差距的新机制。
  突破研究目标（D_strong≥20% vs 强对照）未达：D_strong = −1.309（vs v168）/ −1.382（vs v189）。
- L1_total（仅记录）= 0.421328；L1_negative = 0 < 0.02 ✓。
- 依据任务书不调学习率/步数/seed/自由度；后续增益走新机制（含历史机制 RECOVERY 迁入）。

## A3 六 shard 完整评测（2026-09-06，48 case + OOD）

- **ID**：mean **+0.429820**、median +0.465527、42 正/6 零/0 负、L1_neg **0.000000**；
  split：test +0.434912 / validation +0.424728 均正。
- **逐层部署状态**（gain=0 即 identity 回退）：层 0/2/8 → identity；其余 21 层部署旋转，
  逐层 mean +0.077（层23，H 臂）～ +0.749（层3）。门控按层独立决策，无正层挑选。
- **OOD**：candidate in-dist +0.429820 / ood **+0.438459**；Δgap_mean = **−0.008639**，
  |Δgap| ≤ 0.01 → **OOD 门通过**（未 blocked）；无分布拟合失败特征。
- 强对照（同协议 A2 面板读数）：v168 +0.7494 / v189 +0.7571；本机制从 v162 零点单独贡献
  +0.43 量级，未超强对照栈。
- control 与脱离仓库单文件导入检查：见下节。

## A3 default 面板、计时与门禁裁决（2026-09-06）

- **fresh default（兼容后端 168+120，nvfp4-cache hit，总实测 56.5s）**：
  Attention mean **0.422443**、Linear 0.0（standard 冻结 ✓）、Overall 0.176018。
  与六 shard 面板 +0.4298 一致。
- 六 API 计时（default 面板实测）：W_calib 0.680s、A_calib **23.706s**（24 层 × ~1s 训练）、
  dyn_act 1.697s、dyn_q 0.352s、dyn_k 0.260s、dyn_v 0.231s。
- **官方时间预测**：`T ≈ 170.3 + 0.115·0.680 + 0.694·23.706 + 0.734·1.697 − 1.58·0.843
  = 186.744s < 280s` ✓（官方硬限 300s；对 v189 290s 锚点余量充足）。
- **真实 control**：60 项比较（4 层 × {weight calib, dyn act} × 7 role + dyn V）全部
  与 v162 逐位一致 ✓（`REAL-INPUT CONTROL PASS`）。
- **脱离仓库单文件导入**：临时目录独立导入，六 API 组全部可用 ✓。
- **门禁裁决（run_id a3-id/a3-ood/a3-default）**：

  | 门 | 结果 |
  |---|---|
  | 合法性/five-field/state | PASS（evaluator 全 case 校验 + reference_hif4.validate_state）|
  | 未修改侧 control | PASS（60 项逐位）|
  | 校准/holdout 隔离 | PASS（gate 窗独立；val/test 双 split 正）|
  | L1_negative < 0.02 | PASS（= 0.000000；L1_total 0.429820 仅记录）|
  | mean Δ>0 vs v162 与直接父 | PASS（直接父=v162 零点）|
  | OOD \\|Δgap\\| ≤ 0.01 | PASS（−0.008639）|
  | 时间预测 < 280s | PASS（186.744s）|
  | 单配置/无邻域扫描 | PASS（config.json 冻结）|
  | reachability | 21/24 层部署旋转（attempted 24，accepted 21），层 0/2/8 gate 回退 identity |
  | 强对照 | 未超 v168（+0.7494）/ v189（+0.7571）→ RECOVERY_ONLY 标签维持 |
  | 官方 | unregistered/NA（提交由用户执行）|

- **结论**：候选 `workbench/v162_attention/candidate/solution.py`（arm=gate）通过全部本地门，
  记 CLEAN_ROOM_PROGRESS / RECOVERY_ONLY，作为 v162 Attention 分支第一个可登记机制。
  分支本地最高 Attention：default 0.422443 / shard48 0.429820（父 v162=0）。
  官方贡献待用户提交后按 `S_A − 1001` 登记。

## R1 RECOVERY：v189 Attention 栈迁入 + 标准 Linear（2026-09-06）

- 机制来源与归属：**RECOVERY**——整体迁入 v189（v186 Attention 栈：Smooth-QK/pair-matrix/
  block-smooth/logit-gain/V refinement 等，官方锚 v189=17616 总分、Attention 侧 default 0.752173）。
  按总计划 §1 允许"明确机制来源后单独迁入；已知收益记 RECOVERY，不记新突破"。
- 手术方式：`workbench/v162_attention/candidate_v2/solution.py` = v189 源文件 + 文件尾追加
  v162 标准 Linear 覆盖定义（`hif4_calibration_and_quantize_weight`/`hif4_dynamic_quantize_activation`
  shadow 定义 + 独立 `_branch_*` codec helper，零删除、零注意力侧改动）。
  SHA 前缀 `3619BFEB0E017555BD8F`。
- **逐位验证**：Linear 56 项比较 == v162 standard ✓；Attention 12 项 API 探针 == v189 ✓；
  state 过 reference 合法性 ✓；脱离仓库单文件导入 6/6 API ✓。
- **评测**（r1-screen / r1-id / r1-ood / r1-default）：
  - ID 48-case：mean **+0.752772**、median +0.724958、48 正/0 零/0 负、L1_neg 0；
    split test +0.753569 / validation +0.751976 均正。
  - OOD：in +0.752772 / ood +0.751857，Δgap **+0.000915** ≤ 0.01 → 过门。
  - **default：attention_mean 0.752173**（与 v189 历史值 0.752173407020 完全一致）；
    Linear 0.0 ✓；Overall 0.313406。
  - 时间：A_calib 65.710s、W_calib 0.668s、dyn_act 1.466s、dyn_q/k/v 合计 3.306s →
    **预测 211.832s < 280s** ✓。
- **分支本地最高 Attention 更新：default 0.752173 / shard48 +0.752772**（前值 0.4224/0.4298 旋转 gate 臂）。
- 下一步（R2）：在该栈之上训练旋转（surrogate 走完整部署路径），gate 逐层决策，
  目标超过 0.752173；此后继续新机制逼近 0.9。

## R1 RECOVERY：v189 Attention 栈迁入 + 标准 Linear（2026-09-06）

- 机制来源与归属：**RECOVERY**——整体迁入 v189（v186 Attention 栈：Smooth-QK/pair-matrix/
  block-smooth/logit-gain/V refinement 等，官方锚 v189=17616 总分、Attention 侧 default 0.752173）。
  按总计划 §1 允许"明确机制来源后单独迁入；已知收益记 RECOVERY，不记新突破"。
- 手术方式：`workbench/v162_attention/candidate_v2/solution.py` = v189 源文件 + 文件尾追加
  v162 标准 Linear 覆盖定义（`hif4_calibration_and_quantize_weight`/`hif4_dynamic_quantize_activation`
  shadow 定义 + 独立 `_branch_*` codec helper，零删除、零注意力侧改动）。
  SHA 前缀 `3619BFEB0E017555BD8F`。
- **逐位验证**：Linear 56 项比较 == v162 standard ✓；Attention 12 项 API 探针 == v189 ✓；
  state 过 reference 合法性 ✓；脱离仓库单文件导入 6/6 API ✓。
- **评测**（r1-screen / r1-id / r1-ood / r1-default）：
  - ID 48-case：mean **+0.752772**、median +0.724958、48 正/0 零/0 负、L1_neg 0；
    split test +0.753569 / validation +0.751976 均正。
  - OOD：in +0.752772 / ood +0.751857，Δgap **+0.000915** ≤ 0.01 → 过门。
  - **default：attention_mean 0.752173**（与 v189 历史值 0.752173407020 完全一致）；
    Linear 0.0 ✓；Overall 0.313406。
  - 时间：A_calib 65.710s、W_calib 0.668s、dyn_act 1.466s、dyn_q/k/v 合计 3.306s →
    **预测 211.832s < 280s** ✓。
- **分支本地最高 Attention 更新：default 0.752173 / shard48 +0.752772**（前值 0.4224/0.4298 旋转 gate 臂）。
- 下一步（R2）：在该栈之上训练旋转（surrogate 走完整部署路径），gate 逐层决策，
  目标超过 0.752173；此后继续新机制逼近 0.9。

## R2 REJECTED（OOD_GATE）：旋转部署进 v189 栈（2026-09-06）

- 机制：R1 栈之上，按 A2 冻结配置训练每 KV group 旋转（标准编码器 surrogate），部署在
  `_nvfp4_to_hif4` 全部栈变换之后、编码之前（连续 QK 与 R1 严格不变），gate 在真实部署路径上
  逐层决策（identity 严格更优则回退）。候选 `candidate_v3/solution.py`，评测版完整 SHA `0B56CCA1F557E44CC6409822C64D29919D6955ABC164C59105138594022A9C7C`（typing 导入修正后、评测前定稿）。
- **ID 48-case：+0.773821**（vs R1 +0.752772，+0.021），48 正/0 负，双 split 正
  （test +0.772 / val +0.775）；**default attention_mean 0.767021**（vs R1 0.752173，+0.0148）。
- 时间：A_calib 89.873s（含 24 层训练+gate 评估），**预测 228.936s < 280s** ✓。
- **OOD 门未过**：candidate gap = +0.773821 − (+0.757615) = **+0.016207**；
  相对直接父 R1（同 SHA in-dist/OOD 配对，gap +0.000915）的 Δgap = **+0.015292 > 0.01** → BLOCKED。
  旋转收益主要留在 WikiText 分布内，构成分布拟合特征（与 09-04 OOD 标定的 gap 家族
  Attention +0.015~0.022 区间一致）。
- **裁决：REJECTED / OOD_GATE**。不晋级、不替换 R1；不通过重训/缩容量/换折等邻域手段修复
  （按"失败换机制"纪律）。教训：标准编码器 surrogate 训练的旋转与 v189 栈组合后，
  分布鲁棒性下降；后续新机制需在校准目标中直接体现部署路径或使用分布更稳的参数化。
- **分支状态保持：最佳候选 = R1（`candidate_v2/solution.py`，default 0.752173 / shard48 +0.752772），
  全门通过，官方 unregistered/NA。**

## R2 REJECTED（OOD_GATE）：旋转部署进 v189 栈（2026-09-06）

- 机制：R1 栈之上，按 A2 冻结配置训练每 KV group 旋转（标准编码器 surrogate），部署在
  `_nvfp4_to_hif4` 全部栈变换之后、编码之前（连续 QK 与 R1 严格不变），gate 在真实部署路径上
  逐层决策（identity 严格更优则回退）。候选 `candidate_v3/solution.py`，评测版完整 SHA `0B56CCA1F557E44CC6409822C64D29919D6955ABC164C59105138594022A9C7C`（typing 导入修正后、评测前定稿）。
- **ID 48-case：+0.773821**（vs R1 +0.752772，+0.021），48 正/0 负，双 split 正
  （test +0.772 / val +0.775）；**default attention_mean 0.767021**（vs R1 0.752173，+0.0148）。
- 时间：A_calib 89.873s（含 24 层训练+gate 评估），**预测 228.936s < 280s** ✓。
- **OOD 门未过**：candidate gap = +0.773821 − (+0.757615) = **+0.016207**；
  相对直接父 R1（同 SHA in-dist/OOD 配对，gap +0.000915）的 Δgap = **+0.015292 > 0.01** → BLOCKED。
  旋转收益主要留在 WikiText 分布内，构成分布拟合特征（与 09-04 OOD 标定的 gap 家族
  Attention +0.015~0.022 区间一致）。
- **裁决：REJECTED / OOD_GATE**。不晋级、不替换 R1；不通过重训/缩容量/换折等邻域手段修复
  （按"失败换机制"纪律）。教训：标准编码器 surrogate 训练的旋转与 v189 栈组合后，
  分布鲁棒性下降；后续新机制需在校准目标中直接体现部署路径或使用分布更稳的参数化。
- **分支状态保持：最佳候选 = R1（`candidate_v2/solution.py`，default 0.752173 / shard48 +0.752772），
  全门通过，官方 unregistered/NA。**

## 官方回传与 WA 根因（2026-09-06 深夜）

- **R1 官方 `14009 / 211s`**：`C_A = 13008`，与 v189 加性 Attention 侧锚一致；时间预测
  211.8s vs 实测 211s。R1 成为分支官方锚（归档 manifest 已更新）。
- **A2 / R2 官方 attention wrong answer**（同 SHA 重提交仍 WA，确定性非法）。
- 根因（本地复现，工具 `workbench/v162_attention/fuzz_official_contract.py`）：
  1. **官方 harness 在 inference 上下文调用六 API**——旋转训练 `backward()` 在 inference
     张量/模式下抛 `RuntimeError`（`torch.enable_grad()` 无法逃逸 inference_mode）→ A2 未设防 → WA；
  2. R2 的 v189 校准调用在总 try 之外、动态注入路径无守卫 → WA；
  3. 次要雷：L_q<L_kv 时 `index_select` 越界 → CUDA device-side assert 毒化后续全部用例。
- 按 v107 判例（任一用例异常 = 整次失败；回退应产生负分而非异常）完成加固：
  - **A2b**（`CDB49A02...A879`）：训练气泡（inference-off/grad-on + normal 张量重建）+
    校准/动态全路径 `except Exception` 回退 v162 标准 + 越界修复；
  - **R2b**（`58B1214F...57A8`）：同气泡 + v189 调用入 try + 注入守卫 + 失败时精确回退 R1。
- 验证：模糊测试全 CLEAN（inference_mode/no_grad/变长/5 几何/5000 长序列/极端值/重复性）；
  screen 精度逐位一致（A2b +0.421328 = A2；R2b +0.776965 = R2）。
- 官方 WA 的备选假设（state 自定义键触发官方校验）未排除——若 A2b/R2b 重提交仍 WA，
  则下一轮把自定义 state 键全部移除后重验。
