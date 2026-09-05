# NVFP4 Codebook 精确转换计划（2026-09-05）

> 状态：**ACTIVE**（P0 PASS → **P1 CLOSED-W → P4 评估中**）。前两个研究计划
> （同坐标系误差诊断 + 官方贡献探针；定向 fc/proj·Q/K 解剖）均已闭环，本计划是其后续
> 机制方向。父版本：v186 = **17599 / 272s**（完整官方父，SHA `F8495DCA...7EB8`）；
> 时间预算父 v180 = 17597 / 242s。距榜首 21765 差 4166 分。
>
> **P0 完成（2026-09-05）**：
> G0 **PASS → P1**。
> - W fc_gate/fc_up/proj 聚合：best-sf 非零精确占比 = **0.789**，表法 MSE 比 = **0.476**
>   （远过阈值 0.20 / 0.85）。
> - 全 role exact 均值：fc_gate 0.7746 / fc_up 0.7903 / proj 0.8017 / q 0.7376 / k 0.7046
>   / v 0.8090 / o 0.8384。
> - X（calibration 缓存抽样）：fc_gate 0.4668 / proj 0.5291（n=1/role，但已证明结构上限）。
> - F1 偶八分尾数比例 ~0.49、次正规 scale 占比 fc_gate 0%、proj 6.5%、attention ~0%；
>   exp_spread p50=1, p90=2；跨行对偶相等性低（pair_m4_eq_mean 0.13）——说明 16 粒度
>   perm 信号弱，与计划 §2 推断一致。
> - 实测报告：`artifacts/proxy_v3/cb0-codebook-proof-20260905/run-001/cb0_report.json`，
>   56 秒，40 W + 2 X 样本；脚本 `workbench/cb0_codebook_proof.py`。
>
> **P1 完成（2026-09-05）：G1 CLOSE_W → P4 评估**。
> - 混合 W pipeline（per-(row, 64-block) sf 选择 + mant RTN）在 dense 输入上跑通，
>   但 MSE 是 v186 baseline 的 58–60 倍（fc_gate 60.6 / proj 57.7），精确占比 = 0。
> - **机制证伪**：P0 精确占比 0.789 建立在 NVFP4 (quant, scale) 严格码本结构上；
>   P1 hybrid 在 dense 输入（含 BF16 snap）下，mant = RTN(dense/denom) 不再保证
>   落位 HiF4 格点。**新机制没有提供超越 `_dense_to_hif4` 联合搜索 sf+mant 的解空间**。
> - W 侧机制族**正式关闭**，不作为可提交候选。
> - 实测报告：`artifacts/proxy_v3/cb1-exact-encoder-20260905/run-001/cb1_report.json`，
>   14 秒 quick；脚本 `workbench/cb1_exact_encoder.py`；日志
>   `logs/execution/2026-09-05-cb1-exact-encoder.md`。
>
> **P4 评估中（待启动）**：attention 侧运行时精确转换。开启条件已部分满足
> （P0-F2 W 侧 Q/K 精确占比 0.74/0.70 > 0.20）；但 P1 证伪了"dense → HiF4 严格
> 精确"路径——P4 应改为对 **Q/K NVFP4 (quant, scale) 输入端**做精确路径分支。
> 启动前需复用 cb0 脚本扩抽样 attention X 的 NVFP4 数据复测精确占比。

## 1. 机制来源与证据链（全部可复查）

本计划不是任何已关闭家族的重开，机制由以下三方代码/文档交叉推导：

1. **`evaluator/nvfp4_sim.py`**：输入数据是 NVFP4 对（carrier, scale），16 值一块沿最后一维，
   `scale = E4M3_round_up(amax/6)`，carrier ∈ E2M1 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`。
   权重 scale 形状为 **per-row 的 (out, in/16)**。
2. **`evaluator/reference_hif4.py::dequantize_hif4`**：裁判解码
   `v = sign × mant × lv3 × lv2 × sf`，**全程 FP32 精确算术、无中间舍入**——落在格点上的值
   解码逐位无损。
3. **`evaluator/official_eval.py` L2543-2564**：计分参考
   `reference = NVFP4反量化X @ NVFP4反量化Wᵀ`（FP32），标准与玩家同源同起跑。
   玩家把 c×s 精确落在 HiF4 格点上时，与参考的残差只剩 NVFP4 反量化的 BF16 snap
   （≤2⁻⁹ 相对），约为基线格式误差的 0.3%——可视为零。

**Codebook 精确引理**：相对块基 scale `sf`，HiF4 可表示值集为
`S = {0.25k:k=1..7} ∪ {0.5k:k=1..7} ∪ {k:k=1..7}`（15 个非零幅值 + 0）。
7 个非零 E2M1 码 **全部** ∈ S（0.5, 1, 1.5, 2, 3, 4, 6 均为格点）。
因此子块 b（16 值、单一 E4M3 scale s_b）**精确可转换 ⟺ ∀出现码 c: c·s_b/sf ∈ S**。

**比值结构**（表驱动，脚本内用精确有理数算术重建）：

- 尾数匹配（m4(s_b) = m6(sf)）时，每码有指数窗 J(c)：
  0.5:[−1,3]、1:[−2,2]、1.5:[−1,2]、2:[−3,1]、3:[−2,1]、4:[−4,0]、6:[−3,0]；
  全码子块通常只需 j ∈ {−1, 0}（即 s_b/sf ∈ {0.5, 1}）。
- **奇八分尾数 E4M3 scale（1.125/1.375/1.625/1.875）对任何码都不可能精确**
  （9/8、11/8、13/8、15/8 不在任何合法比值集中）；偶八分尾数失配也只在
  受限码集下合法（如 5/4 只对码集 ⊆{0.5,1,1.5,2}@j=1 或 ⊆{1,2,3,4}@j=0）。
- **E4M3 次正规 scale（k·2⁻⁹, k=1..7）全部精确落在 E6M2 格点上**（k=3→1.5·2⁻⁸
  等）——小量级权重子块天然尾数兼容（nvfp4_sim 注释：小权重大多数块为次正规）。
- **Per-row 约束（本计划与早期构想的分野）**：s 依赖 (row, 子块)，per-column 的
  smooth d 无法跨行补偿尾数失配；精确性只能按 (row, 64-块) 机会性取得——每块选
  一个 sf（E6M2 格），匹配到的子块精确、其余最优 snap。**精确占比是数据量，
  由 P0 实测，结构上限 = P(子块尾数与所选 sf 兼容)。**

## 2. 与 v186 变换栈的冲突面（整合约束，非否决项）

- W 路径（v186）：smooth(per-col d) + permutation + block_smooth(4,8) + rank-1/2
  residual + GPTQ。精确块要求：d 退化为 per-column 2 的幂（或恒等）、permutation
  退化为 16 列粒度（或恒等）、block_smooth/rank 不作用于精确块。
  **GPTQ 与精确块天然可组合**：精确块 δ=0，不向后续块传播补偿扰动；连续精确块
  级联保持原始 c×s 结构。
- 动态 X 路径：state smooth_inv + permutation + per-call GPTQ，同上约束。
- Attention 路径（Q/K）：center(K)/rotation/pair_transform 破坏码本结构 → 仅 P4
  条件开启（按 64-块分支：对齐块走精确路径、其余走现有变换）。

## 3. 预注册门槛（P0–P4，全部先算后测，不回填）

### P0 数据事实（`workbench/cb0_codebook_proof.py`，只读，cached proxy-v2 数据）

- **F0 结构核验**：quant 值域 ⊆ E2M1；scale 可精确分解为 (e, m4∈8格)；
  形状 (rows, in/16)。
- **F1 对齐结构**（W，全 6 shard × 7 role；X/QKV 抽样）：每 64-块 4 个子 scale 的
  尾数分布——P(全同)、众数计数直方图 {1..4}、指数跨度、跨行尾数众数占比
  （16 列粒度置换分组的潜在收益）、偶八分（E6M2 可表）占比。
- **F2 精确占比上限**：全 E6M2 sf 候选格（每子块指数窗 [−4,2]×4 尾数，K≤112）
  表驱动搜索：best-sf 精确值占比（含/不含零值）、MSE-best sf 下的精确占比。
- **F3 码集统计**：每子块非零码数分布、零值占比、逐码频率。
- **F4 编码器级 MSE 对比**（同未变换矩阵、同为 free-a snap，只比 sf 选择）：
  baseline sf = `_standard_e6m2_scale`(amax/7)（含 BF16 中间量，调 solution 原函数）
  vs 候选格 best-sf；另用真实 `_dense_to_hif4`（refine 关）锚定 free-a 乐观差。
- **门槛 G0**（W 侧 fc_gate/fc_up/proj 聚合）：
  - best-sf 非零精确占比 ≥ **20%** 且表法 MSE 比 ≤ **0.85** → 进 P1；
  - 占比 ∈ [10%, 20%) 或 MSE 比 ∈ (0.85, 0.95] → 不做 W 侧候选，转 P4 前置评估
    （用 P0 的 X/QKV 统计决定 attention 分支去留）；
  - 否则记录数据结论并**关闭计划**（无提交）。

### P1 单元证明（`workbench/cb1_exact_encoder.py`）

- 实现真实精确优先编码器（sf 候选 × 4-组 a 兼容 DP × 非精确值 RTN/AdaRound），
  与 GPTQ 组合；smootth/perm 退化为 2 幂/恒等的混合臂。
- 在 ≥3 层 × 3 角色（fc_gate/fc_up/proj）上做**乘积级**对比：缓存 v186 校准态
  （v186 臂）vs 混合臂（自建 state），固定同一组测试 X 窗口。
- **门槛 G1**：混合臂乘积 MSE ≤ 0.90× v186 臂且在 ≥2/3 单元上不劣 → P2；
  (0.90, 1.0] → 计划修订（2 幂 smooth 变体 / 部分保留 perm 的 16 粒度版本），
  修订只许一次；> 1.0 → 关闭 W 侧，转 P4 评估。
- **护栏**：若 P0-F2 的 attention 侧精确占比 ≥ 20% 且 G1 失败，P4 评估仍然开启。

### P2 成对本地评测（eval-v3 default 面板 + OOD）

- v189 候选 vs v186，Linear 侧改动、Attention 侧逐位一致。
- **硬门槛 G2**（2026-09-04 规则）：Δmean > 0 且 L1 < 0.02；OOD
  |Δ(gain_in − gain_ood)| ≤ 0.01。任一不过 → 不提交，归因记录。
- 分数不换算官方分；本地只做符号/风险判读。

### P3 v189 构建与官方提交

- 单文件从 v186 构造；官方时间模型预测 **T < 280s** 才提交
  （W_calib 系数 0.115，是六 API 中最便宜侧；新增搜索在校准期，不进动态路径）。
- 提交后按 step_gain 裁决：≥0 → RETAINED；负 → 回滚根 solution.py 至 v186，
  负值幅度 ≥ 100 视为机制反证、< 100 视为噪声带内（v188 教训），家族裁决记录。

### P4（条件开启）attention 侧运行时精确转换

- 开启条件：P0-F2 attention 侧（Q/K）精确占比 ≥ 20% **且**（G1 通过 或 G0 落入
  attention 转移分支）。
- 内容：Q/K 动态编码器按 64-块分支——对齐块（子 scale 尾数与所选 sf 兼容）走
  码本精确路径（sf = 匹配子块 scale×2^j），其余块走现有 rotation/pair_transform
  路径；每块分支判定为 O(1) 比较，官方动态成本按时间模型 0.734/−1.58 系数预算。
- 独立 G2'/G3' 门槛同上；任何 per-call 动态细化族超时历史（v128-131/v161/v165）
  不重开——本机制无 per-call 迭代求解，只有查表分支。

## 4. 关闭族合规表

| 已关闭家族 | 本计划是否触碰 | 说明 |
|---|---|---|
| L-R2 rank-3/系数/fold | 否 | 残差低秩族不动 |
| scale 窗口 ≠ +4 | 否 | 在线 scale 窗口不动 |
| v187 Jacobian 移植 | 否 | 静态 importance 族不动 |
| v183 refine 覆盖率扩展 | 部分相邻 | 本计划的 sf 候选搜索与 v183 的
  block-smooth refine 不同（v183 是 attention 侧平滑候选全覆盖、官方 0 增益）；
  W 侧 sf 选择差异由 P0-F4/P1 直接实测裁决，不凭 v183 外推 |
| v185 balance/gamma/refine | 否 | clean-room 家族不动 |
| A4/L4/C1、V 侧全部 | 否 | 不注册 V 候选 |
| per-call 动态细化（超时族） | 否 | 无 per-call 迭代求解 |
| ROAB/CAT/BOAT/stored-scale 等 | 否 | — |

机制类别：**解析/结构等价变换**（历史 6/6 官方非负：v158A/v168A/v166L/v180A/
v182L/v183A 中解析类全部非负）——格点精确落位是代数事实，不是统计拟合。

## 5. 风险登记

- **R1 精确占比数据依赖**（主风险）：尾数对齐是随机事件，占比可能 <10%；
  G0 以实测关闭，成本一次脚本运行。
- **R2 变换栈让渡**：2 幂 smooth / 16 粒度 perm 会损失现有平滑/置换价值；
  P1 混合臂直接量化净效应。
- **R3 v183 类比**：更优编码搜索可能官方无感；缓解：W 侧误差池是 X 侧 1.9 倍、
  官方时间系数最低，且 P2/P3 门槛在结论之前。
- **R4 时间预算**：sf 搜索增加 W_calib 时长；时间模型预测 <280s 硬门禁。
- **R5 符号门禁失手带**（v188 教训）：本机制目标是大结构削减（非近零信号），
  预期本地 Δmean 显著为正；若 P2 出现近零信号（|Δmean| < 0.002），按预注册规则
  不提交、先做 P1 复核。

## 6. 生命周期

- 本计划是当前唯一活动计划；状态更新写回本文档（P0 结果 → G0 裁决 → …）。
- 裁决完毕（RETAINED/REJECTED/关闭）后：结果记入
  `docs/current-solution-status.md` 与 `solutions/README.md`，本文档移入 archive，
  AGENTS.md 指针更新。
- 运行日志归 `logs/execution/2026-09-05-cb0-codebook-proof.md` 等，不混入 docs/。
