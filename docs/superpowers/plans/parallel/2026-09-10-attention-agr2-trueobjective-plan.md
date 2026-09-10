# Attention 一般互逆残差的真实目标训练计划（A-GR2）

> 状态：CLOSED / NO_EFFECT（六 shard 72/72 case 与父逐位相同，六层 gate 全 parent），2026-09-10。不占版本号、不归档；实现关闭，不以缩步/缩窗/调 lr/改归一化重试。
> 从属于当前活动总计划。当前完整根为 v231 Linear (L-EM3 K=2) + v195 Attention，
> 官方 `18518/291s`，SHA256 `EA79A1C12DC66714…`。本文件只负责 Attention A-GR2；
> 版本登记、组合与根切换由总协调线处理。
> 前序：A-GR1（v234，侧隔离官方 `14455/263.7s`，相对 v195 侧 **+29** 已确认）。

## 1. 定位

瓶颈审计 §1：代理目标与部署目标错位是核心问题。A-GR1 已证明"一般互逆矩阵 M"这个
自由度有官方价值；A-GR2 测试**同一自由度上"训练目标对齐"的增量**：训练前向从
逐 64 块 amax 尺度比代理损失换成 gate 同款的完整部署路径真实输出 MSE（STE 反向，
复用 A-FIX1 已验证的工程件，但训练对象只有 M——A-FIX1 的"重训 rotation"已关闭，
本卡不触碰）。除训练目标外全部与 A-GR1 相同，不做窗口/步数/lr/gate 任何邻域。

## 2. 机制定义与合法性

与 A-GR1 完全一致：M = I + N（N 无约束，per KV group），Q@M、K@M⁻ᵀ（校准期精确
求逆一次，编译进既有 `learned_rotation`/`learned_center`，center 同步编译为
c@M⁻ᵀ），动态路径只有一次 matmul、无求逆；冻结根全部 state；量化前 logits 精确
不变，收益只来自量化输入分布的移动。

## 3. 固定算法（相对 A-GR1 的单变量）

1. 候选 = v231 归档逐位副本 + A-GR2 段（前缀逐字节 `cmp` 验证）。
2. 训练：fit windows 0,1,2；每步每窗口用 player state（learned_rotation=当前
   R_q@M / R_k@M⁻ᵀ，learned_center=c@M⁻ᵀ）走完整 `hif4_dynamic_quantize_q/k` 部署
   编码 + 部署 V（与 N 无关，每窗只算一次），真实输出对 dense reference 的 MSE；
   STE 反向经 `_m_attention_backward` 得 d_qhat/d_khat，线性链
   `G_m = q_coord^T·d_qhat`、`G_p = k_coord^T·d_khat`，合成
   `grad_N = G_m − P·G_p^T·P`（P=M⁻ᵀ；公式级 float64 有限差分 worst relerr
   2.9e-7 < 1e-6）。
   **归一化说明**：训练损失按根 A2/A-FIX1 惯例除以每窗标准 HiF4 MSE（逐窗正常数，
   不改变逐窗最优解；裸 MSE 的梯度量级 ~1e-15 会被 Adam 固定 eps=1e-8 与 fp32
   在 1.0 附近的分辨率（6e-8）双重吞掉，N 恒为 0——control 实测证实）。gate 仍用
   A-GR1 原样的裸 MSE。
3. 32 步 Adam（lr 0.01、clip 1.0、reg 1e-3、β 0.9/0.999），每步 M 奇异值钳
   [1/√2, √2]；gate windows 3,4 全部严格改善才 arm，否则保持父。
4. 不以缩步/缩窗/调 lr 适配成本。

## 4. Control（全部通过并留证）

1. 0 训练步：Q/K/V 五字段与最终输出与父（v231 归档）逐位一致；
2. 互逆性 fp 精度；量化前 dense logits 不变 <1e-6；
3. 六 API 脱离仓库独立导入；`validate_state` 通过；
4. V/Linear 与父逐位不变；训练可达（attempted=1，n_norm 非零，gate 双臂出现）；
5. 解析梯度 FD 验证 worst relerr < 1e-6。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-agr2-trueobjective/`。
日志：`logs/execution/2026-09-10-attention-agr2-trueobjective.md`。
输出：`artifacts/proxy_v3/attention-agr2-<run>/`。
评测的 `--baseline-solution` 指向 v231 归档（工作区根被并行 Linear 会话弄脏，不用）。
GPU 串行（<2000 MiB 才启动，后台 30s 轮询）；先 shard0，再六 shard
`--stop-after-nonpositive 6`；本地数值只作诊断。

## 6. 完成条件

- 72 case 与父逐位相同或六层全 parent：`NO_EFFECT`，不占版本号，只写日志；
- 否则归档 v235（官方 `unregistered/NA`，用户统一评测）；
- 官方 TIMEOUT 或负向：只关闭该实现，不以缩步/缩窗/调 lr 重试。

## 7. 执行结果

**NO_EFFECT**（2026-09-10）：72/72 case 与父逐位相同（六层 gate 全 parent），
不占版本号、不归档、不开探针条目，只留执行日志。

- 候选 SHA256 `0c84601d8a8d7b69db915d5c61b21d92833befd2827b12b276dbb9658c2b350b`
  （v231 归档纯追加；未归档，仅存 workbench）。
- Control 全过：0 步逐位恢复父；互逆性 fp64 2.95e-7；量化前 logits 不变 1.18e-8；
  六 API 独立导入；validate_state；V/Linear 逐位不变；FD worst relerr 2.89e-7。
- 实现过程修正两处：(a) raw MSE 梯度 ~1e-15 被 Adam eps=1e-8 与 fp32 在 1.0 附近
  分辨率（6e-8）双重吞掉（N 恒为 0）→ 按根 A2/A-FIX1 惯例改每窗标准 HiF4 MSE
  归一化（逐窗正常数，不改逐窗最优解；gate 仍为裸 MSE）；(b) final_loss 初版未在
  训练后重算（恒 0.0 报告 bug）→ 修正为训练后完整前向重算。
- 真实 4B 六层：attempted 6/6，n_norm 14.5–17.8（真实移动、奇异值钳位生效），
  但**六层 gate 全拒**（候选 gate MSE 层层劣于父）；fit 窗训练损失三层升三层降
  （层0 0.4555→0.4577、层1 0.4998→0.5224、层15 0.5798→0.5833 升；层5 0.3011→
  0.2748、层8 0.4494→0.3877、层22 0.6926→0.6917 降）——STE 穿过完整部署编码的
  梯度在 32 步内连 fit 窗真实 MSE 都未稳定下降。
- 结论：在 M=I+N 同一自由度上，"训练目标对齐真实部署 MSE"这一单变量改动否定了
  代理目标错位假设（A-GR1 的 amax 代理反而训出了 2 层可过 gate 的 M）；按 §6 只
  关闭该实现，不以缩步/缩窗/调 lr 重试。
- 执行日志：`logs/execution/2026-09-10-attention-agr2-trueobjective.md`。
