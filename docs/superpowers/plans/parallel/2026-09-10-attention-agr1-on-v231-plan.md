# Attention A-GR1 落到 v231 完整根（A-GR1-on-v231）

> 状态：开发结束，归档 **v236**，官方 `unregistered/NA`（用户统一评测）。
> 从属于当前活动总计划。当前完整根为 v231 Linear (L-EM3 K=2) + v195 Attention，
> 官方 `18518/291s`，SHA256 `EA79A1C12DC66714…`。本文件只负责这一张卡；
> 版本登记、组合与根切换由总协调线处理。
> 前序：A-GR1（v234，侧隔离官方 `14455/263.7s`，相对 v195 侧 **+29** 已确认）、
> A-GR2（真实目标训练，`NO_EFFECT`，`12d7c7d`）。

## 1. 定位：这是"未兑现项"，不是新机制

A-GR1 的官方价值已经测得，但**从未落到完整根上**：v234 的父是 v230，而根此后晋级为
v231（官方 18518/291s）。Attention 侧因此没有任何"已官方定价但未兑现"的正向机制
——这一项是唯一的一个。

本卡唯一动作：**把 v234 的 A-GR1 块原样重新挂到 v231 根上**，取得一次完整包官方读数。
机制、窗口、步数、lr、gate、正则、谱钳位全部不动。

### 与已关闭实现的差异（AGENTS §3 要求）

- 与 A-GR1（v234）：**零差异**，只是父从 v230 换成 v231。
- 与 A-GR2：A-GR2 换训练目标，`NO_EFFECT`；本卡**不碰训练目标**。
- 与 A-FIX1（v230 Attention）：A-FIX1 重训 rotation，官方 TIMEOUT；本卡**冻结根全部已有
  state**，rotation/center 只被复合编译，不被重训。
- 与 v192：v192 是 A-GR1 的对称零迹特例（官方侧 +22，完整包 TIMEOUT）。

## 2. 机制定义与合法性

与 A-GR1 完全一致（见 `attention-general-reciprocal-plan.md` §2）：per KV group
`M = I + N`（N 无约束），Q@M、K@M⁻ᵀ，校准期精确求逆一次并编译进既有
`learned_rotation`/`learned_center`（center 同步编译 `c@M⁻ᵀ`）；动态路径只有一次 matmul、
无求逆；量化前 logits 精确不变，收益只来自量化输入分布的移动。五字段格式不变。

## 3. 固定配置（无扫描）

1. 候选 = v231 归档逐位副本 + A-GR1 块（前缀逐字节 `cmp` 验证，纯追加）。
2. 训练 fit windows 0,1,2；gate windows 3,4；32 步 Adam（lr 0.01、β 0.9/0.999、clip 1.0、
   正则 1e-3）；逐 64 块 amax 尺度比损失；STE 反向，`grad_N = G_q − P·G_k^T·P`（P=M⁻ᵀ）；
   每步把 M 奇异值钳到 `[1/√2, √2]`；逐层全部 gate 窗口严格改善才 arm。
3. 不缩步、不缩窗、不调 lr、不改 gate 粒度。

## 4. Control

1. 前缀逐字节相同（纯追加）；与 v234 归档候选的 `diff` 等于 v230→v231 根的 `diff`；
2. 六 API 脱离仓库独立导入；`validate_state`/`validate_hif4_params` 通过；
3. `agr1_attempted=1`、训练可达（损失下降）、逆误差 fp 级；
4. **Linear 与 v231 父逐位相同**（本卡唯一改动在 Attention 段）。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-agr1-on-v231/`。
日志：`logs/execution/2026-09-10-attention-agr1-on-v231.md`。
输出：`artifacts/proxy_v3/attention-agr1-on-v231-<run>/`。

GPU 串行（<2000 MiB 才启动）。先 shard0 排接口错误，再六 shard；**必须显式传
`--stop-after-nonpositive 6`**——评测器默认值是 2，而本候选 shard0/shard1 恰好精确零，
默认参数会在 shard2 前停住并**正好藏掉唯二两个 arm 的层 15/22**。

## 6. 完成条件

- 72 case 与父逐位相同或六层全 parent：`NO_EFFECT`，不占版本号，只写日志；
- 否则归档 v236（官方 `unregistered/NA`，用户统一评测）；
- 官方 TIMEOUT 或负向：只关闭该实现，不以缩步/缩窗/改 lr 重试；
- 完成后本文件归档。

## 7. 执行结果

- 候选 `3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07` / 520955 B，
  自 v231 根 `ea79a1c1…` / 505762 B **纯追加**（+15193 B，前缀 `cmp` 逐字节相同）。
- **与 v234 同构（机器测量）**：`diff(本候选, v234归档候选)` 与 `diff(v231根, v230根)`
  是同一个 hunk——只有 Linear 的 `_EM1_PASSES = 1 → 2` 及注释；该常量只在 Linear 路径被引用。
- Control `OVERALL: PASS`：六 API 独立导入；state 合法；`agr1_attempted=1`、loss 2.0→1.2170、
  逆误差 2.98e-07；Linear 与 v231 父逐位相同。控件输出同时打印 `passes=2`，机械确认建在 v231 上。
- 六 shard（`attention-agr1-on-v231-sixshard-20260910`，`--stop-after-nonpositive 6`）：
  等权 **`+0.003845`（21/3/48）**；层15 `+0.017449`（10/2/0）、层22 `+0.005618`（11/1/0）接受，
  层 0/1/8/5 gate parent 逐位不变。**与 v234 六 shard 逐项完全相同**（含 L1/tail/计数与
  72 case 绝对分布），即同构推论被独立证实，本卡不产出新的机制证据。
- 归档 v236：`solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/`。
- **官方 `unregistered/NA`**，交用户统一评测。**不写官方秒数预测**：侧隔离 +20.7s，
  完整根余量仅 9s，而侧时间对完整包无预测力（v194/v229 双向反例）。晋级规则事前固定：
  官方分数高于 18518 且 <300s 才晋级，超时或更低则不晋级。
