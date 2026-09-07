# L-R1 执行报告：真实输入/部署闭环

> 日期：2026-09-07。侧：Linear。run_id：`repair-r1`。
> 契约：[evidence-repair-next-cycle.md](../../docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md) L-R1。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。全部数字来自真实 API 路径。

## 1. 目标与技术口径

- 输入：`evaluator.prepare_shard`（`nvfp4_encode`/`NVFP4_MODE=amax6`，
  `NVFP4_INPUT_CODEC=e4m3-subnormal-ceil-v1`）与官方 evaluator 完全一致的
  carrier/scale。**不用 round(x/scale) 自制输入**。
- 权重/激活：调用父实际 `hif4_calibration_and_quantize_weight` 与真实
  `hif4_dynamic_quantize_activation`（含 static-actorder hdiag 路径），解码
  返回五字段；不简化编码器。
- 面板：层 0、11；role o/fc_up/proj；每个 state 两个 in-dist compact holdout
  （与纯正验 shard 一致：validation/test），**训练窗口不入 holdout**。

## 2. 结果

### 2.1 输入/部署闭环（12/12 逐位一致）

`probe_real_api_closure.py`：6 state × 2 holdout = 12 case，全部
`mse_player` 与 evaluator `candidate-linear-shard*.json` 逐位一致
（rel_diff = 0.00e+00）。

| state | w | split | mse_player | expected | diff |
|---|---|---|---|---|---|
| L0-o | 2 | validation | 2.0812e-07 | 2.0812e-07 | 0 |
| L0-o | 7 | test | 2.1533e-07 | 2.1533e-07 | 0 |
| L0-fc_up | 2 | validation | 9.4426e-04 | 9.4426e-04 | 0 |
| L0-fc_up | 7 | test | 9.0966e-04 | 9.0966e-04 | 0 |
| L0-proj | 1 | test | 1.1787e-04 | 1.1787e-04 | 0 |
| L0-proj | 6 | validation | 1.2251e-04 | 1.2251e-04 | 0 |
| L11-o | 2 | validation | 1.0467e-04 | 1.0467e-04 | 0 |
| L11-o | 7 | test | 1.5737e-04 | 1.5737e-04 | 0 |
| L11-fc_up | 2 | validation | 2.2528e-03 | 2.2528e-03 | 0 |
| L11-fc_up | 7 | test | 2.2104e-03 | 2.2104e-03 | 0 |
| L11-proj | 1 | test | 1.8319e-04 | 1.8319e-04 | 0 |
| L11-proj | 6 | validation | 1.8835e-04 | 1.8835e-04 | 0 |

### 2.2 坐标语义/rank 审计（`probe_rank_semantics.py`）

| state | 不含 rank 连续乘积 rel err | 含 rank 连续参照 rel err | final_mse |
|---|---|---|---|
| L0-o | 4.4e-14 | 4.5e-05 | 2.08e-07 |
| L0-fc_up | 3.3e-13 | 1.7e-04 | 9.44e-04 |
| L0-proj | 1.3e-13 | 8.5e-06 | 1.18e-04 |
| L11-o | 1.0e-13 | 1.9e-04 | 1.05e-04 |
| L11-fc_up | 2.8e-13 | 1.2e-04 | 2.25e-03 |
| L11-proj | 2.3e-13 | 1.9e-05 | 1.83e-04 |

结论：
- **smooth/perm/block-hadamard 是可逆等价变换**：不含 rank 的连续乘积
  `X_hat @ W_smooth^T` 与 `X @ W^T` 相对误差 1e-13 级。
- **rank-2 残差补偿不是纯可逆变换**：含 rank 的连续参照 `X_rank @ W^T`
  相对 reference 偏差 1e-4 级（fc_up 层 1.2-1.7e-04，o 层 1.6-1.9e-04，
  proj 1.9e-05）。这正是 L-R1 要求"若存在非等价补偿，单列 reference 偏差"
  的情形；**不在该坐标下强行构造"连续零误差"**。

### 2.3 四臂归因：UNIDENTIFIABLE（`probe_real_api_four_arm.py`）

从真实 API 返回值用官方 `_linear_error_source_details` 构造 E00/E10/E01/E11：

| state | E00 | E10(W-only) | E01(A-only) | E11(both) | inter_gain |
|---|---|---|---|---|---|
| L0-o | 9.1e-07 | 3.7e-04 | 2.5e-04 | 2.1e-07 | +679 |
| L0-fc_up | 1.9e-03 | 2.4e-01 | 1.8e-01 | 9.4e-04 | +220 |
| L0-proj | 2.9e-04 | 6.4e-02 | 5.2e-02 | 1.2e-04 | +399 |
| L11-o | 2.6e-04 | 7.3e-02 | 1.6e-02 | 1.0e-04 | +345 |
| L11-fc_up | 4.1e-03 | 6.1e-01 | 4.4e-01 | 2.3e-03 | +256 |
| L11-proj | 4.7e-04 | 7.2e-02 | 5.6e-02 | 1.8e-04 | +269 |

由于候选 W 在变换坐标（smooth/perm/hadamard + rank gram）下编码，而
"W-only"臂把候选 W 与 standard A 相乘是**混合坐标**（E10 比 E00 大 3–300×），
"A-only"同理；interaction 巨大（+220~+679）说明两侧编码必须配对才能抵消坐标
变换，单独一侧不构成同坐标纯量化分量。

**判定：W-only/A-only/interaction 百分比不可报告（UNIDENTIFIABLE）；完整
E11/both 是唯一有效读数。** 上一轮 `probe_fcproj_decomposition.py` 用简化
编码器（无 deploy 坐标、无 rank gram）报告的"A-only 61–79%"是伪精确贡献，
**撤销**，不迁移为任何方向结论。

## 3. 错误对照（evidence-repair R0 表）

| 旧结论 | 修复后状态 |
|---|---|
| L1 34/35 退化关闭族 | PROBE_INVALID_FOR_DEPLOYMENT（旧探针输入/部署不一致）→ L-R2 重做 |
| 激活误差占 61–79% | ATTRIBUTION_UNVERIFIED → UNIDENTIFIABLE（四臂混合坐标） |
| GPTAQ-JDRQ 同目标 | DEDUP_UNRESOLVED → L-R3 公式级对照 |
| 合法编码全负/全空间覆盖 | SCOPE_CORRECTION（只保留原固定求解器关闭边界） |
| 官方 +742s | NONDEFAULT_EXTRAPOLATION_INVALID → L-R2 只做本机成本诊断 |

## 4. 产物

- `workbench/continuous_linear/repair-r1/probe_real_api_closure.py`
- `workbench/continuous_linear/repair-r1/probe_real_api_four_arm.py`
- `workbench/continuous_linear/repair-r1/probe_rank_semantics.py`
- `artifacts/proxy_v3/continuous/linear/repair-r1/rank-semantics.json`

验收达成：同输入、同父源码、同 API 路径的最终输出与 evaluator 一致（12/12
rel_diff=0）；坐标语义 single 列 reference 偏差；四臂归因改为 UNIDENTIFIABLE。
下一步：L-R2 修正 L1 的梯度/输入/硬前向口径与成本诊断（不在本闭环上直接
宣布任何机制失败）。