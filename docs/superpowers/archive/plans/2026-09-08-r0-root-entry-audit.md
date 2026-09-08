# R0：零 API 根入口审计（L-C3 objective-only 可执行性）

依据 [单一完整方案优化计划 §7 R0](2026-09-08-single-solution-optimization-plan.md)。零模型 API，纯代码审计。

## 审计结论：**L-C3 EXECUTABLE**（同构入口存在）

根 `solution.py`（SHA `D66128A6…`，官方 17636/264s）存在与 L-C1 原卡（fold 权重
`1/‖Y_f‖²` → `1/(F·numel·MSE_STD_f)`）**同构的 fold 加权输出目标选择入口**。

## 同构入口：`_linear_output_candidate_metrics`（def 约 L5056）

**当前公式**（per-case score，L5165-5167）：

```text
score_f = ‖ref_f − recon_f‖²_F / (‖ref_f‖²_F + ε)
aggregate = (1/F) Σ_f score_f          （fold 均匀平均）
```

其中 `ref_f = 变换后稠密激活 × 变换后稠密权重`、`recon_f = 量化激活 × 量化权重`
（per fold sample）。**这正是 L-C1 原卡描述的 `ω_f = 1/‖Y_f‖²` 结构**：
per-case 参考能量归一化 + fold 等权。此前 L-C1 实现审计遗漏此入口
（只查了主 fold 循环与 sample-energy 块序），导致误加 rank-8 求解器。

**L-C3 objective-only 替换**（LC0 官方 +3 验证的归一化）：

```text
score_f = ‖ref_f − recon_f‖²_F / (numel(ref_f) × MSE_STD_f + ε)
aggregate 不变（fold 均匀平均）
```

MSE_STD_f 逐 fold 一次预计算（LC0 同口径）：

```text
std_act_f = decode(standard_encode(x_f));  std_w = decode(standard_encode(w_orig))
MSE_STD_f = mean((std_act_f @ std_wᵀ − x_f @ w_origᵀ)²),  clamp ≥ 1e-12
```

## 调用图（全部在 `_COMPILED_SAMPLE_BASE_CALIBRATION` 流程内）

| 调用点 | 用途 | 当前生效 |
|---|---|---|
| `_linear_smooth_hybrid_metrics`（~L5205） | smooth_d/perm 候选择 e2e 分量（`(1−0.02)·proxy + 0.02·e2e`） | **是**（`_LINEAR_E2E_WEIGHT=0.02`） |
| metric-mode 分派（~L7864） | `_LINEAR_SMOOTH_END_TO_END is True` 时直接用 e2e | **否**（当前 `"hybrid"`） |
| 窄层联合选择（~L8051） | 双窄维 joint 变换直接按 e2e 分选择 | **是**（`< _WIDE_LAYER_MIN_DIM`） |
| block_smooth 判定（~L8127/8135） | selected vs identity 的 e2e 比较 | **是** |

## 标准编解码可用性

- 根已含完整标准 HiF4 编码器 `_branch_encode_standard_hif4`（文件尾部 ~L11130 系）
  与 `_branch_standard_params`/`_branch_solve_standard_hierarchy`。
- LC0 用 `_ref_encode_standard_hif4`（correctness hardening 时加入）。
- R1 移植前先零 API 对比两者是否逐位等价；若不等价，**逐字复制 LC0 的
  `_ref_encode_standard_hif4`** 以保持与官方 +3 机制完全一致。

## 最小 diff 边界（L-C3 objective-only）

1. 在主校准 fold 循环后（`weight`、fold activations 可用处）**一次**计算
   `mse_std_folds`（每 weight state 一次，非每候选）。
2. `_linear_output_candidate_metrics` 增加可选参数 `norm_denoms`
   （= numel_f × MSE_STD_f per fold）；分母从 `reference.square().sum()+ε`
   改为 `norm_denoms[i]+ε`。**分子、聚合、候选集、搜索空间全部不变**。
3. 4 个调用点传入预计算 denoms（含未生效分支，保持一致）。
4. **禁止**：rank、残差子空间、邻域搜索、新正则、新候选循环、activation 路径、
   Attention、动态 API 改动。

## 预期 changed/attempted

- 选择决策变化仅在 `MSE_STD_f/‖Y_f‖²·numel` 逐 fold 比例失衡时发生
  （fold0=10 行 vs fold1=128 行，能量与标准误差不同 → 预期有真实重排）。
- 记录：受影响选择点数（smooth/joint/block_smooth 决策翻转数）、
  最终五字段变化数、activation_state 逐位对照。

## 与旧实现审计的关系

- L-C1 实际运行（rank-8 求解器叠加）只关闭"该 rank-8 后处理"，不否定本目标归一化。
- LC2 实际运行（整块同步 ±1）只关闭那两个整块提案，不证明逐元素/目标级饱和。
- 本审计找到原卡真正假定的接口；R1 只替换该接口的 fold 归一化。