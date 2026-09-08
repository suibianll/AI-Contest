# L-C3 objective-only（READY_FOR_OFFICIAL）

单一完整方案计划 R0/R1。目标公式、代码映射、验证证据见本文件与 manifest。

## 目标公式（R0 审计确认的入口）

`_linear_output_candidate_metrics`（及 `_combos`）当前 per-case：

```text
score_f = ‖Y_f − Ŷ_f‖² / (‖Y_f‖² + ε)          （fold 均匀平均）
```

L-C3 只替换分母（LC0 官方 +3 验证的归一化）：

```text
score_f = ‖Y_f − Ŷ_f‖² / (numel_f · MSE_STD_f + ε)
MSE_STD_f = mean((std_act_f @ std_wᵀ − x_f @ w_origᵀ)²)
```

标准编解码用根自带 `_branch_encode_standard_hif4`——与 LC0 `_ref_encode_standard_hif4`
数值等价（随机 3 形状 max diff = 0.0，机制一致性门通过）。

## 代码映射（diff 59 行，全部 L-C3）

| 位置 | 改动 |
|---|---|
| `_linear_output_candidate_metrics` | 加 `mse_std_folds=None` 参数；分母按 flag 切换（None 时保留旧公式） |
| `_linear_output_candidate_metrics_combos` | 同上 |
| `_linear_smooth_hybrid_metrics` | 加参数并透传给 e2e |
| 主校准 fold 循环后 | `_LC3_MSE_STD_NORMALIZE` 时预计算 `mse_std_folds` |
| 7 个调用点 | 传入 `mse_std_folds` |

**禁止项未添加**：rank、残差子空间、邻域搜索、新正则、新候选循环、activation/Attention 改动。

## 验证

| 项 | 结果 |
|---|---|
| 合成：uniform MSE_STD | 候选排名与父等价（rank preserved） |
| 合成：非 uniform | 仅 fold 分母按 `numel×MSE_STD` 改变（hand-match 0.29649000） |
| 合约：六 API | 全部存在、签名正确 |
| 合约：合法/control | 五字段合法、Attention 空 state 合法 |
| shard0 paired（根 12352EFD） | candidate 0.50098 vs root 0.51735（Δ-0.0164，非灾难性）；api 190.5s |
| 机制一致性 | diff 59 行全 L-C3；flag 关闭时与根逐位一致 |

## 官方提交

- 计分 SHA：`c231328D1F8EE5B2DB02869EF071838F862EFE893F0DBAE9CF59A6293D440CC8`
- 包路径：`solutions/continuous_linear_lc3-objective-only/solution.py`
- 唯一代表（计划 §3.5）：合法、可达、非 no-op、实现忠实、复杂度有界。
- 官方 300s 唯一时间裁决；本地 Δ-0.0164 不预测官方（计划 §2/§3）。
- 结果分支：正 → 六 shard 归档 + interaction audit + 根切换；
  负/超时/WA → 只关闭"根上 MSE_STD+numel fold 归一化移植"这一实现。