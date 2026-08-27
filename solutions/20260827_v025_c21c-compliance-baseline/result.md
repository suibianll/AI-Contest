# v025 — C21-C Compliance Baseline (Phase 0)

- Date: 2026-08-27
- Candidate ID: `C21-C`（Compliance；26000 计划 Phase 0）
- Parent: `C21` / v024，SHA256
  `40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`
- Unique mechanism: delete every Linear output-supervision path
  (`_linear_output_candidate_metrics`、`group_cross8`/`cross8` state、
  `_ACTIVATION_QUADRATIC8_CROSS_*` 开关、Linear 输出评分 gate），替换为
  operand-local 指标（`_linear_candidate_metrics`、activation-only 重构
  损失 gate）。不新增任何精度机制。
- Source SHA256: `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`
- Local status: `local-champion`（唯一合规 Champion，后续所有候选从 C21-C 派生）
- Official status: 未提交（本基线预期低于 16043，为合规必须接受的代价）

## Phase 0 验收（§4.6，全部通过）

1. `pytest -q`：39/39 通过（含 reference codec、error decomposition、
   compliance guard、holdout ledger 测试）。
2. 合规门禁（`evaluator/linear_compliance_guard.py`）：静态 + 运行时
   全过，`violations=[]`；activation_state 无 Weight residual / Linear
   output / cross operator。
3. reference standard 与候选实现解耦（`evaluator/reference_hif4.py`
   冻结 codec；C21 台账在同评测器下逐位复现，见下）。

## 开发结果（offset 0, amax6, CUDA, attn_mask=both）

| Component | C21 | C21-C | Delta |
|---:|---:|---:|---:|
| q | 0.6459 | 0.6008 | −4.51pp |
| k | 0.7125 | 0.5936 | −11.89pp |
| v | 0.6009 | 0.5940 | −0.69pp |
| o | 0.5567 | 0.5178 | −3.89pp |
| fc | 0.5066 | 0.4749 | −3.17pp |
| proj | 0.5354 | 0.4058 | −12.96pp |
| Linear mean | 0.5930 | 0.5311 | −6.19pp |

- Attention 精确不变：MHA offset 0 causal `0.4497` / non-causal `0.4942`，
  与 C21 逐位一致。
- CUDA algorithm-stage `24.03s` vs C21 `26.59s`（ratio `0.904`，删除
  路径变快，符合预期）。

## 固定回归矩阵（§10.2，C21 同日同评测器重跑，与台账逐位一致）

| Case | C21 Linear mean | C21-C Linear mean | Delta | Attention |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5930 | 0.5311 | −6.19pp | identical |
| amax6 offset 97 | 0.5747 | 0.5148 | −5.99pp | identical |
| amax6 offset 193 | 0.5928 | 0.5319 | −6.09pp | identical |
| amax6 offset 389 | 0.5912 | 0.5235 | −6.77pp | identical |
| amax4 offset 0 | 0.4973 | 0.4663 | −3.10pp | identical |
| pow2 offset 0 | 0.5575 | 0.5454 | −1.21pp | identical |

- 6/6 配置均低于父版本：这是移除违规输出监督的真实分数代价，如实记录，
  不为保分保留灰色路径。本表是后续所有候选 ROI 比较的强制基线。
- GQA kv_heads=6 offset 193：Attention causal `0.4169` / non-causal
  `0.4928`，与 C21 逐位一致（Linear 分项同 offset 193 MHA）。
- 各配置 Attention（causal/non-causal mean 与 min/max）与 C21 全部
  逐位一致。

## 合成安全矩阵（§10.3，576 case）

- C21-C 与 C21 逐 case 对比：576/576 一致，最大绝对差 `0`（容差 1e-6）。
- 汇总与 C21 相同：overall causal_mean `0.282448` / noncausal_mean
  `0.299711`；worst `heavy_tail_h4_kv4_d128_s32` pow2 seed 2：
  causal `−0.939570` / noncausal `−1.075382`（inherited，非新增）。
- 两方案 `RESULT ok`（state 合法性与动态五字段全过）。

## Holdout 台账（项目约束：预算 3 次）

```text
holdout_runs_used          1
holdout_runs_remaining     2
holdout_seed_hash          96dd4ed70a0597a0060fe696557d3a330af22e3d273e6676a501d7bfb4b589fc
```

### 最终 holdout 终验收（2026-08-27，官方提交前）

- 触发：Checkpoint B 暂停后，对 Champion 做官方提交前的最终验收
  （`--reason` 已记入 ledger；solution sha `83AB4864…` 与本归档一致）。
- 冻结配置：amax6 causal offset 0，12 层，4 test 窗口，CPU。
- 结果（仅聚合，台账 `evaluator/holdout_ledger.json`）：
  - `linear_mean 0.523558`
  - `attention_mean 0.441898`
  - `algorithm_stage_seconds 69.29s`（calibration 53.26s / dynamic 9.94s）
- 与开发集对照：Linear `0.5311 → 0.5236`（−0.75pp）、Attention causal
  `0.4497 → 0.4419`（−0.78pp）——未见文本上降幅 <1pp，无过拟合信号，
  开发集结论（含固定矩阵）可信。
- Phase 0 归档时未消耗 holdout（当时 0/3）；本次为第 1 次消耗。

## Decision

`accepted as the compliance Champion`. C21-C 付出 Linear mean
−1.2pp 至 −6.8pp 的真实代价（cross8/输出监督机制的虚假增益被移除），
换取：违规路径清零、标准分母冻结解耦、双层合规门禁、冻结 holdout
与 operand-local 归因工具链。Attention 与 576 合成 case 逐位不变，
时序更快。后续所有候选（C22 起）必须从 C21-C 派生并通过同一合规门禁。

Next direction: C22 Linear R64 Incoherence Transform（26000 计划 §5），
在新 operand-local 基线上重建增益。
