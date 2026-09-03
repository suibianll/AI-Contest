# 21765-C Linear cross-fold minimax — REJECTED

日期：2026-09-03

## 结论

C1 的 `1.20×` 仅为工程风险目标，不是硬否决线；单 state `1.584×` 因而只记录为时间高风险。
候选继续完成了预注册 C2，随后因 holdout 大范围回归正式 `REJECTED`：Linear compact mean
delta `-0.088774702`、median `-0.088582739`、`4+/52-/0=`，worst delta
`-0.216586346`，worst-quartile mean `-0.164813211`。七个 role mean 全负，test/validation
分别为 `-0.093781334/-0.083768070`，W-only delta `-0.057872378`，不满足任何主要晋级门。

## 固定实现

- parent：v160，SHA256
  `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
- research source：`workbench/score21765_linear_crossfold_minimax.py`
- source SHA256：`3E469337BFCDDA9D53BCC288FF4A57CCDC59C107C48E3E8ECB7ABB5F7256AEBE`
- fold：两个 calibration window 各自确定性采样至 128 行，按采样后行号 `mod 5`，相同余数
  跨 window 合并
- 固定 scale/lv2/lv3/Activation/坐标，只允许 signed-mantissa parent、相邻低、相邻高 code
- 五折绝对 normalized loss 按 `(max, median, mean)` 字典序；固定 64-coordinate 单 sweep；
  row-block 完整复核后严格改善才接受
- online Activation 与 Attention 路径不变

## C0/C1

单个真实 `fc_gate [4864,896]` state：

| 项目 | 结果 |
|---|---:|
| fold rows | `[52,52,52,50,50]` |
| attempted row-blocks | 68096 |
| accepted row-blocks | 65460 |
| changed codes | 547226 |
| rejected blocks with residual changes | 0 |
| Activation state equal to parent | true |
| scale/lv2/lv3 equal to parent | true |
| parent calibration | 2.160s |
| candidate calibration | 3.421s |
| ratio | 1.584×（高风险，非硬否决） |

校准内 Both ratio `0.005619→0.003011`，但 W-only `0.001864→0.002432`。这预示候选在记忆
同一 calibration 的 `Q(A)` 残差；该现象只作风险诊断，正式裁决仍由独立 compact holdout 给出。

Artifacts：

- `artifacts/official_eval/score21765-c01-linear-audit-smoke.json`
- `artifacts/official_eval/score21765-c01-linear-audit-blockbatched-smoke.json`

## C2 compact

Parent 使用已有不可变同 panel JSON
`artifacts/official_eval/v159-l1-batch-compact-parent.json`；v159/v160 Linear 路径相同。候选结果：

- `artifacts/official_eval/score21765-c2-linear-compact.json`
- `logs/official_eval/score21765-c2-linear-compact.md`

| gate | result | verdict |
|---|---:|---|
| mean delta | -0.088774702 | FAIL |
| median delta | -0.088582739 | FAIL |
| positive / negative / zero | 4 / 52 / 0 | FAIL |
| minimum delta | -0.216586346 | FAIL |
| worst-quartile mean | -0.164813211 | FAIL |
| role mean nonnegative | 0 / 7 | FAIL |
| test mean delta | -0.093781334 | FAIL |
| validation mean delta | -0.083768070 | FAIL |
| W-only mean delta | -0.057872378 | FAIL |
| interaction mean delta | -0.030902324 | FAIL |

Cross-holdout 的 28 个 layer/role pair 虽然全部同号，但分布为 `2` 对双正、`26` 对双负，说明
一致性来自系统性回归而非泛化收益。最坏 role 是 `fc_up -0.155896`，其后为
`proj -0.128753`、`fc_gate -0.126498`；其余四个 role 也全部为负。

同机时间：Weight calibration `96.699s` 对 parent `35.550s`（`2.720×`）；六 API total
`115.755s` 对 `53.701s`（`2.156×`）。时间只作风险信息，不是本次拒绝依据。

## 决策

- 状态：`REJECTED`
- 停止点：C2
- 不运行 C3 default、C4 跨模型、C5 集成/官方
- 不改变 fold 聚合、Jacobi/Gauss-Seidel、coverage、邻域或 role 路由
- 官方：`unregistered/NA`
- 根 `solution.py`：未修改
- 不分配版本号
