# HiF4 Progressive Candidate Ledger

本表是 2026-08-27 本地优化链的顺序索引。每个候选只改变一个主机制；完整配置、分项、时间与源码 SHA 以对应 archive 的 `result.md` 为准。B0 官方结果已闭环为 `15313 / 137s`；C10 / v013（提交 `a2e0ed3`）官方结果为 `15799 / 144s`，为当前官方最优；其余候选尚无官方结果，继续保留 `NA`，未来只追加、不覆盖本地证据。

| Candidate | Parent | 唯一变化 | offset 0 主效应 | 固定回归 | 结论 | Archive | Git commit |
|---|---|---|---:|---|---|---|---|
| C1 / v003 | B0 | output-aware Attention selector | causal Attention +7.12pp | 通过 | Champion | [v003](../../../solutions/20260827_v003_a1-real-attention-local_scoreNA_timeNA/result.md) | `cf15c7c` |
| C2 / v004 | C1 | independent segment CVaR | causal -3.42pp | 未运行 | rejected | [v004](../../../solutions/20260827_v004_c2-segment-cvar-local_scoreNA_timeNA/result.md) | `cee81c1` |
| C2a / v005 | C1 | query-aligned segment CVaR | causal -0.53pp | 未运行 | rejected | [v005](../../../solutions/20260827_v005_c2a-query-segment-cvar-local_scoreNA_timeNA/result.md) | `ecf6f3c` |
| C3 / v006 | C1 | top 5% 8×8 weight quadratic | Linear +1.10pp | 通过 | Champion | [v006](../../../solutions/20260827_v006_c3-topk-8x8-quadratic-local_scoreNA_timeNA/result.md) | `89d6865` |
| C4 / v007 | C3 | 8×8 coverage 5%→10% | Linear +0.092pp | 未运行 | accepted, not promoted | [v007](../../../solutions/20260827_v007_c4-8x8-coverage10-local_scoreNA_timeNA/result.md) | `469d419` |
| C5 / v008 | C3 | top 2% 16×16 weight quadratic | Linear +0.23pp | 6/6 改善 | Champion | [v008](../../../solutions/20260827_v008_c5-topk-16x16-quadratic-local_scoreNA_timeNA/result.md) | `a61bff1` |
| C6 / v009 | C5 | 16×16 coverage 2%→4% | Linear +0.063pp | 未运行 | accepted, not promoted | [v009](../../../solutions/20260827_v009_c6-16x16-coverage4-local_scoreNA_timeNA/result.md) | `66c5643` |
| C7 / v010 | C5 | top 1% 32×32 weight quadratic | Linear +0.123pp | 未运行 | accepted, not promoted | [v010](../../../solutions/20260827_v010_c7-topk-32x32-quadratic-local_scoreNA_timeNA/result.md) | `ce612c2` |
| C8 / v011 | C5 | top 0.5% 64×64 weight quadratic | Linear +0.090pp | 未运行 | accepted, not promoted | [v011](../../../solutions/20260827_v011_c8-topk-64x64-quadratic-local_scoreNA_timeNA/result.md) | `040e9e6` |
| C9 / v012 | C5 | 16×16 sweep 1→2 | Linear +0.025pp | 未运行 | accepted, not promoted | [v012](../../../solutions/20260827_v012_c9-16x16-second-sweep-local_scoreNA_timeNA/result.md) | `375f778` |
| C10 / v013 | C5 | activation quadratic 覆盖 3072-wide FFN | proj +0.54pp; Linear +0.090pp | 6/6 改善 | Champion; 官方 15799/144s（+486 vs B0） | [v013](../../../solutions/20260827_v013_c10-wide-activation-quadratic_score15799_time144s/result.md) | `a2e0ed3` |
| C11 / v014 | C10 | wide activation 8×8 residual | proj +0.31pp; Linear +0.052pp | 6/6 改善 | Champion | [v014](../../../solutions/20260827_v014_c11-wide-activation-8x8-local_scoreNA_timeNA/result.md) | `6cee2f0` |
| C12 / v015 | C11 | wide activation 16×16 residual | proj +0.07pp; Linear +0.012pp | 未运行 | accepted, not promoted | [v015](../../../solutions/20260827_v015_c12-wide-activation-16x16-local_scoreNA_timeNA/result.md) | `37dde76` |
| C13 / v016 | C11 | all-width activation 8×8 | Linear +0.463pp | aggregate 6/6; amax4 o -0.91pp | accepted, not promoted | [v016](../../../solutions/20260827_v016_c13-all-width-activation-8x8-local_scoreNA_timeNA/result.md) | `148029b` |
| C14 / v017 | C11 | calibration-gated all-width activation 8×8 | Linear +0.450pp | 6/6、全分项安全 | Champion | [v017](../../../solutions/20260827_v017_c14-gated-all-width-activation-8x8-local_scoreNA_timeNA/result.md) | `9e10d33` |
| C15 / v018 | C14 | quantized-weight activation Gram | Linear ~0.000pp | 未运行 | accepted, not promoted | [v018](../../../solutions/20260827_v018_c15-quantized-weight-activation-gram-local_scoreNA_timeNA/result.md) | `e6ce6c0` |
| C16 / v019 | C14 | gated activation 8×8 coverage 4% | Linear +0.148pp | 未运行 | accepted, not promoted | [v019](../../../solutions/20260827_v019_c16-gated-activation-8x8-coverage4-local_scoreNA_timeNA/result.md) | `c6b4edd` |
| C17 / v020 | C14 | gated activation 8×8 coverage 8% | Linear +0.285pp | 6/6、36/36 分项改善 | Champion | [v020](../../../solutions/20260827_v020_c17-final-gated-activation-8x8-coverage8-local_scoreNA_timeNA/result.md) | `ff3b624` |
| C18 / v021 | C17 | activation/weight-error cross term | Linear +0.077pp | 未运行 | accepted, not promoted | [v021](../../../solutions/20260827_v021_c18-activation-cross-term-local_scoreNA_timeNA/result.md) | `f367c5d` |
| C19 / v022 | C17 | cross-aware Newton gain selection | Linear +0.152pp | 未运行 | accepted, not promoted | [v022](../../../solutions/20260827_v022_c19-cross-aware-gain-selection-local_scoreNA_timeNA/result.md) | `37acc42` |
| C20 / v023 | C17 | exact discrete cross-gain selection | Linear +0.413pp | 5/6 positive; pow2 proj -5.87pp | accepted, not promoted | [v023](../../../solutions/20260827_v023_c20-exact-discrete-cross-gain-local_scoreNA_timeNA/result.md) | 本次提交 |

## 当前状态

- 官方最优：v013（C10，提交 `a2e0ed3`）`15799 / 144s`，较 B0 `15313 / 137s` 提升 `+486`。
- 当前根 `solution.py`（HEAD `ff3b624`）：C17 / v020，SHA256 `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`；本地 Linear mean `0.5890`，高于 v013 的 `0.5811`，尚未官方评测。
- 工作区提示：`solution.py` 与 `tests/test_release_candidate.py` 存在未提交修改（超出 C17 约 +298 行，疑似 C21 预注册候选的开发中版本），尚未归档、尚未评测。
- 相对 C1 的累计主线：Attention 保留 `+7.12pp` causal 增益；Linear 从 C1 `0.5668` 提升到 C17 `0.5890`，约 `+2.22pp`。
- 已关闭路线：segment CVaR、继续扩大 8×8/16×16 coverage、增加 16×16 sweep、继续扩大 weight quadratic group size。
- 当前已预注册候选：C21 all-width gated exact cross refinement；coverage 路线保持关闭。
