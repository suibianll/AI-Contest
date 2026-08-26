# HiF4 Progressive Candidate Ledger

本表是 2026-08-27 本地优化链的顺序索引。每个候选只改变一个主机制；完整配置、分项、时间与源码 SHA 以对应 archive 的 `result.md` 为准。官方评测不可用，因此 Official 始终保留 `NA`，未来只追加、不覆盖本地证据。

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
| C10 / v013 | C5 | activation quadratic 覆盖 3072-wide FFN | proj +0.54pp; Linear +0.090pp | 6/6 改善 | Champion | [v013](../../../solutions/20260827_v013_c10-wide-activation-quadratic-local_scoreNA_timeNA/result.md) | `a2e0ed3` |

## 当前状态

- 当前根 `solution.py`：C10 / v013，SHA256 `DD8587257299626718A24EB89013447DA9105E8884F391104A6B350607399E44`。
- 相对 C1 的累计主线：Attention 保留 `+7.12pp` causal 增益；Linear 从 C1 `0.5668` 提升到 C10 `0.5811`，约 `+1.43pp`。
- 已关闭路线：segment CVaR、继续扩大 8×8/16×16 coverage、增加 16×16 sweep、继续扩大 weight quadratic group size。
- 当前已预注册候选：C11 wide activation 8×8 residual；不得在开发结果出来前改变 coverage、sweep 或 gate。
