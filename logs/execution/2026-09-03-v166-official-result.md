# v166 官方回传记录（RETAINED，新 Linear 父侧）

日期：2026-09-03
来源：用户回传官方评测结果。
候选：`20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA`
源码 SHA256：`9C0EAC6A7CA883A1F8962C11735744271259460F5EBBF23D530A5BBCF12B4646`

## 官方结果

| 候选 | Linear | Attention | 官方分数 | 官方时间 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| v166 | rank-1 残差重分布 | standard | `4590` | `226s` | `RETAINED` |

相对 v163（v160 Linear + standard Attention）的 `4587 / 202s`：step_gain `+3`、时间
`+24s`，`226s < 300s` 通过。按侧向计划 §3.1 判读 `S_L = 4590 > 4587` 为官方正向，
**成为新 Linear 父侧 `P_L = v166`**。

## 归因

- `C_L = S_L − 1001 = 3589`（相对 v162 双标准零点的 Linear 总贡献，v163 为 3586）；
- `G_L = S_L − 4587 = +3`（rel v163 step_gain）；`P_L = 3/3586 ≈ 0.0008`；
- 收益与本地 default 归因方向一致：本地 168 case 的 paired 收益全部来自
  proj role（`+0.025142`，宽输入层无 GPTQ/Hessian 保护），90/168 case 本地微回归
  与 median 微负未在官方体现；
- 时间上官方 `226s` 相对 v163 `202s` 增加 `24s`，与本地增量主要落在 calibration
  的预期一致（动态路径仅一次 O(TD) 外积）。

## 后续裁决

- 父侧更新：`P_L：v163 → v166`，当前活动计划 §16 与状态文档已同步；后续 Linear
  工作包（L1–L4）从 v166 构造，同时保留 v160 固定口径比例
  `(S_new − 4587)/3586`。
- Attention 父侧 `P_A = v164（13945/204s）` 不变；下一个动作仍为 A1 解析 logits
  增益校正。
- 根 `solution.py` 保持不动：v166 是侧向隔离单侧版本（standard Attention），
  不是新的完整方案父版本。

## 证据

- 本地 Linear default：`artifacts/official_eval/sidecal-v166-linear-default.json`
- 本地 Attention control：`artifacts/official_eval/sidecal-v166-attn-default.json`
- 候选结果：`solutions/20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA/result.md`
- 判读规则：`docs/superpowers/archive/plans/2026-09-03-v162-official-side-isolation-optimization-plan-superseded.md` §3.1