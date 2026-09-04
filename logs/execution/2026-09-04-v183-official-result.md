# v183 官方回传记录（REJECTED）

日期：2026-09-04

来源：用户回传官方评测结果。

候选：`20260904_v183_attn-bsm-full-refine_rejected`

源码 SHA256：`D94F37CC7B5370B1C2BC070157166D060936371C4E65E354DAD3746090771F24`

## 官方结果

| 候选 | 唯一变化 | 官方分数 | 官方时间 | 状态 |
| --- | --- | ---: | ---: | --- |
| v183 | Attention block-smooth final-quantizer refine 覆盖 0.50→1.00 | `17598` | `279.7s` | `REJECTED` |

相对完整官方父 v182（`17598/273s`），`step_gain=0`、时间 `+6.7s`，且
`279.7s<300s`。因此这不是 timeout，而是覆盖率扩展未产生官方分数收益。按照预注册规则
`S(v183)≤17598`，v183 拒绝，coverage ratio/blocks 邻域关闭；完整官方父保持 v182，
根 `solution.py` 不切换。

本地 Qwen default 的 `+0.000511` 未迁移为官方整数分，GPT-2 为轻微负、OPT 为
no-effect。该结果进一步说明本地 `10^-4` 级 mixed 改善不足以支持覆盖率型局部扩展。

v183 是目标设置后的第 4 个官方候选，提交账本更新为 `4/10`，剩余 6。
