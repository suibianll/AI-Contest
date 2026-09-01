# v129 官方超时记录

日期：2026-09-01
候选：`20260901_v129_fixed-attn-budget-sweep1_scoreNA_timeNA`
来源：用户回传官方评测结果。

## 官方裁决

| 候选 | 官方分数 | 官方时间 | 裁决 |
|---|---:|---:|---|
| v129 fixed-attn-budget-sweep1 | 未返回 | `>300s` | **timeout（官方，用户确认）** |

官方评分没有返回可登记分数，因此 Official score 保持 `NA`；官方时间只登记为
超出 300 秒，不使用本地代理数字替代。

## 本地对应记录

同版本 `official-shape-v1` 本地结果为 Linear `0.4656551226`、Attention
`0.8365785543`、API `248.3630966s`、wall `270.6064164s`，本地 API 与 wall 均低于
300 秒。官方仍超时，说明本地六 API 代理与官方端到端时间之间不能作绝对映射。

## 归档处理

源码 SHA256 为 `7319F00E5259FE15E7C5ECA99E214A8F7482CF5CF066D6E3025E86C92D9095EC`。
历史目录保持 `scoreNA_timeNA`，不因官方回传而重命名；结果通过本记录和
`solutions/20260901_v129_fixed-attn-budget-sweep1_scoreNA_timeNA/result.md` 追加。
