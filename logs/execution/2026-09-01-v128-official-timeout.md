# v128 官方超时记录

日期：2026-09-01
候选：`20260901_v128_fixed-attn-budget_scoreNA_timeNA`
来源：用户回传官方评测结果。

## 官方裁决

| 候选 | 官方分数 | 官方时间 | 裁决 |
|---|---:|---:|---|
| v128 fixed-attn-budget | 未返回 | `>300s` | **timeout（官方，用户确认）** |

官方评分没有返回可登记分数，因此 Official score 保持 `NA`；官方时间只登记为
超出 300 秒，不用本地代理数字替代。

## 本地对应记录

同版本 `official-shape-v1` 本地结果为 Linear `0.4656551226`、Attention
`0.8377892255`、API `310.7324530s`、wall `332.5571037s`。这组数值用于解释为何
该版本被选作时间优化父版本，但不能推导官方分数或官方机器上的精确秒数。

## 归档处理

源码 SHA256 为 `0D4A0E91F6D076A9B694390DAE3D63A931D3D759AB609252AB3B54366F22F638`。
历史目录保持 `scoreNA_timeNA`，不因官方回传而重命名；结果通过本记录和
`solutions/20260901_v128_fixed-attn-budget_scoreNA_timeNA/result.md` 追加。
