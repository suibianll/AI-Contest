# Linear 编译校准稳健窗口极值块序 — REJECTED

- 父版本：v189（官方 `17616/275s`）；根保持 v189。
- 候选 SHA256：`9BBDADDBEB7FB1648758A127D5A14A742F37B9FF7B1AD50345D71E4139BD850C`。
- 官方状态：`unregistered/NA`，未提交。

候选从根 v189 独立编译每个校准窗口的输出压力，并按窗口最大值形成固定
64-block activation-GPTQ 顺序；codec、Attention 和在线算子未改动。

R0 六 API 导入与编译检查通过，R1 order 可达且 control 未改动，但 Linear 两 shard
结果不满足方向门禁：

- shard0：mean `-0.000043038`、median `-0.000117686`、L1 `0.001434351`、
  worst-20% `-0.001098366`、`26+/28-/2=`；
- shard1：mean `+0.000191295`、median `0`、L1 `0.001268531`、
  worst-20% `-0.000263512`、`26+/26-/4=`。

shard0 的负 mean/median 触发 R1 关闭条件，未运行 R2 或 fresh default，不提交官方。
R1 原始证据见 `proxy-r1/`，执行记录见
[`2026-09-06-linear-compiled-robust-window-order-plan.md`](../../logs/execution/2026-09-06-linear-compiled-robust-window-order-plan.md)。
