# Linear 动态 32 行块能量块序执行计划

> 创建：2026-09-06
> 状态：**CLOSED / R3_REJECTED_TIME**
> 父版本：v189 `17616/275s`，根源码 SHA256
> `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

## 1. 唯一假设

上一块级动态候选的排序收益来自当前 activation 的块能量×父重要性，但全量 token 扫描
将时间预测推到 `286.022476s`。本候选固定用确定性 `_sample_rows(..., 32)` 估计当前
块能量，仍按同一 64-channel 块统计排序；不改变权重、编码器、GPTQ 补偿或 Attention。

## 2. 固定门禁

1. R0：单文件导入、`py_compile`、六 API、有限输出和动态顺序 reachability。
2. R1：v189 baseline 的 eval-v3 Linear shard 0/1；若 no-op、系统性回退或非法状态，
   立即关闭。
3. R2：R1 通过后跑 Linear 六 shard 与 OOD，检查 median、L1、尾部、control 和 gap。
4. R3：fresh default proxy-v2 时间审计。只有 Overall 严格超过已测本地最高
   `0.688994940507429` 且预测 `<280s`，才归档并提交；否则记 `REJECTED_TIME`/`REJECTED`，
   根保持 v189。

本计划只验证一个固定的 32 行估计，不扫描行数、权重混合、阈值、角色路由或排序邻域。

## 3. 执行记录

执行结果写入 `logs/execution/2026-09-06-linear-dynamic-block-energy32-plan.md`。

## 4. 裁决

R0、R1、R2 均通过：六 shard Linear delta mean 为 `+0.002531235`，OOD delta mean
为 `+0.003227747`，gap change 为 `-0.000696513`，输出有限且动态块序可达。fresh
default 的 Linear/Attention/Overall 为
`0.643280557360/0.752173407020/0.688652578052`；低于已测本地最高
`0.688994940507`，且时间模型预测 `285.526750s`，未通过 `<280s` 提交门。候选标记为
**CLOSED / R3_REJECTED_TIME**，归档于
`solutions/20260906_linear-dynamic-block-energy32_time-rejected/`，未提交官方，根版本保持
v189。
