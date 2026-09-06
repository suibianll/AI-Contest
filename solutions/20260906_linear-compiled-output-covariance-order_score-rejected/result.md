# Linear 编译校准输出协方差块序 — REJECTED_SCORE

## 来源与裁决

- 父版本：根 v189，官方 `17616/275s`。
- 候选源码：本目录 `solution.py`。
- 候选 SHA256：`52DE271400805249EA0566F963EAE96DF61CE65AF6FEA5DB4D76A83DD0782F02`。
- 官方状态：`unregistered/NA`，未提交；根 `solution.py` 保持 v189。
- 当前本地最高：`Overall=0.688994940507429`。

候选只把最终变换坐标中的校准协方差块与部署量化权重输出 Gram 的块内
`trace(C_A[block]·H_W[block])` 编译为固定 64-block 顺序；完整协方差不可用时
显式回退到 v189 的 hdiag 顺序。codec、合法状态、Attention 和在线算子均未改动。

## 证据

- 初始 R1 暴露宽输入缺少回退，shard0/shard1 mean delta 为
  `-0.000364723/-0.000123270`；该无效运行保留在 `proxy-r1-invalid/`，未作为机制证据。
- 修复回退后 R1 shard0/shard1 mean delta 为
  `+0.000178692/+0.000277935`，L1 分别为 `0.001187468/0.001061508`，
  Attention control 全零，输出协方差顺序仅在窄状态可达。
- R2 六 shard aggregate mean delta `+0.000569404`，L1 `0.001252877`，
  `193+/93-/50=`；OOD mean delta `+0.000685097`，相对 ID/OOD gap 变化约
  `-0.000115693`，在 `0.01` 门内。

## Fresh default

Linear/Attention/Overall 为
`0.640810865681/0.752173407020/0.687211924573`；官方时间模型预测
`279.215656s`，通过时间门。但 Overall 比当前本地最高低约 `0.001783016`，
因此按计划裁决为 `REJECTED_SCORE`，不提交官方。

完整执行记录见 [`2026-09-06-linear-compiled-output-covariance-order-plan.md`](../../logs/execution/2026-09-06-linear-compiled-output-covariance-order-plan.md)。
