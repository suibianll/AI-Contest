# Linear 冻结激活状态的输出感知静态权重编译

- 状态：**REJECTED**（J1 eval-v3 负向）
- 日期：2026-09-06
- 父版本：v189（local Linear `0.640258324430`）
- 候选源码 SHA-256：`aca047987fdffcf1c126a6ee37bca327ba0149e4ba5c65c47ae7b5a6900a9631`
- 根 `solution.py`：v186，未修改

## 结果

J0 单文件/六 API/有限输出 smoke 通过，并确认 JDRQ 对宽下投影可达。J1 使用当前
eval-v3、固定 proxy-v2 cache、CUDA、v189 baseline，在前两个 Linear shard 完成
112 个配对 case 后按计划提前停止：

| 指标 | 候选 | v189 baseline | 候选−baseline |
|---|---:|---:|---:|
| Linear mean | 0.6343498783403304 | 0.634575771204753 | -0.0002258928644226 |

- shard 0：delta mean `-0.0003604818362111497`，median `0`，L1
  `0.0003604818362111497`，正/负/零 `0/8/48`；
- shard 1：delta mean `-0.00009130389263410246`，median `0`，L1
  `0.0002452943856971499`，正/负/零 `3/5/48`；
- candidate API total：`133.8395642 s`（weight calibration `99.9741984 s`、dynamic
  activation `33.8653658 s`）。

结论：固定 activation state 后接入 JDRQ 在当前实际输出目标上没有正收益，关闭该
接入及其邻域；不分配 v190、不提交官方。官方状态为 `unregistered/NA`。

详细逐 case 证据：
[`JDRQ eval-v3 J1`](../../artifacts/proxy_v3/linear-fixed-state-output-aware-jdrq-20260906/j1/candidate/manifest.md)
