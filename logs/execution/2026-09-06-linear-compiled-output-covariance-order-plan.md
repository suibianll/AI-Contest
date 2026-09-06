# Linear 编译校准输出协方差块序执行记录

日期：2026-09-06  
父版本：v189（官方 `17616/275s`）  
候选 SHA256：`52DE271400805249EA0566F963EAE96DF61CE65AF6FEA5DB4D76A83DD0782F02`

## R0

从根 v189 复制单文件候选，使用已有最终坐标校准协方差 `gram_full` 与部署量化权重
输出 Gram `weight_output_gram`，为每个合法 64-channel block 编译固定
`trace(C_A[block]·H_W[block])` 顺序。`py_compile`、六 API 导入及合法状态检查通过。

首轮 R1 发现完整协方差在超宽输入不可用时会错误落入自然序动态路径；该运行记为无效，
保留于候选归档的 `proxy-r1-invalid/`，不作机制结论。随后增加显式 v189 hdiag 回退，
候选仍只有一个固定规则。

## R1/R2/OOD

| 阶段 | 结果 |
|---|---|
| R1 修复后 shard0 | mean `+0.000178692`，median `0`，L1 `0.001187468`，`27+/21-/8=` |
| R1 修复后 shard1 | mean `+0.000277935`，median `+0.000140994`，L1 `0.001061508`，`30+/18-/8=` |
| R2 六 shard | mean `+0.000569404`，L1 `0.001252877`，`193+/93-/50=` |
| OOD | mean `+0.000685097`，L1 `0.001862107`，gap change `-0.000115693` |

Attention control 全程保持父版本；输出协方差顺序在窄状态可达，宽状态使用 hdiag 回退。

## R3 裁决

Fresh default：Linear `0.640810865681`、Attention `0.752173407020`、Overall
`0.687211924573`；时间模型预测 `279.215656s`，低于 `280s` 门槛。尽管时间通过，
Overall 比当前本地最高 `0.688994940507429` 低约 `0.001783016`，故状态为
`REJECTED_SCORE`。未提交官方，根保持 v189；候选与全部原始证据归档于
`solutions/20260906_linear-compiled-output-covariance-order_score-rejected/`。
