# v121 官方运行超时记录

> 日期：2026-08-31  
> 官方反馈：用户确认 v121 显示运行超时。  
> 官方完成时间、分数和提交文件 SHA 未提供，保持 `NA`。

## 判定

v121 的固定 Qwen 本地六 API 时间为 `2180.450151s`，wall time `2212.661980s`，均远超
官方 `420s` 限制。因此官方 timeout 与本地 runtime-invalid 判定一致，不属于新的
Attention wrong-answer 证据。

v121 的本地精度数据仍可作为研究消融：Linear `0.5096135327`、Attention
`0.8420394885`、panel `295.8112808759`；但其状态必须记为
`precision-only / official-timeout`，不能作为提交 parent。

## 对后续路线的影响

1. v121/v124/v125 均不进入官方候选集合；后两者本地时间更长，也继承未获官方通过的
   clean Attention 路径。
2. 官方增强线已更新为 v74 `22750 / 239.387s`，v72/v66 保留为控制组。
3. v121 的 structured refresh 只能作为待移植的 Linear 研究机制；若未来移植，必须
   从 v74 源码开始、保持 v74 Attention 调用闭包不变，并在每个单变量步骤恢复
   `<420s` 后才允许提交。
