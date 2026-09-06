# 64-block 层级分区与激活误差解剖执行记录

日期：2026-09-06。父版本：v186。根 `solution.py` 未修改。

## 固定输入

- 父源码 SHA256：`f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`
- 输入 cache SHA256：`ef278e9dbbf72670d54a502648b055833e081d9ea8097b2ba5b9269eb6f2824b`
- 配置：`workbench/hierarchy_partition_activation_probe_config.json`
- 配置 SHA256：`02af466392649122f19955c9997831a289c44569e9527b9ce3fbde6fede002b6`
- GPU：NVIDIA GeForce RTX 3060 Ti，PyTorch `2.6.0+cu124`，CUDA `12.4`

## D-A 激活解剖

工具 `workbench/hierarchy_partition_activation_probe.py`，结果：
[`d-a/result.json`](../../artifacts/proxy_v3/hierarchy-partition-20260906/d-a/result.json)，
manifest：
[`d-a/run_manifest.json`](../../artifacts/proxy_v3/hierarchy-partition-20260906/d-a/run_manifest.json)。

固定层 `[0,8,15,23]`、七个 Linear role、五个 calibration window 共 140 条记录；使用
完整 v186 动态 API，统计每窗均匀 32 token。clip 元素 fraction `0.0453592052`，clip
block fraction `0.9264153080`，clip SSE / 最近合法网格 SSE `0.0888823185`，父最终坐标
MSE `0.00241995794`。结论：长尾/clip 有可见但非主导的误差占比，未触发新机制卡。

首次运行发现并修复 unit broadcast 与后续 D-B 采样回写问题；修复后的 manifest 与结果为
最终证据，失败运行没有写入结果文件。

## D-B 层级子组分区 oracle

固定 pressure 规则是 activation RMS 与 weight RMS 各自按 median 归一化后取
`max(log2(x/median), log2(w/median))` 降序。每个 Linear state 取 32 输出行、4 个
64-channel block，五个 calibration window 各取 32 token；两臂都调用
`evaluator/reference_hif4.py` 的合法 FP64 exact block solver。Attention 使用 head 0 与最后
KV head，记录 Q-only/K-only/joint GQA output、logits、probability。

结果：[`d-b/result.json`](../../artifacts/proxy_v3/hierarchy-partition-20260906/d-b/result.json)，
manifest：[`d-b/run_manifest.json`](../../artifacts/proxy_v3/hierarchy-partition-20260906/d-b/run_manifest.json)。

| 项目 | 结果 |
|---|---:|
| Linear focus L0/L8/L15/L23 layer-median output gain | `−0.017062 / −0.075558 / −0.014854 / −0.042745` |
| focus layers > 1% | `0/4` |
| Attention Q-only median gain | `+0.000005` |
| Attention K-only median gain | `−0.001716` |
| Attention joint median gain | `−0.001534` |
| 总耗时 | `78.804s` |

固定门要求至少 3/4 层重点 role 中位 output gain >1%，且同时覆盖深层和非深层；实际为
`0/4`，因此状态为 **CLOSED / D_B_REJECTED**、gate `NO_MATERIAL_GROUPING_ORACLE`。
共享压力分区不进入候选；不分配版本号、不运行 eval-v3 candidate、不提交官方。

## 后续

D-A/D-B 已关闭“clip 主导”和“固定 64-block 子组重分组”两条诊断假设，但没有证明合法
HiF4 空间耗尽。下一张活动计划应转向尚未直接测量的“与网格共存的联合输出优化”，并保持
v186 为根父；本地 oracle 仍不能换算官方分数。
