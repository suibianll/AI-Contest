# Linear 编译校准稳健窗口极值块序执行记录

日期：2026-09-06  
父版本：v189（官方 `17616/275s`）  
候选 SHA256：`9BBDADDBEB7FB1648758A127D5A14A742F37B9FF7B1AD50345D71E4139BD850C`

## R0

从根 v189 复制单文件候选，新增一个固定规则：以最终部署坐标中的
`(A_window² * activation_importance)` block sum 为窗口分数，再取所有校准窗口的
最大值排序。六 API 导入、`py_compile`、合法状态和有限完整 permutation 检查通过；
Attention control 未改动，order 在 R1 可达。

## R1

| shard | mean | median | L1 | worst-20% | +/-/0 |
|---|---:|---:|---:|---:|---:|
| 0 | `-0.000043038` | `-0.000117686` | `0.001434351` | `-0.001098366` | `26/28/2` |
| 1 | `+0.000191295` | `0` | `0.001268531` | `-0.000263512` | `26/26/4` |

shard0 的 mean 与 median 均非正，触发固定 R1 方向门禁；按计划不运行 R2、不扫
max/mean/median 或其他聚合邻域。候选完整归档于
`solutions/20260906_linear-compiled-robust-window-order_rejected/`，未提交官方，
根保持 v189。
