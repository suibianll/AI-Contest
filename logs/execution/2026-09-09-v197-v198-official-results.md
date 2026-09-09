# 2026-09-09 v197/v198 官方回传记录

用户 2026-09-09 回传两个候选的官方结果：

| 候选 | 机制 | 本地 shard0 | 官方 | 裁决 |
|---|---|---|---|---|
| v197 `linear-aw1-block-gain` | 64-block 标量增益 A@W 闭式拟合（官方口径归一化、充分统计量、硬门控） | Linear delta mean `−0.2077`（0/56/0，mse ratio 1.45），calibration +15.8s/28 层 | `17277/285s`（相对根 18053/289s 为 **−776/−4s**） | REJECTED |
| v198 `attn-gqa-reciprocal-diag` | GQA 组共享互逆对角（解析 RMS 初始化 + smooth-max 5 步 + 真实输出硬门控，冻结 V） | Attention delta mean `−0.001693`（6/6/0），calibration 无超时信号 | `TIMEOUT（>300s）` | REJECTED_TIME，根不变 |

## 结论

- v197：本地灾难性负向与官方 −776 方向一致，证实门控的二次型评估与部署路径不一致
  （疑似 block 对齐 bug：根的 GPTQ 块重排路径未在拟合/部署间对齐）。只关闭该实现；
  低维 A@W 拟合机制本身未证伪，但修复后重试必须先解决 11s 时间余量问题。
- v198：本地 calibration API 5.9s→5.0s（shard0，12 case）无任何超时信号，官方仍 TIMEOUT。
  连同 v190/v191/v192/v196，**五个新增 Attention 校准机制全部官方超时**——官方隐藏规模下
  任何新增校准计算都超支。在根官方时间明显下降之前，不再提交带新增校准计算的候选。
- 根保持 v195（`18053/289s`，SHA `839ADB1E...761D7F`）。
- 归档：`solutions/20260909_v197_linear-aw1-block-gain_scoreNA_timeNA/`、
  `solutions/20260909_v198_attn-gqa-reciprocal-diag_scoreNA_timeNA/`。
