# 单一完整方案与测试方法清理（2026-09-08）

## 结论

- 当前最优已知可复现完整方案是 compiled sample-energy，用户官方回传 `17636/264s`，相对
  v189 `17616/275s` 为 `+20/-11s`。根 `solution.py` 已提升为该源码并与归档逐位一致，SHA256
  `D66128A62E7E068EDC50C91F4D8E212F586A6EDCAEE5BEA7D3564166E258B0F6`。
- 官方平台没有单独返回计分 SHA；仓库只声称“用户回传与该归档关联”，不把缺失字段伪造成已核验值。
- Linear/Attention 独立父、独立 `gain>=0.9` 目标和持续 probe 队列全部停止。后续只从完整根构建
  一个机制、一个固定配置的候选，并由完整官方总分和官方 300s 裁决。

## `fit_gain` 审计

`workbench/continuous_linear/l28-proj-vectorized/fit_gain_table.py` 的单 case 定义为：

`1 - MSE(Q(XR) @ Q(WR^-T)^T, X @ W^T) / MSE(STD(X) @ STD(W)^T, X @ W^T)`。

候选项使用真实动态激活 API 和最终五字段权重解码；先对同一 `(layer, role)` 的 folds 等权平均，
再对 state 等权平均。因而它作为“校准集最终 Linear 输出相对标准 HiF4 的误差改善”是正确的。

它不正确的用途是把结果当成官方成绩代理或 `>=0.9` 晋级线：L28 完整表为 `0.948587`，但官方
相对 L4 只增加 4 分且耗时从 247s 增至 286s。该排序关系已经证明本地拟合幅度不能映射官方收益。
文档统一改称 `calibration_fit_gain`，仅用于确认拟合、可达性和排查实现错误。

## 为什么此前持续没有进展

1. Linear 与 Attention 各自优化的是侧隔离分数，最终却要提交一个完整方案；两套父、两套时间和
   两套门禁无法可靠相加。
2. 本地指标被同时用于拟合、选卡和宣告达标，出现 L28 本地大幅改善、官方仅 `+4` 的错位。
3. probe、误差账本、holdout、OOD、六 shard 和时间预测重复回答相似问题，消耗运行时间却不产生
   更强的官方判据。
4. 两侧官方时间已接近 300s，继续叠加高成本侧机制缺少完整调用图的时间保证。

## 统一后的流程

1. 唯一父为根完整方案；每个候选只改一侧的一个机制和一个固定配置。
2. 官方前只做六 API contract smoke、合法 state、finite、reachable/control 和目标侧 shard0。
3. 合法、可达、非 no-op 且动态路径复杂度有界的代表候选直接交官方；本地正负不决定提交。
4. 官方正向后才补目标侧六 shard 和必要 interaction 归档；官方失败后只为一个会改变下一卡的
   明确问题运行 probe，不维护永久误差账本。

## 验证

- 根与归档 SHA256 相同：`D66128A6...B0F6`。
- 该归档的六个公开 API、Linear/Attention state reference validation 和随机形状 smoke 已通过；
  根与归档逐位一致，因此同一检查适用于当前根。
- `tests/test_reference_hif4.py` 与 `tests/test_eval_system.py`：`17 passed`。首次运行的 5 个 error
  来自系统 pytest 临时目录权限；切换到仓库内独立 basetemp 后全部通过，不是代码错误。
