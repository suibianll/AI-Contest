# 单一最高分根持续优化与过期计划清理

日期：2026-09-08。性质：执行政策与文档清理；未运行模型评测，未修改根或候选源码。

## 用户指令

不再单独优化 Attention；直接在当前最高分版本上持续优化，并清理过期计划。

## 当前政策

- 唯一父为根 `solution.py`，SHA `12352EFDD4E23CC5E1E17953008664FBAA5EA5D693373635FDAFC4D28CE4E24E`，
  用户官方回传 `18032/280s`。
- 每轮只有一个完整父和一个完整候选。候选可只改变一个子系统，但必须继承当前根其余五/六 API，
  不产生侧父、侧队列、侧提交或侧组合。
- 官方正向且 `<300s` 才切换根；下一轮总是从裁决后的最高分根开始。
- 历史 Linear/Attention 侧结果仅作为机制证据，不提供执行顺序。

## 清理边界

`docs/superpowers/plans/workpackages/` 中全部旧双侧循环、侧执行、面板工作包和已完成 R0 卡移入
`docs/superpowers/archive/plans/`。历史内容不删除、不改写；活动区只保留唯一总计划和 README。
现有未跟踪 `workbench/continuous_linear/lc3-objective-only/` 不删除、不覆盖，下一步先核对其父 SHA。
