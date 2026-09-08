# Linear 当前机制队列

唯一明细：[持续研究循环 §7](../../docs/superpowers/plans/workpackages/2026-09-08-continuous-research-loop.md)。
本文件只保存状态，不定义新门禁。

1. **P0 身份核验**：用户回传的完整候选 `17636/264s` 优于根 v189，但官方计分 SHA 尚未与
   归档 SHA `D66128A6...B0F6` 单独绑定。核验前状态为 `REPORTED_BETTER / IDENTITY_PENDING`。
2. **当前侧父**：L28 `4611/286s`，完整校准 fit_gain `0.948587`；L4 `4607/247s` 只作
   time reference。L29-Q/G 已前置拒绝，L30 已拒绝，LC0 只作正确性审计。
3. **下一卡**：L31 与 L32 先去重，只注册其中一张，并从 L4 构建；不得并行扫参。

Linear 直接用全部 4B 校准数据做 A@W 拟合。独立窗口、split、负向损失和本地 `api_seconds`
只记录，不得否决候选；官方分数和官方 300s 是最终裁决。
