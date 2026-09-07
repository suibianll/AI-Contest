# 过期信息清单：2026-09-07

本修订优先于 09-02、09-04、09-05 清单中的执行要求。当前操作以 [4B 指引](4b-panel-testing-guide.md) 为准。

- 全部新测试使用 Qwen3.5-4B；0.5B、逐候选 OOD、GPT-2/opt、fresh-default 计时退役。
- 六 API 时间公式、预测 <280s、TIME_HOLD 及“必须补 fresh timing”失效；官方唯一时间硬限 300s。
- 当前完整六 shard 为 336 Linear + 72 Attention；48/120 Attention 等数字属于旧协议。
- shard 单侧冒烟只运行目标侧 API；六 API 导入与冻结侧 control 分开检查。
- 历史时间/OOD拦截只说明当时未获官方验证，不自动重开候选；已有官方负结果仍保留。
- 总 L1 通用分析器不替代活动工作包的专项负向损失裁决；本地正向不保证官方收益。

历史源码、JSON/report 和日志不删除、不改写；兼容后端仍被日常评估器调用，不按旧名字删除。
清理范围及验证见[审计记录](evaluation-system-audit-2026-09-07.md)。
