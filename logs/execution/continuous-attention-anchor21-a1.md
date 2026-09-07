# Attention A21-1 执行与状态整理

2026-09-07。已完成联合64块scale互逆变换的单配置实现、梯度与部署验证、六shard ID/OOD、
fresh default计时、跨模型记录和48case真实动态诊断。

完整结果与复现：[report](../../workbench/continuous_attention/anchor21-a1/report.md)，
机器可读裁决：[manifest](../../workbench/continuous_attention/anchor21-a1/manifest.json)。

对完整R3：ID Δmean +0.001233981，negative L1 0.007585332；OOD Δmean −0.002764068，
Δgap +0.003998049；预测官方219.284164s。状态READY_FOR_OFFICIAL_EXPLORATION，
官方NA，根仍v189。源码SHA 870d5848f95887307ad7faa6364b5d7f7480f5b7be6001c812c44ead02bdb48a。

本轮发现：6个接受层的独立case平均+0.02287，但18个回退层因删除旧训练平均−0.00598，
抵消大部分收益。后继分开研究回退保真与scale/output目标失配，不扫描本卡参数。

当前计划入口、状态、队列已清除旧矛盾指令。历史原始JSON/report/log不覆盖。
清理两处本任务pytest临时目录被自动审批策略拦截，原目录保留，不绕过。
