# A21-1 官方回传及下一步裁决

用户本轮回传：**14199分，244秒**。绑定此前唯一待官方候选
`solutions/continuous_attention_anchor21-a1/solution.py`，官方计分/归档源码SHA均为
`870d5848f95887307ad7faa6364b5d7f7480f5b7be6001c812c44ead02bdb48a`，两处源码已核对且未修改。

裁决 **OFFICIAL_REJECTED**：相对R3 14405/238s为−206/+6s；相对A2 14440/274s为−241/−30s。
244未超过300秒硬限；模型预测219.284164，低估24.715836秒，不能把单个残差作为新时间模型。
根v189、研发父R3、高分父A2均保留。

旧报告的READY_FOR_OFFICIAL_EXPLORATION是提交前状态，现已失效；本地Δmean +0.001234、
负向L1 0.007585及原始JSON/report仍保留。此路线再次发生本地与官方排序反转，
停止用其本地均值排序晋级，不把OOD负向恰巧同号解释为OOD有效。
本地回退组损失不能直接解释官方−206，官方没有分项。

当前next：[A22-1及后继计划](../../docs/superpowers/archive/plans/attention-after-14199.md)。
先保持A21-1的scale提案，恢复完整R3校准对照及回退，单独检验替换损失；
再依结果进入完整父坐标的增量变换，最后才研究输出敏感度目标。
不扫描旧参数。本次只登记回传、更新状态和计划，未启动新模型实验或改动提交源码。

机器可读官方事实见归档与工作目录各自的official-result.json；manifest只更新裁决字段，
原始本地report与既有execution日志不覆盖。
