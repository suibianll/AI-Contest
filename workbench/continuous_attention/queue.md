# Attention 当前机制队列

更新：2026-09-07。当前优先工作包：[14199回传后计划](../../docs/superpowers/plans/workpackages/attention-after-14199.md)。
冻结 v162 standard Linear、R3 V。官方成绩为侧隔离整包分，不冒充分项。

| 用途/裁决 | 机制 | 官方分 / 时间 |
|---|---|---|
| 高分对照 | A2 full-KV | 14440 / 274s |
| 低成本研究父 | R3 rotation + center | 14405 / 238s |
| 已完成 | A1 deployed-aligned | 14389 / 256.3s |
| 已完成 | R2c | 14387.8 / 240s |
| 已完成 | R1 | 14009 / 211s |
| 已完成 | A2c standalone | 9538 / 175s |
| 官方负向 | A3sym bounded-mix | 13572 / 288s |
| 官方负向 | A21-1 scale替换 | 14199 / 244s |

源码及完整SHA见[state.json](state.json)。A2曾被本地DOMINATED误判，官方结果已推翻该裁决。
A3虽OOD更好但官方明显退步；OOD只能记录风险，不能提供晋级排序。
旧A3配置关闭，不外推为一般互逆变换无效；原始机制卡/日志留在归档和Git历史。

| 顺序 | 卡 | 状态 / 依赖 |
|---|---|---|
| 1 | A22-1 完整R3回退保护 | READY，保持旧scale提案，先完整父control，后成本/风险/官方检验 |
| 2 | A22-2 父坐标增量互逆变换 | 等A22-1回传；S=0恢复父，K-center同步变换 |
| 3 | A22-3 输出失配目标 | 保留父后仍scale/output失配，再登记公式与去重 |

A21-1已官方REJECTED，不再等待回传。其本地接受组收益不能外推官方分项。
本地与官方排序反转后不再用同一路线proxy排序晋级；A22-1有单独登记的一个机制探索代表，其他卡不自动继承。
