# Linear 当前机制队列

2026-09-07更新。唯一明细：[21071下一轮计划](../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。

1. R0：证据身份、去重和[解释纠偏](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)，零API，不重测旧0.5B。
2. L23：官方TIMEOUT，当前复杂度实现关闭。后继为全校准行求解修正与残差增量降成本，单独新SHA验证归档，不机械减rank/样本/步数重试。
3. 条件后继：仅在本卡误差定位给出新自由度时登记一张，不扫旧rank/步数/窗口/阈值。

当前父与全部历史官方结果见[state.json](state.json)。Linear直接用全部4B校准数据做A@W低维拟合，不考虑泛化、不拆fit/select；
独立窗口均值/split/负向损失只记录，不作门，
本地时间仅记录；官方300s。旧local_highest_reference仅历史参考，不作4B准入门。
L23已获官方超时回传，精确耗时和分数未知。具体旧实现负结果保留，不以此关闭成功机制整族。
