# HiF4 优化实验仓库

v231 Linear（L-EM3 K=2）+ v195 Attention 已由用户官方回传 **18518 / 291s**，相对 v230 完整根 **+90 / −1s**，硬限余量 **9s**。根与归档逐位一致，SHA256 `EA79A1C12DC667142C620975AAB188920FAE7B29988C804F41C1A696CC5754F1`。回退根为 v233（L-TF1 梯度复用）`18428/288s`，SHA256 `0EC89710087D061BF9608196AD4D53A1C6BE98C8A5595596A071ECD05A6821EB`——同分于 v230 但快 4s，严格占优；其 L-TF1 尚未并入根。同编号 v230 Attention A-FIX1 官方状态不受本次回传影响。 官方提交次数无限制。

官方待回传：**v236 Attention A-GR1-on-v231**——把已官方定价 +29 的 A-GR1 原样重挂到 v231 根（唯一改动是父替换），六 shard 复现 v234 每一项 `+0.003845`（21/3/48）；**时间风险：侧隔离 +20.7s vs 根余量 9s**，不写秒数预测。见[v236 执行记录](logs/execution/2026-09-10-attention-agr1-on-v231.md)。

最新官方回传见[v231 Linear 晋级记录](logs/execution/2026-09-10-v231-linear-official-result.md)与[v233 同分提速记录](logs/execution/2026-09-10-v233-linear-official-result.md)；同日 v232 Linear L-QF1 官方 TIMEOUT（>300s）REJECTED，见[超时记录](logs/execution/2026-09-10-v232-linear-official-timeout.md)；v235 Linear L-AD1 亦官方 TIMEOUT（>300s）REJECTED——**它是本轮唯一在本地测出可分辨提速的 Linear 候选（有 gram 层 34–52× null），仍超时**，见[超时记录](logs/execution/2026-09-10-v235-linear-official-timeout.md)。此前 v229 官方超时保留，其标准 Linear 侧隔离官方 `14424/245s`（相对 v195 侧基准 −2），K 平移类已关闭；v230 Attention A-FIX1 官方 TIMEOUT（>300s）REJECTED，见[回传记录](logs/execution/2026-09-10-v230-attention-afix1-official-timeout.md)。v234 Attention A-GR1 侧隔离官方 `14455/263.7s`（相对 v195 侧基准 **+29**），为继 C76.4/A1 后第三大 Attention 官方正向机制，见[回传记录](logs/execution/2026-09-10-v234-agr1-side-official.md)；但其**完整包官方 TIMEOUT（>300s）REJECTED**（2026-09-10 用户回传）——A-GR1 的额外时间在校准侧、是机制自带代价，父 v230 余量仅 8s 装不下，见[超时记录](logs/execution/2026-09-10-v234-agr1-official-timeout.md)。侧隔离 +29 测的是分数、且是更小的包，**不被超时推翻**：结论是"机制有分、代价不可落地"，与 v229 A-MC1、v230 A-FIX1 同类，不缩步/缩窗/减候选重试。这也直接落在 **v236**（A-GR1 重挂 v231 根，唯一改动是父替换）上。

## 工作入口

- [本轮总结与下一步](docs/optimization-round-summary-2026-09-10.md)：先修正Linear局部二次代价，再消除重复计算；官方待定项单列
- [执行规则](AGENTS.md)
- [当前唯一活动计划入口](docs/superpowers/plans/README.md)
- [4B 测试指引](docs/4b-panel-testing-guide.md)：唯一日常测试流程
- [评估系统技术说明](docs/proxy-v3.md)
- [当前状态及历史证据](docs/current-solution-status.md)、[版本索引](solutions/README.md)
- [本次评估系统审计](docs/evaluation-system-audit-2026-09-07.md)

## 当前评测口径

使用 `evaluator/eval.py` 与 Qwen3.5-4B 输入缓存；算法开发阶段按活动计划直接运行目标侧完整六 shard
（336 Linear + 72 Attention），不等待官方回传；本地结果只用于 hard-output 诊断和归档。
4B 是结构代理，不是官方隐藏评测；官方样例数按用户确认的 50 Linear + 250 Attention 记录。
本地只比较同 cache、协议、设备及精确 case 身份的父子最终输出 gain，不换算官方分数。

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution solution.py --linear-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-id>
```

Attention 使用 `--attention-only`。完整双侧只用于集成审计。父源码必须绑定不可变 SHA。
不新增 0.5B、逐候选 OOD、GPT-2/opt 或 fresh-default 计时运行；本地 API 时间仅记录。
Linear `calibration_fit_gain` 与 Attention 4B paired 都只作合法性、可达性和风险诊断；任何本地
分数、时间或按层外推都不是提交门，官方总分与官方 300s 是最终裁决。

`official_eval.py` 是仍被日常入口导入的兼容/参考后端，不能删除；旧测试和历史 JSON/report
仅作回归或证据，不作为当前实验命令。历史结果不重写、不批量重测。
