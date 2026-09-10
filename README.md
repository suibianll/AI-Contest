# HiF4 优化实验仓库

**当前完整根：v237 Linear（L-TF2 首遍梯度复用，K=2）+ v195 Attention，官方 18518 / 289s**，相对父 v231 根**同分 / −2s**，硬限余量 **11s**。根与归档逐位一致，SHA256 `ECB1F9E510B5507E1A2DC8B95F8A84E9B51864A420B3828537A328813E2CE554`。见[v237 官方结果与晋级记录](logs/execution/2026-09-10-v237-linear-official-result.md)。回退根仍为 v233（L-TF1 梯度复用，v230 后代）`18428/288s`，SHA256 `0EC89710087D061BF9608196AD4D53A1C6BE98C8A5595596A071ECD05A6821EB`。所有后续新候选从 v237 构建；v238（A-CT1）/v239（A-CT2）建在旧父 v231 上，保留原父、与 v237 不可相加。官方提交次数无限制。

**一条必须记住的教训**：v237 本地三臂配对四行全部落在同字节 sham 空对照之内（0.52/0.52/1.11/1.00×），官方仍量出 **−2s**；v233 同型（本地全时钟分辨不出，官方 −4s）。**本地测不出不等于收益为零**，本地读数不得用于否决纯降时候选的提交。

官方待回传：**v238 Attention A-CT1**（`146bb715…`，A-GR1 gate 固定计算复用）与 **v239 Attention A-CT2**（`55103e8b…`，训练尾部统计复用）。两张都建在**旧父 v231** 上（不是新根 v237），对同父 A-GR1 旧实现 72/72 逐位零；按[停滞诊断](docs/attention-stall-analysis-2026-09-10.md)测定的量级，去重方向合计约 0.2s，是否各花一次提交由用户决定。

v236 Attention A-GR1-on-v231 官方 **TIMEOUT（>300s）REJECTED**（2026-09-10 用户回传），见[超时记录](logs/execution/2026-09-10-v236-agr1-on-v231-official-timeout.md)：它与 v234 的两次超时构成双向封闭区间（父余量 8s/9s，父差仅 1s），**不存在"换个略好的 Linear 父就能过"的空间**；侧隔离 +29 不受影响，A-GR1 机制本身不关闭。

最新官方回传见[v237 Linear L-TF2 晋级记录](logs/execution/2026-09-10-v237-linear-official-result.md)（18518/289s，同分快 2s，晋级为当前完整根）、[v231 晋级记录](logs/execution/2026-09-10-v231-linear-official-result.md)与[v233 同分提速记录](logs/execution/2026-09-10-v233-linear-official-result.md)；同日 v232 Linear L-QF1 官方 TIMEOUT（>300s）REJECTED，见[超时记录](logs/execution/2026-09-10-v232-linear-official-timeout.md)；v235 Linear L-AD1 亦官方 TIMEOUT（>300s）REJECTED——**它是本轮唯一在本地测出可分辨提速的 Linear 候选（有 gram 层 34–52× null），仍超时**，见[超时记录](logs/execution/2026-09-10-v235-linear-official-timeout.md)。此前 v229 官方超时保留，其标准 Linear 侧隔离官方 `14424/245s`（相对 v195 侧基准 −2），K 平移类已关闭；v230 Attention A-FIX1 官方 TIMEOUT（>300s）REJECTED，见[回传记录](logs/execution/2026-09-10-v230-attention-afix1-official-timeout.md)。v234 Attention A-GR1 侧隔离官方 `14455/263.7s`（相对 v195 侧基准 **+29**），为继 C76.4/A1 后第三大 Attention 官方正向机制，见[回传记录](logs/execution/2026-09-10-v234-agr1-side-official.md)；但其**完整包官方 TIMEOUT（>300s）REJECTED**（2026-09-10 用户回传）——A-GR1 的额外时间在校准侧、是机制自带代价，父 v230 余量仅 8s 装不下，见[超时记录](logs/execution/2026-09-10-v234-agr1-official-timeout.md)。侧隔离 +29 测的是分数、且是更小的包，**不被超时推翻**：结论是"机制有分、代价不可落地"，与 v229 A-MC1、v230 A-FIX1 同类，不缩步/缩窗/减候选重试。这也直接落在 **v236**（A-GR1 重挂 v231 根，唯一改动是父替换）上。

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
