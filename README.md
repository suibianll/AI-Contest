# HiF4 优化实验仓库

当前最优已知可复现完整方案为 v202 Linear + v195 Attention：用户官方回传 `18053 / 281s`，根
`solution.py` 与归档逐位一致，SHA256 `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
相对 v195 同分快 8 秒；上一完整根 current Linear + v195 Attention 为 `18053 / 289s`，再上一根
current Linear + R3 Attention 为 `18032 / 280s`。官方硬限 300s，提交次数无限制。

## 工作入口

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
