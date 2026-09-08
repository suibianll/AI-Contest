# HiF4 优化实验仓库

当前完整官方父为 v189：17616 / 275s，根 `solution.py`。官方硬限 300s，提交次数无限制。
官方计分 SHA256：`261202248A0146A2EE45F3DF60BD1979BB8171B7C162921013B0024C848617AF`。
Linear 侧父：L28（4611/286s，2026-09-08 RETAINED，+4 vs L4；残差交叉子空间 A@W 拟合 + 时间安全重构）。
另有 compiled sample-energy 用户回传 17636 / 264s；其官方计分 SHA 尚未与归档 SHA 单独核验，
当前统一标记为 `REPORTED_BETTER / IDENTITY_PENDING`，核验前不替换根父。

## 工作入口

- [执行规则](AGENTS.md)
- [当前唯一活动计划入口](docs/superpowers/plans/README.md)
- [4B 测试指引](docs/4b-panel-testing-guide.md)：唯一日常测试流程
- [评估系统技术说明](docs/proxy-v3.md)
- [当前状态及历史证据](docs/current-solution-status.md)、[版本索引](solutions/README.md)
- [本次评估系统审计](docs/evaluation-system-audit-2026-09-07.md)

## 当前评测口径

使用 `evaluator/eval.py` 与 Qwen3.5-4B 输入缓存，完整六 shard 为 336 Linear + 72 Attention。
4B 是结构代理，不是官方隐藏评测；官方样例数按用户确认的 50 Linear + 250 Attention 记录。
本地只比较同 cache、协议、设备及精确 case 身份的父子最终输出 gain，不换算官方分数。

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution <parent.py> --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 7 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --reuse-existing --output-dir artifacts\proxy_v3\<run-id>
```

Attention 使用 `--attention-only`。完整双侧只用于集成审计。父源码必须绑定不可变 SHA。
不新增 0.5B、逐候选 OOD、GPT-2/opt 或 fresh-default 计时运行；本地 API 时间仅记录。
Linear 独立窗口只记录；Attention 4B paired 只作合法性/可达性/风险判读；任何本地时间或按层外推
都不构成 `<280s` 提交门，官方分数与官方 300s 是最终裁决。
通用分析器的总 L1 规则与持续优化计划的负向损失规则不同，专项裁决依 4B 指引记录。

`official_eval.py` 是仍被日常入口导入的兼容/参考后端，不能删除；旧测试和历史 JSON/report
仅作回归或证据，不作为当前实验命令。历史结果不重写、不批量重测。
