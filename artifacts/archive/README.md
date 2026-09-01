# artifacts/archive 历史归档（旧评测协议产物）

> 本目录只存放**已废弃评测协议**的机器可读结果，禁止用于排序、调参或时间判定。
> 当前唯一评测协议见根 [`README.md`](../../README.md) 与 `evaluator/official_eval.py`。

| 目录 | 内容 | 对应旧协议 |
|---|---|---|
| `legacy-real-model-suite-20260901/` | 多模型 real_model_suite 评测 JSON（含 sampled-means-v1/v2、oracle anchors） | real_model_suite.py |
| `legacy-oracle-dashboard-20260901/` | e0g/l0/l5e oracle dashboard JSON（多层×多模型误差分解） | linear_ceiling_dashboard.py / cap_oracle.py |
| `legacy-jdrq-diagnostics-20260901/` | jdrq D0 投影诊断 JSON（d0-*.json） | jdrq_diagnostics.py |
| `legacy-official-eval-20260901/` | 早期 official-shape-v1 前身协议 JSON | 旧 official_eval 变体 |

命名规范：`legacy-<主题>-<归档日期>YYYYMMDD`。归档动作记录在
[`docs/current-solution-status.md`](../../docs/current-solution-status.md) 第 7 节与
`logs/archive/legacy-root-files-20260901/`。源码归档在
[`evaluator/archive/legacy-20260901/`](../../evaluator/archive/legacy-20260901/)。
