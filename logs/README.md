# 运行产物归档

本目录存放**运行产物**，与 `docs/` 中的工程文档严格分离。

约定：**任何评测输出、执行日志、候选结果报告都不得写入 `docs/`。**
`docs/` 只保留使用说明、设计方案、规范与文献调研。

## 目录用途

| 目录 | 内容 | 命名 |
|---|---|---|
| `evaluations/` | 评测运行报告（`real_model_suite.py` 输出的 Markdown） | `YYYY-MM-DD-<topic>.md` |
| `candidates/` | 候选的官方结果、校准记录与失败诊断 | `<CANDIDATE>-<topic>.md` |
| `execution/` | 执行日志、顺序实验索引、环境记录 | `YYYY-MM-DD-<topic>.md` |

机器可读的 JSON 结果仍放在 `artifacts/real_model_suite/`，Markdown 报告放本目录，两者一一对应。

## 评测器参数约束

`evaluator/real_model_suite.py` 的 `--report` **没有默认值且为必填**，
因此一次运行不可能静默覆盖已归档的报告。每次评测都显式指定路径：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small --candidates c39 `
  --solution solution.py --candidate-name active `
  --device cpu --algorithm-device cuda --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\baseline-YYYYMMDD.json `
  --report logs\evaluations\YYYY-MM-DD-baseline-vs-c39.md
```

`--output`（JSON）保留默认值 `artifacts/real_model_suite/latest.json`；
需要长期留存时改用带日期的文件名。

## 现有归档

- `evaluations/2026-08-28-official-flow-smoke.md`：协议冒烟（C21-C `151.078193` vs C39 `150.313301`，与官方反序）。
- `evaluations/2026-08-28-calibration-old-protocol.md`：旧评分协议生成，**禁止用于候选排序**。
- `evaluations/2026-08-28-active-vs-c39.md`：活跃文件 vs C39 五模型配对（总分 `975.495261` vs `996.745557`）。
- `evaluations/2026-08-28-baseline-vs-c39.md`：同上，干净工作区下的正式基线，精度逐位复现。
- `evaluations/2026-08-28-archived-official.md`：已归档官方候选的对照评测。
- `candidates/C39-FW-official-calibration.md`：当前合规冠军 `14613 / 159.2s`。
- `candidates/C40-robust-block-ldlq.md`：C40 机制说明（官方 `14432 / 216.667s`，已拒绝）。
- `candidates/C40-official-evaluator-diagnosis.md`：本地评测器失效诊断，旧代理排序不可信的依据。
- `execution/`：2026-08-26 起的执行日志、顺序实验索引与环境记录。
