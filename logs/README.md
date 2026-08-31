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

机器可读的 JSON 结果仍放在 `artifacts/real_model_suite/`，Markdown 报告放本目录，两者一一对应；
这些运行输出保留在本地但已由 `.gitignore` 排除，不提交到 Git。

## 官方评测集修订（2026-08-29）

官方样例扩展为 250 个 Linear case 与 200 个 Attention case，时间限制提升
到 7 分钟（420s）。新版已确认：v031/C39-FW `21864 / 161.3s`、
v034/C41b `21864 / 159.4s`、v051/C47b `22451 / 234s`、v066/C66
`22557 / 217.2s`；外部
[`youxilee/hif4`](https://github.com/youxilee/hif4) 官方结果为 `24153 / 239s`。
未修改的 v2.7 源码在本地 CPU 固定缓存上的最高单模型是 Qwen2.5-0.5B，
native `369.527269`；按固定 250/200 panel 投影后，最高同口径本地基准为
`250.327102`。五模型诊断相加的 `1085.743597` 不是官方分数换算，也不允许作为
排名分。当前根 v110 Qwen panel `295.242780` 比外部最高 panel 高 `44.915678`
（`17.94%`），native 高 `52.240685`（`14.14%`）。当前本地归档官方冠军为
v066/C66，官方仍比外部低 `1596` 分；外部 `24153 / 239s` 是用户提供的官方
参考，不由本地代理推导。

> **2026-08-31 再次修订**：官方端到端限制收紧为 `300s`（5 分钟），且不再限制任何
> `A@W` 拟合用法；用户确认 **v98 官方判为 timeout**（本地 API `406.24s`），见
> [`execution/2026-08-31-v98-official-timeout.md`](execution/2026-08-31-v98-official-timeout.md)。
> v107 并非 timeout，官方结果保持 Attention `wrong answer`（非 timeout）。
> 上方 420s 表述仅记录 08-29 修订时的口径。

## 评测器参数约束

`evaluator/real_model_suite.py` 的 `--report` **没有默认值且为必填**，
因此一次运行不可能静默覆盖已归档的报告。每次评测都显式指定路径：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small --candidates c39 `
  --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model gpt2-small `
  --device cpu --algorithm-device cuda --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\baseline-YYYYMMDD.json `
  --report logs\evaluations\YYYY-MM-DD-baseline-vs-c39.md
```

`--output`（JSON）保留默认值 `artifacts/real_model_suite/latest.json`；
需要长期留存时改用带日期的文件名。

默认 `qwen-official` 面板的主分为 `250 * Linear_mean + 200 * Attention_mean`；
Qwen2.5-0.5B 驱动主排序，其他模型只作软 guardrail。`official_flow_total`
原始逐 case 求和仍记录，但不用于跨模型主排序，也不能换算成官方绝对分数。

## 现有归档

- `evaluations/2026-08-28-official-flow-smoke.md`：旧本地协议冒烟（C21-C `151.078193` vs C39 `150.313301`，与官方反序；输出现仅本地保留）。
- `evaluations/2026-08-28-calibration-old-protocol.md`：旧评分协议生成，**禁止用于候选排序**。
- `evaluations/2026-08-28-active-vs-c39.md`：活跃文件 vs C39 五模型配对（总分 `975.495261` vs `996.745557`）。
- `evaluations/2026-08-28-baseline-vs-c39.md`：同上，干净工作区下的正式基线，精度逐位复现。
- `evaluations/2026-08-28-archived-official.md`：已归档官方候选的对照评测。
- `candidates/C39-FW-official-calibration.md`：新版官方锚点 `21864 / 161.3s`（旧口径为 `14613 / 159.2s`）。
- `candidates/2026-08-29-external-hif4-gap-analysis.md`：v066/C66 与外部
  `youxilee/hif4` 的逐项算法差距、Qwen-30B 特征用例假设与 C70 实验路线。
- `solutions/20260829_v070_c70-joint-refine-rejected_scoreNA_timeNA/` 与
  `solutions/20260829_v071_c71-proj-final-quantizer-rejected_scoreNA_timeNA/`：
  外部联合补偿、proj H32/H64 的真实代理消融，均已拒绝并恢复 C69。
- `candidates/C40-robust-block-ldlq.md`：C40 机制说明（官方 `14432 / 216.667s`，已拒绝）。
- `candidates/C40-official-evaluator-diagnosis.md`：本地评测器失效诊断，旧代理排序不可信的依据。
- `execution/`：2026-08-26 起的执行日志、顺序实验索引与环境记录。
