# 4B 面板测试指引（2026-09-08 单一完整方案版）

> 适用前提：Qwen3.5-4B 结构代理面板 R0–R2 验收完成（执行记录
> `logs/execution/2026-09-07-qwen35-4b-panel-r0-r1-r2.md`）。用户指令链：
> ①「后续测试完全使用4B模型的数据，不需要那么多OOD门禁等测试了」；
> ②「不要用0.5B的，全部使用4B进行测试」（0.5B 面板退役）；
> ③「不设本地时间门禁——本地时间根本不准」（20:26）。
> 本文件回答「现在测试应该怎么测」；与
> [当前唯一活动计划](superpowers/plans/2026-09-10-linear-cross-residual-correction-plan.md)冲突时以本文件为准。

> 当前规则优先级：`AGENTS.md` → 本指引 → 唯一活动计划/当前工作包 → workbench 状态文件。
> 状态文件不得新增门禁；冲突先修正文档和状态，不能在候选间临时切换口径。

## 0. 总原则

| 层 | 面板 | 回答什么 |
|---|---|---|
| 机制判读 | **4B** paired Δ | 候选相对父在同族结构上是否正收益、负向风险多大 |
| 时间 | 官方 300s 硬限（**唯一**） | 本地时间不准，**不设本地门禁、不做本地时间预测** |
| 官方 | 提交回传 | 唯一分数/时间裁判 |

- 机制判读只看 paired Δ（候选减父、同 cache 同面板），不看单侧绝对分。
- 本地 `api_seconds` 只随 manifest 记录；0.5B 公式退役，4B 秒数也不代入任何公式。
- 本地按层开销、父官方时间及复杂度估计只标记 `time-risk` 并决定降时优先级，不得据此恢复
  `<280s`、按层外推或其他官方提交硬门。
- 官方提交次数无限制：时间超不超由官方 300s 硬限直接裁决，超时是回传信息，
  不重复同 SHA 提交即可。
- 0.5B 面板退役：不再新增任何 0.5B 运行；历史记录保留作证据。

## 1. 面板与基线锚点

- Cache：`artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt`（6.28 GiB，
  panel `qwen35-4b-panel-v1`，24 层 = 18 DeltaNet + 6 full-attention，16Q/4KV/hd256）。
- 构成：**Linear 336 例**（24 层×7 role×2 窗）: **Attention 72 例**（6 FA×12 窗）。
  DeltaNet 不进 attention 场景是官方 Q/K/V API 合同的结构约束。
- **历史 v189 基线**（SHA `26120224…`，manifest
  `artifacts/proxy_v3/qwen35-4b-baseline-v189-attn72/`）：

| 侧 | cases | mean | median | min | max |
|---|---|---|---|---|---|
| Linear | 336 | 0.524265 | 0.513478 | 0.280 | 0.820 |
| Attention | 72 | 0.526517 | 0.544639 | 0.228 | 0.818 |

- 同 cache/协议/SHA 的已有结果一律 `--reuse-existing` 零 API 重放，不重跑。
- 当前候选统一以根 `solution.py` 为父；侧隔离归档不再切换为工作父。
- 参考成本（仅量级参考，非门）：v189 fresh 全量 api_total 992.1s（同机；
  校准缓存命中的 run 会显著偏小，属正常）。

## 2. 机制候选测试流程（每个候选）

**第 1 步：确定受影响子系统并冻结完整父。** 所有候选只从当前最高分根完整父构建；不存在
Linear/Attention 侧候选或侧父。机制若只影响 Linear，可用 `--linear-only` 做局部 smoke；只影响
Attention 时用 `--attention-only`。这些命令只减少本地检查范围，正式候选仍保留根的完整六 API，
未修改子系统保持 control。L28/A2/R3/AC0 只作历史机制证据，不再作为并行工作父。

**第 2 步：shard0 冒烟（~3–5 分钟）。** 目标侧 API 检查 legal state、coverage true、
无形状崩溃、机制 reachable；六 API 的导入/接口检查与非目标侧 control 单独完成，单侧运行不调用另一侧 API。

**第 3 步：官方前只做目标侧 shard0。** 全六 shard 不再是提交门；仅在官方正向后做归档复核，
或在官方失败后为回答一个明确诊断问题时运行。

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution solution.py --linear-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-id>
```

Attention 侧把 `--linear-only` 换 `--attention-only`。已有结果的复核用 `--reuse-existing` 零 API 重放。

**第 4 步：只判合法性与可达性。**
- `calibration_fit_gain = mean_case(1-MSE_PLAYER/MSE_STD)` 公式保留，但只是校准集内拟合诊断；
  不要求 `≥0.9`，不用于排序或否决。
- Attention/Linear 的 Δmean、holdout、L1 和误差账本均只记录；本地符号不决定是否提交。
- 附带记录 paired mean、正负零 case、最坏层/role/长度即可；只有明确诊断需要时再扩展统计。
- 门通过 ≠ 官方非负（v188 教训）；本地正向不自动晋级。

**第 5 步：记录。** JSON/report 归档 `artifacts/proxy_v3/<run-id>/`；执行日志写
`logs/execution/`；记录 run_id、目标侧、父子 SHA、case count、reachable/control 与官方状态。
api_seconds 照 manifest 抄录即可；若候选
api_total 相比父明显膨胀（如 >1.5×），提交说明标注「时间风险」——只是提示，
不阻止提交。实质变更 commit & push（origin-ssh）。

## 3. 提交官方前检查单

1. **4B目标侧 shard0**：接口、legal state、finite、reachable、非 no-op、非目标 API control。
2. **合法性**：legal state（评测器强制）+ 脱离仓库单文件导入检查。
3. **候选专用六 API 随机形状 contract smoke**（训练类机制必跑）。
4. （提示，非门）候选 api_total 相比父明显膨胀 → 提交说明标注时间风险。
5. 提交后登记：官方分数/时间/计分 SHA → 状态文档、版本索引、计划进度表；
   同时抄录该候选的 4B api_total（仅为将来积累参考数据，不构成预测）。

## 4. 明确不跑

- **任何 0.5B 面板运行**（退役；历史记录保留作证据，不再新增）。
- **本地时间门禁 / 时间锚点 / Δ 时间预测**（20:26 指令废除；api_seconds 只记录）。
- **逐候选 OOD**：4B 面板未抓 OOD 窗口（capture 只抓 in-dist 12 窗）；如需 4B OOD
  诊断须先扩展 capture（未来项，启用前登记）。
- **跨模型 GPT-2/opt**：已废弃（ρ≈−0.2），不跑、不引用。
- **相同 SHA / 逐位等价 A/B 重复提交**；不同算法 ±1~4 分不证明噪声。

## 5. 规则入口

活动总计划和 AGENTS 已同步本指引；历史 0.5B 数字与时间裁决只作历史证据。
代码审计和兼容边界见[评估系统审计](evaluation-system-audit-2026-09-07.md)。
