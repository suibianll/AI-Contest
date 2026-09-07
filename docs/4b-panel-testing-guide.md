# 4B 面板测试指引（2026-09-07 定版；20:16 用户指令后全面 4B 化）

> 适用前提：Qwen3.5-4B 结构代理面板 R0–R2 验收完成（执行记录
> `logs/execution/2026-09-07-qwen35-4b-panel-r0-r1-r2.md`）；用户批准
> 「后续测试完全使用4B模型的数据，不需要那么多OOD门禁等测试了」，并追加指令
> 「不要用0.5B的，全部使用4B进行测试」（2026-09-07 20:16——0.5B 面板退役，
> 时间门同步切 4B）。
> 本文件回答「现在测试应该怎么测」；与
> [持续优化计划](2026-09-07-continuous-linear-attention-plan.md)冲突时以本文件为准。

## 0. 三层总原则

| 层 | 面板 | 回答的问题 | 不能回答的问题 |
|---|---|---|---|
| 机制判读 | **4B**（qwen35-4b-panel-v1） | 候选相对父在同族结构上是否正收益、负向风险多大 | 官方分数预测、晋级旁证 |
| 提交时间门 | **4B**（paired Δ 时间，父官方时间锚定） | 官方时间预测 <280s（硬限 300s） | 机制优劣 |
| 官方 | 提交回传 | 唯一分数/时间裁判 | — |

- 机制判读只看 paired Δ（候选减父、同 cache 同面板），不看单侧绝对分。
- **4B 秒数不可代入 0.5B 时间公式**（v189 4B 实测代入得 ~465s vs 官方 275s，自证失效）；
  时间门用 paired Δ：所有六 API 测量都在 4B，绝对水平由父的官方时间锚定。
- 历史 0.5B 面板记录（含时间模型 R²=0.799 拟合）保留作证据，不再新增任何 0.5B 运行。

## 1. 面板与基线锚点

- Cache：`artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt`（6.28 GiB，
  panel `qwen35-4b-panel-v1`，24 层 = 18 DeltaNet + 6 full-attention，16Q/4KV/hd256）。
- 构成：**Linear 336 例**（24 层×7 role×2 窗）: **Attention 72 例**（6 FA×12 窗）。
  DeltaNet 不进 attention 场景是官方 Q/K/V API 合同的结构约束。
- **v189 根父基线**（manifest `artifacts/proxy_v3/qwen35-4b-baseline-v189-attn72/`，
  SHA `26120224…`）：

| 侧 | cases | mean | median | min | max |
|---|---|---|---|---|---|
| Linear | 336 | 0.524265 | 0.513478 | 0.280 | 0.820 |
| Attention | 72 | 0.526517 | 0.544639 | 0.228 | 0.818 |

- **v189 4B 时间锚点**（fresh 全量 run，校准实际执行；
  `artifacts/proxy_v3/qwen35-4b-baseline-v189/`，22m10s 那次）：

| API | 秒 |
|---|---|
| hif4_calibration_and_quantize_weight | 697.1 |
| hif4_calibration_attention | 15.0 |
| hif4_dynamic_quantize_activation | 279.2 |
| hif4_dynamic_quantize_q/k/v | 0.4 / 0.2 / 0.2 |
| **api_total** | **992.1** |

- 换父（如 anchor22-a2）时：先跑一次该父的 fresh 全量 run 登记其 4B 时间锚点
  与机制基线，再测候选。
- 已有结果不重跑：同 cache/协议/SHA 下一律 `--reuse-existing` 零 API 重放
  （仅用于机制复核；**缓存命中 run 的 api_seconds 不可用作计时行**）。

## 2. 机制候选测试流程（每个候选）

**第 1 步：定侧与父。** Linear 候选 `--linear-only`；Attention 候选 `--attention-only`。
父源码：根 `solution.py`（=v189）或对应归档父（如 `solutions/continuous_attention_anchor22-a2/solution.py`）。
非目标侧保持冻结，不因候选改动。

**第 2 步：shard0 冒烟（~3–5 分钟）。** 确认六 API 全跑、legal state、coverage true、
无形状崩溃、机制 reachable。

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution <parent.py> --linear-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-id>\smoke
```

**第 3 步：全六 shard paired + 计时行（首个候选 fresh 校准约 22 分钟）。**
**主 run 必须 fresh（不带 `--reuse-existing`、校准缓存未命中），它同时产出**
**① 机制 paired 统计 ② 六 API 计时行。** 计时运行前确认 GPU 空闲（gpu.lock），
不做其他 GPU 任务。

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution <parent.py> --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 7 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-id>\id
```

Attention 侧把 `--linear-only` 换 `--attention-only`、目录侧名换 `attention`。
`--stop-after-nonpositive 7` 防止通用两 shard 截断固定六 shard；不是绕过最终裁决。
后续机制复核可用 `--reuse-existing` 重放，但计时一律以 fresh run 的
`api_seconds` 为准（缓存命中会少算校准时间，不能当计时行）。

**第 4 步：判读门。**
- 通用符号门：`Δmean > 0 且 L1 < 0.02`（L1 = 逐 case gain 平均绝对变化）。
- continuous-linear 计划两个工作包改用专项负向损失 `mean(max(-Δgain,0)) < 0.02`
  （总 L1 只记录）；独立验证/control/隔离纪律不变。
- 附带记录：mean/median/q25/q75、worst-quartile、正负零 case、validation/test 同号率、
  最坏层/role/长度（attention 注意 12 窗含 10/128/512/1024 各长度）。
- 门通过 ≠ 官方非负（v188 教训）；本地正向不自动晋级。

**第 5 步：记录。** JSON/report 归档 `artifacts/proxy_v3/<run-id>/`；执行日志写
`logs/execution/`；按活动计划账本字段（run_id、side、SHA、case count、配对统计、
attempted/accepted、六 API 时间、next_action）。实质变更 commit & push（origin-ssh）。

## 3. 提交官方前检查单（全部满足才提交）

1. **4B paired 机制门通过**（Δmean/L1/负向 case 已记录）。
2. **4B paired Δ 时间门**：

   ```text
   预测官方时间 = 父的官方时间 + ΣΔ（候选 fresh 全量 run 六 API 秒 − 父 4B 锚点六 API 秒）
   提交条件：预测 < 280s（硬限 300s）。
   ```

   - 候选与父的测量必须同机、同面板、同 fresh 条件（校准实际执行；缓存命中 run
     的 api_seconds 不能当计时行）；父锚点见 §1（v189 = 992.1s / 官方 275s）。
   - Δ 按 1:1 传递到官方时间是当前假设（无 4B↔官方时间配对可拟合）；每次官方回传后
     记录预测残差，配对数据积累后评估 4B 系数重拟合或 Δ 传递系数标定。
   - 注意 v189 根父官方 275s：其候选的 Δ 预算只有 +5s（280s 门），时间敏感改动
     优先挂到时间余量更大的父（如 anchor22-a2 官方 271s，预算 +9s）。
   - 例：候选 fresh run api_total 1010s − 锚点 992.1s = ΔT +17.9s →
     预测 275 + 17.9 = 292.9s > 280s → 不提交（<300s 也不建议赌）。
3. **单文件导入检查**（脱离仓库 import 六 API）。
4. **fuzz_official_contract.py**（训练类机制必跑）。
5. 提交后登记：官方分数/时间/计分 SHA → 状态文档、版本索引、计划进度表；
   同时登记 4B 预测残差（官方时间 − 预测），积累 4B↔官方时间配对。

## 4. 明确不跑

- **任何 0.5B 面板运行**（2026-09-07 20:16 用户指令）：机制筛查已被 4B 取代，
  fresh default 时间门已改 4B paired Δ；0.5B 历史 JSON/公式保留作证据，不再新增运行。
- **逐候选 OOD**：4B 面板未抓 OOD 窗口（capture 只抓 in-dist 12 窗）；如需 4B OOD
  诊断须先扩展 capture（未来项，启用前登记）。
- **跨模型 GPT-2/opt**：已废弃（ρ≈−0.2），不跑、不引用。
- **相同 SHA / 逐位等价 A/B 重复提交**；不同算法 ±1~4 分不证明噪声。

## 5. 与活动计划命令的替换关系

持续优化计划 §5 的命令模板写于 4B 面板存在之前，引用 `qwen2.5-0.5b-proxy-v2.pt`：
机制筛查与计时一律用 `qwen3.5-4b-proxy-v2.pt`（机制+计时同 run）；计划中
「A 336 / A 48」的 48 是 0.5B 口径，4B 全量 attention 为 **72**；计划 §5 的
「fresh default 计时（official_eval.py default.json）」整段由本文件 §3 第 2 条
（4B paired Δ 时间门）取代；其余流程（侧隔离、stop-after-nonpositive、
账本字段、组合队列规则）不变。
