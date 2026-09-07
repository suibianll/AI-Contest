# 4B 面板测试指引（2026-09-07 定版）

> 适用前提：Qwen3.5-4B 结构代理面板 R0–R2 验收完成（执行记录
> `logs/execution/2026-09-07-qwen35-4b-panel-r0-r1-r2.md`）；用户批准
> 「后续测试完全使用4B模型的数据，不需要那么多OOD门禁等测试了」。
> 本文件回答「现在测试应该怎么测」；与
> [持续优化计划](2026-09-07-continuous-linear-attention-plan.md)冲突时以本文件为准。

## 0. 三层总原则

| 层 | 面板 | 回答的问题 | 不能回答的问题 |
|---|---|---|---|
| 机制判读 | **4B**（qwen35-4b-panel-v1） | 候选相对父在同族结构上是否正收益、负向风险多大 | 官方分数预测、晋级旁证 |
| 提交时间门 | **0.5B**（fresh default 计时） | 官方时间预测 <280s（硬限 300s） | 任何机制优劣 |
| 官方 | 提交回传 | 唯一分数/时间裁判 | — |

- 4B 与 0.5B 数值**永不混排**（协议断层纪律）；4B 秒数**不得**代入时间模型公式。
- 机制判读只看 paired Δ（候选减父、同 cache 同面板），不看单侧绝对分。

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

- 基线不重跑：同 cache/协议/SHA 下已有结果一律 `--reuse-existing` 零 API 重放。
- 换父（如 anchor22-a2 等归档父）时，用 `--baseline-solution <父.py>` 同 run 配对，
  系统自动算父侧；父的单独基线读数以该 run 的 baseline manifest 为准。

## 2. 机制候选测试流程（每个候选）

**第 1 步：定侧与父。** Linear 候选 `--linear-only`；Attention 候选 `--attention-only`。
父源码：根 `solution.py`（=v189）或对应归档父（如 `solutions/continuous_attention_anchor22-a2/solution.py`）。
非目标侧保持冻结，不因候选改动。

**第 2 步：shard0 冒烟（~3–5 分钟）。** 确认六 API 全跑、legal state、coverage true、
无形状崩溃、机制 reachable。

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution <parent.py> --linear-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-id>\smoke
```

**第 3 步：全六 shard paired（首个候选 fresh 校准约 22 分钟；同 SHA 校准缓存命中后约 10 分钟）。**

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution <parent.py> --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 7 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --reuse-existing --output-dir artifacts\proxy_v3\<run-id>\id
```

Attention 侧把 `--linear-only` 换 `--attention-only`、目录侧名换 `attention`。
`--stop-after-nonpositive 7` 防止通用两 shard 截断固定六 shard；不是绕过最终裁决。

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

1. **4B paired 门通过**（机制证据；记录 Δmean/L1/负向 case）。
2. **0.5B fresh default 时间门**（168L+120A 兼容计时面板，候选校准必须实际执行）：

   ```powershell
   .venv\Scripts\python.exe evaluator/official_eval.py --solution <candidate.py> --name <run-id> --cache-mode read --nvfp4-cache-mode auto --algorithm-device cuda --output artifacts\proxy_v3\<side>\<run-id>\timing\default.json --report artifacts\proxy_v3\<side>\<run-id>\timing\default.md
   ```

   六 API 实测代入 AGENTS.md §4 时间模型，**预测 <280s 才提交**；训练类按
   「预测 +~35s」做情景检查（<300s 硬限）。4B 秒数不可代入。
3. **单文件导入检查**（脱离仓库 import 六 API）。
4. **fuzz_official_contract.py**（训练类机制必跑）。
5. 提交后登记：官方分数/时间/计分 SHA → 状态文档、版本索引、计划进度表；
   4B 读数不写进官方对齐结论。

## 4. 明确不跑

- **逐候选 OOD**：4B 面板未抓 OOD 窗口（capture 只抓 in-dist 12 窗）；如需 4B OOD
  诊断须先扩展 capture（未来项，启用前登记）。0.5B OOD 路径保留但不参与机制判读。
- **跨模型 GPT-2/opt**：已废弃（ρ≈−0.2），不跑、不引用。
- **0.5B 机制筛查**：已被 4B 取代；0.5B 只保留 fresh default 时间门。
- **相同 SHA / 逐位等价 A/B 重复提交**；不同算法 ±1~4 分不证明噪声。

## 5. 与活动计划命令的替换关系

持续优化计划 §5 的命令模板写于 4B 面板存在之前，引用 `qwen2.5-0.5b-proxy-v2.pt`：
机制筛查一律把 cache 换成 `qwen3.5-4b-proxy-v2.pt`；计划中「A 336 / A 48」的 48 是
0.5B 口径，4B 全量 attention 为 **72**；其余流程（侧隔离、stop-after-nonpositive、
账本字段、组合队列规则）不变。
