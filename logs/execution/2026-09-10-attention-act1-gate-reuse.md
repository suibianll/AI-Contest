# A-CT1：A-GR1 gate 的固定计算复用（v238）

计划卡：[`docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md) §4。
归档：[`solutions/20260910_v238_attention-act1-gate-reuse_scoreNA_timeNA/`](../../solutions/20260910_v238_attention-act1-gate-reuse_scoreNA_timeNA/result.md)。
工作目录：`workbench/full_solution/attention-act1-gate-reuse/`。
设备：NVIDIA GeForce RTX 3060 Ti。

本日志记**过程与判定**；逐项证据表在归档的 `result.md`，不在此重复。

## 0. 开工前置

| 项 | 值 |
|---|---|
| 正式父 | v231 完整根，官方 **18518 / 291 s**，余量 9 s，SHA `ea79a1c1…` / 505762 B |
| 装配来源 | v236 = 该根 + A-GR1，SHA `3319fc35…` / 520955 B，`cmp` 确认前 505762 B 逐字节等于根 |
| 对照定位 | v236 是**同父、同算法的实现对照**，官方 `unregistered/NA`；不设为晋级父，不继承其任何读数 |

**装配取自 v236 而非手工重装 A-GR1。** 计划要求"从该根重新装配 A-GR1"；v236 的归档已经证明是
"根字节 + A-GR1 块"的纯追加（本轮再次用 `cmp` 核验前缀），所以取它的文本**就是**那份装配，
且比手工重装更不可能引入差异。正式父仍是根，v236 只是装配来源与实现对照。

**父绑定不可变归档**：`build.py` / `verify.py` / `timing.py` 一律从 `solutions/` 下的归档读，
不读工作区 `solution.py`（前轮有并行会话改脏工作区的先例）。

## 1. 机制与改动点

A-GR1 的 gate 每窗连续调两次 `_agr1_gate_loss`（父臂、候选臂）。两臂**只差** `q_state.learned_rotation`、
`k_state.learned_rotation`、`k_state.learned_center`；`v_state` 是拷贝、值相等。

改动：每窗两次调用 → 一次 `_act1_gate_pair`，它把与臂无关的三件事各算一次（dense 参考 Q/K/V、
参考 `target`、父侧 V 五字段），两臂各自的 Q/K 量化/解码与各自的 Attention 前向**照旧**。

**不是**缩步/缩窗/减候选：窗口、候选数、32 步 Adam、学习率、谱约束、接受逻辑与最终 state 编译全不动。

## 2. 前提核验（先于任何等价性主张）

计划 §4 要求"必须先确认 V API 对输入和 state 无影响后续结果的修改，且两臂使用相同 V state"。
静态上：`_check_attention_state` 只做 `.get` 后原样返回 state；`hif4_dynamic_quantize_v` 不写 state；
`_nvfp4_to_hif4` 无 RNG，唯一的原地 `mul_` 作用在它自己 `_dequantize_nvfp4_float32` 产生的张量上。

但**没有把静态读当结论**，三条都在真实数据上实测（`verify.out` 的 `[P1]/[P2]/[P3]`）：

- P1：两臂差异字段恰为 `q_state.learned_rotation` / `k_state.learned_rotation` / `k_state.learned_center`，
  **v_state 逐字节相同**；
- P2：等值 state 调两次 V API，五字段逐位相同；
- P3：Q/K/V 调完，传入的 state dict 逐字节不变。

若任一条不成立，本卡必须收缩到"只复用参考 `target`"，不能用父臂的 V。

## 3. 构建（`build.py` → `build.json`）

追加式：候选 = 装配字节（520955 B）+ 追加模块（7249 B）= 528204 B。
追加的函数由装配文本派生：切出 A-GR1 的 `hif4_calibration_attention`，做恰好 1 次块替换（6 行，−136 B），
再追加 `_act1_gate_pair` 与该影子函数。还原成两次调用后 AST 与 A-GR1 逐节点相同。

候选 SHA256 `146bb7151f5f2a041b2f1fdbc94e5b370815fb91fa94d58db79384bb7a54fbc7`。

## 4. CPU 验证（`verify.py`，全 PASS）

- **B**：真实数据、真实训练输出下，两条路径的每窗父/候选 loss **精确相等**。
- **C**：独立调用计数——每窗 Attention 前向 4→3、参考解码 12→8、hif4 解码 6→5、V 量化 2→1，
  **Q/K 保持 2/2**。少的是臂无关的量，臂相关的量一个没动。
- **D**：六个真实 attention 层全量校准，q/k/v state **逐字节相同**、审计字段相等。
- **E**：接受（层 0/22）与拒绝（层 1/5/8/15）真实出现；M=I（层 8 天然 identity + 补丁探针）、
  ineligible、异常回退各一致。
- **F**：装配自比逐字节相同。

## 5. GPU 评测

父侧用 v236 归档（**对照是同父 A-GR1 旧实现**，不是根——这样"处处为零"才是本卡的预期读数）。

- **shard0**：`records=1`、`reasonableness_issues: 0`、12 例 `mean_delta_gain=0.0`（0/0/12）。
- **六 shard**：`{'protocol': 'eval-v3', 'records': 6, 'reasonableness_issues': 0}`。

### 5.1 `stopped_early: true` —— 显式标注，并说明它不是截断

与 L-TF2 同型：停止检查在 shard 结果 append **之后**（`eval_system.py:487` 与 `:509`），
本候选每片 `delta_mean` 恰为 0 ⇒ 全部 nonpositive ⇒ 计数器在**最后一个被请求的 shard** 上触顶。
**没有丢帧**：`results` 6 条、全 `ok`、各 12 例、合计 **72**，`analysis-*.json` 6 份。
属纪律的"显式标注"处置，需补齐的 shard 一个都没有。

### 5.2 逐例配对（`pair_sixshard.py`）

六 shard 全部 12/12 恰好零，合计 **72 例 0/0/72**；脚本核对两侧 `source_sha256`
（候选 `146bb715…` 跨 shard 一致、基线 `3319fc35…` 即 v236）。

### 5.3 对正式父的变化：继承而非复测

因候选与 v236 逐例为零，A-GR1 相对根的变化原样继承。本次把 v236 的归档六 shard
（`artifacts/proxy_v3/attention-agr1-on-v231-sixshard-20260910`，基线 `ea79a1c1…` 即正式父）
**按 `case_id` 重新逐例核算**核对：等权 `+0.003845`、**21/3/48**、层 15 `+0.017449`、层 22 `+0.005618`
—— 与 v236 归档一致。这一步是传递性的地基，所以它被 `archive.py` 设为归档的**前置断言**。

### 5.4 `api_total` 与校准秒数不可比（只记录）

两侧受缓存 identity 影响（候选 SHA 新、基线命中），按 AGENTS §5 不跨缓存状态比较，记 0 也不解读。

## 6. 计时（`timing.py`）：改动段可分辨，整调用边际

三臂配对（assembly / candidate / **同字节 sham**），15 轮，轮转顺序，CUDA event 无 profiler。
计划要求"计时区分父校准、训练、gate"，所以分两段测：

| 段 | 层 | assembly 中位 | 候选效应 | 占父 | null | \|effect\|/\|null\| | 更快轮数 |
|---|---|---:|---:|---:|---:|---:|---:|
| gate 窗口对 | 0 | 74.658 ms | **+13.514 ms** | **+18.10%** | −0.883 ms | **15.30×** | **15/15** |
| gate 窗口对 | 22 | 71.057 ms | **+9.749 ms** | **+13.72%** | −0.586 ms | **16.62×** | **15/15** |
| 整校准 | 0 | 6034.325 ms | +34.973 ms | +0.58% | −15.988 ms | 2.19× | 10/15 |
| 整校准 | 22 | 6022.831 ms | +33.417 ms | +0.56% | −22.700 ms | 1.47× | 11/15 |

**本卡改动的那一段干净可分辨（15–17× null，15/15 轮一致）；整调用上效应同量级但只有 null 的
1.5–2.2 倍，属边际。** 照实记两段，不取好看的那一半。显存峰值三臂完全相同（587.1 / 614.6 MiB）。

## 7. 与同日其他回传的关系

- **v235 官方 TIMEOUT**（本轮回传，[记录](2026-09-10-v235-linear-official-timeout.md)）：按计划 §3
  "保留其实际裁决，不自动移植；L-TF2 独立裁决"，本卡与 v235 无继承关系；
- **A-GR1（v234）的官方记录各自保留**：侧隔离 `+29` 与完整包 TIMEOUT。计划 §5 原文
  "v236若超时，只关闭原执行结构，不自动关闭已实质削减计算的A-CT1"——本卡正是"已实质削减计算"的那一类；
- 不把 `+29`、`+0.003845`、本地百分比或秒数相加或换算。

## 8. 裁决

**LOCAL：输出与同父 A-GR1 **逐位等价**（72/72 精确零、六层 state 逐字节相同）、gate 段执行工作实质减少、
改动段墙钟 15–17× 于空对照。官方 `unregistered/NA`。**

计划 §4 开发裁决原文："合法、保持A-GR1输出/接受逻辑且实际减少执行工作的完整候选：归档一个正式代表，
官方状态未知；无须等v236结果才归档或交付。" 本卡照此归档。

## 9. 产物

- 工具：`workbench/full_solution/attention-act1-gate-reuse/`（`parent.json` / `build.py` / `verify.py` /
  `timing.py` / `pair_sixshard.py` / `archive.py`）；
- 证据：`build.json`、`verify.out`、`timing.json`、`timing.out`、`paired_sixshard.json`；
- 评测：`artifacts/proxy_v3/attention-act1-shard0-20260910/`、`artifacts/proxy_v3/attention-act1-sixshard-20260910/`；
- 归档：`solutions/20260910_v238_attention-act1-gate-reuse_scoreNA_timeNA/`；
- 官方字段未知记 `null`。
