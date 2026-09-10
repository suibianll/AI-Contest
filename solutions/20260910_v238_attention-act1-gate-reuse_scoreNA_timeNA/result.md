# v238 — Attention A-CT1：A-GR1 gate 的固定计算复用

活动计划 [`docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md) §4。
执行记录 [`logs/execution/2026-09-10-attention-act1-gate-reuse.md`](../../logs/execution/2026-09-10-attention-act1-gate-reuse.md)。

| | |
|---|---|
| 正式父 | v231 完整根（Linear L-EM3 K=2 + v195 Attention），官方 **18518 / 291 s**（余量 9 s） |
| 正式父 SHA256 | `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1` / 505762 B |
| 装配来源 | v236（= 该根 + A-GR1，逐字节验证）`3319fc35…` / 520955 B，**非晋级父，不继承其任何结果** |
| 候选 SHA256 | `146bb7151f5f2a041b2f1fdbc94e5b370815fb91fa94d58db79384bb7a54fbc7` / 528204 B |
| 改动 | 一次块替换：每窗两次 `_agr1_gate_loss` 调用 → 一次 `_act1_gate_pair`（6 行，−136 B） |
| 本地六 shard | 对**同父 A-GR1 旧实现 v236**：**72/72 精确零**（0/0/72） |
| 计时 | gate 段 **+18.10% / +13.72%（空对照的 15.3× / 16.6×，15/15 轮）**；整校准 **+0.58% / +0.56%（2.19× / 1.47×，边际）** |
| 显存峰值 | 三臂**完全相同**（587.1 / 614.6 MiB）——没有用内存换时间 |
| 官方 | **`unregistered/NA`**，待用户统一评测。不写本地秒数预测 |

## 机制：gate 对每个窗口把与臂无关的工作做了两遍

A-GR1 的 gate 是这两个连续调用：

```python
for index in _AGR1_GATE_WINDOWS:
    parent_loss = _agr1_gate_loss(calib_qkv_list[index], states, ...)
    candidate_loss = _agr1_gate_loss(calib_qkv_list[index], candidate, ...)
```

而 `_agr1_gate_loss` 每次调用都要：从窗口解出 dense 参考 Q/K/V、量化并解码该臂的 Q/K/V、
算两次 Attention 前向（一次 actual、一次 reference `target`），最后返回两者 MSE。

**两臂的差别只有 Q/K 的 rotation 与 K 的 center**：

```python
candidate = _agr1_parent_copy(states)
candidate["q_state"]["learned_rotation"] = tq
candidate["k_state"]["learned_rotation"] = tk
candidate["k_state"]["learned_center"]   = center   # 存在时
```

`v_state` 是拷贝、值相等；V API 对 state **只读**（`_check_attention_state` 只做 `.get` 后原样返回）。
所以 dense 参考 Q/K/V、参考 `target`、父侧 V 五字段**都与臂无关**——每个窗口的 4 次 Attention 前向里有 3 次、
2 次 V 量化里有 1 次，是在重算不随臂变的量。

候选把它换成一次调用：

```python
for index in _AGR1_GATE_WINDOWS:
    parent_loss, candidate_loss = _act1_gate_pair(
        calib_qkv_list[index], states, candidate,
        q_num_heads, kv_num_heads, head_dim,
    )
```

`_act1_gate_pair` 只算一次参考解码、一次 `target`、一次父 V，然后**两臂各自**量化/解码自己的 Q/K、
各自做一次 Attention 前向与同一个 `target` 比较。**每窗的父/候选 loss、逐窗 `candidate_loss < parent_loss`、
跨窗 AND、info 字段与最终 state 编译全部是 A-GR1 的代码，逐字节未动。**

缓存只活在一次调用内：无全局输入缓存、不写入任何部署 state、**不缓存随 M 变化的东西**（Q/K 解码每臂重做，
那正是两臂分歧的部分）。

### 为什么 V 可以复用，而 Q/K 不能

V 的 state 在两臂之间值相等，且 V API 不改写它——所以"父臂的 V"就是"候选臂的 V"，**定义上相等**。
Q/K 不同：`learned_rotation`/`learned_center` 正是两臂唯一的差异，复用它们会把 gate 变成恒真。
这三条不是论证出来的，是**测出来的**：

| 前提 | 测法 | 结果 |
|---|---|---|
| P1 两臂只差 Q/K | 逐字段比对两臂 state | 差异字段恰为 `q_state.learned_rotation`、`k_state.learned_rotation`、`k_state.learned_center`；**v_state 逐字节相同** |
| P2 V 是参数函数 | 等值 state 调两次 V API | 五字段**逐位相同** |
| P3 API 不改 state | 调用前后逐字段比对 | Q/K/V 过后 state 逐字节不变 |

若 P1–P3 任一不成立，本卡就必须收缩到"只复用参考 `target`"，不能用父臂的 V——计划对此有明确要求。

## 构建证据（`build.py` / `build.json`）

候选 = **v236 装配字节 + 追加模块**，父字节一个未动；影子的 `hif4_calibration_attention` 在调用时
从模块全局解析，自然覆盖前面那份。

- 装配与正式父的 SHA/字节数都按记录核对；`cmp` 确认装配前 505762 B 就是 v231 根；
- 被追加的函数**从装配文本派生**：切出 A-GR1 的 `hif4_calibration_attention`，做**恰好一次**块替换，再追加；
- 字节级：改动 6 行、**−136 B**；替换前先确认该块唯一、且替换后形态不存在；
- 语法级：把 `_act1_gate_pair` 调用**还原成两次 `_agr1_gate_loss`** 后，与 A-GR1 函数的 AST **逐节点相同**；
- 线性侧与控制无关的 Attention 面：`hif4_calibration_and_quantize_weight` / `hif4_dynamic_quantize_activation`
  与根**字节码相同**；`hif4_dynamic_quantize_q/k/v` 与装配字节码相同。

## 等价性（`verify.py` / `verify.out`，CPU）

**逐位，不是容差。** 在真实校准数据（4B 面板 pack 的 `calibration_qkv`，5 窗，即评测器自己喂的形状）上：

- **`[B]` 每个 gate 窗口的父 loss 与候选 loss 在两条路径之间精确相等**（层 0 两窗；`float` 精确比较）。
- **`[D]` 六个真实 attention 层（0/1/5/8/15/22）全量校准**：返回的 q/k/v state **逐字节相同**、
  `agr1_*` 审计字段全部相等。
- **`[F]`** 装配自比逐字节相同，所以上一条不是运行间差异的产物。

**覆盖**（计划 §4 要求接受/拒绝/M=I/异常回退）：

| 情形 | 来源 | 结果 |
|---|---|---|
| 接受 | 真实层 0、22（`agr1_arm='accepted'`） | 一致 |
| 拒绝 | 真实层 1、5、8、15（`agr1_arm='parent'`） | 一致 |
| M = I | 层 8 真实出现 `agr1_parent_arm='identity'`；另用补丁探针强制 | 一致 |
| ineligible | 2 窗列表（不足 `max(gate_windows)+1`） | 一致 |
| 异常回退 | 两模块同样打补丁令训练抛 `ValueError` | 一致（`agr1_arm='fallback'`，错误串相同） |

## 调用计数（`[C]`，独立计数器）

对模块自身的名字计数，每**一个 gate 窗口**的新旧调用数：

| 量 | A-GR1 | A-CT1 | 变化 |
|---|---:|---:|---:|
| `_a2_attention_forward` | 4 | 3 | **−1** |
| `_dequantize_nvfp4_float32` | 12 | 8 | **−4** |
| `_dequantize_hif4` | 6 | 5 | **−1** |
| V 量化（`_AGR1_PARENT_V`） | 2 | 1 | **−1** |
| **Q 调用** | 2 | **2** | **0** |
| **K 调用** | 2 | **2** | **0** |

`_dequantize_nvfp4_float32` 减 4 而非 3，是因为 V 量化器内部也会调它：3 个参考解码，
另加 1 个随 V 调用数下降而省下的内部解码。旧两窗配置把每个数翻倍：少 2 次参考 Attention 前向、
2 次 V 量化、2 组参考解码。**Q/K 必须保持 0 变化**——它们是臂相关的部分，减掉就把 gate 变成恒真。

## 计时（`timing.py` / `timing.json`）：区分整调用与 gate，带 sham 空对照

三臂配对（`assembly` / `candidate` / **同字节 sham**），15 轮，顺序轮转，CUDA event，无 profiler。

| 段 | 层 | assembly 中位 | 候选配对效应 | 占父 | null(sham) | \|effect\|/\|null\| | 候选更快轮数 |
|---|---|---:|---:|---:|---:|---:|---:|
| **gate 窗口对** | 0 | 74.658 ms | **+13.514 ms** | **+18.10%** | −0.883 ms | **15.30×** | **15/15** |
| **gate 窗口对** | 22 | 71.057 ms | **+9.749 ms** | **+13.72%** | −0.586 ms | **16.62×** | **15/15** |
| 整校准调用 | 0 | 6034.325 ms | +34.973 ms | +0.58% | −15.988 ms | 2.19× | 10/15 |
| 整校准调用 | 22 | 6022.831 ms | +33.417 ms | +0.56% | −22.700 ms | 1.47× | 11/15 |

**本卡改动的那一段是干净可分辨的**：15–17 倍于空对照、15/15 轮方向一致。整调用上效应同量级
（33–35 ms ≈ 两窗 × 单窗 10–14 ms）但只有空对照的 1.5–2.2 倍——6 秒级调用上的热漂移把 null 抬到了
16–23 ms，**这一段只能记作边际，不能记作结论**。显存峰值三臂完全相同。

## 六 shard 本地结果（`pair_sixshard.py` / `paired_sixshard.json`）

`--attention-only` 六 shard，基线 **v236（同父 A-GR1 旧实现）**，逐例重算 `candidate.gain − baseline.gain`：

| shard | 0 | 1 | 2 | 3 | 4 | 5 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 12 | 12 | 12 | 12 | 12 | 12 | **72** |
| Δ mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **+0.000000** |
| +/-/0 | 0/0/12 | 0/0/12 | 0/0/12 | 0/0/12 | 0/0/12 | 0/0/12 | **0/0/72** |

**处处为零是这张卡的预期读数，不是筛不过**：基线就是它要复现的实现，非零才是缺陷。
脚本核对两侧 `source_sha256`（候选 `146bb715…` 跨 shard 一致、基线 `3319fc35…` 即 v236）。

### `stopped_early: true` —— 显式标注，并说明它不是截断

- 停止检查在**当前 shard 结果 append 之后**执行（`evaluator/eval_system.py:487` 与 `:509`）；
- 本候选每片 `delta_mean` 都恰好 0 ⇒ 全部计入 nonpositive ⇒ 计数器**在最后一个被请求的 shard** 上
  刚好触到 `--stop-after-nonpositive 6`。

**没有丢帧**：`results` 6 条、全部 `status ok`、各 12 例、合计 **72**，`analysis-*.json` 6 份。
属纪律里的"**显式标注**"处置，需补齐的 shard 一个都没有；`verification.json → stopped_early_flag`
记录了独立覆盖核验。

## 对正式父的实际输出变化（继承，非复测）

本候选与 v236 在 72 例上逐例为零，**因此 A-GR1 相对 v231 根的输出变化原样继承**。
该读数来自 v236 自己的归档六 shard（`artifacts/proxy_v3/attention-agr1-on-v231-sixshard-20260910`，
基线 SHA `ea79a1c1…` 即正式父），本次按 `case_id` 重新逐例核算确认：

- 等权 **`+0.003845`**，**21 / 3 / 48**（正/负/零）；
- 层 15 `+0.017449`、层 22 `+0.005618` 接受；层 0/1/5/8 gate parent 逐位不变。

即：**这张卡不产出新的机制证据**，它改的是同一个机制的执行成本。

## 边界（这份结果不主张什么）

- **V 的复用依赖 P1–P3**，而 P1–P3 是在本根、本 pack、本数据上**实测**的，不是对任意输入成立的恒等式。
  若换根或换标定数据后两臂的 `v_state` 不再等值，复用即失效——这正是 `verify.py` 把前提单独成组的原因。
- **整校准调用上的时间收益是边际的**（1.5–2.2× null）。本卡的时间结论只由 gate 段的 15–17× null 支撑，
  整调用读数照实记为边际，不取其中好看的那一半。
- **本地秒数不是官方秒数。** 官方 300 s 是唯一时间门；不换算、不预测。
- **A-GR1 的官方记录不被本卡继承**：侧隔离 `+29`（14455/263.7s）与完整包 TIMEOUT 两个事实各自保留。
  计划 §5 原文："v236若超时，只关闭原执行结构，不自动关闭已实质削减计算的A-CT1"。
  本次回传的 v235 超时同样不改变本卡定位。
- **六 shard 的时间读数不构成时间证据**：本轮 `api_total` 与校准秒数两侧都受缓存 identity 影响
  （候选 SHA 新、基线命中），按 AGENTS §5 只记录不解读。
- 等价性覆盖六个真实 attention 层（面板的全部 FA 层）× 5 窗，加上合成与补丁覆盖的各条分支；
  没有扫其他面板或别的根。

## 复现

```powershell
.venv\Scripts\python.exe workbench/full_solution/attention-act1-gate-reuse/build.py
.venv\Scripts\python.exe workbench/full_solution/attention-act1-gate-reuse/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attention-act1-gate-reuse/candidate/solution.py --baseline-solution solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/attention-act1-shard0-20260910
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attention-act1-gate-reuse/candidate/solution.py --baseline-solution solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py --attention-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 6 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/attention-act1-sixshard-20260910
.venv\Scripts\python.exe workbench/full_solution/attention-act1-gate-reuse/pair_sixshard.py
.venv\Scripts\python.exe workbench/full_solution/attention-act1-gate-reuse/timing.py
```
