# v237 — Linear 首遍梯度复用 (L-TF2)

活动计划 [`docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md) §2。
执行记录 [`logs/execution/2026-09-10-linear-tf2-retained-root-gradient-reuse.md`](../../logs/execution/2026-09-10-linear-tf2-retained-root-gradient-reuse.md)。

| | |
|---|---|
| 父根 | v231 Linear (L-EM3 K=2) + v195 Attention，官方 **18518 / 291 s**（余量 9 s） |
| 父 SHA256 | `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1` / 505762 B |
| 候选 SHA256 | `ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554` / 516697 B |
| 改动 | 一次子串替换，循环头加 `if _pass:`（5 行，+34 B） |
| 本地六 shard | 等权 `+0.000000`，**0 / 0 / 336**，六个 shard 的 min 和 max 都是 `0.0` |
| 时间 | 每次下降调用**少 2 个矩阵乘**（实测 38 → 36）；本地墙钟效应**低于本机分辨力**，带同字节 null 对照记录 |
| 官方 | **18518 / 289s，RETAINED，晋级为当前完整根**（2026-09-10 用户回传）——同分、比父 v231 快 **2s**、余量 9s→**11s** |

> **官方结果（2026-09-10 用户回传）：18518 / 289s，RETAINED，已晋级为当前完整根。**
> 事前固定的晋级规则两条同时满足：计划 §2"相对届时当前根同分更快且 `<300s`：晋级"，
> 以及 AGENTS §2 的 v202 先例"同分的实质提速候选可在官方更快且 `<300s` 时晋级，**只在其分数不低于当前根时**"
> ——分数同为 18518（不低于），289 < 300。见 [`official-result.json`](official-result.json) 与
> [官方结果记录](../../logs/execution/2026-09-10-v237-linear-official-result.md)。
>
> **本卡的时间读数全部是否定的，而官方仍量出 −2s。** 三臂配对四行 |effect|/|null| 为
> 0.52/0.52/1.11/1.00×、符号行间翻转、更快轮数 10–17/31。计划 §2 步骤 6 原文
> "本地分辨不出差异不否决正式成本验证"在此第三次被验证（v233 的 −4s 是第二次）。
> 本仓库在提交建议阶段曾据本地读数推出"v237 大概率超时、不值得占名额"，**该推断已被本次回传证伪**，
> 更正记在官方结果记录里。下文（含"官方 unregistered/NA，不写本地秒数预测"的提交时措辞）保留为
> **提交时**的记录，不追改。

## 机制：父根把同一个梯度算了两遍，第一遍的结果被丢掉

`_em1_dynamic_descent` 在进入 pass 循环之前先算一次梯度，**只为做有限性检查**：

```python
gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
if not torch.isfinite(gradient).all():
    diagnostics["em1_dynamic_arm"] = "nonfinite-gradient"
    return result
```

进入循环后，**每一个 pass 的第一条语句又把它重算一遍**，把循环前那个张量覆盖掉：

```python
for _pass in range(_EM1_PASSES):
    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)   # pass 0 时结果被丢弃
    if not torch.isfinite(gradient).all():
        aborted = True
        break
```

两次求值是**同一个表达式、同一组操作数**，所以循环前那次的结果可以直接复用：

```python
for _pass in range(_EM1_PASSES):
    if _pass:
        gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
        if not torch.isfinite(gradient).all():
            aborted = True
            break
```

### 为什么是 `if _pass:` 而不是把表达式提出去

因为 pass body 会**原地改写**这两个张量：

```python
deployed.add_(row_delta.mul_(keep_scale))
gradient.add_(g_delta.mul_(keep_scale))
```

从 pass 0 结束后，`gradient` 建立时所用的张量已经不存在了，**pass 1 起必须照父的原地顺序重算**。
所以复用只能止于 pass 0——守卫写成 `if _pass:`，而不是整段提升到循环外。

### 与 v233（L-TF1）的关键区别：K 不同

v233 的父是 **K = 1**，守卫分支**永不执行**，那次只证明了"能省"。本卡的父是 **K = 2**，
**pass 1 会真正走进这个分支**，所以这张卡证明的是"省了，而且后续 pass 行为未变"。
算子账也随之不同：**38 → 36**（每次下降调用），不是 v233 的 20 → 18。

## 构建证据（`build.py` / `build.json`）

候选是**父根字节 + 追加的一个模块**，父根字节一个未改：追加的定义在调用时从模块全局解析，
自然覆盖前面那个，不需要 hook。

- 父 SHA256 与字节数符合预期（`ea79a1c1…` / 505762）；
- 被追加的模块**由父自身的函数源码派生**，不是手写：切出父的 `_em1_dynamic_descent`，
  做**恰好一次**子串替换（循环头 → 带守卫的循环头），再追加；
- 字节级：改动 5 行、+34 B；执行前先确认该子串在父函数里**唯一**、且替换后形态不存在；
- 语法级：把插入的 `if _pass:` 节点**溶解回 pass body** 后，候选的 AST 与父**逐节点相同**
  （`ast_identical_after_dissolving_pass_guard: true`）——这比"改动了几行"更强，
  它说明被执行语句树除守卫外未变；
- 候选里 `def _em1_dynamic_descent(` 出现 2 次（父的 + 影子），调用点 `corrected = ...` 恰好 1 次。

## 等价性与算子账（`verify.py` / `verify.out`，CPU）

七组控制，全部 PASS。要点：

| 控制 | 内容 | 结果 |
|---|---|---|
| A | 血统与唯一改动 | 候选以父字节为前缀（505762 B）；**Attention 四个 API 字节码相同**；影子定义确实是活的那个（`co_firstlineno` 判定）；shipped **K=2** 被断言而非读回 |
| B | 输出等价（逐字节） | 合成层、真实 layer0/q 128 行、无 `em1` payload、超范围旁路——**五字段全部逐位相同**（比对原始字节，非容差） |
| C | 前提实测 | 循环前 vs pass-0 的**两个矩阵积逐位相同**（合成 + 真实），求和后梯度亦相同 |
| D | 调用计数 | K=2：父 **38** 个 `mm`，候选 **36**；候选序列 = 父**去掉第 2、3 个积**、其余不变 ⇒ **pass 1 的梯度积仍在** |
| E1 | 循环前 NaN 梯度 | 双方同走 `nonfinite-gradient`，输出逐字节相同 |
| E2 | **pass 1 NaN 梯度** | 父 22 个积（中止于 index 21）、候选 20 个（中止于 index 19）——双方都在第二遍任何分组步之前停下，输出逐字节相同 |
| F | 确定性 / state 往返 / K=1 回归 | 全部成立；K=1 探针只在内存里改并在结束后校验已恢复 |

`premise.py` 另作**静态**前提确认，结论由 AST 算出而非断言：P1 两次梯度之间**无值破坏性改写**
（唯一一条是 `deployed = deployed.clone()`，逐元素保持）；P2 梯度是 `_pass` 循环的首条语句；
P4 pass body 确实原地改写了 `deployed`/`gradient`——这正是复用必须止于 pass 0 的原因。

E2 在本卡不是探针而是**发布配置**：K=2 就是根实际运行的 K，守卫的中止分支就是这条。

## 计时（`timing.py` / `timing.json`）

三臂同状态同输入配对，31 轮，`torch.cuda.Event`，不挂 profiler、不重放，每臂先无计时预热，
顺序每轮轮转。`sham` 臂是**同一份归档再加载一次**、与 parent 逐字节相同——它与 parent 的配对差
就是"什么都没改时本方法会报出的数"，是本机此刻的**经验零假设**。

| 行 | parent 中位 | 候选配对效应 | null(sham) | \|effect\|/\|null\| | 候选更快轮数 | null 更快轮数 |
|---|---:|---:|---:|---:|---:|---:|
| layer0/q 2560ch, 128 行 | 532.090 ms | −4.106 ms | −7.774 ms | **0.53×** | 13/31 | 13/31 |
| layer0/q 2560ch, 512 行 | 889.200 ms | −0.757 ms | +1.467 ms | **0.52×** | 13/31 | 17/31 |
| layer0/o 4096ch, 128 行 | 483.432 ms | −10.071 ms | −9.038 ms | **1.11×** | 10/31 | 8/31 |
| layer0/o 4096ch, 512 行 | 559.735 ms | +9.505 ms | +9.527 ms | **1.00×** | 17/31 | 23/31 |

**四条行的效应都落在 null 之内**（0.52–1.11×），符号在行与行之间翻转，更快轮数是抛硬币。
配对差的离散也远大于任何效应：候选侧 p25–p75 分别跨 −21.6…+12.5、−13.6…+12.0、
−17.3…+10.7、−11.2…+28.2 ms。

**结论是"本机分辨不出"，不是"没有收益"。** 算子确实少了 2 个（控制 D 有实测序列），
但这点工作量在 480–890 ms 的调用里低于本机噪声。不写官方秒数预测：
官方 300 s 硬限是唯一时间门，本地读数不换算官方时间。

## 六 shard 本地结果（`pair_sixshard.py` / `paired_sixshard.json`）

`--linear-only` 六 shard，单进程双侧，按 `case_id` 逐例重算 `candidate.gain − baseline.gain`：

| shard | 0 | 1 | 2 | 3 | 4 | 5 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 56 | 56 | 56 | 56 | 56 | 56 | **336** |
| Δ mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **+0.000000** |
| +/-/0 | 0/0/56 | 0/0/56 | 0/0/56 | 0/0/56 | 0/0/56 | 0/0/56 | **0/0/336** |

`min = max = 0.0`，按 role 全 `0.0`，`reasonableness_issues: 0`。
配对脚本核对两侧记录的 `source_sha256`：候选侧 `ecb1f9e5…` 跨 shard 一致，
基线侧 `ea79a1c1…`（即 v231 根）——这排掉了"基线其实是别的根"这类污染。

**这不是容差口径。** 这里报的是评测器自己的配对统计量（不是逐位比较），而它在 336 个 case 上
**处处精确落在零**；逐位的证据在同目录的控制 B（原始张量逐字段比字节）。

### 关于 `stopped_early: true`：逐条说明，不是默认值无害

manifest 里 `stopped_early` 为 **true**，必须显式标注并解释，不能当作没发生：

- 计数器的停止检查在**当前 shard 的结果 append 之后**执行（`evaluator/eval_system.py:487` vs `:509`）；
- 本卡输出逐位等价 ⇒ 每个 shard 的 `delta_mean` 都恰好 0 ⇒ 全部计入 nonpositive ⇒
  计数器在**最后一个被请求的 shard** 上刚好触到 `--stop-after-nonpositive 6`；
- 因此这次 **没有截断**：`results` 6 条、全部 `status ok`、各 56 例、**合计 336**，
  `analysis-*.json` 6 份，六个 shard 一个不少。

即：这是"等零候选 + 该参数"的**结构性读数**，不是丢帧。独立核验记录在
`verification.json → stopped_early_flag`。

## 时间侧的另一条读数（只作诊断）

`api_total` 两侧**不可比**：候选 SHA 是新值、标定缓存冷，基线侧命中缓存。
六 shard 候选 `42.1/169.5/172.5/169.5/171.0/171.6 s` vs 基线 `43.7/42.1/41.3/41.6/41.6/43.1 s`
——差值主要是候选侧 shard 1–5 未命中的权重标定，不是算法时间。按 AGENTS §5
**不跨缓存状态比较 `api_seconds`**，此处只记录、不解读。

## 边界（这份结果不主张什么）

- **本地秒数不是官方秒数。** 上面的毫秒是本机 CUDA event 读数，官方 300 s 硬限是唯一时间门。
- **本次回传背景（重要）**：同一根上的 v232、v234、v235 均已官方 TIMEOUT，
  其中 **v235 是本轮唯一在本地测出可分辨提速的候选（34–52× null）却仍超时**
  （[超时记录](../../logs/execution/2026-09-10-v235-linear-official-timeout.md)）。
  本卡的效应比 v235 小一到两个数量级、本机分辨不出，因此**不主张它能在官方计时上产生可分辨差异**。
  它作为"实质执行成本变化的纯提速候选"按 AGENTS §2 可作一次官方验证，是否提交由用户决定。
- **不是"整模型省 2 个矩阵乘"**：省下的是每次**进入下降的**动态激活调用 2 个 `mm`；
  未进入下降的调用（no-metric / 非有限回退 / 超范围）一分不省，本卡也不声称省。
- 等价性证据覆盖：合成层 + 真实 layer0/q 128 行 + 无 payload + 超范围旁路，以及 K=1 回归。
  **没有**扫全部 24 层全部 role——主张的根据是"两次求值是同一表达式、同一操作数、
  且中间无值破坏性改写"这条对任意输入成立的论证，实测是它的对照，不是它的替代。
- 六 shard 的时间读数没有 null 臂，**不构成时间证据**；时间结论只由三臂配对支撑（且结论是"分辨不出"）。

## 复现

```powershell
.venv\Scripts\python.exe workbench/full_solution/linear-tf2-retained-root-gradient-reuse/build.py
.venv\Scripts\python.exe workbench/full_solution/linear-tf2-retained-root-gradient-reuse/premise.py
.venv\Scripts\python.exe workbench/full_solution/linear-tf2-retained-root-gradient-reuse/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/linear-tf2-retained-root-gradient-reuse/candidate/solution.py --baseline-solution solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py --linear-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/linear-tf2-shard0-20260910
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/linear-tf2-retained-root-gradient-reuse/candidate/solution.py --baseline-solution solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 6 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/linear-tf2-sixshard-20260910
.venv\Scripts\python.exe workbench/full_solution/linear-tf2-retained-root-gradient-reuse/pair_sixshard.py
.venv\Scripts\python.exe workbench/full_solution/linear-tf2-retained-root-gradient-reuse/timing.py
```
