# L-TF2：在保留的 K=2 根上复用首遍梯度（v237）

计划卡：[`docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md) §2。
归档：[`solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/`](../../solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/result.md)。
工作目录：`workbench/full_solution/linear-tf2-retained-root-gradient-reuse/`。
设备：NVIDIA GeForce RTX 3060 Ti。

本日志记**过程与判定**；逐项证据表在归档的 `result.md`，不在此重复。

## 0. 开工前置核验

| 项 | 值 |
|---|---|
| 当前根 | v231 Linear (L-EM3 K=2) + v195 Attention，官方 **18518 / 291 s**，余量 9 s |
| 根 SHA256 | `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1` / 505762 B |
| 工作区 `solution.py` | 与归档**逐字节相同**（同 SHA），已登记进 `parent.json` |
| 回退根 | v233 `0ec89710…`，18428 / 288 s |

**父绑定不可变归档，不绑定工作区文件。** 前一轮的 A-GR2 日志记过"工作区根被并行 Linear 会话改脏，
未用作父或 baseline"。本卡把这个风险按构造消除：`build.py` / `premise.py` / `verify.py` / `timing.py`
一律从 `solutions/20260910_v231_.../solution.py` 读父，工作区 `solution.py` 只在核验时被读一次
（`build.json` 里记 `live_root_matches_archive`），**不作为任何构建或对照输入**。

## 1. 机制与改动点

`_em1_dynamic_descent` 里同一个梯度算两遍：循环前算一次（做有限性检查），pass 循环头又算一次
（pass 0 时覆盖掉前一次的结果）。改动是把循环头重算连同其有限性检查放进 `if _pass:`。

守卫**不能**写成"把表达式提到循环外"，因为 pass body 原地改写 `deployed` 和 `gradient`
（`deployed.add_` / `gradient.add_`），pass 1 起必须照父的原地顺序重算。这一点由 `premise.py` 的
P4 从 AST 算出，不是读代码的印象。

**与 v233 的实质差异是 K**：v233 父为 K=1、守卫分支永不执行；本卡父为 K=2、pass 1 真正走进该分支。
算子账由 20 → 18 变为 **38 → 36**。计划 §2 明确"不能把 v233 的 K=1 测试当作该分支的验证"，
本卡的验证因此把 K=2 当作**发布配置**来测（控制 D / E2），并把 K=1 降为回归探针（控制 F）。

## 2. 构建（`build.py` → `build.json`）

追加式：候选 = 父字节（505762 B，逐字节保留）+ 追加模块（10933 B），总计 516697 B。
追加的模块**从父的函数源码派生**——切出父的 `_em1_dynamic_descent`，做恰好 1 次子串替换（+34 B / 5 行），
再追加；`HEADER + 函数` 的拼接被断言拆得开，替换前先确认子串在父函数里唯一。
语法级：溶解插入的 `if _pass:` 后 AST 与父逐节点相同。

候选 SHA256 `ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554`。

## 3. CPU 验证（`premise.py` / `verify.py`）

`premise.py`：P1（两梯度之间无值破坏性改写，唯一是 `deployed = deployed.clone()`）、
P2（梯度是 `_pass` 循环首语句）、P4（pass body 原地改写 ⇒ 复用止于 pass 0）全成立，
并打印父模块的 `_EM1_PASSES`=2 与"守卫在此根上是活路径"的结论。

`verify.py` 七组控制全 PASS，要点：

- **D（本卡的核心读数）**：K=2 父 38 个 `mm`、候选 36 个；候选序列 = 父去掉第 2、3 个积。
  这一条同时证明两件事——**省了 2 个**，且 **pass 1 的梯度积仍在**（没有把重算错误地提出循环）。
- **E2**：pass 1 注入非有限梯度时，父 22 个积（中止于 21）、候选 20 个（中止于 19），
  双方都在第二遍任何分组步之前停下、输出逐字节相同。E2 在本卡是发布配置而非探针。
- **B**：合成层 / 真实 layer0/q 128 行 / 无 `em1` / 超范围旁路，五字段**逐字节**相同。
- **A**：Attention 四 API 字节码相同（未改子系统 control）；shipped K=2 被**断言**而非读回。
- **F**：确定性、state 往返、K=1 回归探针（内存内改动并校验恢复）。

## 4. GPU 评测

按 [4B 指引](../../docs/4b-panel-testing-guide.md)：先 shard0 排接口，再六 shard 一次。
父侧一律用归档路径（字节与工作区根相同，但不受并行会话影响）。

- **shard0**：`records=1`、`reasonableness_issues: 0`、56 例 `mean_delta_gain=0.0`（0/0/56）。
- **六 shard**：`{'protocol': 'eval-v3', 'records': 6, 'reasonableness_issues': 0}`。

### 4.1 `stopped_early: true` —— 显式标注，并说明它不是截断

manifest 的 `stopped_early` 为 **true**，必须交代而不是绕开：

- 停止检查在**当前 shard 结果 append 之后**执行（`evaluator/eval_system.py:487` 与 `:509`）；
- 本卡输出逐位等价 ⇒ 每 shard 的 `delta_mean` 恰好 0 ⇒ 全部计入 nonpositive ⇒
  计数器**在最后一个被请求的 shard 上**刚好触到 `--stop-after-nonpositive 6`。

所以**没有丢帧**：`results` 6 条、全部 `status ok`、各 56 例、合计 **336**，`analysis-*.json` 6 份。
这不是"默认值无害"的推断，而是逐条核验过的结构性读数，并已写进
`verification.json → stopped_early_flag`（含独立覆盖核验：六个候选 JSON 的 `source_sha256` 一致）。
按纪律处置属于"**显式标注**"这一条——需要补齐的 shard 一个都没有。

### 4.2 逐例配对（`pair_sixshard.py`）

六个 shard 全部 **56/56 恰好零**，合计 **336 例 0/0/336**，`min = max = 0.0`，逐 role 全 0。
脚本核对两侧 `source_sha256`（候选 `ecb1f9e5…` 跨 shard 一致、基线 `ea79a1c1…` 即 v231 根）。

### 4.3 `api_total` 两侧不可比（只记录）

候选侧 `42.1/169.5/172.5/169.5/171.0/171.6 s`，基线侧 `43.7/42.1/41.3/41.6/41.6/43.1 s`。
差值主要来自候选 SHA 是新的、shard 1–5 的权重标定未命中缓存。按 AGENTS §5
**不跨缓存状态比较 `api_seconds`**，此条只作诊断记录。

## 5. 计时（`timing.py`）：本机分辨不出，带 null 对照

三臂配对（parent / candidate / **同字节 sham**），31 轮，顺序轮转，CUDA event，无 profiler。

| 行 | parent 中位 | 候选效应 | null | \|effect\|/\|null\| |
|---|---:|---:|---:|---:|
| layer0/q 128 行 | 532.090 ms | −4.106 ms | −7.774 ms | 0.53× |
| layer0/q 512 行 | 889.200 ms | −0.757 ms | +1.467 ms | 0.52× |
| layer0/o 128 行 | 483.432 ms | −10.071 ms | −9.038 ms | 1.11× |
| layer0/o 512 行 | 559.735 ms | +9.505 ms | +9.527 ms | 1.00× |

四行都落在 null 之内，符号行间翻转，更快轮数 10–17/31 即抛硬币。

**判定：算子确实少了（控制 D 有实测序列），但这点工作量在本机噪声之下，测不出。**
写"分辨不出"而不是"没有收益"，也不据此预测官方秒数——官方 300 s 是唯一时间门。

## 6. 与同日其他官方回传的关系（本卡不做加法）

本卡开发期间收到 **v235 官方 TIMEOUT**（[超时记录](2026-09-10-v235-linear-official-timeout.md)）。
按计划 §3 预定表，该行触发"**保留其实际裁决，不自动移植；L-TF2 独立裁决**"，已照此执行：
v237 不含 v235 的 L-AD1、不含 v232 的 L-QF1、不含 v233 的 L-TF1（v237 是 v231 的后代，v233 是 v230 的后代）。

值得如实记下的是**它对本卡的期望值是负面的**：v235 是本轮唯一在本地测出可分辨提速的 Linear 候选
（34–52× null，折算预期省 8–13 s）却仍官方超时；本卡效应小一到两个数量级、本机分辨不出。
所以本卡**不主张**能在官方计时上产生可分辨差异。它作为"实质执行成本变化的纯提速候选"按 AGENTS §2
可作**一次**官方验证（不是重复等价 A/B），提交与否由用户决定。

## 7. 裁决

**LOCAL：输出逐位等价、算子少 2 个、本地墙钟效应低于本机分辨力（有 null 对照）。官方 `unregistered/NA`。**

计划 §2 的晋级规则（事前固定，原文）："相对届时当前根同分更快且 `<300s`：晋级；
分数提高且 `<300s`：按官方规则晋级；同分不更快：不晋级；分数更低或超时：不晋级，保留当前根。"
读数是哪个由官方数字决定，本卡不预填。

## 8. 产物

- 候选与工具：`workbench/full_solution/linear-tf2-retained-root-gradient-reuse/`
  （`parent.json` / `build.py` / `premise.py` / `verify.py` / `timing.py` / `pair_sixshard.py` / `archive.py`）。
- 证据：`build.json`、`verify.out`、`timing.json`、`paired_sixshard.json`。
- 评测：`artifacts/proxy_v3/linear-tf2-shard0-20260910/`、`artifacts/proxy_v3/linear-tf2-sixshard-20260910/`。
- 归档：`solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/`
  （`solution.py` / `build.json` / `config.json` / `verification.json` / `verify.out` / `timing.json` / `result.md`）。
- 官方字段未知记 `null`，不填本地秒数预测。
