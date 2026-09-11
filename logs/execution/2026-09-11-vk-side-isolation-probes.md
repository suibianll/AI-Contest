# VK 核侧隔离探针构建（2026-09-11）

用户指令原话："给我将这两个的优化配合标准的linear/attention，组合出新版本，去官方评测上看看是不是有优化效果"。

"这两个"指 v241 / v242——**同一 VK 核机制的两种粒度**，两张的完整包都已官方 TIMEOUT
（见[超时记录](2026-09-11-v241-v242-vk-official-timeout.md)）。本记录登记的是
**换用仓库既有的侧隔离通道**给同一机制定价的探针：标准 Linear + 候选 Attention。

## 1. 为什么走侧隔离

完整包在 v237 根上余量只有 **11 s**，而官方评测机是鲲鹏 920B（CPU）、硬限 300 s，
两次尝试都没进限、**都没拿到分数**。侧隔离通道是仓库既有的、**已被官方裁决过 7 次**的
定价口径：包更小、余量充裕，只测"该 Attention 机制本身值多少官方分"。

**已登记口径**（`logs/execution/2026-09-09-standard-linear-attention-side-scores.md`）：

| 侧隔离口径 | 官方分 | 官方时间 | Δ vs 14405 |
|---|---|---|---|
| 标准 Linear + **R3** attn（基线） | `14405` | 238s | — |
| 标准 Linear + **v195** attn | `14426` | 243s | **+21** |
| 标准 Linear + **v234** attn | `14455` | 263.7s | **+50**（相对 v195 行 +29） |

余量约 **57～62 s**——正是完整包缺的那个量级。

**这是诊断定价，不是晋级路径**：AGENTS §2"侧隔离只作诊断，不建立侧父或侧晋级线"；
侧隔离分**不晋级**、**不可与完整分相加或互推**（同一文件 §1 已写死）。

## 2. 探针怎么造的（复用仓库既有构建器，未新写拼接逻辑）

工具：`workbench/standard_linear_attention_probes/build.py`（+ `verify.py`、`build-manifest.json`）。
构造式就是文件里那一行：

```
探针 = 候选 solution.py 全文(rstrip)  +  固定的标准 Linear 尾块
```

尾块取自 `solutions/v162_attention_r1-v189-attnstack-recovery_officialNA_timeNA/solution.py`
（SHA `3619bfeb0e017555bd8fe31410888f80950a0128e3a2b94f643bd3f20b30bfc8`），
是 R1 标记之后的**全部内容**，SHA `cb265612adc030d8d2f301c184d8290cf1b11dc2904da8f94d68084777be3a58`。
它**只 shadow 两个 Linear API**（`hif4_calibration_and_quantize_weight`、
`hif4_dynamic_quantize_activation`），Attention 四 API 不被重新定义。

本次只往 `build.py` 的 `CANDIDATES` 表追加两项，改动的只有该表与重新生成的 manifest：

| 探针 | Attention 源（= 完整候选） | 候选 SHA256 | 输出 SHA256 |
|---|---|---|---|
| `standard-linear_v241-attn` | `20260911_v241_attention-vk-kernel_scoreNA_timeNA` | `8d364b3d…1db186cd` | `c86c972a…749a45b6a` |
| `standard-linear_v242-attn` | `20260911_v242_attention-vk-kernel-shared_scoreNA_timeNA` | `b11ba4f2…9fb1e9227` | `61cc4887…f8334782c4` |

## 3. 独立复核（不采信 `verify.py` 的自述结论）

`verify.py` 是纯静态的、且它检查的正是构建器自己的构造式——**它自证**。所以另外跑了几项：

**一 · 六个旧探针的磁盘内容未被扰动。** 构建前对 15 个 `standard-linear*` 探针取 SHA 快照，
构建后逐一比对：只有两张新探针出现，其余**全部逐字节不变**（构建是确定性的）。

**二 · 前缀逐字节相同。** 对四张探针（含**已官方跑过**的 v195 `14426` 与 v234 `14455`）核：
探针文件 **以候选源码 `rstrip()` 开头**，余数**同为 6475 字节**——即拼接器插入的分隔线+标记+恒定尾块。
新旧探针同型，构造式一致。

**三 · 尾块在结构上无法影响 Attention。** 从四个 Attention API 出发做**传递闭包**，收集 head 内
可达的顶层名字：v241/v242 各 **97 个**（同型对照 v234 为 104 个），
**其中被追加尾块 shadow 的：0 个**。尾块只提供 Linear 路径，Attention 路径在探针里
就是候选自己的那段代码——**不是"测出来一样"，是"没有通道能不一样"**。

**四 · 脱离仓库单文件导入。** 把两张探针复制到仓库外的独立目录，清空 `PYTHONPATH` 后导入：
无仓库依赖 import（`solutions`/`workbench`/`evaluator` 均为 0），六 API 齐备且位置参数个数正确
（3/3/4/5/5/5），`sys.modules` 中无仓库模块。两张均通过。

**五 · SHA 唯一。** 两张的 `solution.py` SHA256 在**全部 300 个归档 `solution.py` + 根**中唯一，
不构成 AGENTS §2 禁止的"重复相同 SHA 或逐位等价 A/B"。

## 4. 官方前的目标侧 shard0（AGENTS §4.3 / 4B 指引 §3.1）

命令（4B 指引 §2 第 3 步，`--linear-only` 换 `--attention-only`），
基线取**官方比较口径** `standard-linear_v195-attn`：

```
evaluator/eval.py --solution solutions/20260911_standard-linear_v24N-attn_scoreNA_timeNA/solution.py \
  --baseline-solution solutions/20260908_standard-linear_v195-attn_scoreNA_timeNA/solution.py \
  --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt \
  --calibration-cache-mode auto --algorithm-device cuda \
  --output-dir artifacts/proxy_v3/side-vk24N-shard0-20260911
```

| 探针 | shard0 delta_mean | +/-/0 | L1 | decision | 完整候选的六 shard（对照） |
|---|---|---|---|---|---|
| `standard-linear_v241-attn` | `+0.003940` | 11/1/0 | 0.004163 | `continue_next_shard` | `+0.004547`（六片全正） |
| `standard-linear_v242-attn` | `+0.005159` | 12/0/0 | 0.005159 | `continue_next_shard` | `+0.004602` |

两张均 `reasonableness_issues: 0`，无 blocker、无 warning。两条 `standard-linear` 的
Linear 侧 API 调用数均为 **0**（`hif4_calibration_and_quantize_weight` 0 calls），
与"单侧运行不调用另一侧 API"一致。

**运行账里的一条附带读数**：`hif4_dynamic_quantize_v` 在 shard0 上是
**v241 `7.094s / 12 calls`（0.591 s/次）** 对 **v242 `1.100s / 12 calls`（0.092 s/次）**，
本地实测比值约 **6.4×**。这独立印证了"共享核确实把 V API 降下来了"（v242 声称 ÷5.7）。
**同时它再次给出"12 个 case / 12 次 V 调用"**，与成本算术第一条支点
（一 case 一次 V 调用）在 shard0 尺度上一致。
**但这条读数只在本地 GPU 上成立，不用于官方秒数预测**——见 §6。

## 5. 归档内容

每张探针目录**只含 `solution.py`**，与既有侧隔离探针（`standard-linear_v195-attn`、
`standard-linear_v234-attn`）同规格。

| 探针 | `solution.py` 大小 | SHA256 |
|---|---|---|
| `standard-linear_v241-attn` | 537267 B | `c86c972a75b26bc5f7e0a6eafc2ce9f65183be434bdb831f434d2d6749a45b6a` |
| `standard-linear_v242-attn` | 537466 B | `61cc48878275e881c72db37d9439ad4c197487bb4504f9ee9b71cfb8334782c4` |

**未打包 `solution.zip`**：用户 2026-09-11 明确"压根不用你打包"。
本仓交汇流程以 `solution.py` 为准，需要 zip 时按先例
`solutions/20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/solution.zip`
（zip 根下单个 `solution.py`，deflate）当场生成即可——本记录不保留该步骤。

## 6. 这次能回答什么、不能回答什么

**能回答**：**VK 核机制在官方侧到底值多少分**——相对 `14426`（v195 行）与 `14405`（R3 基线）。
这是完整包形态两次都拿不到的那个数。**VK 机制至今没有任何官方分数**。

> **回填（2026-09-11 回传后，按本节 §7 的约定）：这个"能回答"没有兑现——问题仍然没有答案。**
> `standard-linear_v241-attn` 官方 **`TIMEOUT (>300s)`，无分**（见
> [超时登记](2026-09-11-side-vk241-attn-official-timeout.md)）。VK 机制现在两条通道四次尝试
> 全部无分。**并且本次证伪的正是本文 §1 用来开这条通道的那个前提**——本文 §1 写的
> "侧隔离通道……包更小、余量充裕……余量约 57～62 s，正是完整包缺的那个量级"
> **不成立**：那 57～62s 是"此前被测机制都便宜"的结果，不是通道的属性；
> 侧隔离包同样有 300s 硬限，**能否拿到分数由机制成本决定，不由通道决定**。
> **`standard-linear_v242-attn` 同日回传 `14098 / 250s`**（对照 R3 基线 `14405` 为 **−307**）：
> **VK 机制的唯一官方分，且是负向**，故机制关闭。详见
> [回传登记](2026-09-11-vk242-side-isolation-official.md)。

**不能回答（逐条写明，禁止外推）**：

- **不能回答完整包能否落地。** 侧隔离时间对完整包**无预测力**——仓库有双向反例：
  v194 侧隔离 −4 s、完整包却 `+5s`；v190 侧隔离 246 s、完整包 `TIMEOUT`。
  本次侧隔离无论跑出多少秒，**都不能推出 v241/v242 完整包进不进 300 s**。
- **不能推出官方调用次数。** 本次 shard0 的 12 次 V 调用是**本地 12 case 的读数**，
  不是官方 case 数；`250` 与 `200` 两个数字在仓库内仍互相冲突、均非实测（见超时记录）。
- **不推导本地→官方换算系数，不从本地秒数预测官方时间。**
- **侧隔离分不晋级、不与完整分相加**（AGENTS §2）。
- **本次不关闭也不重开任何机制。** VK-1 在计划 §7 下关闭的只是 v241/v242 两个**实现**。

## 7. 待办

官方回传后，按探针名绑定计分 SHA，登记到 `solutions/README.md` 的侧隔离表与
[侧隔离分登记](2026-09-09-standard-linear-attention-side-scores.md)口径，
并回填本节 §6 的"能回答"栏。**回传前本文不写任何分数或秒数预测。**

**已完成（2026-09-11，`standard-linear_v241-attn` 回传后）**：

- §6 已按上述约定回填——答案是**没有拿到分数**，问题仍未回答。
- 计分 SHA 已按探针名绑定：`c86c972a75b26bc5f7e0a6eafc2ce9f65183be434bdb831f434d2d6749a45b6a`。
- 已登记到 `solutions/README.md` 侧隔离表、`docs/current-solution-status.md`、
  计划入口与活动计划 §6.1，并另写
  [超时登记](2026-09-11-side-vk241-attn-official-timeout.md)与探针目录 `result.md`。
- **未改**[侧隔离分登记](2026-09-09-standard-linear-attention-side-scores.md)本身：
  该文是 2026-09-09 的时点记录（其 §2 表本就不含 v229/v234），实时索引以
  `solutions/README.md` 为准。按 AGENTS §5「不覆盖原始执行日志，修正另写日志」处理。

**亦已完成（2026-09-11，`standard-linear_v242-attn` 同批回传）**：

- 计分 SHA 已绑定：`61cc48878275e881c72db37d9439ad4c197487bb4504f9ee9b71cfb8334782c4`。
- 官方 `14098 / 250s`；对照 R3 基线 `14405` 为 **−307**、对照 v195 `14426` 为 **−328**。
- 已登记到 `solutions/README.md` 侧隔离表、`docs/current-solution-status.md`、
  计划入口、活动计划 §6.1 与 §5 关闭表，并另写
  [回传登记](2026-09-11-vk242-side-isolation-official.md)与探针目录 `result.md`。
- **两条探针均已回传，本文无遗留待办。**

**结果提示（与本文 §1 的前提更正互为印证）**：便宜的粒度（per-KV-group）在时间上**进得了限**
（`250s`），但**精度是倒扣 307 分**。至此"能不能进限"对 VK 已不是待解问题，
该机制的降本线结束——详见[回传登记 §5](2026-09-11-vk242-side-isolation-official.md)。
