# L-TF1：梯度复用（v233）

计划卡 [`docs/superpowers/plans/2026-09-10-linear-correctness-and-runtime-plan.md`](../../docs/superpowers/plans/2026-09-10-linear-correctness-and-runtime-plan.md) §3（第二卡）。

父根：v230 Linear L-EM2 + v195 Attention，官方 **18428 / 292 s**，
SHA256 `0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc`，505496 B。
本卡不改父根；候选是父根的**纯追加**（append-only），父根字节一个未动。

工作区：`workbench/full_solution/linear-tf1-gradient-reuse/`。
候选：`0ec89710087d061bf9608196ad4d53a1c6be98c8a5595596a071ecd05a6821eb` / 515985 B，
实现模块 `dd5e436ebc6952f5441349ae418e9ecbe7d8fe645d3b8e3e41767a68db0ce3bf` / 10487 B。

## 1. 机制：父根把同一个梯度算了两遍，第一遍的结果被丢掉

`_em1_dynamic_descent`（`solution.py:12067`）在进入 pass 循环**之前**先算一次梯度：

```python
gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
if not torch.isfinite(gradient).all():
    state["em1_dynamic_arm"] = "nonfinite-gradient"
    ...
    return activation
```

这一次的唯一用途是**有限性检查**——判完就 `return` 或继续，结果本身不再被读。
随后循环体的第一句把同样的表达式**再算一遍**，并把那个名字重新绑定：

```python
for _pass in range(_EM1_PASSES):
    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)   # 父根 12158-12160
    if not torch.isfinite(gradient).all():
        aborted = True
        break
```

`_EM1_PASSES = 1`（`solution.py:11941`），所以每次调用这两遍**都跑**，第一遍的乘积作废。

候选在循环里那一遍上加保护：

```python
for _pass in range(_EM1_PASSES):
    if _pass:
        gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
        if not torch.isfinite(gradient).all():
            aborted = True
            break
```

pass 0 复用父根刚算过、刚检查过的那一份；pass 1 及以后照旧重算。

### 1.1 为什么是 `if _pass:` 而不是把表达式提出去

pass 体里有原地写（`solution.py:12190-12204`）：

```python
deployed.add_(row_delta.mul_(keep_scale))
gradient.add_(g_delta.mul_(keep_scale))
```

梯度所依赖的 `deployed` 在 pass 0 之后已经不是构建它时的那个张量。把整个表达式提到循环外，
**每一个**后续 pass 都会用错的值。`if _pass:` 是让 pass 0 复用、pass ≥ 1 仍按父根原序重算的
唯一写法。这一点是机制成立的前提而非实现细节，因此静态与动态各测一次：

- `premise.py`（判据 P4）：两点之间对被观察名字的重新绑定**有且仅有一次**——`deployed =
  deployed.clone()`，逐元素保持原值；`deployed`/`reference` 无原地写。
- `verify.py` 控件 E2/F：K=2 时后续 pass 确实仍然重算（见 §4）。

### 1.2 两遍为什么必然相同

同一表达式、同一批张量、同一顺序。`verify.py` 控件 C 在运行时逐位复核：
两遍各自的两个 `mm` 乘积**逐位相同**。

```
[C:synthetic]    product 0 identical: True, product 1 identical: True  (12x2560 @ 2560x2560 两次)
[C:real/layer0/q] product 0 identical: True, product 1 identical: True  (128x2560 @ 2560x2560 两次)
```

## 2. 算子账：每次调用少 2 个矩阵乘，且只有这 2 个

`verify.py` 控件 D 把 `torch.Tensor.mm` 包起来，记录双方**完整的**乘积序列（形状 + 结果），
而不是只数个数。K=1：

| | 循环前 | 一个 pass 的梯度 | 一个 pass 的步进 | 合计 |
|---|---:|---:|---:|---:|
| 父根 | 2 | 2 | 16 | **20** |
| 候选 | 2 | 0（pass 0 复用） | 16 | **18** |

候选的形状序列**等于父根的序列删掉第 2、3 项**，别无其他变化——合成层与真实 layer0/q 各验一次：

```
[D:synthetic]     K=1: parent 20 mm products, candidate 18; the candidate's sequence
                  equals the parent's with products 2 and 3 removed and no other change
[D:real/layer0/q] K=1: parent 20 mm products, candidate 18; ...
```

去掉的 FLOP：`2 × 2 × rows × channels²`——128 行 3.36 GFLOP，512 行 13.4 GFLOP。

### 2.1 E2：证明保护没有把重算提出循环

这是本次验证里最关键的**负向**对照。把 K 在内存里改成 2，并给 pass 1 的梯度在父根会走到的
那个 `mm` 上投毒成 NaN：

```
[E2:real/layer0/q] K=2 with a non-finite gradient at the second pass:
   parent issues 22 products (abort at index 21) and candidate 20 (abort at index 19)
   -- both stop before any second-pass group step, and their outputs are byte-identical
```

偏移恰为 2（= 被删掉的那两个乘积），两个都在第二个 pass 的任何步进之前停住。
也就是说 K ≥ 2 时候选**仍然重算**，并且**停在父根停的地方**。

> 这个控件第一次跑时失败过：断言期望父 22 / 候选 20，实际报"父发了 20 个，没停在 pass 1"。
> 原因是前一个控件 `control_f_k2` 已经把 `_EM1_PASSES` 还原成 1，而 E2 仍假设 K=2。
> 修法是让 E2 自己在 try/finally 里把两边的 `_EM1_PASSES` 设成 2 并做泄漏检查，
> 两个投毒下标**由期望调用数推导**并断言等于 21 和 19（不是写死的魔数）。

## 3. 候选构建：AST 级相等，而不是"逐字节差异数"

`build.py` 不手写覆盖函数，而是从父根字节里切出 `_em1_dynamic_descent` 的源码、
替换一个子串、追加回去。于是"唯一改动"是**测出来的**。

**这张卡不报逐位置字节差异数。** 改动是**插入一行**：插入点之后所有字节整体位移，
逐位置比较会把整条尾巴报成"不同"——一个看着像证据、其实不是数字的数。改报：

| 量 | 值 |
|---|---|
| `extracted_parent_function_bytes` | 7999 |
| `substitutions_performed` | **1**（未加保护的循环头出现 1 次、已加保护的 0 次，否则拒绝构建） |
| `extracted_function_changed_lines` | 5 |
| `extracted_function_byte_delta` | +34 |
| `ast_identical_after_dissolving_pass_guard` | **true** |

最后一条比字节计数强：把插入的 `if _pass:` 节点溶回父块后，两棵 AST 的 `ast.dump` **完全相等**
——执行的语句树除该保护外一个节点没变。这个等式由 `verify.py` 控件 A **独立重算**，
所以归档不建立在构建脚本的自述之上。

构建脚本的拒绝条件：父 SHA256/字节数不符、未加保护的循环头出现次数 ≠ 1、已加保护的 ≠ 0、
AST 不等、生成模块 ≠ HEADER+函数、模块内 `if _pass:` ≠ 1 处、候选内该函数定义 ≠ 2 个、
调用点 ≠ 1 处、候选不以父根字节开头——任一不满足即非零退出。

`_em1_dynamic_descent` 由父根 hook 在调用时从模块全局解析（`co_names` 含该名字），
所以追加定义遮蔽父定义，父根两个 hook 一行未碰。候选里唯一的变化是"父定义变成死代码"。

## 4. 验证（`verify.py`，CPU）

`verify.out` 结尾 `ALL L-TF1 CONTROLS PASSED`。

| 控件 | 结果 |
|---|---|
| A | 前缀 505496 B 与父根全同；候选 515985 B；六 API 独立导入（无仓库同级文件在路径上）；Attention 四 API 源码与字节码相同；活体解析到追加定义；`if _pass:` 恰好 1 处；AST 相等由 verify.py 独立重算 |
| B | 四条路径**逐位相同**：合成层（16 步、2586 组移动、2586 个 mantissa 码改变）、无 `em1` 载荷状态（双方 `no-metric`）、`in_features=4160` 的 out-of-scope 旁路、真实 layer0/q 128 行（16 步、40167 组移动、2044 行保留） |
| C | 循环前与 pass 0 的两个梯度乘积**逐位相同**（合成 + 真实） |
| D | K=1：父 20 个 mm、候选 18 个；候选序列 = 父序列删掉第 2、3 项（合成 + 真实） |
| E1 | NaN reference → 双方都走父根自己的 `nonfinite-gradient` 分支，输出逐位相同 |
| E2 | K=2 投毒：父 22 个乘积止于第 21、候选 20 个止于第 19，偏移恰为 2，输出逐位相同 |
| F | 两次调用逐位一致；`torch.save/load` 的 state 往返逐位一致；K=2 内存态复原后 K=1 行为不变；`accepted_steps=16`（上限 16），即父根 16 组调度的**一个** pass |

两处实现细节值得记下，都是"看起来等价、其实不等价"的坑：

- **`torch.no_grad()` 遮蔽来源身份。** 直接对装饰过的函数做 `inspect.getsource` 或比较
  `__code__`，两边都会报 torch 自己的装饰器文本，A 的"Attention 字节码相同"就变成空断言。
  必须先 `inspect.unwrap` / `.__wrapped__`。
- **`torch.equal` 认为 NaN ≠ NaN。** 逐位比较一律走 `numpy().tobytes()`，不用 `torch.equal`，
  否则 E1/E2 这两个含 NaN 的路径会被误判成"不同"。

被测对象是**发布源码里真实存在的那段函数**，参考量独立——控件不会与发布物漂移。

## 5. 时间：算子确实少了，墙钟收益本机测不出来

卡片要求"同状态同步长、成对、中位数加离散度、不用 profiler"，并允许"输出等价但执行不同的
候选官方计时"。本机的结论是：**删掉的工作量是真的，它的墙钟值低于本地所有时钟的分辨力。**

### 5.1 删掉的工作量，直接测

`where_the_time_goes.py` 在同样设备、同样形状下把下降过程自身的算子拆开**单独计时**
（复刻，不改动任何发布代码）：

| 项 | 128 行 | 512 行 |
|---|---:|---:|
| `gradient_two_mm`（**本卡删掉的就是它**） | **0.3860 ms** | **2.0316 ms** |
| `one_step_mm` | 0.1861 ms | 0.8571 ms |
| `sixteen_step_mm` | 5.0504 ms | 16.9072 ms |
| `metric_cholesky_inverse` | 20.3827 ms | 19.1066 ms |
| `group_gram_index` | 0.2058 ms | 0.2028 ms |

### 5.2 三臂成对测量，带同字节 null 对照

`timing.py` 每轮跑三臂：**父根 / 候选 / sham**。sham 是把 `solution.py` **再加载一遍**，
与父根逐字节相同；它与父根的成对差就是"这台机器、这个时段、这个热漂移下什么都没测时，
估计器报出来的数"。每轮轮换执行顺序（没有哪一臂永远最后跑），CUDA event，
每次测量前显式 synchronize，31 轮，每臂 3 次无计时预热。

| 状态 | 父根中位 | 效应中位 | 效应 % | null 中位 | \|效应\|/\|null\| | 候选更快轮数 | null 更快轮数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer0/q/128 | 516.138 ms | **+3.028 ms** | +0.587% | +0.324 ms | 9.34× | 19/31 | 16/31 |
| layer0/q/512 | 864.229 ms | **+7.608 ms** | +0.880% | +6.939 ms | 1.10× | 17/31 | 20/31 |
| layer0/o/128 | 489.817 ms | **−8.387 ms** | −1.712% | −5.172 ms | 1.62× | 13/31 | 13/31 |
| layer0/o/512 | 529.109 ms | **−9.333 ms** | −1.764% | +0.943 ms | 9.90× | 13/31 | 17/31 |

**读法**：效应的**符号在四个状态之间不一致**；量级（3–9 ms）比本卡真正删掉的工作
（0.4–2.0 ms）大一到两个数量级；null 自己的成对中位数就漂到 ±6.9 ms，
q/512 那一行的 |效应|/|null| 只有 1.10×。**估计器分不开这个效应**，
因此不从这个测量主张任何墙钟节省。

### 5.3 六 shard 与同进程两种测法互相矛盾

动态激活 API（唯一可比口径）：

| | 候选 | 基线 | 差 |
|---|---:|---:|---:|
| shard 0（同进程） | 41.9098 s | 43.9954 s | **−2.0856 s** |
| shard 1（同进程） | 41.4343 s | 42.5994 s | **−1.1651 s** |
| 六 shard（跨进程） | 253.7036 s | 251.0651 s | **+2.6385 s** |

跨进程逐 shard 差：`+0.8973 / −0.1780 / +0.9225 / +2.6025 / +0.8805 / −2.4861` s
（每次调用 +7.85 ms）。

同一个量、两种测法、**相反的符号**；跨进程逐 shard 的散布从 −2.49 s 到 +2.60 s。
机制每次调用的全部孤立刻度是 0.386 ms（128 行），336 次调用造不出 2.6 s 的差。
尺度参照：v232 的同进程 shard0 配对，对一次**可证执行完全相同**的改动
（下标字符串里的一个字符）报出过 0.89 s 的差。

**这些数字记为"测量分辨率"的证据，不记为关于候选的证据。**

> **`api_total_seconds` 不可比，照录但不当证据。** 它含
> `hif4_calibration_and_quantize_weight`，该校准缓存命中时 0.0 s、未命中时 128–143 s。
> 候选的 shard 0/1 命中（总计 42.8 / 41.5 s），shard 2–5 与全部基线未命中（169–185 s）。
> 两列之间 226 s 的差**全部**是缓存命中造成的，与本机制无关。
> v232 的执行记录里已经写过同一条结论。

### 5.4 一次调用的时间到底在哪（分母）

`attribute.py` 先量整调用，再把某个部件打桩后重量一次。
**打桩会改变算出来的值，所以这个脚本的任何数字都不是候选测量**；它给的是
"这张卡最多够得着多少"的分母。

| 行数 | 整调用 | 关掉下降 | 度量打桩 | mm 全部打桩 | 父根动态 API 占 | 下降占 | 度量占 | 全部 mm 占 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 534.0 ms | 480.6 ms | 516.7 ms | 522.0 ms | **90.0%** | 10.00% | 3.24% | 2.24% |
| 512 | 873.6 ms | 814.3 ms | 843.2 ms | 849.0 ms | **93.2%** | 6.79% | 3.48% | 2.81% |

父根自己的动态激活 API 就占一次调用的 **90–93%**；本卡编辑的那段下降占 7–10%，
而下降里**所有**矩阵乘加起来不到 3%。删掉 18 个里的 2 个，上限就是一次调用的千分之一量级。

独立 `perf_counter` 与 event 时钟一致到 2% 以内（0.981× / 0.991×），
所以"每次调用 500–870 ms"本身是真的，不是 event 时钟的伪影。

**这一节顺带回答了计划 §4 那张架构卡的问题**（父激活 GPTQ、新增下降、度量求逆各自占多少
无 profiler 调用时间）：父激活 GPTQ/hook ≈ 90–93%，新增下降 6.8–10.0%，度量求逆 3.2–3.5%，
全部 mm 2.2–2.8%。答案已在本卡内取得，§4 无需再单独测一次。

## 6. 六 shard 本地结果

`pair_sixshard.py` 按 `case_id` 配对本卡候选单侧进程与归档 `linear-em3-v230base` 基线，
脚本拒绝 `source_sha256` 不一致（或一致）的文件。

| shard | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| n | 56 | 56 | 56 | 56 | 56 | 56 |
| Δ gain | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| +/-/0 | 0/0/56 | 0/0/56 | 0/0/56 | 0/0/56 | 0/0/56 | 0/0/56 |

等权 `+0.000000`，**0 / 0 / 336**，`min = median = max = 0.0`，按角色全部 `0.0`。
`reasonableness_issues: 0`。

**这不是容差口径。** 这里报的是评测器自己的配对统计量（不是逐位比较），
而它在 336 个 case 上**处处精确落在零**。控件 B 更进一步，在原始张量上逐字段比字节。

**零 delta 是预期结果，不是筛不过。** 机制删掉的是一个**结果被丢弃**的运算，
按构造不可能改变任何 gain；这里出现非零 delta 才是缺陷。本卡的全部价值就在于没有。

### 6.1 配对方法与同基准校验

- shard 0/1：**同进程**配对（那次调用随后 `stopped_early: true`，只写了两个 shard，
  所以同进程路径覆盖不了全盘——这一点与 QF1 工作区 `pair_sixshard.py` 文档里的警告一致）。
- shard 0–5：另跑一个候选单侧进程，跨进程配对。

**同基准校验**：本卡同进程 shard0/1 基线与归档 `linear-em3-v230base` 的 shard0/1 文件
（同为 `0f1af6db`）在 56 个 case 上逐一比较，**max |gain difference| 恰为 `0.0`**。
所以 shard 2–5 的跨进程配对与 v231、v232 用的是同一个基准，不是近似。

## 7. 裁决

**LOCAL：输出等价、算子更少、本地时间不可分辨（有 null 对照）。官方 PENDING。**

- 336 个配对 case **全部精确零**，0 正 0 负；同进程 shard0/1 各自 56/56 零。
- 机制被证明**恰好**删掉每次调用 2 个矩阵乘（20 → 18），别无其他；E2 证明 K ≥ 2 的行为未变。
- 墙钟收益低于本机所有时钟的分辨力，这一点**带 null 对照记录在案**而不是绕开。
  不写官方秒数预测：合理陈述是"应当仍是父根的 292 s，期望上略少"，少多少本机测不出。

本地工具的 `no_effect` 是它自己 `delta_mean > 0 and L1 < 0.02` 的**小趋势筛选门**——
对一个按构造输出等价的候选报 `no_effect` 是**设计中的结果**，不是被拒。
归档照录工具原判，不重述、不辩解。

**晋级规则（计划 §3 原文，事前固定）**：官方同分且更快可作为完整工作父；官方分数下降或超时则不晋级。
两条都记在这里，读数是哪个由官方数字决定。

未对任何候选跑官方评测（用户批量统一评测）。

**血统**：v233 是 v230 的后代，**不含** v231 的 K=2 机制、也**不含** v232 的 L-QF1 二次代价修正；
它是父根加一行保护。三个待评候选是**兄弟不是叠加**，本地 Δ 不可相加、也不可当序列读。
Attention 半边与父根完全一致（v195）。

## 8. 产物

| 文件 | 内容 |
|---|---|
| `verify.py` / `verify.out` | 控件 A–F，结尾 `ALL L-TF1 CONTROLS PASSED` |
| `premise.py` / `premise.out` | 静态 AST 前提检查（P1/P2/P4） |
| `build.py` / `build.json` | 追加式构建与拒绝条件 |
| `timing.py` / `timing.json` / `timing.out` | 三臂成对计时 + sham null |
| `where_the_time_goes.py` / `.json` / `.out` | 下降过程各算子单项定价 |
| `attribute.py` / `attribute.json` | 整调用打桩归因（分母） |
| `pair_sixshard.py` / `paired_sixshard.json` | 六 shard 跨进程配对（由 QF1 工作区复制，文档串与默认路径已改） |
| `candidate/solution.py` | 候选本体（归档 `solutions/` 下的副本与之同 SHA） |
