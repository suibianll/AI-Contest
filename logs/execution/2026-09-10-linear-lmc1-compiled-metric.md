# L-MC1（把固定度量重建移到校准）—— 审计记录：一个阻塞性的缺失事实

计划卡：[`docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md) §4。
工作目录：`workbench/full_solution/linear-lmc1-compiled-metric/`。设备：CPU（未用 GPU）。

**状态：审计完成、实现完成、验证全过（2026-09-10）。** 阻塞点是一个仓库没有记录的事实（见 §2），
按计划 §5 的纪律"登记最小缺失证据"处理，不拿口径推测关卡。

## 1. 前提成立（依赖那一半）

`_em1_metric`（根 `solution.py:12021`）每次动态调用做：

```python
inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
ridge   = mean(diag(inverse)) - gram_diag_mean
metric  = inverse; metric.diagonal().sub_(ridge)
```

`metric`（即 G）只由 `h_inv` 与校准期存下的 `gram_diag_mean` 标量决定，**与 activation 无关**。
所以"可以前移"这一半成立。

**但根里那段注释已经写明了当前的取舍**（`_em1_compile_metric`，`solution.py:11995-12001`）：

> "removes an n^3 Cholesky inverse from every calibration call -- measured at 39.9 ms (in=4096) /
> 15.3 ms (in=2560), i.e. ~2.8 s over the official 144 in-scope calibrations"

即：现在的设计是**故意**不在校准端求逆，把它推到每次动态调用。所以 L-MC1 不是"消除"这个求逆，
而是**把它从动态端搬到校准端**。净账取决于两侧的调用次数比。

## 2. 阻塞点：官方形状下这个比值的符号无法从仓库记录判定

调用模式（`evaluator/proxy_v3_eval.py` 实测）：

- **校准**：每 `(layer, role)` 一次 —— 24 层 × 7 role = 168 次，其中在范围内（channels ≤ 4096）**144 次**；
  `proj`（9216）超范围。
- **动态**：每 **case** 一次。

据此从 pack 实测本地面板：

| | 在范围内校准 | 在范围内动态 | 移动的净效果 |
|---|---:|---:|---:|
| **本地面板**（24 层 × 6 role × 2 窗） | 144 | 288 | **省 144 次求逆** |

**官方形状则无法判定。** README 记"官方样例数按用户确认的 50 Linear + 250 Attention"，
但**仓库里没有任何文档写明一个官方 Linear"样例"的结构**：它是 `(layer, role, 窗)` 三元组
（则动态 = 50），还是一次整模型前向、每个样例会流过全部 144 个在范围内 `(layer, role)`（则动态 = 7200）？
两种读法给出方向相反的结果：

| 读法 | 官方动态次数 | 移动的净效果 |
|---|---:|---:|
| 50 个三元组 | 50 | **净亏 94 次**（144 次校准求逆 − 50 次动态求逆） |
| 50 个样本 × 144 个位置 | 7200 | 净赚 7056 次 |
| 官方只为它的样例标定所需位置 | ≈50 | 中性 |

**缺失的事实（最小证据）**：官方评测中 `hif4_dynamic_quantize_activation` 的调用次数，
或其与 `hif4_calibration_and_quantize_weight` 调用次数的比例。在拿到它之前，本卡的核心前提
（"这是一处值得消除的重复工作"）**无法评估**。

**不拿推测关卡。** 本仓库刚有一次教训：v237 的本地墙钟不可分辨曾被外推成"官方大概率超时"，
被官方 −2s 证伪。此处若按"官方只跑 50 个样例"推断净亏并据此关闭，是同一类错误——
而且这次连测量都没有，只有口径假设。故按计划 §5 登记缺失证据，交用户裁决。

## 3. 与口径无关的已测边界（三项）

**3.1 设备/dtype 边界（已实测，结论：可行但有契约条件）**
`h_inv` 由 `_cpu_state_tensor` 存为 **CPU float32**；动态路径在 `_em1_metric` 里
`h_inv.to(device=device, dtype=torch.float32)` 搬到运行设备后才做 Cholesky。实测（2560 与 4096 两个尺寸）：

| 比较 | 结果 |
|---|---|
| CPU 上的 `cholesky_inverse(cholesky(h))` vs CUDA 上的 | **不逐位相同**，相对差 ~1.0e-06 |
| float32 张量 CUDA→CPU→CUDA 往返 | **精确**（逐位无损） |

故本卡逐位可行的**充分条件**是：G 在**运行设备**上算、以 CPU 副本入 state、动态端再搬回同设备。
评测器两侧用的是同一个 `--algorithm-device`，且**校准缓存 identity 本就包含 device**
（AGENTS §5），所以该条件在当前口径下自动成立；但必须写进契约——若某一流程出现校准 CPU / 动态 CUDA，
预存的 G 会与现路径差 ~1e-6，逐位等价即不成立。

**3.2 原地修改风险（静态发现，实现时必须处理）**
`_em1_metric` 现在对 `inverse` 做的是**原地** `metric.diagonal().sub_(ridge)`。若 G 改为从 state 读出，
而同一次运行里同一个 state dict 被多个 case 复用（评测器每 case 调一次动态 API），
第二次调用会把 ridge **再减一遍**——这正是计划 §4 点名的"使用方不得原地修改持久G……避免每次再减ridge"。
最干净的解法是把**已减 ridge 的最终 G** 存进 state，使动态端只读。

**3.3 `validate_state` 接受度（已核对）**
`evaluator/reference_hif4.validate_state` 是通用遍历：tensor 须为 **CPU**、strided、无梯度、实数、
dtype 在允许集内、有限，且深度 ≤8、节点 ≤4096。新增一个 CPU float32 的 `metric` 张量**可以通过**
（每 state 只 +1 节点）。注意"必须 CPU"这一条，正是 3.1 里"存 CPU 副本"的来源。

**3.4 state 体积接近翻倍（已测）**
`metric` 与已存的 `h` 同为 channels² float32，故每个在范围内 state 的 em1 载荷翻倍：

| role | channels | 单个新增 | ×24 层 |
|---|---:|---:|---:|
| q/k/v/fc_gate/fc_up | 2560 | 26.2 MB | 5 × 24 × 26.2 MB |
| o | 4096 | 67.1 MB | 24 × 67.1 MB |

合计新增约 **4.75 GB**。对照现状：六个 linear 校准缓存已是 **38.05 GB**（每个 6.81 GB），
故本次增加约 +12%（每个 shard +0.79 GB）。计划要求把这一项"与旧路径峰值内存、累计 state 大小、
传输次数同表记录"——此处先给磁盘侧的数，内存/传输侧待实测。

## 4. 生效路径与两处 metric 调用（已核对）

`_em1_metric` 在根里有**两处**调用：`solution.py:12108`（原始 `_em1_dynamic_descent`）与 `:12423`
（L-TF2 追加的影子定义）。**只有影子是活的**——模块全局名在调用时解析，后定义者胜出（v237 已用
`co_firstlineno` 验证过同一机制）。故本卡只需改影子路径；但必须显式核对这一点，
因为 L-TF2 的纯追加让根里同时存在两份 descent 定义（计划 §4 点名的"包装/覆盖关系"）。
`_EM1_PARENT_LINEAR_DYNAMIC` 是校准/动态分离用的包装别名，与 metric 无关。

## 5. 仍未做

- 峰值内存与 state 搬运次数的实测（磁盘侧已给：+4.75 GB / 现 38.05 GB）；
- 逐位等价实测（需先有实现）；本记录只给可行性的充分条件与三条边界。

## 6. 实现与验证（已完成）

**实现**（`build.py`）：两个追加影子，父字节一个未动。
`_em1_compile_metric` 在校准端算出并存入最终 G（已减 ridge）；`_em1_metric` 动态端只加载，无 Cholesky、无求逆、无写入。
候选 `2c344722…`，524220 B。

**验证**（`verify.py`，CUDA，真实权重）——**全部通过**：

| 控制 | 结果 |
|---|---|
| A | 纯追加；Attention 与动态激活字节码相同；两个影子都是活定义 |
| B | **3 个 (layer,role) × 3 次动态调用，五字段逐字节相同**；存储的 G 与父每次重建的 G **逐位相同**（3/3） |
| C | 详见 §7 的次数账 |
| D | 同一 state dict 连调三次，字段逐字节相同，**存储的 G 未被改动**（ridge 只减一次） |
| E | 超范围宽度（proj 9216）两侧都不存；无 em1 载荷两侧同样穿透；无 metric 的 state 两侧 arm 同为 `no-metric` |
| F | 校准确定性：两次运行存的 G 逐字节相同 |

### 6.1 最隐蔽的一处：内存布局也是等价性的一部分

第一版实现**通过了"存储的 G 与父重建的 G 逐位相同"这一条，但五字段仍然不同**——`scale_factor`/`scale_lv2`/`scale_lv3` 全同，只有 `sign`/`mant` 不同。逐步定位后：

| | 存储的 G | 父重建的 G |
|---|---|---|
| 数值字节 | **相同** | **相同** |
| stride | `(2560, 1)` 行主序 | **`(1, 2560)` 列主序** |
| contiguous | True | **False** |

**`torch.cholesky_inverse` 返回的是转置 stride 的张量（LAPACK 惯例）**，而 `_cpu_state_tensor` 里的 `.contiguous()` 把它拍成了行主序。
数值完全一样，但下游 `(deployed - reference).mm(metric)` 面对不同 stride 会走不同的 cuBLAS 路径，舍入因此不同——
足以改变离散的 mantissa 码分配。

修法：存 G 时**不强制 contiguous**（`nan_to_num` 与 CPU↔CUDA 往返都保 stride，已实测），
动态端 `.to(device)` 把 stride 一并带回来。

**这条值得单独记**：计划 §4 写的是"按原表达式、原float32顺序构建G"——方向对，但只说到了"顺序"，
没说**布局**。本案里只保数值不保布局，等价性会以一个非常隐蔽的方式失败（前三个字段全对，只有离散码不同）。

## 7. 次数账与实测代价（含一处自我更正）

**次数**（`verify.py`，1 次校准 + 4 次动态调用）：

| | 校准端求逆 | 动态端求逆 | 合计 |
|---|---:|---:|---:|
| 父 | 2 | 4 | 6 |
| 候选 | 3 | 0 | 3 |

父的校准端本来就有 2 次（底层校准路径自带，与本卡无关），两侧共有。**本卡的边际效果是"每校准 +1、每动态 −1"。**

**代价（`timing.py`，三臂配对含同字节空对照，15 轮，layer0/o，4096 通道）**：

| 段 | parent 中位 | 候选效应 | 空对照 | \|effect\|/\|null\| | 候选更快轮数 |
|---|---:|---:|---:|---:|---:|
| 校准 | 1151.030 ms | **−50.045 ms（候选更慢）** | −12.311 ms | 4.06× | **1/15** |
| 动态 | 484.594 ms | **+28.301 ms（候选更快）** | +2.457 ms | 11.52× | **14/15** |

### 7.1 我此前说错了一处，此处更正

我在 §7 的初稿里写过"盈亏平衡点是每个校准对应 1 次动态调用"。**那是错的**——它默认两侧的求逆代价相同。
实测两侧**并不相同**：校准端 +50.0 ms、动态端 −28.3 ms（同尺寸 4096）。故正确的关系是：

- **加**：每次校准 50.0 ms
- **省**：每次动态调用 28.3 ms
- **平衡点 ≈ 255 次动态调用**（不是 144），即**每个校准平均要对应 ≈1.77 次动态调用**才不亏

本地面板：144 校准 × 50.0 ms = **7.21 s 加**；288 动态 × 28.3 ms = **8.15 s 省** → **净省约 0.95 s**。
官方形状若只有 50 次动态调用：7.21 s 加 vs 1.42 s 省 → **净增约 5.8 s**；若每个样例流过全部 144 个位置
（7200 次）则大幅净省。**"官方有多少次动态调用"因此仍是决定符号的缺失事实**，且门槛现在是**255 次**而不是 144 次。

**限制**：只测了 layer0/o（4096 通道）一个 (layer, role)；2560 通道的角色两侧代价都会更小，平衡点需另行测量。
按计划，本地墙钟既不构成提交门也不构成关闭条件，此处只作账目记录。

## 8. 产物

- `workbench/full_solution/linear-lmc1-compiled-metric/audit.py` / `audit.json`（本地面板计数与官方口径引用）。
- 本记录；未占版本号、未跑 shard、未产出候选。

- `workbench/full_solution/linear-lmc1-compiled-metric/`
  （`audit.py` / `audit.json` / `build.py` / `build.json` / `candidate/solution.py` /
  `implementation.generated.py` / `verify.py` / `verify.json`）。
- `logs/execution/linear-lmc1-sixshard.out`、`artifacts/proxy_v3/linear-lmc1-{shard0,sixshard}-20260910/`。
- 归档见 `solutions/`（版本号在归档时分配）。
