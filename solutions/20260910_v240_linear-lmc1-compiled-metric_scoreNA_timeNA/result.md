# v240 — Linear L-MC1：把固定度量重建移进校准

活动计划 [`docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md) §4。
执行记录 [`logs/execution/2026-09-10-linear-lmc1-compiled-metric.md`](../../logs/execution/2026-09-10-linear-lmc1-compiled-metric.md)。

| | |
|---|---|
| 父根 | **v237**（L-TF2 首遍梯度复用，K=2 + v195 Attention），官方 **18518 / 289 s**（余量 11 s） |
| 父 SHA256 | `ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554` / 516697 B |
| 候选 SHA256 | `2c344722e3cbf2a32b72c89319309c47a54b9fcd16dd36111491f60248d01994` / 524220 B |
| 改动 | 两个追加影子：`_em1_compile_metric` 在校准端算出并存入最终 G；`_em1_metric` 动态端只加载 |
| 本地六 shard | 对 v237 根 **336/336 精确零**（0/0/336） |
| 代价账 | 校准 **+50.0 ms**、每次动态调用 **−28.3 ms**（4096 通道）→ **平衡点 ≈255 次动态调用** |
| 官方 | **`unregistered/NA`**。不写本地秒数预测 |

## 机制：这不是消除工作，是搬运工作

`_em1_metric` 每次动态调用重建 G：

```python
inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
ridge   = mean(diag(inverse)) - gram_diag_mean
metric  = inverse; metric.diagonal().sub_(ridge)
```

**这些只依赖 `h_inv` 与标量 `gram_diag_mean`，与 activation 无关**，所以确实是重复工作——但根自身的注释
（`_em1_compile_metric`）写明当前是**故意**不在校准端求逆（"~2.8 s over the official 144 in-scope
calibrations"）。所以本卡把这次求逆**从动态端搬到校准端**：

- `_em1_compile_metric`：校准结束时按**同一表达式、同一输入、同一运行设备**算出 G，减掉 ridge 后存入 state；
- `_em1_metric`：只加载，无 Cholesky、无求逆、**无写入**（ridge 已减，所以"第二次调用再减一遍"的坑不存在）。

## 最隐蔽的一处：内存布局也是等价性的一部分

第一版实现**通过了"存储的 G 与父重建的 G 逐位相同"这一条，五字段却仍然不同**，而且
`scale_factor`/`scale_lv2`/`scale_lv3` 全同、只有 `sign`/`mant` 不同。逐层定位后：

| | 存储的 G | 父重建的 G |
|---|---|---|
| 数值字节 | **相同** | **相同** |
| stride | `(2560, 1)` 行主序 | **`(1, 2560)` 列主序** |
| `is_contiguous()` | True | **False** |

**`torch.cholesky_inverse` 返回的是转置 stride 的张量（LAPACK 惯例）**，而 `_cpu_state_tensor` 里的
`.contiguous()` 把它拍平了。数值完全一样，但下游 `(deployed - reference).mm(metric)` 面对不同 stride
会走不同的 cuBLAS 路径，舍入因此不同——**足以改变离散的 mantissa 码分配**。

修法：存 G 时**不强制 contiguous**（实测 `nan_to_num` 与 CPU↔CUDA 往返都保 stride），
动态端 `.to(device)` 把 stride 一并带回来。

**计划 §4 写的是"按原表达式、原 float32 顺序构建 G"——方向对，但只说到了顺序，没说布局。**
本案里只保数值不保布局，等价性会以"前三个字段全对、只有离散码不同"的形态失败。

## 验证（`verify.py` / `verify.out`，CUDA，真实权重，全 PASS）

| 控制 | 结果 |
|---|---|
| A | 纯追加；Attention 与动态激活字节码相同；两个影子都是活定义 |
| B | **3 个 (layer, role) × 3 次动态调用，五字段逐字节相同**；存储的 G 与父每次重建的 G **逐位相同**（3/3） |
| C | 次数账见下 |
| D | 同一 state dict 连调三次字段逐字节相同、**存储的 G 未被改动**（ridge 只减一次） |
| E | 超范围宽度（proj 9216）两侧都不存；无 em1 载荷两侧同样穿透；无 metric 的 state 两侧 arm 同为 `no-metric` |
| F | 校准确定性：两次运行存的 G 逐字节相同 |

**等价性必须在 CUDA 上判**：实测 CPU 与 CUDA 的 `cholesky_inverse` **不逐位相同**（相对差 ~1.0e-06），
而 float32 张量的 CUDA→CPU→CUDA 往返**精确**。所以 G 在**运行设备**上算、以 CPU 副本入 state
（`validate_state` 要求 CPU）、动态端搬回同设备——评测器两侧用的是同一个 `--algorithm-device`，
且校准缓存 identity 本就含 device，故该条件在现行口径下自动成立。

## 代价账（`timing.py`，三臂配对含同字节空对照，15 轮，layer0/o，4096 通道）

| 段 | parent 中位 | 候选效应 | 空对照 | \|effect\|/\|null\| | 候选更快轮数 |
|---|---:|---:|---:|---:|---:|
| 校准 | 1151.030 ms | **−50.045 ms（更慢）** | −12.311 ms | 4.06× | **1/15** |
| 动态 | 484.594 ms | **+28.301 ms（更快）** | +2.457 ms | 11.52× | **14/15** |

**两侧的求逆代价并不相等**（50.0 ms vs 28.3 ms），所以：

- **加**：每次校准 50.0 ms；**省**：每次动态调用 28.3 ms
- **平衡点 ≈255 次动态调用**（每个校准平均对应 ≈1.77 次）
- 本地面板：144 × 50.0 ms = 7.21 s 加；288 × 28.3 ms = 8.15 s 省 → **净省约 0.95 s**
- 官方若只有 50 次动态调用：7.21 s 加 vs 1.42 s 省 → **净增约 5.8 s**

**执行记录里有一处自我更正**：初稿写过"平衡点是每个校准对应 1 次动态调用"——那是错的，
默认了两侧代价相同。更正已写进记录，不静默改掉。

**限制**：只测了 4096 通道的一个 (layer, role)；2560 通道两侧代价都更小，平衡点需另测。

## 六 shard 本地结果

`--linear-only` 六 shard，基线 **v237 根**（本卡的直接父）：**336/336 精确零**（0/0/336），
六条记录全 `ok`，`analysis-*.json` 6 份。

`stopped_early: true` 同前几卡的结构性读数：停止检查在 shard 结果 append **之后**，等零候选使计数器
**在最后一个被请求的 shard** 上触顶；**无截断**。

## 边界（这份结果不主张什么）

- **符号未定，且已登记为缺失事实。** 本卡搬运而非消除工作，所以赚不赚取决于调用次数比。
  **官方评测调用 `hif4_dynamic_quantize_activation` 多少次，仓库里没有任何记录**：
  "50 Linear"样例是一个 `(layer, role, 窗)` 三元组（则动态 50 次、净增约 5.8 s），
  还是一次整模型前向、每个样例流过全部 144 个位置（则大幅净省）？**不拿口径推测关卡**——
  本仓库刚有一次同类教训（v237 的本地不可分辨曾被外推成"官方大概率超时"，被官方 −2 s 证伪）。
- **不换算官方秒数**：官方 300 s 是唯一时间门。
- **state 体积接近翻倍**：`metric` 与已存的 `h` 同为 channels² float32，新增约 **+4.75 GB**；
  对照六个 linear 校准缓存现有 38.05 GB，约 +12%。
- 等价性覆盖三个 (layer, role)（含 4096 与 2560）× 各 3 次动态调用，加超范围/无 em1/无 metric/确定性；
  没有扫全部 24 层全部 role。

## 复现

```powershell
.venv\Scripts\python.exe workbench/full_solution/linear-lmc1-compiled-metric/audit.py
.venv\Scripts\python.exe workbench/full_solution/linear-lmc1-compiled-metric/build.py
.venv\Scripts\python.exe workbench/full_solution/linear-lmc1-compiled-metric/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/linear-lmc1-compiled-metric/candidate/solution.py --baseline-solution solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py --linear-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/linear-lmc1-shard0-20260910
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/linear-lmc1-compiled-metric/candidate/solution.py --baseline-solution solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 6 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/linear-lmc1-sixshard-20260910
.venv\Scripts\python.exe workbench/full_solution/linear-lmc1-compiled-metric/pair_sixshard.py
.venv\Scripts\python.exe workbench/full_solution/linear-lmc1-compiled-metric/timing.py
```
