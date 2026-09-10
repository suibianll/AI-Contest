> 生命周期：SUPERSEDED，2026-09-10。开发结果归档、未知官方回传单列；不再提供当前指令。下一步只见[计划入口](../../plans/README.md)。旧计时外推、DD1等价性/成本归因按本轮总结修正。

# Linear 动态下降的派发削减：把 K 的边际价格打下来（L-DD1）

> 状态：ACTIVE，2026-09-10。L-EM3/v231 之后的**重新规划卡**。
> 当前完整根：v230 Linear（L-EM2，K=1）+ v195 Attention，官方 **18428 / 292s**，
> SHA256 `0F1AF6DBC207FF32B2C6BE16987E9C4FE50F3F10747DE26782EF52A6F2FAB7BC`。
> 前一臂 [L-EM3 K=2](2026-09-10-linear-k2-timed-arm-plan-superseded.md) 已归档 v231
> `solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/`，**官方 `PENDING`**，折算 `295.2s`。
> 侦察记录见 `workbench/full_solution/linear-em4-metric-arch/FINDINGS.md`。

## 1. 为什么是"派发"，而不是钩子

计划卡 L-EM3 登记"校准钩子折算 **+5.8 ~ +6.2 s**"，并把它当作机制剩余的最大单项成本。
L-EM4 侦察**推翻了这个数**：钩子真实成本约 **+2.56 s**（折 144 次在范围内校准）。

误差不在测量本身，在探针：`profile_hook_breakdown.py` 的**分量**用 GPU params 计时（≈2.4 s），
而**整钩子**用的是校准缓存里的 **CPU** params（5.59 s）。缓存是 `_cpu_params` 落盘的格式
（`official_eval.py:2583`），但官方路径上 params 在 GPU——`probe_real_calibration_path.py` 按
官方调用形状调用真实候选，返回的五个 param 全部 `cuda:0`。设备一错配：

| | CPU | GPU |
|---|---:|---:|
| `_dequantize_hif4` | 20.97 ms | **0.74 ms** |
| 整钩子 in=2560 / in=4096 | 39.11 / 52.05 ms | **15.94 / 26.87 ms** |

所谓"无法归因的 3.2 s 缺口"就是这一次错配，**在官方根上并不存在**。

修正后，Linear 线的成本重心落在**动态 API**：

| API | 绝对成本 | 次数 |
|---|---:|---|
| `hif4_dynamic_quantize_activation` | **0.45–0.51 s / call** | 每个 Linear case 一次 |
| 校准钩子 | 0.016–0.027 s / call | 144（在范围内） |

而且它是**派发受限**的，不是算力受限：单次调用（in=2560）发出约
**19 269 次 `as_strided`**、6 852 `view`、6 768 `permute`、6 554 `unsqueeze`、5 598 `reshape`、
4 149 `copy_`、1 504 `bmm`、752 `einsum`，平均单 kernel **5–90 µs**。

### 为什么这直接关系到精度

现场核对（同一 state、同一 activation，同进程交替）：

```
0/q in=2560  ROOT(v230 K=1) 530.04 ms    v231 (K=2) 528.68 ms
0/o in=4096  ROOT(v230 K=1) 433.86 ms    v231 (K=2) 423.05 ms
```

**K=1 与 K=2 的绝对成本在噪声内相同**，第二遍 pass 只值 **+3.2 s**（折 144 次）。
即：**每调用的固定开销主导，K 的边际很小**。K 的阶梯是 p1 上
`-19.10%`（K=1）→ `-26.06%`（K=2）→ `-31.23%`（K=4），而 K=3 在 295.2 s 的投影上会落到
≈298.4 s，**只剩 1.6 s 余量**，不敢提。

**因此削减固定开销是唯一同时改善两件事的杠杆**：它降低基座，也直接降低 K 的价格。
本卡的目标就是把"下一个精度台阶"缺的那笔预算腾出来。

## 2. 预注册（在测量之前写定，不得事后更改）

| 项 | 值 |
|---|---|
| 目标 | `hif4_dynamic_quantize_activation` 的**每调用固定开销**（不是 K 的边际） |
| 手段 | 削减小算子派发与中间张量重排：合并逐 group/逐 coordinate 的循环、消除 `permute`/`unsqueeze`/`reshape` 中转、复用已分配的缓冲 |
| 不改的 | 机制本身：精确度量 `G`、交叉矩阵 `H`、理想目标 `T`、pm1 候选集、整行联合精确二次接受——**逐条不变** |
| 不改的 | `_EM1_PASSES` 仍为 2；不得扫描 K；不得改动 `solutions/` 任何归档源码 |
| 精度要求 | **首选逐位相同**：六 shard 336 case 与 v231 候选 `ea79a1c1…` 的 `source_sha256` 结果逐位比对 |
| 精度回退 | 若重排无法避免 fp32 重结合，则以六 shard 等权 Δ 与 v231 对比，容差 **|Δ| ≤ 0.002**（L-EM3 的 case 级噪声地板量级），并在归档中**披露重结合点** |
| 时间 | 记录，不设门（官方时间是唯一的门） |
| 成功判据 | 每调用 `as_strided` 计数下降 ≥ 50%，且单次调用 wall 时间中位下降 ≥ 30% |
| 失败判据 | 派发计数下降但 wall 不变（说明瓶颈不在派发）→ 记 `NO_EFFECT`，只存 `workbench/result`，**不占版本号** |
| 归档条件 | 仅当精度要求满足**且**时间判据满足才归档；否则按失败判据处理 |

> **为什么"不改机制"这条要写死。** 本卡是**纯时间卡**。K 的阶梯已经证明精度还在涨，
> 但涨的每一步都要 3.2 s 而余量只有 5 s。把机制和时间同时改动，就再也分不清
> 精度变化来自谁——L-EM3 的换算口径教训（一个数据点、非全局结论）已经吃过一次
> 归因不清的亏。本卡只动时间，精度必须守成。

## 3. 方法

1. **建立基线**：在真实 state 上跑 `profile_dynamic_api.py`，记录每调用的
   `as_strided / view / permute / unsqueeze / reshape / copy_ / bmm / einsum` 计数与 wall 中位。
   基线已存在于 `FINDINGS.md` §3。
2. **定位派发源**（已初定位，见下）。`_em1_dynamic_descent`（solution.py:12070）的内层是
   `for _pass in range(2): for step in range(16):`，共 **32 步**。L-EM3 卡片 §3 已实测单步
   稠密路径在 in=2560/rows=128 下 **9.47 ms**，即 **32 × 9.47 ≈ 303 ms/call**，与本次实测的
   ~0.50 s/call 吻合（其余为 setup/梯度/反量化）。

   单步内约 35 个 Python 级算子，但 profiler 记到 **1 504 次 `bmm`/call ≈ 47 次/步**、
   **752 次 `einsum`/call ≈ 23.5 次/步**——派发量远超算子数，来源是

   ```python
   quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)
   ```

   `k=8, b=40` 的广播把这一句拆成几十次微型 `bmm`。等价改写为
   `(delta @ local_gram * delta).sum(dim=-1)`（一次广播 `bmm` 加两次逐元素）
   预计把 ~47 次 `bmm` 降到 1 次，是单步内最大的一处派发削减。
   **该改写是同一表达式的重结合，不保证逐位相同**——按 §2 走精度回退条款。

   > 已排除的方向：L-EM3 卡片 §3 已实测"稠密 `row_delta` 换成列积"在 rows=128 下**更慢**
   > （1.10×，官方面板以 rows=128 为主），该卡已据此决定不做。本卡不重开该方向，
   > 原因是它**增加**而非减少派发。
3. **改写**：以"同一批张量、同一个数学、更少的算子"为唯一目标。逐步骤做，每步后重跑计数，
   记录该步的计数下降与 wall 变化。
4. **精度验证**：六 shard，与 v231 候选逐 case 比对。首选逐位相同；不逐位则按 §2 回退并披露。
5. **归档**：满足判据才建版本号；否则存 `workbench/result`。

### 已知的顺带项（本卡内做，量小）

- `_cpu_state_tensor`（solution.py:6468）在 D2H 之后对 host 上的 n² 矩阵做 `nan_to_num`
  再 `.contiguous()`；钩子在调用前已用 `torch.isfinite(h_matrix).all()` 验过有限性，
  故该 `nan_to_num` 在此路径**可证为空操作**。去掉它，以及调用点
  `_cpu_state_tensor(h_matrix.contiguous())` 的冗余 `.contiguous()`。
  **注意 `_cpu_state_tensor` 是共享助手**（Attention 线也用），须逐个调用点确认前提，
  不能全局改。
- 该项折 144 次约 **-0.33 s**，且按构造逐位不变。

## 4. 边界与纪律

- 只动 `workbench/full_solution/linear-em4-metric-arch/` 下的工作副本与候选；`solutions/` 只读。
- 单张 GPU 串行：**不得与 Attention 进程同时跑 4B**。跑前查 `nvidia-smi`。
- 不扫描参数/seed/阈值/粒度邻域。
- 无效机制只保存 `workbench/result`，不占版本号。
- 官方 300s 硬限是唯一时间门；本地 Δ 与本地时间只作诊断。
