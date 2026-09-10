# L-MC1（把固定度量重建移到校准）—— 审计记录：一个阻塞性的缺失事实

计划卡：[`docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md) §4。
工作目录：`workbench/full_solution/linear-lmc1-compiled-metric/`。设备：CPU（未用 GPU）。

**状态：审计未完成，卡未关闭也未开工。** 阻塞点是一个仓库没有记录的事实（见 §2），
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

## 3. 另外两个已测到的成本（与上面的口径问题无关）

**3.1 设备/dtype 边界（待测，计划已点名）**
`h_inv` 由 `_cpu_state_tensor` 存为 **CPU float32**；动态路径在 `_em1_metric` 里
`h_inv.to(device=device, dtype=torch.float32)` 搬到运行设备后才做 Cholesky。
前移到校准端只有在校准与之后每一次动态调用**同设备同 dtype** 时才逐位安全——
计划 §4 已点名"校准与动态设备不同可能造成数值差异，按实际contract列明"。**尚未实测。**

**3.2 state 体积接近翻倍（已测）**
`metric` 与已存的 `h` 同为 channels² float32，故每个在范围内 state 的 em1 载荷翻倍：

| role | channels | 单个新增 | ×24 层 |
|---|---:|---:|---:|
| q/k/v/fc_gate/fc_up | 2560 | 26.2 MB | 5 × 24 × 26.2 MB |
| o | 4096 | 67.1 MB | 24 × 67.1 MB |

合计新增约 **4.75 GB**。对照现状：六个 linear 校准缓存已是 **38.05 GB**（每个 6.81 GB），
故本次增加约 +12%（每个 shard +0.79 GB）。计划要求把这一项"与旧路径峰值内存、累计 state 大小、
传输次数同表记录"——此处先给磁盘侧的数，内存/传输侧待实测。

## 4. 尚未做的（审计未完成）

- 设备/dtype 逐步等价实测（§3.1）；
- `validate_state` 是否接受更大的 em1 载荷（新增字段合法性）；
- 校准端 h_inv 生成/修改顺序、两处 metric 调用（`solution.py:12108` 原始 descent 与 `:12423`
  L-TF2 影子，**只有影子是活的**）的最终生效路径与共享 state 调用者；
- 峰值内存与 state 搬运次数。

## 5. 产物

- `workbench/full_solution/linear-lmc1-compiled-metric/audit.py` / `audit.json`（本地面板计数与官方口径引用）。
- 本记录；未占版本号、未跑 shard、未产出候选。
