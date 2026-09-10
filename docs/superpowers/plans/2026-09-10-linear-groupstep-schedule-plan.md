# Linear 精确度量下降的组号主序调度（L-EM2）

> 状态：CLOSED / RETAINED，v230 Linear 官方18428/292s，2026-09-10。
> （`solutions/20260910_v230_linear-em2-groupstep-schedule_scoreNA_timeNA/`，
> `official_status: PENDING`）。执行记录见
> [L-EM2 执行记录](../../../logs/execution/2026-09-10-linear-groupstep-schedule.md)。
> 六 shard `288/0/48`、等权均值 `+0.079454`；K 回退为 1，折算官方 `≈294 s`。
> 本卡是 L-EM1 的时间判定重规划卡。
> 当前完整根：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256
> `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
> 前一张卡
> [L-EM1 精确度量精化计划](2026-09-10-linear-exact-metric-refinement-plan.md)
> 因 §3.3 的时间估计被实测证伪（低估约 40 倍）而**不提交官方、不占版本号**；
> Attention 侧见 [Attention Q 侧加性 logit 偏置计划](parallel/2026-09-10-attention-qk-logit-bias-plan.md)。

## 1. 为什么重规划

L-EM1 的机制（精确度量 `G`、理想目标 `J`、pm1 候选、精确二次接受）**精度侧成立**：
shard0 配对 linear mean `+0.107791`（48/0/8 over layers 0/6/12/18），单 case 真实输出误差
`−21.80%`。判定它的是**时间**：

| 项 | 实测 |
|---|---|
| 逐组序贯动态开销 | `+0.5671 s/call`（最坏 +1.0193），unit 0.93–1.00 ms/组 |
| 折算官方 168 次 | `+95.9 s` |
| 校准钩子 | `+4.8 s` |
| 官方合计 | **≈382 s vs 300 s 门 → 必然 TIMEOUT** |

微优化救不回来：去掉两次 host sync 只省 ~10%（开销是**派发受限**，~25 µs/算子），
逐组改解析闭式仍折算 `+44.7 s`。**逐组 Python 循环在 300 s 门下不可行，与实现细节无关。**

## 2. 精度前沿探测（`probe_frontier.py`，真实 4B 数据，只读 CPU）

四个变体共享 L-EM1 的全部预注册固定选择（理想目标 `J`、pm1 候选、自然升序、
精确二次接受规则），**只改下降调度的顺序迭代结构**。`dL_true` 为真实输出平方误差下降。

| 变体 | 每 pass 顺序迭代 | layer0/q p1 / p2 / p4（in=2560） | layer0/o p1 / p2 / p4（in=4096） |
|---|---:|---|---|
| `seq`（L-EM1） | 640 / 1024 | −21.80% / −28.16% / −32.42% | −42.06% / −57.35% / −69.43% |
| `blockseq`（每 64 块） | 40 / 64 | −3.10% / −3.60% / −3.68% | — |
| `jacobi`（整层一次） | 1 | **0.0000%**（全部 pass） | — |
| **`groupstep`（组号主序 ×16）** | **16** | **−19.10% / −26.06% / −31.23%** | **−26.95% / −37.30% / −47.73%** |

1. `seq` p1 逐位复现已归档读数（`−21.8002%` / `−42.0623%`），探针口径无误；
2. `jacobi` 恒为 0（`accepted_rows=0`）：整层同时提议被整行精确校验一律否决 ——
   机制的价值来自**顺序刷新**，不是提议本身；
3. `groupstep` 用 **1/40** 的顺序迭代拿回 `seq` p1 的 88%（in=2560）/ 64%（in=4096）；
   in=2560 上 K=2（−26.06%）已超过 `seq` p1（−21.80%）；
   in=4096 上 K=4（−47.73%）才超过 `seq` p1（−42.06%）。

## 3. 调度成本（`bench_seq_gpu.py`，同 GPU、真实 shape）

| 实现 | in=2560 rows=128 | in=4096 rows=512 |
|---|---:|---:|
| `impl`（逐组，L-EM1） | 507.6 ms | 838.5 ms |
| `groupstep` K=1 | **19.4 ms** | **52.6 ms** |
| `groupstep` K=2 | **29.9 ms** | **95.7 ms** |

折算官方 168 次动态调用（120 次 in=2560 + 24 次 in=4096）+ 校准 4.8 s：

| 调度 | 官方合计 | 余量 |
|---|---:|---:|
| `groupstep` K=1 | ≈289.4 s | +10.6 s |
| **`groupstep` K=2（选定）** | **≈291.7 s** | **+8.3 s** |

## 4. 预注册固定选择

除下降调度外，全部与 L-EM1 逐条相同，**不重新搜索**：

| 项 | 值 |
|---|---|
| 目标 | 理想 `J(X)=||(X−T)W_hat^T||²`，`T = X_ref C G^{-1}` |
| 度量 | `G = h_inv^{-1} − c I`，`c = mean(diag(h_inv^{-1} − W_hat^T W_hat))` |
| 交叉矩阵 | `H = (W_hat − W)^T W_hat`，`W` 取输入 weight 对的 nvfp4 解码 |
| 候选集 | pm1：每个自然 4 元素组内一个元素，mantissa 码 ±1，固定三级 scale |
| 接受规则 | 精确二次 `2⟨δ, g⟩ + δ^T G_δδ δ < 0`（整行联合，若被否决则该行本 step 不动） |
| **下降调度** | **每 pass 16 步，step `k` = 全部 64 块的**第 `k` **组同时提议（组号主序）** |
| **passes** | **K = 2**（pass 之间重算精确梯度）→ **实测回退为 K = 1**，见下 |
| 层覆盖 | `in_features <= 4096`（`proj` 9216 不做） |

**时间驱动的回退（预先声明，非精度搜索）**：若真实配对实测折算官方 > 296 s，则 K 降为 1；
仍超则本卡关闭为 `not-submitted/time-infeasible`。

> **【回退触发，2026-09-10】K = 1。**
> `time_paired.py` 同进程配对实测（8 次中位）：动态 Δ K=2 `+11.6 s`（最小值估计量 `+11.0 s`）、
> K=1 `+7.8 s`（`+7.6 s`）。校准钩子用 `profile_compile_hook.py` 隔离测（`time_calibration.py`
> 配对整个校准调用时父单独就要 2–10 s，~40 ms 的 hook 不可分辨）：`+5.8 ~ +6.2 s`。
> 合计 K=2 `281 + 11.6 + 6.0 ≈ 298.3 s > 296 s`，**回退规则触发**；K=1 `≈294.7 s`，余量 ~5.3 s。
> 除规则外，1.7 s 余量对上 ±1.5 s 的噪声地板不构成余量，而越过 300 s 得分为 0。
> `official_status: submitted`，记于
> `logs/execution/2026-09-10-linear-groupstep-schedule.md`。
>
> 同次执行还**取消了校准侧的 Cholesky 逆**：`c = mean(diag(h_inv^{-1} − gram))` 拆成
> 校准侧存 `mean(diag(gram))`、动态侧取 `mean(diag(h_inv^{-1}))`（动态侧本来就要建 `h_inv^{-1}`）。
> 实数恒等、fp32 只差 ~1e-7，`verify.py` control C 实测 `diag_gap = 0`、`path_rel = 0`（逐位一致）。
> 故 §6 步骤 1 的"校准钩子逐字节相同"不再成立——校准钩子也已改动。

## 5. 单调性

每个 step 的行内联合移动都过**精确二次整行校验**（`row_delta @ G`，不是逐组近似），
被否决的行本 step 完全不动，因此每个 step 都不会增大 `J`，从而不会增大真实输出误差
`L = J + const`。与 L-EM1 同一条不变式，只是接受粒度从"组"提到"行内一步"。

## 6. 执行步骤

1. 建 `workbench/full_solution/linear-em2-groupstep-schedule/`，`implementation.py` 相对
   L-EM1 替换 `_em1_dynamic_descent` 的下降体，**并重构校准钩子的 ridge**（见 §4 回退注：存储
   `mean(diag(gram))` 取代校准侧求逆；§4 表格中的 `c` 公式在数学上不变、实现上拆成两半）；
2. `verify.py` CPU 控制 A–E（追加/隔离导入、父关闭逐位、H·ridge·G 恢复、单调性 +
   代价恒等式 + 梯度有限差分、确定性 + 状态往返）；真实 case 期望值与
   `probe_frontier` 的 `ideal/groupstep/pm1/p1`（`−19.0998%`）一致；
3. `time_paired.py` 同进程配对实测动态与校准，判定 K；
4. `run_shard.py` 跑 shard0 配对精度，折算时间合格后跑六 shard；
5. 归档官方提交候选（v230；v229 已被并行 Attention 线占用，两者同为根 56dc805d 的兄弟）、`archive.json`、日志、config，提交推送。

## 7. 与 Attention 计划的协调

- 开发阶段不占 GPU；任何 4B 评测前先 `nvidia-smi` 确认 Attention 线未在跑，单卡串行；
- 官方定价与组合顺序沿用 L-EM1 卡 §6.2：Linear 与 Attention 两条线各自独立消融，
  组合顺序以官方实测为准；
- calibration cache 继续按 SHA 修剪，`--min-age-hours 2`。

## 官方裁决（2026-09-10）

v230 Linear（L-EM2）+ v195 Attention 已由用户官方回传 **18428 / 292s**，相对 v202 完整根 **+375 / +11s**，硬限余量 **8s**。根与归档逐位一致，SHA256 `0F1AF6DBC207FF32B2C6BE16987E9C4FE50F3F10747DE26782EF52A6F2FAB7BC`。回退根 v202 保留 `18053/281s`。同编号 v230 Attention A-FIX1 官方状态不受本次回传影响。 计分/归档 SHA `0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc`；根原样晋级，既有控制与六 shard 结果复用，未重跑评测。历史时间预测不据此恢复为门禁。
