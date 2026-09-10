# Linear 精确度量 Activation 码序下降与双线协调计划（L-EM1）

> 状态：ACTIVE，总协调计划，2026-09-10。
> 当前完整根：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256
> `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
> 本计划负责 Linear 执行和两条优化线的统一父版本、版本登记、组合顺序与最终归档；Attention 算法见
> [Attention Q 侧加性 logit 偏置计划](parallel/2026-09-10-attention-qk-logit-bias-plan.md)。
> 上一张 Linear 卡
> [L-XR1 完整输出交叉残差纠码](../archive/plans/2026-09-10-linear-cross-residual-correction-plan-rejected.md)
> 已本地关闭为 `REJECTED`。

## 1. 目标与算法判断

L-XR1 的归因是明确的：块外 `G` 分量主导 `offblock_rel` 中位 0.82，固定 rank-4 只捕获 39%，
冻结梯度系统性低估纠码代价。本轮不再使用任何低秩近似或冻结梯度，而是实现精确度量本身：

**L-EM1：理想输出可达目标上的精确度量 Activation 码序下降。**

只改部署 Activation 的 mantissa 码，Weight 五字段、hierarchy、scale 层级、permutation 全部不变。
校准阶段编译一个固定矩阵 `H`；动态阶段用父状态里已有的 `h_inv` 精确重建度量 `G`，按自然 64 通道块
序、每个 4 元素组用固定 `±1` 码候选做一次**精确序贯**下降，每次接受后刷新梯度。

与已关闭机制的区别：不是静态 Weight 边界（L-RB1）、不是 A@W gain、不是共享 threshold/offset/rank
调整（L-JRB1）、不是低秩/冻结梯度纠码（L-XR1），也不扫描任何参数、层名单、阈值或粒度邻域。

## 2. 精确目标与单调性

记 dense/部署 Activation 为 `X`、`X_hat`，dense/部署 Weight 为 `W`、`W_hat`，
`X_ref` 为样本参考 Activation（动态 API 的输入），

`L(X) = || X W_hat^T - X_ref W^T ||^2`，

`G = W_hat^T W_hat`，`C = W^T W_hat`，`H = G - C = (W_hat - W)^T W_hat`，

`P = W_hat G^{-1} W_hat^T`（到 `range(W_hat)` 的正交投影），`T = X_ref C G^{-1}`。

定义

`J(X) = || X W_hat^T - T W_hat^T ||^2 = tr((X - T) G (X - T)^T)`。

则

`L(X) - J(X) = || X_ref W^T (I - P) ||^2`，

**与 `X` 无关**：因为 `(I-P) W_hat = 0`（故 `<X W_hat^T, X_ref W^T(I-P)> = 0`）且
`(I-P) P = 0`（故 `<X_ref W^T P, X_ref W^T (I-P)> = 0`）。因此

> 对 `J` 的任何下降都**恰好**是真实输出平方误差 `L` 的下降。

`J` 对 `X` 的半梯度（后文统一用此约定）为

`g = (X - T) G = (X - X_ref) G + X_ref H`，

单个 4 元素组的候选移动 `delta`（码 `±1`，`scale_factor/lv2/lv3` 固定）的精确代价为

`DeltaJ = 2 <delta, g_cur> + delta^T G_delta delta`，

其中 `g_cur` 是**当前点**的梯度；接受后按 `g += delta G[sl,:]` 精确刷新，所以序贯下降是精确的
坐标下降，接受条件 `DeltaJ < 0` 使 `L` 严格不增。数值上 `g` 的两项都是小量
（`X - X_ref` 是量化误差、`H` 是 Weight 量化误差），无灾难性抵消；`H` 以 fp32 保存。

`G` 不需要额外存储：父状态保存 `h_inv = (G + c I)^{-1}`，`c` 是父自适应选择的 ridge
（`c = reg * mean(diag G)`）。由 `h_inv^{-1} - G = c I` 得
`c = mean(diag(h_inv^{-1} - W_hat^T W_hat))`，随后 `G = h_inv^{-1} - c I`。
本机 CPU 实测（shard0 校准缓存，float64 对照）：

| 层/角色 | n | c | `||D - cI||/||G||` | `G` 重建相对误差 |
|---|---|---|---|---|
| layer0/q | 2560 | 6.659657e-02 | 1.116e-06 | 1.116e-06 |
| layer0/o | 4096 | 5.011905e-02 | 5.068e-06 | 5.069e-06 |

即 `G` 可从父状态以 fp32 精确重建（相对误差 `~1e-6`），无需修改父校准口径，也无需额外存 `G`。

## 3. 探测证据（CPU 只读，float64 真值重算）

`workbench/full_solution/linear-em1-exact-metric-refinement/probe_headroom.py`
在真实 shard 校准缓存 + `qwen3.5-4b-proxy-v2.pt` 上重建父部署，报告**真实重算**的输出平方误差变化
（`dL_true`），不是代理量。候选集 `pm1`（每个自然 4 元素组内一个元素码 `±1`，8 候选/组，16 组/64 块）。

误差分解与下降（rel 相对该 case 的 `L0`）：

| 层/角色 | n | `|A|^2` | `|B|^2` | dense/seq | ideal/seq |
|---|---|---|---|---|---|
| layer0/q | 2560 | 0.385 | 0.613 | −6.45% | **−21.80%** |
| layer0/o | 4096 | 0.189 | 0.809 | −8.46% | **−42.06%** |
| layer0/proj | 9216 | 0.190 | 0.809 | −12.31% | **−75.82%** |
| layer2/q | 2560 | 0.388 | 0.608 | −13.63% | **−37.69%** |

结论与固定选择：

1. `|B|^2`（Weight 量化误差）占 61–81%，dense 目标（只优化 A）只能拿到 −6%…−14%；
   ideal 目标能同时吸收 `B` 的可补偿分量，收益是 dense 的 3–6 倍，因此**目标固定为 ideal `J`**；
2. 运行时可行的**张量化块内同时提议**变体（`blockseq`）在 layer0/q 只有 −3.10%（vs 精确序贯
   −21.80%），因此**放弃该变体**，固定使用精确序贯刷新；
3. 精确序贯的调度开销实测（CPU，逐组 9 个算子，`rows=8` 纯调度）：640 组 `66.7 ms`、
   1024 组 `122.9 ms`；折算 GPU 每调用约 50–100 ms，按每 shard 4 层 × 7 角色共 28 次调用估计
   **新增 ~2–2.5 s**，官方 300s 硬限下当前余量约 19 s，满足；
4. 覆盖范围固定为 `in_features <= 4096`（`q/k/v/o/fc_gate/fc_up`）。`proj`（9216）本轮**不做**：
   其 `H` 需要 340 MB/状态、每 shard 1.36 GB，且每次调用 `n^3` 求逆约 0.15 s；作为后续卡
   L-EM2 单独处理（可用 `A=(W_hat-W)^T`、`F=W_hat^T` 两个 fp16 因子把 `H` 压到 94 MB/状态）。

## 4. 固定实现

### 4.1 校准编译（`hif4_calibration_and_quantize_weight` 后处理）

1. 完整执行父校准/Weight 编码，取得最终 `W_hat` 与父状态（`h_inv` 已在父状态中）；
2. 用父的 smooth/permutation 口径重建 dense `W`，计算 `C = W^T W_hat`、`G = W_hat^T W_hat`、
   `H = G - C`；
3. 由 `h_inv` 与 `W_hat` 求 `c = mean(diag(h_inv^{-1} - G))`，只保存标量 `c` 与矩阵 `H`（fp32）；
4. 仅对 `in_features <= 4096` 的层保存 `H`；宽层保持父实现（无新状态）；
5. 不搜索 rank、阻尼、层名单、覆盖率、接受阈值；不保存 `G`（动态阶段重建）。

新增状态字段只有 `em1_h`（`in x in` fp32）、`em1_ridge`（标量）、`em1_version`。
Weight 五字段、Linear transform、permutation 和父 activation state 其他字段保持不变。
每 shard 新增约 `(5 x 26 + 68) x 4 = 0.79 GB`（相对父状态 2.15 GB 为 +37%），
磁盘校准缓存由 6.0 GB 增至约 6.8 GB。

### 4.2 动态下降（`hif4_dynamic_quantize_activation` 后处理）

1. 完整执行父动态编码，解码得到 `X_hat`；由输入对得到 `X_ref`；
2. `G = h_inv^{-1} - c I`（一次 Cholesky + 逆，`n^3`：narrow ~3 ms、o ~14 ms）；
3. 半梯度 `g = (X_hat - X_ref) G + X_ref H`（两次 `rows x n x n` 矩阵乘）；
4. 按自然升序处理 64 通道块；每块取 `G` 的 64×64 子块，按组 0→15 序贯：
   候选 `delta` 一次张量化（8 个候选 × 全部行），代价 `2<delta,g_cur> + delta^T G_gg delta`
   （组内 4×4 子块），取最小代价且 `< 0` 的候选接受，接受后 `X_cur` 与 `g` 精确刷新
   （`g += delta G[sl,:]`）；
5. 只走一遍，不重复、不刷新到初始梯度、不改 hierarchy/scale、不做 Python 候选轮询；
6. 写回 `mant = code * 0.25`，码为 0 时写规范零 sign；`sign/scale_factor/lv2/lv3` 结构不变。

动态 API 只做固定矩阵乘与组内二次比较，不做校准搜索、矩阵分解或候选邻域扫描。

## 5. 执行步骤

工作目录：`workbench/full_solution/linear-em1-exact-metric-refinement/`。
日志：`logs/execution/2026-09-10-linear-exact-metric-refinement.md`。
输出：`artifacts/proxy_v3/linear-em1-<run-id>/`。

1. 从本计划启动时记录的完整根复制单文件候选（append-only 覆盖两个 API），不修改根和任何归档源码；
2. `verify.py` 做六 API 独立导入、合法 state、finite、父关闭 control、有限差分校验梯度恒等式、
   单调性断言（`DeltaL == DeltaJ <= 0`）、`G` 重建误差、`H` 保存/加载一致性与确定性；
3. 运行 Linear shard0 排除接口错误和死分支，并记录每 case 的 `DeltaL` 与真实重算对照；
4. 只要真实调用形成合法、非等价硬输出（`changed_mantissa > 0` 且输出非逐位相同），即固定运行
   Linear 六 shard 一次；本地正负和时间只记录；
5. 保存源码、配置、SHA、attempted/accepted、五字段 changed count、336 case 配对结果和单文件导入结果；
6. 将唯一代表候选交官方裁决。只有官方分数提高且时间 `<300s` 才晋级。

无真实码变化记 `NO_REACHABILITY`；官方负向或超时关闭 L-EM1，不缩覆盖、不改候选步长或块序重试。
`proj` 层不在本卡范围内，若需要另开 L-EM2，不改本卡的固定选择。

## 6. 与 Attention 计划的协调

### 6.1 开发隔离

- 两条线都冻结同一份启动根 `R0`；任何一边先完成都不得改变另一边的源码父或结果口径；
- Linear 只修改 `hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation` 及其私有 helper；
- Attention 只修改 calibration attention、动态 Q/K 及其私有 helper，V 保持父实现；
- 两条线使用独立 workbench、日志、artifact 和校准缓存；单张 GPU 上评测串行，禁止同时跑 4B；
  启动任何 GPU 运行前先 `nvidia-smi` 确认无 Attention/其他 4B 进程占用（显存 <2 GiB 且无 torch 进程）；
- Linear 候选必须证明 Attention 三 API 在固定输入上与 R0 一致；Attention 候选必须证明 Linear 两 API
  与 R0 一致。

### 6.2 官方定价与组合

两个单侧候选都从同一完整 R0 构建，并分别作为完整六 API 文件提交，不建立 Linear/Attention 侧父：

| Linear 官方结果 | Attention 官方结果 | 动作 |
|---|---|---|
| 非正向 | 非正向 | 两个机制分别关闭，根保持 R0 |
| 正向 | 非正向 | L-EM1 成为新完整根，Attention 机制关闭 |
| 非正向 | 正向 | A-QB1 成为新完整根，Linear 机制关闭 |
| 都正向 | 都正向 | 先选官方分更高的单机制完整候选为父，再在该源码上重新应用另一机制，构造一个组合候选 |

事实（2026-09-10）：A-G1 已本地关闭为 `REJECTED`（v227，六 shard 等权 `-0.005294`，未提交官方），
根保持 R0；Attention 线由 A-QB1 承接。L-XR1 已本地关闭为 `REJECTED`
（shard0 配对 `delta_mean=-0.040082`、`0/40/16`），不分配版本号、不提交官方，根保持 R0。

组合候选不是复制粘贴两个归档文件。它必须从较高分完整父重新构建，重算全部 calibration state，先做
目标两侧 control，再运行一次 `--scenario both` interaction audit，最后交官方。组合是否晋级仍只看
完整官方分数和 `<300s`；不得用两个单候选分数相加预测组合收益。

官方结果回传可能异步，但版本号、归档和根切换由本计划串行登记，避免两个执行线抢占同一版本或覆盖
`solution.py`。

## 7. 缓存、提交与结束

- 两条线存活期间，缓存清理同时保留 R0、L-EM1、L-XR1 和 A-QB1 前缀，并使用 `--min-age-hours 2`；
  绝不删除 `qwen3.5-4b-proxy-v2.pt`；
- 每个机制只有一个固定配置和一个正式代表，不按本地 shard 结果扫描参数邻域；
- 本地 paired、holdout、API 时间均为诊断，不是提交门；官方分数与 `<300s` 是唯一晋级依据；
- 实质实现和结果分别提交，归档目录按实际结果标记，拒绝候选名包含 `rejected`；
- Linear 与 Attention 单机制均获裁决，且必要的组合候选完成裁决后，本计划结束并移入 archive。
