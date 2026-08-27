# HiF4 26000 分算法实施计划

日期：2026-08-27
目标定位：主目标为官方 22000~25000；26000 为 stretch 目标（详见第 1 节目标分级）
状态：已按官方 `A @ W` 禁令完成合规修订；2026-08-27 评审后补充目标分级、Q0 教训、向量化硬性要求与官方时间上限，待实施
历史分数锚点：C21 / v024，官方 `16043 / 173.8s`；后续父版本必须先替换为 C21-C 合规基线
当前源码 SHA256：`40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`

## 0. 给实施 AI 的执行指令

本计划用于指导后续 AI 在现有仓库中实现新算法。实施时必须遵守以下规则：

1. **规则零：Linear 校准、参数搜索、gate、fallback 和动态激活量化中，不得显式或隐式计算 `A @ W`，也不得用与 `A @ W` 数学等价的监督信号拟合、选择或倒推出 `Q(A)`。合规优先于分数、速度和历史 Champion。**
2. 根目录 `solution.py` 始终是唯一可提交算法文件，不得依赖本地模块、文件、网络或 NumPy。
3. 每个候选只引入一个主要机制；候选失败时回退该候选，不回退合规 Champion。
4. 任何改动前先在执行日志预注册 candidate ID、父 SHA、唯一机制、评测矩阵、时间预算和晋级门。
5. 所有新增机制必须有 feature flag；flag 关闭时与父版本字段级、数值级等价。
6. 所有 calibration state 必须是 CPU、contiguous、strided、finite、无梯度的普通数据。
7. 六个官方 API、HiF4 五字段名称和参数顺序不得改变。
8. Linear-only 候选必须保证 Attention 全矩阵逐 case 不变，容差 `1e-6`。
9. 不得把本地 Linear/Attention 指标换算成官方绝对分数用于归档；本文分数换算仅用于战略目标判断。
10. 不得同时实现 C22、C23、C24。必须按顺序完成、评测、归档，再开始下一候选。
11. C21 的 pow2 fallback 和 A1 Attention selector可以保留；C21 的 Linear output gate、exact-cross activation 路径和所有由 Linear 输出决定 `Q(A)` 的逻辑必须先删除。

### 0.1 规则零的不可规避解释

以下做法全部按违规处理，即使没有在内存中生成名为 `A @ W` 的 Tensor：

- 计算 `A @ W.T`、`Q(A) @ Q(W).T` 或两者残差，并据此选择 activation scale、mantissa、rotation、seed、coverage、gate 或 fallback；
- 用 trace 展开、分块乘法、低秩分解、采样行、缓存输出、teacher label 等形式恢复同一 Linear 输出监督；
- 使用包含精确 Weight 残差的 cross term，例如 `A (W_q-W)^T`、`A (W_q-W)^T W_q`，去更新或选择 `Q(A)`；
- 在校准阶段先用输出指标选出 activation 候选，再声称动态路径本身没有矩阵乘法；
- 由评测器把参考 Linear 输出、输出残差或其梯度回传给 `solution.py` 或写入 activation state。

无条件允许且作为本计划默认白名单的统计只有：

- 为拟合 `Q(A)` 使用 `A` 自身的 amax、分位数、均值、方差、`A^T A`、块内协方差和重构误差；
- 为拟合 `Q(W)` 使用 `W` 自身统计，以及由 activation calibration 构造的 `H_A=A^T A/N+lambda I`；这里优化对象是 `Q(W)`，不生成 Linear 输出；
- 对 Weight 与 Activation 分别计算 operand-local HiF4 误差，再以预注册规则组合候选排名；组合过程中不得出现两操作数的收缩乘积；
- Attention API 内按赛题允许的 Q/K/V Attention 目标工作；该路径不得被复用于 Linear activation 拟合。

`W_q^T W_q` 虽不等于 `A @ W`，但它会让 `Q(A)` 依赖 Weight。为采用最保守解释，本计划默认也不把它用于 activation candidate 的拟合或 gate；只有取得可引用的官方书面确认后，才能作为单独候选重新预注册，不能由实施 AI 自行放宽。

### 0.2 独立评测器边界

独立评测器可在候选 state 已冻结后，按官方评分方式计算最终 Linear 指标。该指标只能用于候选级验收和归档，不能进入同一次校准调用、不能逐层返回、不能成为 `Q(A)` 的参数、标签、梯度或 gate。提交文件的任何执行路径都不能访问该结果。

## 1. 目标与量化约束

### 1.0 目标分级与官方时间上限（评审修订）

- **主目标：官方 22000~25000**，按当前四锚点近似对应 Linear mean `0.790~0.890`、相对标准 HiF4 残余 MSE约 `21%~11%`；按 Checkpoint C/D 的决策点逐步官方提交、建立锚点。
- **stretch 目标：官方 26000**，对应 Linear mean `≈0.923`、残余 MSE `≈7.7%`。26000 不是本计划的承诺交付；任何情况下不得因 stretch 压力引入第 14 节禁止的捷径。

只计算各候选的相对增量门，下限之和约为
`0.593 + 0.5pp(C22) + 2pp(C23) + 4pp(C24) + 4pp(C25) + 2pp(C26a) + 1pp(C26b) ≈ 0.73`。
但这不是“全部阶段通过”的真实下界，因为 C24 另有 Linear `>=0.78` 的绝对门。若 C24 恰好以
`0.78` 通过、C25 走相对 `+4pp` 分支、C26a/C26b 分别踩线 `+2pp/+1pp`，最终下界约为
`0.85`；若 C25 走绝对 `>=0.85` 分支，C26 后约为 `0.88`。因此全部阶段通过后仍距
`0.923` 约 `4.3~7.3pp`，26000 只有在若干主机制显著超过晋级门时才可达。主交付区间为
Linear `0.790~0.890`（约 22k~25k）；达到后应先官方提交确认兑换率，再决定是否继续冲 stretch。

官方评测时间硬上限为 **300 秒**（已确认；项目归档用 `time300plus` 标记官方超时）。
C21 当前官方时间 `173.8s`；本计划各候选的 CPU 推算目标 `<205s / <225s / <250s / <270s`
均由 300s 上限反推，预留至少 30s 余量。任何候选推算官方时间超过 `270s` 一律不得晋级，
无论分数多高。`173.8s → 270s` 允许相对 C21 最多增加约 `55.4%`；`270s` 占官方上限
`90%`，并消耗 C21 原剩余时间余量的约 `76.2%`。各阶段时间目标可以逐级提高，但距
300s 的剩余余量只减不增。

### 1.1 当前锚点

| 版本 | Linear mean | Attention | 官方分数 | 官方时间 |
|---|---:|---:|---:|---:|
| v001 | 0.3993 | 0.2768 | 10250 | 127s |
| v002 | 0.5668 | 0.3786 | 15313 | 137s |
| v013 | 0.5811 | 0.4497 | 15799 | 144s |
| C21/v024 | 0.5930 | 0.4497 | 16043 | 173.8s |

四锚点最小二乘近似：

```text
official_score ≈ -1783 + 29991 * linear_mean + 224 * attention
```

在 Attention 固定为 `0.4497` 时：

| Linear mean | 预测分数 | 相对标准 HiF4 残余 MSE |
|---:|---:|---:|
| 0.593 | 16100 | 40.7% |
| 0.650 | 17800 | 35.0% |
| 0.750 | 20800 | 25.0% |
| 0.800 | 22300 | 20.0% |
| 0.850 | 23800 | 15.0% |
| 0.900 | 25300 | 10.0% |
| 0.923 | 26000 | 7.7% |

26000（stretch）的本地工程含义不是“再增加几个百分点”，而是把当前残余 MSE 从 `40.7%` 压到约 `7.7%`；主目标区间 22k~25k 对应残余 MSE 压到约 `21%~11%`。

### 1.2 当前误差归因

旧分析曾通过“一个操作数精确、另一个量化”的 Linear 输出 oracle 得到 `0.7916/0.7979`。该方法需要构造额外 Linear 输出，并可能诱导后续 `Q(A)` 拟合，因此从本计划删除，不得重跑或作为 activation gate。

Phase 0 重新建立两个完全分离的指标：

| 指标 | 数据依赖 | 用途 |
|---|---|---|
| activation local normalized error + tail CVaR | 仅 `A` | 评价 `Q(A)` |
| Weight full-H normalized error | `W` 与 `H_A=A^T A/N` | 评价 `Q(W)` |

26000 仍要求 Weight 与 Activation 两侧都出现机制级改善，但不再根据 output oracle 给两侧分配 `3%~4%` 的伪精确预算。C21-C 建立后，应记录上述两个合法指标，后续只与合规父版本比较。

## 2. 当前实现的算法缺口

### 2.1 丢弃了 64 元素 block 内的大部分相关性

当前代码构造完整 covariance/Gram 后，只通过：

```text
_flat_group_gram    -> 4x4 block diagonal
_flat_group_gram8   -> 8x8 block diagonal
_flat_group_gram16  -> 16x16 block diagonal
```

进入离散求解器。HiF4 的一个顶层 block 有 64 个元素，当前求解器看不到跨 8/16 group 的协方差和误差传播。

### 2.2 当前 8/16 refinement 不是完整 GPTQ

当前 refinement 固定 scale/lv2/lv3 后做局部坐标更新。它使用 `H*e`，但没有：

- Cholesky/Schur 顺序；
- quantization error 向未量化坐标传播；
- scale/hierarchy 与 full-64 mantissa 的交替优化；
- 跨 16 通道的完整目标。

### 2.3 等价变换只搜索到 16 维

当前允许 block transform size 为 `4/8/16`，每个 size 仅少量 sign seed。实际 q/k/fc/proj 多数层已经选择 8 或 16，说明变换有效且搜索空间触顶。

### 2.4 当前实现使用了不合规的联合输出监督

当前流程不仅先量化 Weight，还在三个位置让 Linear 输出监督影响 `Q(A)`：

- `_linear_output_candidate_metrics` 显式生成参考与量化 Linear 输出，用它选择变换；
- `_activation8_gate_decisions` 显式生成参考输出，并用输出误差决定 activation refinement 是否启用；
- `group_cross8` 由精确 Weight 与量化 Weight 的残差构造，再与 activation 相乘影响离散更新。

这些路径在严格官方口径下不能保留。问题不是 C21 的 8x8 近似“不够完整”，而是这类目标无论近似还是 full-64 都不应被用于拟合 `Q(A)`。

### 2.5 calibration 泛化能力不足

- calibration 只有两个 batch；
- gate 改善阈值低至 `0.05%`；
- offset 97/193/389 已被观察；
- 学习型旋转若继续使用同一数据选择，容易过拟合。

### 2.6 evaluator standard 与候选 codec 耦合

`evaluator/real_data_eval.py::std_hif4` 调用候选自己的 `_dense_to_hif4`。任何底层 codec 改动都可能同时改变 standard 分母。Phase 0 必须先修复。

## 3. 总体算法架构

合规修订后的新主线命名为：

```text
HiF4-OSQ
Operand-Separated Orthogonal and Second-Order HiF4 Quantization
```

对一层 Linear：

```text
X: [tokens, in_features]
W: [out_features, in_features]
```

定义严格等价变换：

```text
X_t = X * D^-1 * P * R
W_t = W * D    * P * R
```

其中：

- `D`：正对角 Smooth/LET scale；
- `P`：置换矩阵；
- `R`：正交矩阵；
- `R * R^T = I`。

因此：

```text
X_t * W_t^T = X * W^T
```

优化必须拆成互不使用 Linear 输出监督的两个目标：

```text
L_A = sum_i rho_i * robust_error(X_t[i], Qx(X_t)[i])
L_W = sum_rows (W_t[row]-Qw(W_t)[row])^T
               H_A
               (W_t[row]-Qw(W_t)[row])
H_A = X_t^T X_t / N + damping * I
```

其中：

- `L_A` 只允许使用 activation 本身；`rho` 必须由 activation-only calibration 冻结；
- `L_W` 可以使用 `H_A`，因为它只拟合 `Q(W)`；
- 变换候选用预注册的 operand-separated metric 排序，例如 `max(L_A/L_A_base, L_W/L_W_base)`；
- 禁止构造 joint output objective，也禁止用 trace/cross 形式间接恢复它。

严格等价变换的作用只通过代数结构保证，不通过在 `solution.py` 中计算变换前后的 `XW^T` 来验证。实现测试检查 `R^T R=I`、变换 round-trip 和 shape/state，不计算 Linear 输出。

## 4. Phase 0：建立 C21-C 合规基线并加固评测

当前 C21/v024 的 `16043` 只能保留为历史分数锚点，不能直接作为后续提交父版本。Phase 0 必须产生一个新的 C21-C（Compliance）基线；在它通过规则零门禁前，不得开始 C22。

### 4.1 删除当前违规路径

修改 `solution.py`：

1. 删除 `_linear_output_candidate_metrics` 及其所有调用；
2. 删除 `_activation8_gate_decisions`、`_activation_cross8_is_safe` 中的 Linear 输出评分逻辑；
3. 删除 `group_cross8`、`cross8` state、`_ACTIVATION_QUADRATIC8_CROSS_*` 开关及 cross coordinate update；
4. 禁止在 Linear calibration 路径出现 `activation.mm(weight.T)`、`activation_hat.mm(weight_hat.T)` 或语义等价 contraction；
5. 保留 Attention 专用的 `Q @ K.T`、`P @ V`，但禁止 Linear 代码调用 Attention scorer。

替换函数：

```python
def _linear_operand_candidate_metrics(
    weight,
    activation_samples,
    transform,
) -> tuple[float, dict[str, float]]:
    # A-side: activation-only hard HiF4 reconstruction loss
    # W-side: H_A-weighted Q(W) loss
    # Never produce [tokens, out_features].
```

建议指标：

```text
activation_score = mean_case(
    mean((Q(A_t)-A_t)^2) / (mean(A_t^2)+eps)
    + 0.10 * CVaR_95(per_block_normalized_error)
)

weight_score = sum_row(e_w^T H_A e_w)
             / (sum_row(w^T H_A w)+eps)

candidate_score = max(
    activation_score / activation_score_identity,
    weight_score / weight_score_identity
)
```

候选只有在两个 activation folds 都不退化，且 `candidate_score` 优于 identity 时才能采用。不得用最终 Linear 输出分数做逐层回退。

Activation 8x8 gate 改成：

```text
base_loss    = activation_only_loss(A_t, Q_base(A_t))
refined_loss = activation_only_loss(A_t, Q_refined(A_t))
```

只在 held-out activation fold 上 `refined_loss < base_loss` 时启用；该 gate 不读取 Weight、Weight state 或 Linear 输出。

### 4.2 新增独立 standard HiF4 codec

文件：

```text
Create: evaluator/reference_hif4.py
Modify: evaluator/real_data_eval.py
Create: tests/test_reference_hif4.py
```

要求：

1. 从 C21 中复制最小、冻结的标准 HiF4 编码/解码逻辑到 evaluator。
2. 禁止调用候选 `_dense_to_hif4` 生成 standard。
3. reference codec 只实现标准 amax/7、E6M2、lv2/lv3、mantissa，不包含 offset/refinement。
4. 用 v000/v001/v002/v013/C21 重新评测，确认分数与当前记录在 `1e-6` 内一致。
5. 在后续 candidate 修改 `_dense_to_hif4` 时，standard 输出必须保持逐位不变。

### 4.3 固化合规误差归因工具

文件：

```text
Create: evaluator/linear_error_decomposition.py
Create: tests/test_linear_error_decomposition.py
```

`solution.py` 内只允许输出 operand-local 诊断：

```text
activation_local_error
activation_tail_cvar
weight_hessian_error
weight_plain_error
transform_orthogonality_error
```

独立 evaluator 可以在候选冻结后报告最终官方式 Linear ratio，但不得把逐层输出误差传回算法。归因工具不得计算或记录可被 `Q(A)` 重用的参考 Linear 输出、cross residual 或其低秩表示。

### 4.4 新增规则零静态与运行时门禁

文件：

```text
Create: evaluator/linear_compliance_guard.py
Create: tests/test_linear_compliance_guard.py
```

静态检查至少拒绝：

- `_linear_output_candidate_metrics`、`group_cross8`、`cross8` 等已知违规符号重新出现；
- Linear calibration 函数内 activation-derived `[N,K]` 与 weight-derived `[M,K]` 的收缩乘法；
- 把 evaluator 输出、reference output 或 output residual 写入 state；
- 名称变化后的等价实现，因此不能只做字符串黑名单，还要做运行时 shape/taint 测试。

运行时测试构造带 provenance 的 activation/weight，记录 Linear calibration 内的 contraction。允许 `A.T @ A`；允许仅用于 `Q(W)` 的 Hessian loss；拒绝任何产生 `[tokens,out_features]` 的 contraction，以及任何由精确 Weight 残差和 activation 共同生成并流入 activation state 的 Tensor。

### 4.5 新增真正冻结的 holdout

要求：

1. 新增一段未用于当前候选的固定文本；
2. 生成至少 4 个新 token windows；
3. 文件中只记录 seed/hash，不在开发输出中展示逐层数据；
4. 每个 candidate 最多运行一次最终 holdout；
5. holdout 不用于 seed、threshold、coverage 搜索。

### 4.6 Phase 0 验收

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluator\real_data_eval.py --solution solution.py --device cuda
```

验收条件：

- C21-C 可能低于 16043；必须记录移除违规监督造成的真实分数变化，不得为保分保留灰色路径；
- Attention 与 C21 逐 case 不变；
- reference standard 与候选实现完全解耦；
- `linear_compliance_guard` 静态与运行时检查全部通过；
- activation state 中不存在 Weight residual、Linear output 或其等价 cross operator；
- C21-C 归档为新的唯一合规 Champion，后续所有候选从它派生。

## 5. C22：Linear R64 Incoherence Transform

### 5.1 唯一机制

在当前 Linear transform candidate 中增加 64 维 signed Hadamard：

```text
R64(seed) = S(seed) * H64
```

不修改：

- Attention；
- weight/activation refinement；
- scale offset；
- C21-C operand-local gate；
- coverage；
- sweep。

### 5.2 需要修改的代码

文件：

```text
Modify: solution.py
Modify: tests/test_release_candidate.py
```

新增常量：

```python
_LINEAR_R64 = True
_LINEAR_R64_BLOCK = 64
_LINEAR_R64_STAGE1_SEEDS = tuple(range(32))
_LINEAR_R64_STAGE2_KEEP = 4
_LINEAR_R64_MIN_IMPROVEMENT = 0.005
_LINEAR_R64_WORST_TOLERANCE = 0.002
```

新增函数：

```python
def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor
def _linear_r64_signs(channels: int, seed: int, device) -> torch.Tensor
def _apply_linear_r64(x: torch.Tensor, seed: int) -> torch.Tensor
def _rank_r64_seeds(...cheap sampled operands...) -> list[int]
def _select_r64_candidate(...operand-separated metrics...) -> tuple[int, metrics]
```

### 5.3 FWHT 实现要求

禁止通过构造 `[64,64]` dense Hadamard 后 `matmul` 实现动态 R64。使用 butterfly：

```text
for width in 1,2,4,8,16,32:
    reshape (..., 64/(2*width), 2, width)
    a = left
    b = right
    left  = a + b
    right = a - b
divide by sqrt(64)
```

要求：

- 支持 CPU/CUDA；
- 输入不被原地破坏，除非调用者明确传入 clone；
- float32/BF16 均可运行；
- 与 `_hadamard_matrix_unchecked(64)` 的 dense 结果误差 `<1e-5`。

### 5.4 两阶段 seed 选择

直接对 32 个 seed 跑完整 C21-C 太慢，采用两阶段：

#### Stage A：cheap rank

对每个 seed：

1. 使用最多 64 activation rows；
2. 使用最多 128 weight rows；
3. 只运行标准 HiF4 或关闭 8/16 refinement；
4. 分别计算 activation-only hard reconstruction loss 与 `H_A`-weighted Weight loss；
5. 保留最优 4 个 seed，加 identity candidate。

#### Stage B：deployed validation

对 top-4 + identity：

1. 使用当前完整 C21-C 合规 Weight quantization；
2. 构造当前完整 activation state；
3. 使用两个 calibration batch 做双向验证：
   - fold 0：batch 0 统计/选择，batch 1 评分；
   - fold 1：batch 1 统计/选择，batch 0 评分；
4. seed 必须在两个 fold 的 activation-only 指标均不劣于 identity，且 operand-separated robust metric 优于 identity；
5. 通过后在全部 calibration 数据上重建最终 state。

robust metric：

```text
ratio_A = activation_local_loss / identity_activation_local_loss
ratio_W = weight_hessian_loss / identity_weight_hessian_loss
metric  = max(ratio_A, ratio_W)
        + 0.10 * max(0, activation_tail_ratio - 1)
```

禁止用最终 Linear output ratio替换这个 seed metric；最终 Linear ratio只由独立 evaluator 在 seed/state 完全冻结后报告。

### 5.5 state

复用现有：

```text
block_smooth_size = 64
block_smooth_seed = selected_seed
```

不得保存 dense R64。

### 5.6 C22 单元测试

新增：

```text
test_fwht64_matches_dense_hadamard
test_linear_r64_is_orthogonal
test_linear_r64_activation_roundtrip
test_linear_r64_weight_roundtrip
test_linear_r64_state_is_seed_only
test_linear_r64_disabled_matches_c21c
test_linear_r64_candidate_falls_back_on_regression
test_linear_r64_is_deterministic_cpu_cuda
```

不变量测试不得通过计算 Linear 输出实现，改为：

```text
max_abs(R.T @ R - I) < 1e-5
max_abs(inverse_transform(transform(X)) - X) < 1e-5
max_abs(inverse_transform(transform(W)) - W) < 1e-5
```

### 5.7 C22 晋级门

开发集：

- Linear mean 相对 C21-C `>= +0.5pp`；
- fc/proj/o 三项平均 `>= +1.0pp`；
- 任一分项不低于 `-0.1pp`；
- Attention 逐 case 一致。

固定矩阵：

- 6/6 配置 Linear mean 为正；
- win rate `>=70%`；
- tail mean `>=-1pp`；
- CPU algorithm-stage ratio `<=1.12`；
- 推算官方时间目标 `<205s`。

未达到开发门时，归档 rejected，停止 seed 扩展，不直接实现双 Hadamard。

## 6. C23：Full-64 Weight Schur/GPTQ

前置条件：C21-C 已通过规则零门禁。若 C22 rejected，C23 从 C21-C 构建，并只保留已经通过合规审计的 identity/R4/R8/R16 变换。

### 6.1 唯一机制

只替换 weight 的 full-64 refinement；dynamic activation 仍使用父版本。

### 6.2 数学目标

对每个 transformed weight row 的 64 元素 block：

```text
w: [64]
H: [64,64] = X_t^T X_t / N + damping * I
q: legal HiF4 decoded block
loss(q) = (q-w)^T H (q-w)
```

`H` 必须使用完整 64x64 block，不截断为 4/8/16。

damping：

```text
damping = 0.01 * mean(diag(H))
```

如果 Cholesky 失败，依次尝试 `0.03/0.1`；仍失败则回退父版本。

### 6.3 scale beam

每个 64 block 的候选 scale code：

```text
standard_code + {-2,-1,0,1,2,3}
```

先使用现有 exact hierarchy proxy 排序，保留 loss 最低的 4 个 code。命中边缘时允许现有 edge extension，但最终 beam 不超过 4。

C23 scale beam 安全要求：

1. 固定回归必查对 scale 敏感的 Linear 配置（`amax4 offset 0`、`pow2 offset 0`）；任一
   固定回归 case 的 Linear 相对指标不得低于父版本 `2pp`，防止扩展 beam 改善均值但
   破坏尾部；
2. beam 接受前逐 block 验证五字段合法、full-H loss finite 且不高于父版本；不满足时
   该 block 回退父版本参数（逐 block fallback，不是逐层）；
3. 若固定回归失败，当前候选立即归档为 rejected。需要把 beam 收窄为 `{-1,0,+1}` 时，
   必须分配新的 candidate ID、重新预注册并从开发评测开始，不能在看过固定回归结果后
   原地修改同一候选；
4. variantH 的 `saturated_logits_h4_kv2_d64_s32`（seed 307）`0.0000` 是 Attention
   calibration 的扩展 offset 未经验证进入运行时 state 所导致，不能作为 C23 Weight beam
   的直接因果证据。该教训由第 10.3 节 Attention 合成安全门单独处理。

### 6.4 每个 beam 的求解流程

对每个 candidate scale：

1. 使用现有 `_solve_exact_hierarchy` 初始化 lv2/lv3/mantissa；
2. 计算 `H^-1` 的 upper Cholesky factor；
3. 按 `diag(H)` 从大到小确定 processing order；
4. 固定 scale/lv2/lv3，执行一次 GPTQ sequential mantissa initialization；
5. 使用完整 H 做一次 64-coordinate exact discrete descent；
6. 逐个枚举 16 个 lv3 bit toggle，接受 full-H loss 下降的 toggle；
7. 逐个枚举 8 个 lv2 bit toggle，接受 full-H loss 下降的 toggle；
8. 再做一次 64-coordinate descent；
9. 记录最终 full-H loss；
10. 四个 beam 中选择最优。

### 6.5 GPTQ initialization 伪代码

```python
adjusted = w.clone()
q = zeros_like(w)
for i in processing_order:
    denominator = scale * lv2[group8(i)] * lv3[group4(i)]
    q[i] = nearest_legal_mantissa(adjusted[i], denominator)
    error = (adjusted[i] - q[i]) / chol_inv[i, i]
    for j in remaining_after_i:
        adjusted[j] -= error * chol_inv[i, j]
```

processing order 只影响求解过程，不改变最终 tensor 坐标。

### 6.6 full-H coordinate descent

维护：

```text
e = q - w
g = H @ e
```

对坐标 i 枚举合法 signed mantissa code：

```text
delta = candidate_q_i - q_i
loss_change = 2 * delta * g_i + delta^2 * H_ii
```

选择最小 loss_change；接受后：

```text
q_i += delta
e_i += delta
g += delta * H[:, i]
```

### 6.7 代码结构

新增：

```python
def _full64_hessian_blocks(cov: torch.Tensor, channels: int) -> torch.Tensor
def _cholesky_inverse_factor(h: torch.Tensor) -> Optional[torch.Tensor]
def _gptq_initialize64(...)
def _coordinate_descent64(...)
def _hierarchy_toggle_refine64(...)
def _refine_weight_blocks64(...)
```

内存要求：

- weight rows 必须 chunk，默认 128 rows/chunk；
- 不得展开 `[rows, blocks, beams, 64,64]`；
- H 每层共享，不按 weight row 复制；
- beam 按顺序执行，避免 4 倍峰值内存。

向量化硬性要求（`CPU ratio <=1.15` 晋级门的先决条件）：

- 禁止按 64-block 逐个执行 Python 循环求解；一个 row chunk 内的全部 blocks 必须
  合并为批量张量运算，例如把 descent 状态组织为 `[rows*blocks, 64]`、loss 与
  接受判定组织为 `[rows*blocks]` 向量；
- coordinate descent 只允许在坐标维（64）与 beam 维上循环，rows/blocks 维度必须
  保持批量；hierarchy toggle 的接受判定使用布尔掩码批量更新，不得逐 block 分支；
- 交付前必须先通过向量化正确性探针：批量路径与逐 block 参考实现数值一致（容差
  `1e-6`）；rows/blocks 维仍存在逐项 Python 循环时视为实现未完成，不得用“功能正确”
  豁免；
- micro-benchmark 固定在 CPU/float32，使用与正式评测相同的 Torch 线程数，测试
  `rows>=2000`、`channels in {768,3072}`、chunk 128；预热 3 次、测量 10 次并报告中位数，
  计时范围覆盖 `_refine_weight_blocks64` 的生产调用及其必要临时分配；
- 批量路径相对逐 block 参考实现 `>=10x` 作为诊断目标而非独立晋级门。即使达到 10x，
  端到端 `CPU ratio >1.15` 仍不得晋级；若未达到 10x 但端到端 ratio 合格，必须记录瓶颈
  分析后才允许进入固定回归，不能隐藏或更换 benchmark 口径。

### 6.8 C23 测试

```text
test_full64_hessian_extraction
test_gptq64_initialization_returns_legal_codes
test_coordinate_descent64_is_monotonic
test_hierarchy_toggle64_is_monotonic
test_weight64_final_loss_not_above_parent
test_weight64_chunking_is_exact
test_weight64_fallback_on_non_psd
test_weight64_deterministic
```

### 6.9 C23 晋级门

- Linear mean 相对父版本 `>= +2pp`；
- fc/proj/o 平均 `>= +3pp`；
- 固定矩阵上的 Weight full-H normalized error 相对父版本下降 `>=20%`；
- 6/6 固定配置正向；
- CPU ratio `<=1.15`；
- 推算官方时间 `<225s`。

若 full-64 weight 后总 Linear 仍 `<0.68`，记录为 26000 风险检查点：不立即进入 C24，先分析 weight residual 是否真的下降。

## 7. C24：Top-K Full-64 Activation-Only Solver

前置条件：C23 晋级且 weight residual 明显下降。

### 7.1 唯一机制

保持 C23 weight params 不变，只增加 activation full-64 dynamic refinement。

### 7.2 合规 activation-only objective

在 transformed 坐标中只定义：

```text
x: 当前 activation 的一个 64 元素 block
q: 合法 HiF4 decoded block
rho: 仅由 calibration activation 统计得到的 64 维非负权重
```

首版目标：

```text
L_local(q|x) = sum_i rho_i * huber(q_i-x_i; delta_i)
             + lambda_tail * CVaR_90(group_error_4)
```

推荐固定参数：

```text
rho_i = clamp(1 / (EMA_abs_i + eps), 0.25, 4.0)
delta_i = 2 * median_abs_deviation_i
lambda_tail = 0.10
```

`rho/delta` 只能从 activation calibration 计算。实现不得接收 Weight、`W_q^T W_q`、Weight residual、Linear 输出或 evaluator 指标。若 Huber/CVaR 增加过多运行时间，首个候选先使用加权平方误差，但仍须优化完整 64 元素共享层级。

### 7.3 activation state

state 只新增 activation-only 字段：

```text
local_weight64  # [blocks,64], rho
local_delta64   # [blocks,64], Huber delta；平方误差版本可省略
```

明确禁止出现：

```text
cross8 / cross64
weight_residual_operator
reference_output
output_gradient
teacher_linear_target
```

所有 state 保存为 CPU contiguous float32。动态 API 必须仅凭当前 activation、activation state 和合法 HiF4 参数完成量化。

### 7.4 Full-64 hierarchy solver

64 元素共享一个 HiF4 顶层 scale，8 个 lv2 与 16 个 lv3。求解流程：

1. 用父版本生成 base scale/lv2/lv3/mantissa；
2. 对 scale code `{base-2,...,base+2}` 建 beam，最多 3 个；
3. 固定 scale，交替枚举每个 lv2、lv3 合法 toggle；
4. 每次 hierarchy 变化后，对受影响的 4/8 元素重新选择最小 activation-local loss 的 mantissa；
5. 执行一轮 64-coordinate mantissa descent；
6. 每次接受必须使 `L_local` 单调下降；
7. 与 base hard loss 比较，不下降则逐 block 回退。

该算法的“full-64”来自共同优化整个 HiF4 hierarchy，而不是引入 Weight 或输出 Hessian。

### 7.5 动态 top-K

动态量化流程：

1. 使用父版本 `_dense_to_hif4` 得到 base params；
2. 用 activation-only `L_local` 计算每个 64 block 的当前损失；
3. 用 scale/hierarchy 邻域的 activation-local 下界估计 gain；
4. 按 gain 排序选择 top-K；
5. 只对 top-K block运行一次 full-64 hierarchy solver；
6. 未入选 block 保持父版本参数；
7. final loss 必须下降至少 `1e-5 * max(abs(initial_loss), eps)`；
8. 未下降逐 block 回退 base params。

首个候选固定：

```text
max_ratio = 0.05
max_blocks = 4096
sweeps = 1
```

不得在 C24 同时搜索 10%/20%；coverage 扩展必须作为后续独立候选归档。

### 7.6 C24 测试

```text
test_activation64_state_contains_no_weight_or_output_data
test_activation64_local_objective_is_monotonic
test_activation64_full_hierarchy_codes_are_legal
test_activation64_topk_cap
test_activation64_parent_fallback
test_activation64_state_legality
test_activation64_dynamic_params_legality
test_activation64_disabled_matches_c23
test_activation64_does_not_contract_activation_with_weight
```

### 7.7 C24 晋级门

- Linear mean 相对 C23 `>= +4pp`；
- fc/proj/o 平均 `>= +5pp`；
- activation local mean error 相对 C23 下降 `>=15%`，tail CVaR 下降 `>=10%`；
- overall Linear 目标 `>=0.78`；
- 6/6 固定配置正向；
- CPU algorithm-stage 推算 `<250s`；
- dynamic 时间增量必须单独记录。

若 C24 后 Linear `<0.75`，26000 路线暂停，先优化 activation-local hierarchy search、R64 或 clipping，而不是引入 output/cross 监督或盲目扩大 coverage。

## 8. C25：Learned Equivalent Scale（LET）

前置条件：C24 晋级。

### 8.1 唯一机制

固定 C24 的 rotation、weight solver 和 activation solver，只把手工 alpha grid 替换为 calibration-time learnable `log_d`。

### 8.2 参数化

```text
d = exp(log_d)
log_d ∈ [-log(8), log(8)]
```

初始化为父版本选择的 `d_parent`。

### 8.3 STE proxy

定义：

```python
def q_ste(x):
    q_hard = deployed_hif4_quantize(x)
    return x + (q_hard - x).detach()
```

只对 `log_d` 求梯度；R/P、离散 scale/hierarchy 不参与梯度。

### 8.4 优化配置

```text
optimizer = Adam
steps = 8
lr = 0.03
activation_rows = 64
weight_rows = 128
regularization = 1e-3 * ||log_d - log_d_parent||^2
gradient_clip = 1.0
```

loss：

```text
L = lambda_A * normalized_activation_local_loss
  + lambda_W * normalized_weight_hessian_loss
  + regularization
  + 0.1 * fold_variance
```

其中 `normalized_activation_local_loss` 只比较 `A_t` 与 `Q(A_t)`；`normalized_weight_hessian_loss` 只优化 `Q(W_t)`，使用合法的 `H_A=A_t^T A_t/N+damping*I`。不得添加 joint output、trace cross term、reference Linear output 或由 Weight residual生成的 activation 梯度。

优化后必须用真实离散 C24 路径重新量化，并在两个 operand-local hard loss 上 gate；STE loss 与独立 evaluator 的 Linear output score都不得用于同一次逐层 gate。

### 8.5 双 fold gate

- batch0 用 operand-separated loss optimize，batch1 用 operand-separated hard loss evaluate；
- batch1 optimize，batch0 hard evaluate；
- 两个 fold 的 `log_d` 取均值后在全部 calibration 上做 operand-local hard evaluate；
- 两个 fold 的 activation loss 均不退化，且组合 metric 改善才采用；
- 否则逐层回退 parent `d`。

### 8.6 C25 晋级门

- Linear mean `>=0.85`，或相对 C24 `>=+4pp`；
- 固定矩阵 6/6 正向；
- holdout 为正；
- CPU 推算 `<270s`；
- 任何层 optimization non-finite 必须回退，不得崩溃。

## 9. C26：双 Hadamard与 HiF4 adaptive headroom

C26 只在 C25 达到 `>=0.85` 后开始。它包含两个独立候选，禁止合并实现。

### 9.1 C26a：双 Hadamard离散旋转

```text
R = S1 H64 P64 S2 H64
```

- 先固定第一阶段 C22 seed；
- 搜索第二 sign seed 和 4 个预定义 permutation family；
- cheap rank 后只 hard evaluate top-4；
- state 保存 `seed1/seed2/permutation_id`；
- 使用两次 FWHT；
- CPU 推算必须 `<270s`。

晋级目标：相对 C25 `>=+2pp`。

### 9.2 C26b：HiF4 adaptive headroom

在 C26a 或 C25 Champion 上，单独增加：

```text
inner_target ∈ {4,5,6,7}
```

对每个 64 block：

1. 为四个 target 生成初始 E6M2 code；
2. 每个 target 调用 exact hierarchy/full64 solver；
3. 分别计算 activation-local 与 Weight-Hessian objective，按预注册 operand-separated metric 选择；
4. 输出仍必须是现有合法五字段，不新增格式字段；
5. target 只影响 candidate scale 初始化，不写入输出。

该机制借鉴 NVFP4 Four-Over-Six，但必须以 HiF4 合法码本重新求解，不能直接复制 NVFP4 scale 公式。

晋级目标：相对父版本 `>=+1pp`，时间增量 `<5%`。

## 10. 统一评测矩阵

### 10.1 开发筛选

```powershell
.\.venv\Scripts\python.exe evaluator\real_data_eval.py `
  --solution solution.py --model models\gpt2 --device cuda `
  --mode amax6 --token-offset 0 --attn-mask both --verbose
```

开发门未通过时立即归档，不运行固定回归和 CPU 时间。

### 10.2 固定回归

必须运行：

```text
amax6 offsets 0/97/193/389
amax4 offset 0
pow2 offset 0
MHA causal/non-causal
GQA kv_heads=6 causal/non-causal
```

Linear-only candidate 的 Attention 必须与父版本逐 case 一致。

### 10.3 合成安全

```powershell
.\.venv\Scripts\python.exe evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

Linear-only candidate 必须与父版本 576 case 一致，容差 `1e-6`。

Attention 历史硬门：variantH 曾因扩展 dynamic offset 未经 calibration gate 验证就进入
runtime state，使 `saturated_logits_h4_kv2_d64_s32`（seed 307）官方指标退化到 `0.0000`。
因此任何未来 Attention 候选必须同时满足：

1. extended 与 conservative offset set 分别接受 gate，只有实际通过的集合才能写入 state；
2. 上述 saturated-logits case 必须不低于父版本，且合成矩阵不得出现新的 `<=0` case；
3. Linear-only 候选不得触碰 Attention offset/state，576 case 必须按本节开头的逐 case 一致门执行；
4. 失败候选归档后才能以新 candidate ID 修改 offset，不得依据同一次固定安全矩阵原地调参。

### 10.4 CPU 时间

父子必须串行、同环境运行：

```powershell
.\.venv\Scripts\python.exe evaluator\real_data_eval.py `
  --solution <parent> --device cpu --attn-mask both

.\.venv\Scripts\python.exe evaluator\real_data_eval.py `
  --solution solution.py --device cpu --attn-mask both
```

记录：

```text
algorithm-stage
calibration
dynamic
api-total
nested calls
```

## 11. 统一正确性门禁

每个候选必须满足：

1. `pytest -q` 全部通过；
2. `git diff --check` 通过；
3. 无文件 I/O、网络、debug print；
4. API 参数与返回 keys 不变；
5. state 合法；
6. 五字段 shape/dtype/finite 合法；
7. feature-off 与父版本字段级等价；
8. 数学不变量误差在约定阈值内；
9. objective 单调测试通过；
10. CPU 时间未超过候选预算；
11. 源码归档 SHA 与实际评测 SHA 一致。

## 12. 归档模板

每个 result.md 至少记录：

```text
Candidate ID
Parent ID / SHA256
Unique mechanism
Changed flags/constants
Source SHA256
Development matrix
Fixed regression matrix
Per-component deltas
Per-layer tail
Error decomposition before/after
State size
CUDA timing
CPU timing
Official status
Decision
Next direction
```

新增的关键指标：

```text
linear_mean
activation_local_normalized_error
activation_local_tail_cvar
weight_full_h_normalized_error
weight_plain_normalized_error
R64 selection rate
full64 fallback rate
activation64 selected block rate
compliance_guard_static_pass
compliance_guard_runtime_pass
```

## 13. 决策检查点

### Checkpoint A：C22 后

- 若 R64 `<+0.5pp`：停止多 seed/双 Hadamard，直接测试 C23 full64 weight；
- 若 R64 `>=+0.5pp`：以 C22 为 C23 父版本。

### Checkpoint B：C23 后

- 若 Linear `<0.68` 或 Weight full-H normalized error 未下降 `20%`：完整 weight solver 未产生机制级突破，暂停 C24；
- 若达到目标：进入 activation full64。

### Checkpoint C：C24 后

- 若 Linear `<0.75`：26000 暂不可达，优先检查 activation operator 压缩和动态预算；
- 若 Linear `0.78~0.83`：官方目标约 22k~23k，进入 LET；
- 若 Linear `>0.83`：优先官方提交建立新锚点，再继续。

### Checkpoint D：C25 后

- 若 Linear `<0.85`：不要继续堆 headroom；分别分析 activation-local 与 Weight-Hessian 残差；
- 若 `0.85~0.90`：目标约 24k~25k，进入 C26；
- 若 `>=0.90`：先提交官方确认兑换率，再决定是否冲 0.923。

### Checkpoint E：26000 stretch 提交门

主目标轨道（22k~25k）的官方提交由 Checkpoint C/D 决定，不依赖本门。只有同时满足以下条件才把候选视为 26000 stretch 级：

```text
local Linear mean >= 0.92
activation local error 与 C21-C 相比下降 >= 70%
weight full-H error 与 C21-C 相比下降 >= 70%
6/6 fixed matrix positive
new frozen holdout positive
no component below parent by 0.1pp
CPU projected official time <= 270s
all compliance tests pass
```

未达到本门不视为失败：只要候选优于当前官方锚点且全部合规测试通过，仍应按
Checkpoint C/D 的节奏提交官方、建立新锚点；本门仅决定是否继续冲击 stretch。

## 14. 明确禁止的实现捷径

后续 AI 不得：

1. 只扩大现有 8/16 coverage 并称为 full second-order；
2. 把 C8 的旧 64-coordinate refinement 直接复活并称为 GPTQ；
3. 在 C22 同时修改 rotation、scale、coverage；
4. 计算 Linear reference/reconstructed output并用其选择 `Q(A)`、rotation、seed、gate 或 fallback；
5. 直接保存 dense layer-wide rotation；
6. 在 dynamic path 对全部 block 无界运行 full64 solver；
7. 用 STE proxy 分数直接晋级；
8. 根据 offset 97/193/389 调 seed 或 threshold；
9. 修改 evaluator standard 以提高候选分数；
10. 用本地线性拟合伪造官方分数记录；
11. 用 trace、cross、低秩、采样或改名函数绕过规则零；
12. 把 Weight residual、输出 teacher、evaluator 分数或其梯度写入 activation state；
13. 因为违规路径曾获得官方分数，就把它当作合规先例。

## 15. 参考算法

- GPTQ：完整二阶信息与顺序误差反馈，https://arxiv.org/abs/2210.17323
- QuaRot：严格等价随机旋转消除 4-bit outlier，https://arxiv.org/abs/2404.00456
- SpinQuant：学习旋转优于固定随机旋转，https://arxiv.org/abs/2405.16406
- OmniQuant：learnable clipping 与 equivalent transformation，https://arxiv.org/abs/2308.13137
- Four Over Six：自适应 FP4 block headroom，https://arxiv.org/abs/2512.02010
- NVIDIA NVFP4 recipe：RHT 与 4over6 官方实现说明，https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html

## 16. 最终执行顺序

```text
Phase 0  删除违规输出监督 + 建立 C21-C + compliance guard + holdout
  ↓
C22      Linear R64
  ↓
C23      Full-64 Weight Schur/GPTQ
  ↓
C24      Top-K Full-64 Activation Solver
  ↓
C25      Learned Equivalent Scale
  ↓
C26a     双 Hadamard离散旋转
  ↓
C26b     HiF4 Adaptive Headroom
  ↓
主目标轨道：按 Checkpoint C/D 在 22k~25k 区间逐步官方提交、建立锚点
  ↓
stretch 轨道：达到 local Linear >=0.92 后才进入 Checkpoint E（26000 stretch 提交门）
```

该顺序必须保持。每个箭头都代表：预注册、实现、测试、开发评测、固定回归、CPU 计时、归档和 Champion 决策，不允许跨候选合并。
