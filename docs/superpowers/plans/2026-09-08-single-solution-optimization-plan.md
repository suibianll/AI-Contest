# 当前最高分完整版本持续优化计划

> ACTIVE，2026-09-08。计划只说明怎么优化、怎么执行、怎么归档。

## 1. 当前安排

- 根 `solution.py` 是当前最高分完整版本，后续所有候选都从开始该轮时的根构建。
- LC3 已由用户提交，不再作为本计划的前置任务，也不等待它的结果。
- 当前重点是继续改进完整方案中的 Attention，但不建立独立 Attention 父线；每个候选仍是完整六 API
  `solution.py`。
- 一个候选进入官方评测后，可以继续实现下一个方向，不因等待官方结果而停工。
- 若多个独立候选分别取得官方提升，再把有效改动合入最高分根，提交一次组合版本。

## 2. 每一轮怎么执行

每轮使用一个独立目录：`workbench/full_solution/<candidate-name>/`。

1. `build.py` 从当前根复制完整候选，并加入本轮算法。
2. `verify.py` 检查算法公式、父状态回退和实际参数变化。
3. `check_math_and_import.py` 检查六 API 单文件导入、合法状态和有限输出。
4. 运行 Attention shard0，确认候选确实生效且没有明显实现异常。
5. 将候选归档到 `solutions/<candidate-name>/`，然后提交官方。
6. 官方提高就替换根；没有提高就保留原根，继续下一个方向。

统一评测命令：

```powershell
.venv\Scripts\python.exe workbench/full_solution/<candidate-name>/build.py
.venv\Scripts\python.exe workbench/full_solution/<candidate-name>/verify.py
.venv\Scripts\python.exe workbench/full_solution/<candidate-name>/check_math_and_import.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/<candidate-name>/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/<candidate-name>-shard0
```

## 3. Attention 方向一：逐通道闭式 Q/K 互逆平衡

候选名：`attn-diag-reciprocal-balance`

这是当前优先执行方向。它只做一次统计和一次硬编码，不运行 32 步矩阵训练，适合当前完整版本的时间
余量。

### 算法

在根已经得到的 Q/K rotation 和 K center 后，为每个 GQA group、每个 head channel 计算 Q、K
量化误差对最终输出的传播能量 `a_j` 和 `b_j`，闭式求解：

```text
loss_j(d_j) = a_j * exp(2*d_j) + b_j * exp(-2*d_j)
d_j = 1/4 * log((b_j + 1e-12) / (a_j + 1e-12))
```

组内对 `d` 去均值，然后限制在 `[-log(2)/2, log(2)/2]`。部署为：

```text
Q_new = Q_parent * exp(d)
K_new = K_parent * exp(-d)
c_new = c_parent * exp(-d)
```

连续 QK 结果保持不变，收益来自硬量化后的动态范围重新分配。

### 代码实现

在候选中新增：

- `_attn_diag_error_energy`：从前三个校准窗口累计 `a_j/b_j`。
- `_attn_diag_reciprocal_balance`：计算唯一 `d` 并编译 rotation/center。
- `_attn_true_output_loss`：用真实 HiF4 Q/K/V 计算最终 Attention 输出误差。

在 `hif4_calibration_attention` 原逻辑得到父状态后调用上述函数。第 4 个窗口比较候选与父状态，第 5
个窗口复核；候选更差时该层直接保留父状态。

### 本轮记录

- `d` 的范数与最大值。
- Q/K 中实际改变的量化码数量。
- 尝试层数、采用层数。
- shard0 相对根的结果与运行时间。

## 4. Attention 方向二：64 维块间稀疏三角搬运

候选名：`attn-block-triangular-transport`

这个方向不再只缩放通道，而是在 head_dim 的四个 64 维块之间搬运量化压力，表达能力高于对角平衡，
计算量又远小于全矩阵迭代训练。

### 算法

固定两个非重叠块对：`0 → 1` 和 `2 → 3`。构造：

```text
T = I + N
N = 两个 rank-1 上三角 64x64 块
N^2 = 0
T_inverse = I - N
```

Q 使用 `T`，K 使用 `T^{-T}`，因此连续 QK 保持不变。每个 rank-1 块的左右方向来自父版本最终硬
Attention 输出误差对该块的一次梯度，取最大奇异向量；步长取沿该方向第一次触发真实 HiF4 码变化的
距离，只生成一个候选。

### 代码实现

在候选中新增：

- `_attn_block_output_gradient`：计算两个块对的最终输出梯度。
- `_attn_rank1_triangular_transform`：两次 64×64 SVD 后生成 `N/T/T_inverse`。
- `_attn_first_code_boundary_step`：计算唯一有效步长。
- `_attn_compile_pair_transform`：把 `T/T^{-T}` 和 K center 合入现有状态。

仍由 `hif4_calibration_attention` 先生成根状态，再生成这个唯一候选，并用后两个校准窗口的真实最终
输出误差决定每层是否采用。

### 本轮记录

- 两个块对的主奇异值和采用步长。
- 真实翻码数量。
- 尝试层数、采用层数。
- shard0 相对根的结果与运行时间。

## 5. Attention 方向三：全矩阵 Q/K 互逆残差

候选名：`attn-full-reciprocal-residual`

这是已有官方正向证据的高表达能力方向。参考实现：
`solutions/continuous_attention_anchor22-a2/solution.py`。只移植算法增量，不把历史版本当父。

### 算法

在根的完整 Q/K 坐标后，为每个 GQA group 学习一个对称、零迹矩阵 `S`：

```text
Q_new = Q_parent @ exp(S)
K_new = K_parent @ exp(-S)
Rq_new = Rq_parent @ exp(S)
Rk_new = Rk_parent @ exp(-S)
c_new  = c_parent  @ exp(-S)
```

训练目标是 Q/K 两侧相对父状态的 64-block `amax²` 之和。最后两个校准窗口只用于比较真实硬编码后的
Attention 输出误差。

### 固定实现

直接移植参考实现中的：

- `_a21_exp`
- `_a21_exp_backward`
- `_a21_project`
- `_a21_scale_loss_grad`
- `_a21_matrix_grad`
- `_a22b_train`
- `_a21_gate_loss`

固定参数：

```text
训练步数       32
学习率         0.01
梯度范数上限   1.0
S 正则         0.001
Adam beta      0.9 / 0.999
谱范围         ±log(2)/2
候选数量       1
```

### 本轮记录

- `S` 范数、Q/K range loss 变化和互逆误差。
- 每层 Q/K 翻码数量与采用情况。
- shard0 相对根的结果与校准时间。

如果该实现出现官方超时，保留前两个低成本方向继续推进，不围绕训练步数做扫描。

## 6. Attention 方向四：联合 Q/K 块尺度乘积目标

候选名：`attn-joint-qk-product`

这个方向与方向三使用相同的互逆矩阵，但训练目标不同。方向三分别缩小 Q 和 K 的范围；本方向直接
优化同一 GQA group、同一 64-block 的 Q/K 尺度乘积，减少两侧分别改善但相互抵消的问题。

### 算法

```text
aQ(g,b) = Q group g、block b 的平均 amax²
aK(g,b) = K group g、block b 的平均 amax²
loss = mean(aQ_new * aK_new / (aQ_parent * aK_parent + 1e-12))
       + 0.001 * mean(S²)
```

仍使用：

```text
Q_new = Q_parent @ exp(S)
K_new = K_parent @ exp(-S)
c_new = c_parent @ exp(-S)
```

训练配置与方向三相同：32 步、学习率 `0.01`、梯度裁剪 `1.0`、谱范围 `±log(2)/2`。实现时从当前
最高分根直接加入完整互逆残差训练，不要求方向三先成为父版本。

### 代码实现

复用方向三的矩阵指数、反向和状态编译函数，将 `_a21_scale_loss_grad` 替换为：

- `_attn_group_block_amax2`：按 GQA group 和 head 内 64-block 聚合 Q/K。
- `_attn_qk_product_loss_grad`：计算联合乘积目标和手工梯度。

最终仍用真实 Attention 输出误差选择层，不用乘积目标直接决定部署。

### 本轮记录

- Q/K 尺度乘积的前后变化。
- `S` 范数、翻码数量和采用层数。
- shard0 相对根的结果与校准时间。

## 7. 执行顺序与持续推进

当前执行顺序：

1. `attn-diag-reciprocal-balance`
2. `attn-block-triangular-transport`
3. `attn-full-reciprocal-residual`
4. `attn-joint-qk-product`

顺序按计算成本从低到高排列，不构成结果依赖。上一候选已经完成本地检查并提交官方后，就可以从当时
的最高分根开始实现下一项。若期间根因新的官方结果发生变化，只需让尚未构建的候选使用新根；已经完成
的候选不作废。

四个方向完成后，根据官方结果继续：

- 有明确正向机制：围绕该机制做一次组合或降成本实现。
- 全部无提升：转向低维 A/W 互逆拟合，不继续扫描 Attention 参数。
- 多个方向正向：在最新最高分根上合并这些独立改动，提交完整组合版本。

## 8. 怎么归档

每个实际运行候选保存：

```text
solutions/<candidate-name>/
  solution.py
  config.json
  result.md
```

- `solution.py`：完整六 API 文件。
- `config.json`：算法名称、固定参数和实际采用层数。
- `result.md`：改动说明、本地检查、官方分数、官方时间、是否替换根。

工作脚本留在 `workbench/full_solution/<candidate-name>/`；评测 JSON 放在
`artifacts/proxy_v3/full_solution/<candidate-name>/`，不复制进 Git。

官方提高时更新根 `solution.py`、`docs/current-solution-status.md` 和 `solutions/README.md`。失败候选保留
归档，但不继续作为父版本。
