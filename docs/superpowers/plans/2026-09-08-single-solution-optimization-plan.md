# 当前最高分版本持续优化执行计划

> ACTIVE，2026-09-08。只回答三件事：怎么改、怎么跑、怎么归档。

## 1. 执行方式

始终把根 `solution.py` 当作当前最高分完整版本。每轮只做一个明确改动，但提交物始终是包含六个 API
的完整 `solution.py`，不再建立 Linear 版、Attention 版或两条并行路线。

每轮按下面五步执行：

1. 从根 `solution.py` 复制候选，写清本轮唯一算法改动和固定参数。
2. 完成实现后，先跑合成检查与六 API 导入检查。
3. 只跑受影响场景的 4B shard0，排除实现错误、完全无效和严重退化。
4. 将候选归档为完整包并提交官方；本地小幅正负不决定是否提交。
5. 官方分数更高且未超时就替换根版本；否则保留原根，从原根开始下一轮。

不再维护候选池、门禁表、误差账本或两侧阶段计划。一次只推进下面列出的当前轮次。

## 2. 第一轮：提交 LC3 objective-only

### 优化内容

当前候选已经完成，路径为：
`solutions/continuous_linear_lc3-objective-only/solution.py`。

它只修改 Linear 校准时的候选评分：

```text
原评分 = 输出误差平方和 / 原输出能量
新评分 = 输出误差平方和 / 标准 HiF4 输出误差平方和
```

对应代码入口：

- `_linear_output_candidate_metrics`
- `_linear_output_candidate_metrics_combos`
- `_linear_smooth_hybrid_metrics`
- Linear 校准 fold 中标准 HiF4 误差的预计算与传递

这轮不再改代码，也不再增加本地实验。已有合成检查、六 API 检查和 Linear shard0 足以确认实现可以
提交；下一动作就是上传该完整候选。

### 执行

1. 提交 `solutions/continuous_linear_lc3-objective-only/solution.py`。
2. 将官方分数和时间写入同目录 `result.md` 与 `manifest.json`。
3. 按以下方式处理：
   - 分数提高且运行未超时：把该文件复制为根 `solution.py`，它成为下一轮父版本。
   - 分数未提高：根保持不变，LC3 标记为 `REJECTED`。
   - 超时或运行错误：记录实际错误，根保持不变。
4. 结果登记完成后直接进入第二轮，不围绕 LC3 调权重、fold 或归一化系数。

## 3. 第二轮：在最新根上加入 Q/K 互逆残差变换

这一轮不是单独维护 Attention 版本，而是在第一轮裁决后的最高分完整根上增加一个已经出现官方正收益
的 Q/K 机制。参考实现为
`solutions/continuous_attention_anchor22-a2/solution.py`，只作为代码来源，不作为父版本。

### 算法

保留根版本已经学到的 Q/K rotation 和 K center，再学习每个 GQA group 的对称矩阵 `S`：

```text
Q_new = Q_parent @ exp(S)
K_new = K_parent @ exp(-S)
```

部署时同步编译：

```text
Rq_new = Rq_parent @ exp(S)
Rk_new = Rk_parent @ exp(-S)
c_new  = c_parent  @ exp(-S)
```

这样连续 QK 乘积不变，优化目标只是把 Q/K 的动态范围重新分配到更适合 HiF4 编码的位置。最后一个
校准窗口比较新状态与根状态的真实 Attention 输出误差；只有新状态更好时才保存它，否则该层继续使用
根状态。

### 固定实现

从参考实现移植以下函数及其调用，不重新设计另一套训练器：

- `_a21_exp`
- `_a21_exp_backward`
- `_a21_project`
- `_a21_scale_loss_grad`
- `_a21_matrix_grad`
- `_a22b_train`
- `_a21_gate_loss`

在根的 `hif4_calibration_attention` 中，先按原逻辑得到完整父状态，再调用 `_a22b_train` 生成候选状态，
最后用 `_a21_gate_loss` 比较两者。Q/K 动态 API 不增加训练，只读取校准后保存的 rotation/center。

参数直接固定为参考实现已验证的一组：

```text
训练步数       32
学习率         0.01
梯度范数上限   1.0
S 正则         0.001
Adam beta      0.9 / 0.999
谱范围         ±log(2)/2
矩阵约束       对称、零迹
候选数量       1
```

### 实现目录

新建 `workbench/full_solution/qk-reciprocal-residual/`：

- `build.py`：从当时的根生成候选并移植上述增量。
- `verify.py`：检查 `S=0` 能恢复父状态、QK 连续乘积保持、K center 同步变换。
- `check_math_and_import.py`：检查矩阵梯度、六 API 单文件导入和合法 state。
- `candidate/solution.py`：待评测的完整候选。
- `config.json`：只记录上面的固定参数。

### 运行顺序

```powershell
.venv\Scripts\python.exe workbench/full_solution/qk-reciprocal-residual/build.py
.venv\Scripts\python.exe workbench/full_solution/qk-reciprocal-residual/verify.py
.venv\Scripts\python.exe workbench/full_solution/qk-reciprocal-residual/check_math_and_import.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/qk-reciprocal-residual/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/qk-reciprocal-residual-shard0
```

执行者只需确认：检查脚本通过、训练确实产生非零 `S`、至少有层接受候选、shard0 没有严重异常。满足后
就归档并提交官方，不继续试学习率、步数、group 数或其他变体。

### 结果处理

- 官方提高：候选替换根，下一轮继续从新根优化。
- 官方不提高：根不变，关闭这次“完整根 + Q/K 互逆残差”的实现。
- 官方超时：根不变；下一轮优先选择不增加校准训练的机制。
- 运行错误：只修复明确的实现错误并重新验证，不趁机改变算法。

## 4. 后续怎么继续优化

第二轮官方结果回来后再写下一轮，不提前堆一长串候选。选择规则很简单：

- 如果 Q/K 互逆残差有效，下一轮优先减少它的校准成本或把同一变换更好地编译进现有状态。
- 如果无效，回到最新根，下一轮改做低维 A/W 互逆拟合；届时先固定唯一的低维基和训练参数，再开始实现。
- 如果超时，下一轮只选不会新增在线计算、且校准开销明显更小的变换。

每次只把“下一轮马上要执行的算法”写成上述详细程度；尚未开始的方向不再写成大段限制条件。

## 5. 归档方式

每个实际运行的候选只保留一份归档：

```text
solutions/<candidate-name>/
  solution.py     完整六 API 提交文件
  config.json     本轮算法和固定参数
  result.md       本地检查、官方分数、时间和结论
```

工作脚本放在 `workbench/full_solution/<candidate-name>/`，评测 JSON 放在
`artifacts/proxy_v3/full_solution/<candidate-name>/`。评测产物不复制进 Git。

`result.md` 只写五项：改了什么、本地是否正常、官方分数、官方时间、是否替换根。官方结果登记后更新：

- `solution.py`（仅成功时替换）
- `docs/current-solution-status.md`
- `solutions/README.md`
- 本计划的“当前轮次”

失败候选保留归档，不删除、不继续作为父版本，也不为它追加新的调参分支。
