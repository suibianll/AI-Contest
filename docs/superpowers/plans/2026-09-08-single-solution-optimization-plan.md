# 当前根提速与有效 Attention 优化计划

> ACTIVE，2026-09-08。当前根官方 `18032/280s`。目标是先去掉现有实现中的重复计算，再做真正改变
> 当前根输出的 Attention 优化。旧 v190–v193 已执行完毕，不再调参重试。

## 1. 总体执行顺序

只做三个候选：

1. **v194：A2/R3 校准等价提速**——输出与当前根完全相同，只减少重复计算。
2. **v195：修复 K-center 梯度聚合**——修复现有训练器只使用最后一个训练窗口 center 梯度的问题。
3. **v196：按原始正向配置移植 Q/K 互逆残差**——使用历史 A22-2 的 `4窗训练+1窗选择`，不再使用
   v192 的 `3窗训练+2窗全通过` 改版。

三个候选分别从当前根构建，不互相等待。某个候选官方正向后，尚未提交的候选再合入该正向改动；已经
提交的候选保持独立结果，不重复提交。

本计划不运行 OOD、跨模型、全六 shard、参数扫描或额外诊断面板。每个候选只做与其目标直接相关的
一次验证和一次 Attention shard0。每个验证脚本同时完成六 API 单文件导入和有限输出检查，不再另建
一层检查脚本。

## 2. v194：A2/R3 校准等价提速

候选名：`attn-a2-calibration-fused`

### 为什么先做

当前根只剩约 20 秒官方时间余量。v190 最终回退父状态仍超时，说明任何新算法之前必须先减少已有
Attention 校准开销。当前 `_a2_train_rotation` 和 gate 中存在三处可直接消除的重复计算。

### 代码修改

只改当前根的 `_a2_train_rotation`、`_a2_true_path_gate_loss` 和
`hif4_calibration_attention`：

1. 每个训练 step 的 `theta` 对所有窗口相同。把 `_m_cayley_pair(theta)` 和
   `rotation = base @ cayley(theta)` 从窗口循环内移到循环外，每 step 只计算一次。
2. 预处理窗口时 `std_v` 和 `v_hat` 是同一个 `_dense_to_hif4(v_sub)` 结果。只编码一次，直接令
   `v_hat = std_v`。
3. 当前 gate 为 identity 和 rotation 各调用一次 `_a2_true_path_gate_loss`，而每次函数内部又重新计算
   父输出。改成一次函数同时返回 `parent_mse` 和 `candidate_mse`，父状态只执行一次 Q/K/V 编码和
   Attention 前向。

不改变训练步数、采样 token、rotation、center、loss、优化器、最终 state 字段或动态 API。

### 执行

工作目录：`workbench/full_solution/attn-a2-calibration-fused/`。

```powershell
.venv\Scripts\python.exe workbench/full_solution/attn-a2-calibration-fused/build.py
.venv\Scripts\python.exe workbench/full_solution/attn-a2-calibration-fused/verify_equivalence.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attn-a2-calibration-fused/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/attn-a2-calibration-fused-shard0
```

`verify_equivalence.py` 只验证一件事：同一真实 4B 校准输入下，父子返回的部署 state 和 Q/K/V 动态输出
逐位一致。shard0 只确认分数相同并记录双方 calibration API 时间。

逐位一致且校准更快后直接归档并提交官方。同分但更快就替换根；若没有明显变快，结束该候选，不继续做
微型性能改写。

## 3. v195：修复 K-center 梯度聚合

候选名：`attn-a2-center-gradient-aggregate`

### 问题

当前 `_a2_train_rotation` 每个 step 会遍历全部训练窗口：`grad_theta` 对所有窗口累加，但
`grad_center = dk3.sum(dim=0)` 在循环内反复覆盖，所以 center 实际只使用最后一个训练窗口的梯度，
与当前多窗口平均输出 loss 不一致。

### 代码修改

在每个训练 step 开始时增加：

```text
grad_center = zeros_like(center)
```

把窗口内的：

```text
grad_center = dk3.sum(dim=0)
```

改成：

```text
grad_center += dk3.sum(dim=0)
```

其余训练代码完全不变：仍为当前步数、学习率、Cayley rotation、最终输出 MSE 和最后一个窗口选择。
这一改动几乎不增加计算量，只让 center 与 rotation 使用相同的多窗口信息。

### 执行

工作目录：`workbench/full_solution/attn-a2-center-gradient-aggregate/`。

```powershell
.venv\Scripts\python.exe workbench/full_solution/attn-a2-center-gradient-aggregate/build.py
.venv\Scripts\python.exe workbench/full_solution/attn-a2-center-gradient-aggregate/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attn-a2-center-gradient-aggregate/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/attn-a2-center-gradient-aggregate-shard0
```

`verify.py` 只确认三个事实：center 梯度确实包含全部四个训练窗口、输出有限、至少一个校准层的 center
或最终选择发生变化。随后直接看 shard0：候选不是 no-op、没有严重错误即可归档并提交官方。本地小幅
正负不用于挑参数。

## 4. v196：原始配置的 Q/K 互逆残差移植

候选名：`attn-reciprocal-residual-original-split`

### 为什么重做这一项

A22-2 历史上在 R3 上取得过官方正收益。v192 并没有复现它：原实现用前四个窗口训练、最后一个窗口
选择，v192 改成前三个窗口训练，并要求后两个窗口同时严格改善，导致提案最终回退。v196恢复原始配置，
但仍从当前完整根构建。

### 代码修改

从 `solutions/continuous_attention_anchor22-a2/solution.py` 移植以下完整增量：

- `_a21_exp`
- `_a21_exp_backward`
- `_a21_project`
- `_a21_scale_loss_grad`
- `_a21_matrix_grad`
- `_a22b_train`
- `_a21_gate_loss`

父状态仍由当前根原有 `hif4_calibration_attention` 产生。然后：

```text
fit  = calib_qkv_list[:-1]
gate = calib_qkv_list[-1]
Q_new = Q_parent @ exp(S)
K_new = K_parent @ exp(-S)
c_new = c_parent @ exp(-S)
```

固定沿用原实现：32步、学习率 `0.01`、梯度裁剪 `1.0`、正则 `0.001`、对称零迹 `S`、谱范围
`±log(2)/2`。只比较最后一个窗口的真实 Attention 输出 MSE；更好就采用，否则回父。不增加第二个
gate，不改变训练窗口，也不尝试联合乘积目标。

### 执行

工作目录：`workbench/full_solution/attn-reciprocal-residual-original-split/`。

```powershell
.venv\Scripts\python.exe workbench/full_solution/attn-reciprocal-residual-original-split/build.py
.venv\Scripts\python.exe workbench/full_solution/attn-reciprocal-residual-original-split/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attn-reciprocal-residual-original-split/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/attn-reciprocal-residual-original-split-shard0
```

`verify.py` 只验证互逆关系、K-center 同步编译和训练分支实际执行。shard0 中至少有一个层采用残差且
输出发生变化后归档。由于该算法增加校准训练，优先将 v194 的等价提速合入提交包；如果仍发生官方超时，
结束全矩阵残差实现，不再缩窗或减步数。

## 5. 归档

每个候选只保存：

```text
solutions/<candidate-name>/
  solution.py
  config.json
  result.md
```

`result.md` 记录算法改动、shard0结果、校准时间、官方分数/时间和是否替换根。工作脚本留在
`workbench/full_solution/<candidate-name>/`；评测 JSON 留在 `artifacts/proxy_v3/full_solution/`。

v194–v196完成后，无论结果正负，都先根据官方结果重新选择下一种机制，不在这三个实现上继续调参数。
