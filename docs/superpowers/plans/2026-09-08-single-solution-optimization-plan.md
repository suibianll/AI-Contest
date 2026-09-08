# 当前根提速与低增量 Attention 优化计划

> ACTIVE，2026-09-08。当前根官方 `18032/280s`。v190、v191、v192 均已官方超时；当前不再追加
> 新的 Attention 校准循环，只执行输出等价提速和一个近零成本的训练修复。

## 1. 当前只执行两项

1. **v194：A2/R3 校准等价提速。** 删除当前根已有流程中的重复矩阵求解、重复 V 编码和重复 gate
   前向，输出与当前根保持一致。
2. **v195：K-center 梯度聚合修复。** 修复 center 梯度在窗口循环中被覆盖的问题，不增加训练轮数、
   窗口或额外前向。

此前准备的 v196 全矩阵互逆残差草稿不提交：它仍包含与 v192 同级的32步全矩阵训练，改变窗口划分
不能解决官方超时。工作目录保留作草稿，不纳入当前执行。

每项只做一次直接验证和一次 Attention shard0。验证脚本同时完成六 API 单文件导入和有限输出检查；
不运行 OOD、跨模型、全六 shard、参数扫描或额外诊断面板。

## 2. v194：A2/R3 校准等价提速

候选名：`attn-a2-calibration-fused`

### 修改内容

只改当前根的 `_a2_train_rotation`、`_a2_true_path_gate_loss` 和
`hif4_calibration_attention`：

1. 每个训练 step 中，`theta` 对全部窗口相同。将 `_m_cayley_pair(theta)` 和
   `rotation = base @ cayley(theta)` 移出窗口循环，每 step 只执行一次。
2. 窗口预处理中的 `std_v` 与 `v_hat` 来自同一次 `_dense_to_hif4(v_sub)`。保留一次编码，直接复用
   `std_v`。
3. 当前 identity 与 rotation gate 重复计算父状态。改成一次函数同时返回父 MSE 和候选 MSE，父
   Q/K/V 编码及 Attention 前向只执行一次。

训练步数、token采样、rotation、center、loss、优化器、最终state和动态API全部不变。

### 执行

```powershell
.venv\Scripts\python.exe workbench/full_solution/attn-a2-calibration-fused/build.py
.venv\Scripts\python.exe workbench/full_solution/attn-a2-calibration-fused/verify_equivalence.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attn-a2-calibration-fused/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/attn-a2-calibration-fused-shard0
```

`verify_equivalence.py` 对同一真实4B输入比较父子部署state和Q/K/V动态输出。逐位一致且本地校准时间下降
后归档、提交官方。同分但更快则替换根；没有变快就结束，不继续微调性能实现。

## 3. v195：K-center 多窗口梯度修复

候选名：`attn-a2-center-gradient-aggregate`

### 修改内容

当前 `_a2_train_rotation` 每个 step 内，`grad_theta` 累加全部训练窗口，但：

```text
grad_center = dk3.sum(dim=0)
```

在窗口循环中反复覆盖，导致 center 只使用最后一个窗口。修改为每个 step 开始初始化一次，然后累加：

```text
grad_center = zeros_like(center)
grad_center += dk3.sum(dim=0)
```

其余代码不变：不增加训练步数、校准窗口、gate或动态计算。

### 执行

```powershell
.venv\Scripts\python.exe workbench/full_solution/attn-a2-center-gradient-aggregate/build.py
.venv\Scripts\python.exe workbench/full_solution/attn-a2-center-gradient-aggregate/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attn-a2-center-gradient-aggregate/candidate/solution.py --baseline-solution solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts/proxy_v3/full_solution/attn-a2-center-gradient-aggregate-shard0
```

`verify.py` 确认center梯度包含全部四个训练窗口、输出有限，并且至少一个校准层的center或最终选择发生
变化。候选不是no-op且没有运行错误就归档、提交官方；本地小幅正负不用于调参。

## 4. 后续决定

v194与v195完成官方裁决后再制定下一轮：

- v194明显降低官方时间：从更快根重新选择一个输出优化机制。
- v195提高分数：将其作为新根，并继续寻找不增加校准循环的修改。
- 两项均无收益：停止继续堆Attention校准计算，转向当前根已有计算的目标或实现修正。

在获得更快官方根之前，不提交v196，也不提交v193或其变体。

## 5. 归档

```text
solutions/<candidate-name>/
  solution.py
  config.json
  result.md
```

`result.md`只记录修改内容、shard0、校准时间、官方分数/时间和是否替换根。工作脚本留在
`workbench/full_solution/<candidate-name>/`，评测JSON留在`artifacts/proxy_v3/full_solution/`。
