# Linear carrier-energy act-order 计划（REJECTED_TIME）

> 状态：`REJECTED_TIME / LINEAR-CARRIER-ENERGY-ACTORDER`
>
> 日期：2026-09-06
>
> 父版本：v189 `static-actorder-hdiag-recovered`（本地 Linear
> `0.6402583244`，Attention `0.7521734070`；官方 `unregistered/NA`）。

## 1. 目标与边界

本计划只验证一个低开销 Linear 机制：用校准期实际输入 NVFP4 carrier 的能量，结合已编译的
部署权重重要性，静态决定 64-channel GPTQ block 的顺序。目标是保留上一轮
`calibration-energy-actorder` 的正向输出误差信号，同时去掉其为计算残差/层级变换而付出的
额外校准时间，使官方时间预测回到 `<280s`。

本计划不改 Attention、不改量化码本、不改 block 内编码器、不增加在线动态算子，也不扫描
sample 数、混合系数、阈值、fold、邻域或候选路由。失败后关闭该统计族，不从失败结果反推参数。

## 2. 单一机制

沿用 v189 的全部实现和状态，只在校准结束后替换 `gptq_block_order`。对每个校准 activation
pair 固定抽取 32 行，解码输入 carrier，并只应用部署中已有的逐元素 `smooth_inv` 与固定
`permutation`；不调用 v189 的 residual/rank/block-smooth 变换。令

```text
E_j = mean_i carrier[i,j]^2
score_b = sum_{j in block_b} E_j * importance_j
```

按 `score_b` 降序得到 block 顺序，随后复用 v189 的静态 act-order GPTQ 路径。该定义让排序
统计直接来自真实 NVFP4 carrier，而不是重新构造完整部署 dense 矩阵。

## 3. 固定父项与证据

- 父代码：`workbench/linear_static_actorder_hdiag_recovered_solution.py`。
- 父固定结果：`artifacts/official_eval/static-actorder-hdiag-recovered-fresh-default-r9.json`。
- 上一候选 `calibration-energy-actorder` 的完整 Linear eval-v3 为正（`+0.000995975`），
  但默认时间预测 `281.515s`，因此已按 `REJECTED_TIME` 归档；本计划只改变排序统计的
  计算路径，不重跑上一候选或其参数邻域。

## 4. 执行门禁

### R0：接口与 reachability

使用 CUDA venv 做单文件编译、六 API 导入和最小 Linear smoke；确认 state 合法、
`gptq_block_order` 为固定 64-block permutation、实际 attempted/accepted block 非零，
并确认 Attention API 与父版本逐位不变。

### R1：两 shard 配对

使用 eval-v3、同一 proxy-v2 cache、Linear-only、父版本作为 baseline，先运行 shard `0,1`。
只检查接口、合法性、control、reachability、逐 case delta 和时间；若出现接口错误、非法
state、无 reachability 或明显非目标 control 变化，立即归档并关闭。

### R2：六 shard Linear

R1 通过后运行 Linear shards `0,1,2,3,4,5`，记录 mean/median、q25/q75、worst-quartile、
负 case、validation/test 同号率、未修改 control、最坏 layer/role/shape/split/length，
以及 API 时间。`L1 < 0.02` 是过拟合门禁；本地正向仅作机制证据，不能换算官方分数。

### R3：OOD 与默认集成

R2 通过后运行同 SHA 的 OOD Linear 面板，计算 `Δ(gain_in - gain_ood)`；只有
`|Δgap| <= 0.01` 才允许保留为待裁决候选。随后运行 fresh default，确认 Attention 逐位
不变，并用六 API 分解模型预测官方时间；预测必须 `<280s` 才具有官方提交资格。

## 5. 归档与提交

候选无论本地正负都保存独立 source、manifest、JSON/report、执行日志和本计划的最终状态。
时间预测不满足时记 `REJECTED_TIME`；机制证据否定时记 `REJECTED`。只有 R0–R3 全部通过且
本地 default 高于 v189、预测时间 `<280s`，才生成只含候选 `solution.py` 的 `solution.zip`
并登记为 `official: unregistered/NA`；根目录 `solution.py` 仍保持官方 v186，不能把本地结果
写成官方分数，也不能在当前工具未提供上传入口时宣称已完成官网提交。

若候选获得官方回传，则按官方总分/时间相对当前完整父版本裁决：提升且不超时才更新根版本；
否则保留根 v186，追加官方结果并关闭本机制。

## 6. 最终裁决

六 shard Linear 与 OOD 均通过，fresh default 的 Overall 为 `0.687776303`，高于 v189 的
`0.686889609`；但时间模型预测为 `280.622241s`，未满足 `<280s` 提交门禁。因此候选记为
`REJECTED_TIME`，未生成或上传官方提交包，根 `solution.py` 保持 v186。
