# Linear 动态样本能量 GPTQ 块序计划

> 创建：2026-09-06
> 状态：**CLOSED / R3_REJECTED_TIME**
> 父版本：v189（官方 `17616/275s`，源码 SHA
> `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`）
> 本地完整最高：Overall `0.687776303`（静态 carrier-energy 候选，时间预测
> `280.622s`，不具提交资格）

## 1. 唯一机制

现有 v189 在校准期编译一个固定的 64-channel activation-GPTQ block 顺序。新候选只
把该顺序改为**每次动态 activation API 调用时**从最终连续变换后的当前输入计算：

```text
score_b = Σ_{j∈block_b} mean_tokens(x_j²) · importance_j
```

其中 `x` 是已有 `smooth_inv/permutation/block-smooth/residual` 变换后的 NVFP4 输入，
`importance` 仍是父版本的静态部署权重重要性。分数降序得到本次 GPTQ 的 64-block
访问顺序，随后复用父版本的合法 block compensation、层级搜索和五字段输出。该规则
不读取其它 case、不会保存 token/output 张量，也不改变权重、Q/K/V、Attention 或
HiF4 编码语义。

这是 activation-GPTQ 的样本条件访问顺序，不是已关闭的 conditional-curvature、
carrier/calibration 静态排序、sample-importance refine 或任何排序参数邻域；配置
固定为上述一个公式，不扫描混合系数、排序方向、采样行数、层/role 路由或 block 数。

## 2. 固定执行顺序

### R0：单文件与合法性

从 v189 根源码生成候选，运行 `py_compile`、六 API 独立导入、最小 Linear smoke、
合法 state/有限输出检查，并确认 Attention 与父版本逐位不变。动态顺序必须是当前
输入维度上的完整 CPU `int16` permutation，父回退路径保持可用。

### R1：Linear eval-v3 前两 shard

使用固定 `proxy-v2` cache、CUDA、`evaluator/eval.py --linear-only --shards 0,1`，
以 v189 为 baseline。记录 paired mean/median、L1、正负 case、reachability、最坏
layer/role 及未修改 Attention control。若无真实变化、接口/有限性失败、或前两片无
一致正向信号，立即关闭。

### R2：六 shard、OOD、default

仅当 R1 通过时运行六片 Linear、OOD 与一次 fresh default；记录完整泛化和时间分解。
要求同一 cache 下 `L1 < 0.02`、OOD `|Δgap| <= 0.01`、Overall 严格超过本地最高
`0.687776303`，且分解模型预测时间 `<280s`。

R2 结果：六 shard、OOD 与接口检查均通过；兼容后端 default 的 Overall 为
`0.688994940507`，但时间模型预测 `285.365171s`，故当前源码只通过分数门，未通过
提交时间门。

### R3：单一实现级降时复核

在不改变上述样本能量公式、输出编码或任何候选参数的前提下，只去掉动态块序路径中
可证明冗余的 CPU permutation 校验/往返，把已由 `argsort` 产生的合法块序直接留在
算法设备上完成重排；以新 SHA 作为一个独立候选。R3 只在 R0 smoke 与 R1 两片配对
保持正向时运行一次 fresh default；不重复运行 R2 源码、不扫描顺序或实现邻域。

R3 结果：R0/R1 通过，fresh default 输出与 R2 候选一致（Linear `0.643867464427`、
Attention `0.752173407020`、Overall `0.688994940507`），但时间模型预测仍为
`284.775756s`。候选归档于
[`20260906_linear-dynamic-actorder_time-rejected`](../../solutions/20260906_linear-dynamic-actorder_time-rejected/)。

## 3. 归档与裁决

候选无论成败都保存 source、SHA、配置、JSON/report 和执行日志。若失败，记
`REJECTED` 或 `REJECTED_TIME`，不扫描邻域，根 `solution.py` 保持 v189。只有严格
超过本地最高且时间通过才建立 v190 归档与提交包；官方回传前不得把本地分数写成官方
分数，官方正向后才切换根版本。

本计划最终状态为 `CLOSED / R3_REJECTED_TIME`：虽然候选超过本地 proxy 最高，但未通过
官方时间预测门禁，未提交官方，根 `solution.py` 仍为 v189。
