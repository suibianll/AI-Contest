# Attention Q 侧 per-call 数据中心化计划（A-QC1）

> 状态：CLOSED / NO_EFFECT，2026-09-10。
> 执行结果：6/6 层 gate 全拒，六 shard 72 case 与根逐位相同；不占版本号、未提交官方。
> 执行记录：`logs/execution/2026-09-10-attention-aqc1-q-mean-center.md`。
> 从属于当前活动总计划（Linear 线）。当前完整根为 v202 Linear + v195 Attention，官方 `18053/281s`。
> 本文件只负责 Attention A-QC1；版本登记、组合与根切换由总协调线处理。
> 前序 Attention 卡：A-G1（v227 REJECTED）、A-QB1（v228 REJECTED）、
> A-MC1（v229 本地正向 `+0.014923`；官方完整包 TIMEOUT、侧隔离 `14424/245s` 即 −2，已关闭）。

## 1. 定位：与 v228 的关键区分

A-QB1/v228 的失败归因是"拟合参数过拟合校准窗口"——`b_q` 是从 5 个校准窗口学出的静态向量。
A-MC1/v229 证明同一轴向上的**数据无关 per-call 规则**（均值来自当前调用本身）可以本地正向
（层 15 全 case 均匀 +0.079）。A-QC1 把同一规则形态移到 Q 侧，不含任何校准拟合参数，
因此不是 v228 的参数/步数/lr/fold/head 粒度重试。

与 A-MC1 的不对称性必须明说：K 再定心有精确 softmax 不变性保护（逐 query 常数列），
Q 中心化**没有**不变性——`mean(Q)·Kᵀ` 随 key 变化，是有意改变真实函数的规则。
本卡的假设是：Q 的逐通道 token 均值在当前窗口内接近一个可加 DC 分量，移除它能压缩量化输入的
有效动态范围（与 K 侧同源的范围效应），代价是 logit 的系统性移动；二者孰胜由全 folds 真实 MSE
gate 逐层裁决。预期价值低于 A-MC1，这是 Attention 规则级空间的最后一张卡。

## 2. 机制定义与合法性

- 部署规则：在根最终部署坐标（`learned_rotation` 之后、`_dense_to_hif4` 之前）对当前调用的 Q
  做 `Q -= mean_tokens(Q)`（按 head 分组，与 A-MC1 的 reduce 形状一致）。
- 动态 API 只做一次 reduce+减法；无候选循环、无 Gram、无求逆，符合 v165 边界。
- state 只新增每层一个 CPU 标量 arm 标志（`q_state`）；五字段格式不变。
- K/V/Linear 完全不动；冻结根全部已有 state。

## 3. 固定算法

与 A-MC1 完全同构：无训练、无步数、无超参。校准期对 6 个 full-attention 层用全部
calibration folds（窗口等权）完整部署路径真实 MSE 比较「父」与「父+中心化」两臂，
逐层严格改善才写 arm 标志，否则保持父。

## 4. Control（全部必须通过并记录）

1. arm 关闭：Q/K/V 五字段与输出与根逐位一致；
2. arm 开启 + 合成非零均值 Q 输入：Q 五字段改变、K/V/Linear 逐位不变；
   （与 A-MC1 不同：Q 中心化无 softmax 不变性，输出允许改变——记录 dense softmax 输出差异量级
   作为机制生效证据，而不是不变性检查）；
3. 六 API 脱离仓库独立导入；`validate_state` 通过；
4. 记录每层 arm、gate 父/候选 loss、Q changed count。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-aqc1-q-mean-center/`。
日志：`logs/execution/2026-09-10-attention-aqc1-q-mean-center.md`。
输出：`artifacts/proxy_v3/attention-aqc1-<run-id>/`。

GPU 串行（`nvidia-smi` 显存 <2GiB 才启动，忙碌时等待）；先 shard0 排除接口错误，再
`--shards 0,1,2,3,4,5 --stop-after-nonpositive 6` 跑满六 shard；本地正负只记录；
合法且可达的非等价候选归档一个版本（`unregistered/NA`，官方评测用户统一进行）。

## 6. 完成条件

- 本地净负或 NO_EFFECT：归档 REJECTED/NO_EFFECT，Q 侧数据中心化关闭，不以
  midrange/中位数/trimmed 变体重试；
- 本地非负：归档待官方；
- 完成后 Attention 规则级已盘点方向全部有裁决记录（不构成完备性证明）：Q/K 不变量四类
  结构变换 + 两侧 per-call 定心规则均有定论（A-MC1 已获官方侧隔离 −2 定价），后续只由
  官方回传或 21071 锚点源码绑定驱动。
