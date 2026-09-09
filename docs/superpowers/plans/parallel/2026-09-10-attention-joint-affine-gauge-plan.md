# Attention Q/K 联合仿射 Gauge 优化计划

> 状态：ACTIVE 执行附录，2026-09-10。
> 从属于[Linear 完整输出交叉残差纠码与双线协调计划](../2026-09-10-linear-cross-residual-correction-plan.md)。
> 当前完整根为 v202 Linear + v195 Attention，官方 `18053/281s`。本文件只负责 Attention A-G1，
> 统一父版本、版本号、组合和根切换由总协调计划处理。

## 1. 目标与新自由度

上一轮 Attention 的 rotation/event/group/threshold 方向没有突破：独立事件搜索成本高，低维共享边界
又在真实 softmax 输出上失真。本计划不增加新的 event 搜索，也不把 v190/v198/v199 的 reciprocal
候选缩步重跑，而是在当前根已经执行的 A2 `rotation + K-center` 同一训练循环里增加一个互逆对角
自由度：

**A-G1：rotation、K-center 与 Q/K reciprocal log-scale 的联合仿射 gauge。**

它对应用户确认的“Q/K 互逆 scale 学习”机制，但与旧实现的区别是：scale 不是独立闭式初始化、外挂
smooth-max 循环、固定 temperature 或逐块 hard event，而是和当前根的 rotation、center 使用同一份
最终 Attention 输出残差、同一步数、同一次参数更新联合求解。

## 2. 数学不变性

对每个 KV group，设当前 A2 正交矩阵为 `R`、K center 为 `c`，新增零均值 log-scale
`s`，`D=diag(exp(s))`。部署前变换为

`Q' = (Q R) D`，

`K' = (K R + c) D^{-1}`。

于是

`Q' K'^T = Q K^T + (Q R)c^T`。

第二项对同一个 query 的所有 key token 都是相同常数，softmax 后严格消失。因此在进入 HiF4 量化前，
该变换不改变 Attention 输出；所有收益或损失只来自 Q/K 的离散量化变化。

每个 KV group 共享一个 `s[head_dim]`，同组 Q heads 复用该向量。每步执行
`s -= mean(s)` 去除无效公共温度方向，并固定截断到 `[-log(2), log(2)]`。配置只用这一组，不扫描范围、
粒度、group、步数或正则。

## 3. 固定训练算法

### 3.1 在现有 A2 循环内联合更新

沿用根 `_a2_train_rotation` 的 calibration windows、token 子采样、训练步数、学习率、梯度裁剪、
K-center 和最终真实部署 gate；只增加 `s` 及其 Adam 一、二阶矩，不新增外层循环。

数据职责保持分离：前 `N-1` 个 calibration folds 只负责拟合，梯度按 fold 等权聚合；最后一个
calibration fold 只在训练结束后选择“父 / 唯一联合候选”。eval-v3 的独立 holdout 只记录结果，
不参与参数、候选选择或本地否决。

每一步、每个窗口执行：

1. 由现有 `theta` 得到 `R`，计算 `d=exp(s)`；
2. 生成 `q_t=(Q R)*d`、`k_t=(K R+c)/d`；
3. 按当前 HiF4 路径量化得到 `Q_hat/K_hat/V_hat`，计算最终
   `softmax(Q_hat K_hat^T)V_hat` 对 dense reference 的归一化 MSE；
4. 使用现有 `_m_attention_backward` 得 `dQ_t`、`dK_t`；
5. 回传到 rotation 和 center：`dQ_R=dQ_t*d`、`dK_R=dK_t/d`，center 梯度为 token 维
   `sum(dK_t/d)`；
6. reciprocal log-scale 的解析梯度为
   `grad_s = sum(q_t*dQ_t) - sum(k_t*dK_t)`，按 token 和同组 Q heads聚合；
7. rotation、center、s 同步做一次现有 Adam 更新，然后对 s 做组内零均值投影和固定范围截断。

训练结束只生成一个 `(R,c,s)` 候选。最后一张 calibration window 通过完整动态 Q/K/V 和最终
causal Attention 输出，与当前根候选进行一次真实 MSE 比较；候选严格改善才把三者一起写入 state，
否则三者全部恢复当前根 state。该 gate 只选择父或唯一候选，不循环搜索 scale。

### 3.2 动态部署

在 `_nvfp4_to_hif4` 的现有 `learned_rotation`、`learned_center` 之后、`_dense_to_hif4` 之前增加：

- Q：乘 `exp(s)`；
- K：乘 `exp(-s)`；
- V：完全不变。

`q_state/k_state` 分别保存同一 CPU float32 `s` 和相反应用方向。动态 API 只做一次逐元素乘法，不做
候选循环、Gram contraction、矩阵求逆或校准搜索。

## 4. 与历史机制的边界

- v190 是逐通道闭式 reciprocal balance，最终被 gate 回退；A-G1 从最终 Attention 输出残差联合更新
  `R/c/s`，不是闭式 balance 邻域；
- v198 是独立 GQA reciprocal 初始化加 5 步 smooth-max，再做 hard gate；A-G1 不增加训练循环，scale
  进入根 A2 的同一优化器并影响 rotation/center 梯度；
- v199 是每 GQA group/64-block 的 hard reciprocal 边界选择；A-G1 是每 KV group 的连续对角 gauge，
  不枚举 hard event；
- v217 是固定 temperature `1.25`；A-G1 去除了公共 temperature 无效方向并学习通道相对尺度；
- A-RB1 改 mantissa threshold；A-G1 保持编码器边界不变，只改变进入量化器的精确等价坐标。

因此本卡只验证“与现有 rotation 和 center 联合训练的互逆对角 gauge”这一实现，不据此重开已关闭的
factor、step、window、rank、block 或 fixed-temperature 邻域。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-ag1-joint-affine-gauge/`。
日志：`logs/execution/2026-09-10-attention-ag1-joint-affine-gauge.md`。
输出：`artifacts/proxy_v3/attention-ag1-<run-id>/`。

1. 从总协调计划登记的 R0 复制完整单文件候选，不修改根或归档源码；
2. 验证 `s=0` 时变换、Q/K 五字段和最终输出逐位恢复当前根；
3. 用非零零均值合成 `s` 验证 Q 乘正尺度、K 乘逆尺度，dense softmax 输出保持不变且 hard Q/K
   编码可发生变化；
4. 验证六 API 独立导入、合法 state、finite、V control、Linear control，并记录训练
   `||s||`、非零通道、Q/K changed count、gate 父/候选 loss；
5. 运行 Attention shard0 排除接口、异常回退和死分支；
6. 只要形成合法、可达、非等价 hard output，就固定运行 Attention 六 shard一次；本地正负、holdout
   和 API 时间只记录；
7. 保存源码、配置、SHA、72 case 配对结果和单文件导入结果，交总协调计划形成唯一完整候选并官方裁决。

如果六层均恢复父或最终输出逐位相同，记 `NO_EFFECT`；官方负向或超时关闭 A-G1。不减少 A2 步数、
缩小尺度范围或拆分 head/block 重试。

## 6. 与 Linear 计划的交互处理

- 本计划开发期间固定 R0，不读取 Linear L-XR1 候选的 state 或源码；GPU 评测与 Linear 串行；
- A-G1 候选的 Linear 两 API必须与 R0一致；不得把 Linear 本地收益写入 Attention 结果；
- 两个单机制先从同一 R0 分别获得官方结果。A-G1 只有在自身官方正向时才有资格进入组合；
- 若两边都正向，由总协调计划选择官方分更高的完整候选为父，在该父上重新训练 A-G1 或重新编译
  L-XR1，运行一次完整双侧 interaction audit，再提交组合；
- 不假设两侧分数可加，也不根据侧隔离时间推算组合时间。

## 7. 完成条件

A-G1 的固定实现获得一次明确裁决即完成：

- 无可达非等价输出：`NO_EFFECT`；
- 官方未提高或 `TIMEOUT`：`REJECTED`；
- 官方提高且 `<300s`：由总协调计划登记为完整根或进入双正向组合。

完成后本文件与日志一并归档，不追加 reciprocal 参数邻域。
