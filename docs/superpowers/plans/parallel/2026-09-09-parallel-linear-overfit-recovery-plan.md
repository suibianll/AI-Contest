# 历史过拟合 Linear 机制的 4B 并行恢复计划

> 状态：并行执行附录（v3，2026-09-09）。
> 本文件从属于[唯一活动总计划](../2026-09-09-output-aware-rounding-and-joint-aw-plan.md)，不建立
> 第二条父版本、版本号或官方晋级线。当前完整父仍为 v202 Linear + v195 Attention，官方
> `18053/281s`。根切换、正式归档和官方结果登记继续由唯一活动总计划统一处理。

## 1. 结论

本计划只执行 **PLA1：带权重误差交叉项的低秩输出纠码**。它恢复 v107-v125 中“用最终部署
权重输出度量修正动态 Activation”的有效部分，但补上旧方法缺少的 dense/deployed Weight 交叉残差，
并把多轮逐坐标搜索压缩为一次 rank-4 后处理。

原 PLW1/C70 整层 offset 重构在实现前关闭：现役权重编码器已经枚举 E6M2 offset 并完整重解
lv2/lv3/mantissa；C70 本身在旧 Qwen/OPT 上负向，后续记录也已关闭该路线。PLW1剩余差异只有
“逐块接受改为整层接受”，属于接受粒度和 offset 邻域，不值得写代码或运行4B。

## 2. 为什么 PLA1 与当前活动计划不冲突

活动总计划的 L-RB1/L-JRB1 改变跨样本共享的 mantissa 舍入边界。L-JRB1 的实际执行又确认：
当 `group_gram` 存在时，活动编码路径由 `_adaround_mantissa` 生成 mantissa，共享 threshold 不会被查询；
4B 的 `in_features=2560` 窄层因此无法被 L-JRB1 的 Activation 边界修改。

PLA1只处理这些 `in_features <= 3072`、已有部署 Gram 的窄层。它不学习 threshold，不改变Weight，
而是在父 Activation 编码完成后，根据当前输入样本的真实量化误差做一次合法纠码。因此两者的作用层、
变量和动态规则都不同。

目录所有权固定如下：

- 活动总计划继续独占 `workbench/full_solution/linear-lrb1-residual-rounding/`、
  `attention-arb1-qk-rounding/`、`linear-jrb1-joint-aw-boundaries/`、根 `solution.py` 和正式归档；
- 本计划只使用 `workbench/parallel_linear_overfit_recovery/pla1-lowrank-output-correction/`、
  `artifacts/proxy_v3/parallel-linear-overfit-recovery/pla1-<run-id>/` 和
  `logs/execution/2026-09-09-parallel-linear-overfit-recovery-pla1.md`；
- 本计划不读取或改写其他 workbench，不直接覆盖根文件。

## 3. PLA1 的输出目标

设当前样本的 dense Activation 为 `X`，父动态编码的解码值为 `X_hat`，dense Weight 为 `W`，
父部署权重为 `W_hat`。父输出残差为

`R = X_hat W_hat^T - X W^T`。

若只改变 Activation，增量为 `DeltaX`，真实 Linear 输出损失变化是

`DeltaL = 2 <R W_hat, DeltaX> + tr(DeltaX G DeltaX^T)`，

其中 `G = W_hat^T W_hat`。完整梯度可改写为

`R W_hat = (X_hat-X)G + X C`，

`C = (W_hat-W)^T W_hat`。

旧的 Gram-only Activation 精修只近似使用 `(X_hat-X)G`，遗漏了 Weight 量化误差与Activation之间的
交叉项。PLA1同时编译 `G` 的块外低秩项和 `C` 的低秩项，使动态纠码显式感知 `W-W_hat`。

## 4. 固定算法

### 4.1 校准阶段

仅对 `in_features <= 3072` 的层执行：

1. 完成当前父Weight编码，得到最终 `W_hat`；
2. 计算 `G=W_hat^T W_hat` 和 `C=(W_hat-W)^T W_hat`；
3. 保存 `G` 的现有4×4 block-local部分；
4. 从去掉4×4 block-local部分的 `G` 中取绝对特征值最大的4个对称方向，保存
   `U_g[channels,4]` 和带符号 `lambda_g[4]`；
5. 对非对称矩阵 `C` 做固定 rank-4 SVD，保存 `U_c[channels,4]`、`S_c[4]`、
   `V_c[channels,4]`；
6. rank固定为4，不扫描rank、收缩、ridge、覆盖率或层名单。宽层保持父状态。

新增状态只包含上述紧凑因子和父已有的4×4局部Gram；不保存完整 `G/C`，不把校准样本或搜索过程
带入动态API。

### 4.2 动态阶段

1. 先完整执行父Activation编码，得到合法五字段和 `X_hat`；
2. 计算 `E=X_hat-X`；
3. 用
   `g_approx = E G_local + (E U_g) diag(lambda_g) U_g^T + (X U_c) diag(S_c) V_c^T`
   得到当前样本的输出梯度近似；
4. 按父block order的反向顺序处理自然64通道块；每个块把当前码与固定
   `scale_factor/lv2/lv3` 下的相邻 mantissa `-1/+1` 一次张量化比较；
5. 每行每个64块最多修改一个4元素组。以冻结的 `g_approx` 和精确4×4局部Gram计算候选二次型，
   只应用其中损失变化严格小于0的最佳组；
6. 所有块只处理一遍，不更新梯度、不做第二轮、不改变scale/lv2/lv3、不增加Python候选轮询；
7. 输出仍只有合法五字段，mantissa为0时使用规范零sign。

这是固定、样本自适应的部署规则。低秩近似与完整输出目标之间的误差只作诊断，不设置人为一致率门
或本地分数门。

## 5. 实际执行

### 5.1 实现与控制

1. 从启动时的当前最高分完整根复制候选到PLA1独立工作目录；
2. PLA1关闭时必须与该父版本六API及输出逐位一致；
3. 用合成窄层输入证明新状态通过合法检查、rank-4交叉项进入动态路径，并产生可解释的
   attempted/accepted和changed-mantissa计数；
4. 对校准样本额外计算一次完整 `G/C` 的真实 `DeltaL`，与低秩近似结果并列记录，用来解释误差，
   但不因某个比例或阈值提前停止候选。

### 5.2 4B与官方

1. 运行Linear shard0，排除接口、状态和死分支问题；
2. 若真实4B调用中从未尝试或从未改变任何码，记录 `NO_REACHABILITY`，不生成无差异候选；
3. 只要形成合法、可达、非等价输出，就运行固定六shard一次；本地正负、L1、分位数和API时间只记录；
4. 保存源码、固定配置、shard0、六shard结果及单文件导入检查，交给活动总计划；
5. 活动总计划在届时最高分完整根上重放PLA1，非等价代表交官方裁决。官方分数提高且时间小于300秒
   才更新根；官方负向或超时后关闭本实现。

不根据4B结果修改rank、邻码范围、处理轮数、每块修改数量或形状范围。

## 6. 记录内容

执行日志记录：

- 父源码SHA、候选源码SHA和固定配置；
- 各层 `G/C` 低秩重建误差；
- attempted/accepted、changed sign/mantissa、修改行数与4元素组数；
- 近似二次型与完整校准输出 `DeltaL`；
- shard0及六shard配对结果、父/候选API时间；
- 最终状态：`NO_REACHABILITY`、`REJECTED`、`TIMEOUT`、`unregistered/NA`或交回总计划。

本计划产生的缓存只能在对应运行结束后清理，并使用 `--min-age-hours 2`；不得删除
`qwen3.5-4b-proxy-v2.pt`，不得删除其他执行线正在写入的缓存。

## 7. 完成条件

PLA1完成一次固定实现的4B与官方裁决后，本计划结束：

- 无真实码变化：以 `NO_REACHABILITY` 关闭；
- 有变化但官方不提高或超时：关闭该实现，不缩rank或减少覆盖重试；
- 官方提高且小于300秒：由活动总计划切换完整根。

完成后将本文件移入 `docs/superpowers/archive/plans/`。本附录不继续追加新的Linear卡。
