# HiF4 持续优化计划：Hard-Output Attention + Linear 降时

> ACTIVE，2026-09-09。当前完整根为 18032/280s。后续不再通过增加 STE 训练、矩阵指数或
> 动态范围代理迭代优化 Attention；Attention 改为低维、离散、真实 HiF4 hard-output 优化，
> 同时从当前 Linear 中释放完整方案时间。

## 1. 当前方法的问题

| 证据 | 结论 |
|---|---|
| v190 改变 Q 约108万、K约27万编码，但两个输出窗口都变差并回退 | RMS/传播能量平衡不能代表最终输出 |
| v191 实际部署后 shard0 仅 -0.000104，主奇异值约 1e-11 | 输出梯度方向太弱，首个翻码步长没有有效收益 |
| v192 训练 loss 2.0→1.2237，两个 hard gate 都变差 | STE/连续训练目标与真实量化输出错位 |
| v195 修复多窗口 center 梯度后 shard0 仅 +0.001935 | bug 存在，但不是 Attention 停滞的主因 |
| v198 草稿 smooth-max loss 迭代前后完全相同 | 解析初始化已把代理目标做饱和，继续迭代没有新信息 |
| v194 本地 calibration 快22.5%，官方却为 18032/285s | 局部计时不能代表官方端到端提速 |

因此下一步不是继续调学习率、训练步数或 gate，而是更换优化方法：

    低维参数 → 强制跨越真实 HiF4 码边界 → hard encode/decode
    → 真实 logits / Attention output 计算 → 选择部署状态

## 2. 固定基线与分工

- 最终晋级根：current Linear + R3 Attention，18032/280s。
- Attention 官方侧对照：标准 Linear + R3，14405/238s。
- 当前 Linear 净贡献：18032-14405=3627，侧等价分4628。
- Attention 净贡献：14405-1001=13404。

分工固定：

- **Attention 负责提分。** 它占当前总增量的78.7%，先解决目标错位。
- **Linear 负责释放时间。** 保留其3627分贡献，只做输出等价的计算合并。
- 标准 Linear 只用于测 Attention 官方侧分，不作为最终父版本。

## 3. 每一轮如何执行

每轮只实现一个算法变化，按以下循环持续推进：

1. 从当前最高分完整根构建候选；Attention 候选同时生成标准 Linear 诊断版。
2. 先跑一次 Attention shard0 排除接口错误，再跑4B Attention六 shard，观察真实 hard-output
   是否确实变化。这里的本地结果用于改算法，不换算官方分数。
3. 用标准 Linear 诊断版提交一次官方，计算 attention_step_gain = score - 14405。
4. 官方侧分为正时，把该 Attention 装回当前 Linear，只提交一次完整组合。
5. 完整组合提分且低于300s就切根；提分但超时则保留算法结论，先执行 Linear 降时卡再组合。
6. 失败后换下一种参数化或目标，不在同一算法上扫描 seed、学习率、步数和 clamp。

每轮只保留四项核心记录：修改内容、六 shard hard-output 变化、官方侧分/时间、完整组合结果。

## 4. 先完成已有结果归因

标准 Linear 组合已经生成。运行：

    .venv\Scripts\python.exe workbench/standard_linear_attention_probes/verify.py

随后按 v195 → v191 → v190 → v192 各提交一次标准 Linear 版本。它们只回答历史算法是否在
官方 Attention 侧有效，不阻塞下面的新算法实现，也不因结果继续调旧实现。

v194 已有完整官方 18032/285s，不再提交标准 Linear 版本。v196 与 v192 同机制，不提交。

## 5. Attention 新算法路线

### A1 / v199：GQA × 64-block hard reciprocal coordinate

这是下一张主卡，替换当前 v198 草稿；v198 原样不提交。

参数只有每个 KV/GQA group、每个64通道块一个 u[g,b]：

    Q' = Q * exp(u)
    K' = K * exp(-u)
    V' = V

执行方法：

1. 从 R3 最终 rotation/center 后的 Q/K 开始，V 完全冻结。
2. 对每个64块直接计算使 Q 或 K 首次发生真实 HiF4 编码变化的正、负 reciprocal 步长。
3. 将所有正负候选批量 hard encode/decode，在 fit 窗口上计算真实最终 Attention output MSE。
4. 每个 GQA group 只采用一个最优块移动；全部 group 完成后，在独立 holdout 窗口重新计算真实输出。
5. 只保存最终 u[g,b] 到 state，动态 API 只做一次乘法，不带搜索和训练。

这一卡不使用 RMS loss、smooth-max、STE、Adam、SVD或矩阵指数。目标是让优化过程直接看到码变化和
最终输出变化，而不是先把代理 loss 做小。

### A2 / v200：64 + 8 hierarchy hard reciprocal

只有 A1 找到真实正向块、但块内剩余误差仍明显时执行。

1. 保留 A1 选中的64块。
2. 在该块的8个八元素组上增加 u8[g,b,i]。
3. 每个八元素组仍只测试首次正/负翻码边界，批量计算 hard-output loss。
4. 固定一次由粗到细的遍历：先64块、再8组，不重复回扫。

如果 A1 在全部六层都找不到正向64块，A2不启动，因为细分只会增加自由度和时间。

### A3 / v201：hard-logit residual weighted reciprocal

当 A1 能改变大量编码、但 final output 仍不改善时执行。此时问题是候选排序目标，而不是可达性。

1. 用父版本 hard Q/K 得到真实 logit residual：E = Q_hat K_hat^T - Q K^T。
2. 通过当前 softmax Jacobian和V把 E 映射成输出误差权重，只用于给64块候选排序。
3. 最终采用与否仍由真实 hard Attention output MSE决定，不对 quantizer 使用 STE。
4. 参数仍是 A1 的 u[g,b]，不增加矩阵自由度。

### A4：联合合法 hierarchy 码选择

如果 A1–A3 的 reciprocal transform 均没有官方正向，停止对角互逆族。下一轮直接在 Q/K 的合法
scale_factor/lv2/lv3 候选中做联合 hard-logit 选择；每个64块只比较父状态和一个相邻合法
hierarchy 状态，mantissa随后一次重编码。该方向改变的是量化码选择，不再继续优化连续变换。

## 6. Linear 降时路线

### L-T1 / v202：sample-energy 与基校准融合

当前 Linear 在基校准结束后，又通过 _combined_sample_energy_block_order 重建校准激活并统计
block energy。v202 将 energy 统计合并进已有 Gram/importance 校准遍历：

1. 在首次解码校准激活时同步累计每个64块 energy。
2. 基校准结束时直接生成并保存 gptq_block_order。
3. 删除第二次 _static_actorder_dense_from_state 重建和遍历。
4. 权重参数、activation state和动态输出必须与当前根逐位一致。

本地只核对等价性和完整调用次数，然后直接提交当前完整组合。官方仍为18032且时间低于280s才保留；
否则回退，不继续做同类微优化。

L-T1 在 A1 的六 shard与标准 Linear侧分完成后立即执行，不等待完整组合超时。若更早出现官方正向
Attention，但装回当前根后超时，也直接提前执行 L-T1。

标准 Linear 不作为降时方案，因为它会损失3627分。

## 7. Linear 提分草稿的处置

现有 v197 linear-aw1-block-gain 只改权重并增加 A@W 拟合计算，尚未完成真实4B验证。当前不提交，
也不让它阻塞 Attention A1和 Linear L-T1。

只有完整根已经释放出明确时间后才继续 Linear A@W：

1. 先在真实4B数据运行一次，记录合法投影前后的实际 A@W output loss。
2. 若连续闭式解改善、合法重编码后改善消失，下一张卡改成 hierarchy-aligned legal proposal，
   不增加输出组自由度。
3. 若合法重编码后仍有明显改善，再交完整官方；不按 fit_gain 推算官方分。

## 8. 当前执行队列

| 顺序 | 工作 | 当前状态 | 完成后动作 |
|---:|---|---|---|
| 1 | 标准 Linear 组合静态核对 | 已生成，待运行verify | 提交 v195 侧分 |
| 2 | v195/v191/v190/v192 标准 Linear 官方归因 | 待回传 | 只记录，不调旧实现 |
| 3 | A1 / v199 hard reciprocal 64-block | 下一张实现卡 | 六 shard后提交标准 Linear侧分 |
| 4 | L-T1 / v202 Linear等价降时 | A1后或组合超时时执行 | 完整官方确认时间 |
| 5 | A2或A3 | 根据A1失败类型二选一 | 不同时启动 |
| 6 | 最佳 Attention + 当前最快 Linear | 待侧分正向 | 一次完整官方晋级 |

## 9. 归档

    workbench/full_solution/<candidate>/           构建和验证脚本
    artifacts/proxy_v3/full_solution/<candidate>/  4B本地结果
    solutions/<candidate>/                         单文件、result.md、official-result.json

候选失败后在 result.md 写清楚是“无翻码、hard-output无收益、官方负向或官方超时”中的哪一种，
然后进入队列下一项。活动计划只维护上表，不再堆叠长篇门禁和历史实验流水账。
