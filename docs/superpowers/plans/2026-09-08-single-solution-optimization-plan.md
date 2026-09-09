# HiF4 持续优化计划：Hard-Output Attention + Linear 降时

> ACTIVE，2026-09-09。当前完整根为 v202 Linear + v195 Attention，18053/281s。后续不再通过增加 STE 训练、矩阵指数或
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

- 最终晋级根：v202 Linear + v195 Attention，18053/281s；上一完整根为 current Linear + v195 Attention，18053/289s；再上一根为 current Linear + R3，18032/280s。
- Attention 官方侧对照：标准 Linear + R3，14405/238s。侧隔离归因已于 2026-09-09 完成，
  见 §4：v195 +21、v192 +22、v190/v191/v194 均为 0。
- **缺口量化（2026-09-09，提交前必须看）**：当前根 `18053`，距榜首锚点 `21765` 差 **3712 分**。
  按当前最优效率 4.2 分/秒（v195：+21 分 / +5s）× 剩余余量 19s，时间最多再换 **约 80 分**；
  按机制维度 21 分/机制，需约 **180 个**同量级机制。**任何单机制卡的量级判断都要对标这个数，
  不要再为 ±20 分的机制投入完整轮次。**
- 当前 Linear 与 v195 Attention 的侧向净贡献待标准 Linear v195 诊断版官方归因，不从完整总分反推；
  上一根的 Linear 参考增量为 3627 分。

分工固定：

- **Attention 负责提分。** v195 已在完整组合上取得官方 +21，继续解决目标错位。
- **Linear 负责释放时间。** v202 已完成输出等价的计算合并并将官方时间降至 281s；当前根余量为 19s，可进入下一张 Linear A@W 卡。
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

## 4. 已有结果归因（2026-09-09 已完成，官方分已回传）

五个标准 Linear 诊断版已全部回传。基线 = 标准 Linear + R3 `14405/238s`；
`attention_step_gain = score - 14405`。完整归因见
[侧隔离分登记](../../../logs/execution/2026-09-09-standard-linear-attention-side-scores.md)。

| Attention 变体 | 侧隔离官方分 / 时间 | gain | 裁决 |
|---|---|---|---|
| v195 | `14426 / 243s` | **+21** | 与完整包 `18053−18032=+21` 交叉验证；已在根上兑现 |
| v192 | `14427 / 272s` | **+22** | 有效，但 +34s → 完整包 TIMEOUT；需先降时 ≥40s 才可能回收 |
| v190 | `14405 / 246s` | **0** | **机制证伪**，关闭机制族，不做邻域重试 |
| v191 | `14405 / 264s` | **0** | **机制证伪**，关闭机制族，不重试块对/步长 |
| v194 | `14405 / 234s` | **0** | 侧隔离 −4s 但完整包 +5s（285s）；等价提速路线不成立 |

由此确定的三条硬约束：

1. **Attention 互逆/残差族八个变体的官方结局为
   `0 / 0 / +22 / TIMEOUT / TIMEOUT / TIMEOUT / TIMEOUT / TIMEOUT`（v190/v191/v192/v196/
   v198/v199/v201/v203）。** 该族不再出卡；v192 的 +22 只有在降时 ≥40s 后才值得回收。
2. **标准 Linear 侧隔离时间对完整根时间没有预测力**（v194 侧隔离 −4s / 完整包 +5s；
   v190 侧隔离 246s / 完整包 TIMEOUT，两个方向都不对）。侧隔离只用于测分，
   任何基于它的时间推断都不得写入晋级或否决理由。
3. **上述关闭只针对这些具体实现与其直接邻域**，不推广为"Attention 侧已饱和"或
   "互逆重参数化无效"（与 LC1/LC2 审计同一条纪律）。

v196 与 v192 同机制，不提交侧隔离版。v193 未提交。

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

本地只核对等价性和完整调用次数，然后直接提交当前完整组合。v202 已获官方 `18053/281s`，
相对 v195 同分快 8 秒；
候选需保持分数并低于官方 `300s` 才保留；
否则回退，不继续做同类微优化。

L-T1 在 A1 的六 shard与标准 Linear侧分完成后立即执行，不等待完整组合超时。若更早出现官方正向
Attention，但装回当前根后超时，也直接提前执行 L-T1。

标准 Linear 不作为降时方案，只用于 Attention 侧归因；当前 v195 的完整增量不从侧分反推。

## 7. Linear 提分草稿的处置

现有 v197 linear-aw1-block-gain 已证明当前实现存在部署 block 对齐错误，不能直接重交。v202
官方将完整根从 289s 降到 281s，满足“时间释放后再继续 Linear A@W”的前置条件。已执行的卡为
**L-AW1 / v204：部署坐标对齐的 64-block 标量 A@W 拟合**：

1. 从当前 v202 完整根构建，只修正 A@W 拟合与最终部署 `gptq_block_order`、permutation 及
   weight carrier 的同坐标映射；不复用 v197 的错位拼接。
2. 按用户最新指令使用全部 Qwen3.5-4B 校准数据，不拆 fit/select；每个 64 输入块仅一个
   固定标量增益，直接以 `XW^T - Q(X)Q(W)^T` 的实际输出目标求解。
3. 将拟合后的块重新编码为合法五字段 state，并把部署动态路径保持为已编译状态，无在线搜索、
   候选循环或额外 Attention 校准。
4. 只注册一个固定配置；记录 A@W output loss、accepted/attempted、合法 state、单文件导入和
   六 shard 结果。不扫描 ridge、clamp、窗口或增益邻域。
5. 本地正向只作为机制证据；候选归档后按当前规则记录官方状态，不用本地分数换算官方分数。

v204 已完成上述固定实现：`eval-v3` 4B Linear-only 六 shard 的 336 个 case 与 v202
逐位相同（overall mean `0.5292658476834804`，changed cases `0`）。受 hard-output gate 保护，
没有部署任何 A@W 增益；该具体实现关闭为 `REJECTED`，官方状态为 `unregistered/NA`，不切换根，
也不扫描其 ridge、clamp、窗口或增益邻域。

在不重试 v204 的前提下，L-AW2/v205 已从同一 v202 根执行 HiF4 第一层 8 元素组标量拟合；
六 shard 336 个 case 仍逐位相同，具体实现关闭为 `REJECTED`，并归档到带 `rejected` 的目录。
随后执行的 **L-AW3 / v206：输出组 × 64-block A@W 拟合**固定输出行组大小为 64，
每个输出组共享一组输入 64-block 标量；从 v202 构建，使用全部 API 输入的全部校准行，
一次批量闭式求解、一次合法五字段重编码和全校准 hard-output gate，不扫描组大小或参数邻域。
v206 已完成：六 shard 336 个 case 与 v202 逐位相同，hard-output gate 未接受任何部署变化，
具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续为 v202。
随后执行的 **L-AW4 / v207：4 元素细粒度组 A@W 拟合**固定每个部署自然坐标 4 元素组一个
标量，仍从 v202 使用全部校准行并做一次合法重编码与 hard-output gate；不扫描组大小或参数邻域。
v207 已完成：六 shard 336 个 case 与 v202 逐位相同，hard-output gate 未接受任何部署变化，
具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW5 / v208：输出行 8-group × 64-block A@W 拟合**，固定输出行组大小为 8，
每个输出组共享输入 64-block 标量；使用一次批量闭式求解、一次合法五字段重编码和 hard-output gate，
不扫描组大小或参数邻域。
v208 已完成：六 shard 336 个 case 与 v202 逐位相同，hard-output gate 未接受任何部署变化，
具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW6 / v209：4 元素组广播 additive A@W 拟合**，固定每个部署自然坐标 4 元素
组一个共享于所有输出行的加性参数；以真实输出域一次闭式求解，随后一次合法五字段重编码和 hard-output
gate，不扫描组大小、幅度或参数邻域。
v209 已完成：六 shard 336 个 case 与 v202 逐位相同，hard-output gate 未接受任何部署变化，
具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW7 / v210：按输出行独立的 4 元素组 additive A@W 拟合**，固定每个输出行的
每个部署自然坐标 4 元素组一个加性参数；以真实输出残差一次对角闭式求解，随后一次合法五字段重编码和
hard-output gate，不扫描组大小、幅度或参数邻域。
v210 已完成：六 shard 336 个 case 与 v202 逐位相同，hard-output gate 未接受任何部署变化，
具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续为 v202。
随后执行的 **L-AW8 / v211：冻结 Q(A) 的输出感知 4-code-group 联合更新**，固定每个输出行
选择一个自然 4 元素组；对冻结最终 `Q(A)` 的实际输出残差解一次 4x4 正规方程，直接将连续步
取整到现有 signed-mantissa 整数码并逐行精确接受。v211 在 shard0 实际翻码但 Linear mean
delta 为 `-0.0171391319`（3/53/0），校准 API 约为父级 3.8 倍，具体实现关闭为 `REJECTED`，
归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW9 / v212：64-block 内单组共享整数码偏移**：冻结最终 `Q(A)`，
每个 64 输入块只选一个自然 4 元素组，在所有输出行共享一个 4 维整数 signed-mantissa 偏移；
用全部校准行的一次输出正规方程确定该偏移，直接写回 mantissa/sign 并用实际产品损失保留或回退。
scale/lv2/lv3 不变，不扫描组、偏移、正则或窗口，不把输出张量写入 state。
v212 已完成：shard0 的 56 个 Linear case 全部逐位等价（`0/0/56`），没有 hard-output 变化，
校准 API 约为父级 5.0 倍，具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为
`unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW10 / v213：输出感知 per-group lv3 hierarchy bit toggle**：冻结最终
`Q(A)`，每个 64 输入块只选一个自然 4 元素组，直接比较该组 `lv3=1↔2` 的合法切换在实际
输出残差上的变化；按自然 block 顺序一次写回，scale/lv2/mantissa 不变，不扫描组或邻域，
不把校准输出写入 state。v213 已完成：shard0 的 56 个 Linear case 为 `0/0/56`，没有
hard-output 变化，校准 API 高于 v202，具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，
官方状态为 `unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW11 / v214：输出感知 per-group lv2 hierarchy bit toggle**：冻结最终
`Q(A)`，每个 64 输入块只选一个自然 8 元素组，直接比较该组 `lv2=1↔2` 的合法切换在实际
输出残差上的变化；按自然 block 顺序一次写回，scale/lv3/mantissa 不变，不扫描组或邻域，
不把校准输出写入 state。v214 已完成：shard0 的 56 个 Linear case 为 `0/0/56`，没有
hard-output 变化，校准 API 高于 v202，具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，
官方状态为 `unregistered/NA`，根继续为 v202。
下一张卡注册为 **L-AW12 / v215：输出感知 E6M2 scale_factor 相邻码更新**：冻结最终 `Q(A)`，
对每个自然 64 输入块和输出行依据当前输出残差方向提出一次相邻 E6M2 code 步进，直接用
实际产品残差接受或回退；lv2/lv3/mantissa 不变，不扫描方向或邻域，不把校准输出写入 state。
v215 已完成：shard0 的 56 个 Linear case 为 `0/56/0`，mean delta `-0.150813`、tail
`-0.263206`，具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为
`unregistered/NA`，根继续为 v202。当前三种静态单字段 hard-output 方向（lv3、lv2、E6M2
scale_factor）均已实测关闭；不重复这些实现，等待新的非重复机制计划。

随后注册 **L-T2 / v216：运行时使用 calibration-compiled activation GPTQ block order**：v202
虽在 state 中编译了 sample-energy order，动态 wrapper 仍按每次 activation 重算 order；该卡
固定使用 state order，删除每次 energy ranking/sort，不新增校准搜索。v216 已完成：shard0
为 `13/43/0`，mean delta `-0.003884`、tail `-0.004127`，runtime 仅有约 `0.3s` 诊断差异，
具体实现关闭为 `REJECTED`，归档目录名含 `rejected`，官方状态为 `unregistered/NA`，根继续
为 v202。

下一项新机制注册为 **A5 / v217：单一固定 reciprocal temperature 1.25**：只在 A1 的真实
attention-output 选择轨启用现有 reciprocal head-scale 参数化，固定 `factor=1.25` 一项，
不扫描其它 factor；Q/K 连续点积保持不变，动态 API 和 V 状态不变。

## 8. 当前执行队列

| 顺序 | 工作 | 当前状态 | 完成后动作 |
|---:|---|---|---|
| 1 | 标准 Linear 组合静态核对 | 已完成本地等价核对；按用户指令不等待官方侧分 | 结果归档，不阻塞新机制 |
| 2 | v195/v191/v190/v192 标准 Linear 官方归因 | 待回传；不等待、不重跑 | 只记录，不调旧实现 |
| 3 | A1 / v199 hard reciprocal 64-block | 已完成；边界可达但六 shard 代理 `−0.0000729515` | 归档并切换目标 |
| 4 | L-T1 / v202 Linear等价降时 | 已完成；官方 `18053/281s`，与 v195 同分快 8s | 已归档并切换为当前根 |
| 5 | A3 / v201 + A4 / v203 | 已完成；官方均 `TIMEOUT`；A3 代理 `−0.0001206117`，A4 代理 `−0.0010964882` | 归档，停止 reciprocal/邻码族 |
| 6 | 最佳 Attention + 当前最快 Linear | 当前根为 v202 Linear + v195 Attention，`18053/281s` | 保持根，进入下一张 Linear 卡 |
| 7 | L-AW1 / v204 部署坐标对齐 64-block A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 8 | L-AW2 / v205 HiF4 8 元素层级组 A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 9 | L-AW3 / v206 输出组 × 64-block A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 10 | L-AW4 / v207 HiF4 4 元素细粒度组 A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 11 | L-AW5 / v208 输出行 8-group × 64-block A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 12 | L-AW6 / v209 4 元素组广播 additive A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 13 | L-AW7 / v210 按输出行独立的 4 元素组 additive A@W 拟合 | 已完成；六 shard 与 v202 逐位相同，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 14 | L-AW8 / v211 冻结 Q(A) 输出感知 4-code-group 联合更新 | 已完成；shard0 `-0.0171391319`（3/53/0），`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 15 | L-AW9 / v212 64-block 内单组共享整数码偏移 | 已注册；固定一次输出正规方程与直接码空间回写 | 完成后按 hard-output 结果归档，不等待官方 |
| 16 | L-AW10 / v213 输出感知 per-group lv3 hierarchy bit toggle | 已完成；shard0 `0/0/56`，无 hard-output 变化，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 17 | L-AW11 / v214 输出感知 per-group lv2 hierarchy bit toggle | 已完成；shard0 `0/0/56`，无 hard-output 变化，`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 18 | L-AW12 / v215 输出感知 E6M2 scale_factor 相邻码更新 | 已完成；shard0 `-0.150813`（0/56/0），`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 19 | L-T2 / v216 运行时使用 calibration-compiled activation GPTQ order | 已完成；shard0 `-0.003884`（13/43/0），`REJECTED` | 保持 v202 根；不重试该具体实现 |
| 20 | A5 / v217 单一固定 reciprocal temperature 1.25 | 已注册；仅启用一个固定 factor，动态 API 不变 | 完成后按 Attention hard-output 结果归档，不等待官方 |

## 9. 归档

    workbench/full_solution/<candidate>/           构建和验证脚本
    artifacts/proxy_v3/full_solution/<candidate>/  4B本地结果
    solutions/<candidate>/                         单文件、result.md、official-result.json

候选失败后在 result.md 写清楚是“无翻码、hard-output无收益、官方负向或官方超时”中的哪一种，
然后进入队列下一项。活动计划只维护上表，不再堆叠长篇门禁和历史实验流水账。
