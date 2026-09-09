# 输出感知舍入边界与 A/W 联合量化计划

> ACTIVE，2026-09-09。当前完整根为 v202 Linear + v195 Attention，官方 `18053/281s`，
> SHA256 `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
> 每张正式卡都从执行时的最高分完整根构建，保留六 API；不修改任何 `solutions/` 归档源码。

## 1. 结论：上一轮为什么没有突破

上一轮 R1、R2、R3 已全部结束，但它们不是三个独立的算法族：

| 卡 | 实际自由度 | 结果 | 说明 |
|---|---|---|---|
| v224 / A-H1R | 全局 Q/K 正交切空间上的最近 hard event | 六 shard `≈+1.2e-6`；官方 `TIMEOUT(>300s)` | 修正了父状态错误，但效应在数值噪声底；与 v223 同成本类，只关闭该实现 |
| v225 / A-H3 | 把同一正交事件拆到 GQA group | `+3.11e-5`；官方 `TIMEOUT(>300s)` | 18/24 group 接受，但收益几乎全部来自 shard5；与 v224 同成本类，只关闭该实现 |
| R3 / A-C76.5 | 用输出残差生成 C76.4 signed-Hadamard 候选 | `NO_EFFECT` | 候选可达且非重复，但六层均未被选择 |

三者都使用 `Q' = QR, K' = KR`（或其局部分块形式），在量化前保持 `QK^T` 不变；区别只在
起点、作用范围和候选生成方式。它们没有改变 HiF4 的可表示值，也没有改变 mantissa 的离散决策规则。
因此继续增加 event 数、GQA group 轮数、Hadamard seed、block size 或旋转优化步数，只是在同一自由度
上加密搜索，不可能解释当前与 `21071` 机制锚点的 3018 分差距。

当前根其实已经包含两类成熟算法：

1. Linear 已有逐通道 SmoothQuant、permutation、Hadamard/CAT、full-H GPTQ 和 A@W/JDRQ；
2. Attention 已有 Q/K reciprocal smooth、K-center、hierarchy/permutation、C76.4 rotation 和最终输出选择。

现有失败集中在两种形态：连续变换被自适应 scale 吸收，或高自由度逐码修改在校准窗口过拟合。
下一轮改用中间自由度：**每层只学习少量共享舍入边界，但边界直接决定大量真实 hard code**。

## 2. 新算法的共同原理

### 2.1 当前舍入规则

在一个已确定的合法 denominator

`D = scale_factor * scale_lv2 * scale_lv3`

下，当前 mantissa code 为

`c = clamp(round(4 * |x| / D), 0, 7)`，解码值为 `sign(x) * c * D / 4`。

标准 `round` 等价于在每个相邻整数码 `m` 与 `m+1` 之间固定使用阈值 `0.5`。新算法把它改为

`c = m                         , frac(u) < tau[m]`

`c = m + 1                     , frac(u) >= tau[m]`

其中 `u = 4*|x|/D`，`tau[m]` 由校准输出误差学习。输出字段仍只包含合法的
`scale_factor/lv2/lv3/sign/mant`，解码器、码宽和接口完全不变。

### 2.2 为什么这是新自由度

- 标量 gain/additive 会在重新计算 E6M2 和层级 scale 后被吸收；舍入阈值直接改变 floor/ceil 选择，
  无法由一个公共 scale 还原。
- v224/v225/R3 改的是输入坐标；本计划改的是量化决策边界。
- AW8 为每个权重元素自由贪心，容易过拟合；本计划每层最多 12 个或 18 个共享参数，修改受结构约束。
- v170 是固定 E6M2 offset，v171 是 moment-matched 单阈值；本计划保持 E6M2 不变，以最终
  A@W/Attention output 残差分别学习每个非零 mantissa 区间的边界。
- `m=0` 的零码插入保持父规则 `tau[0]=0.5`，不重试已经失败的 v220 零值到最小非零码机制。

### 2.3 固定搜索方式

不保存、排序数百万个原始 breakpoint，也不逐 threshold 重跑编码器。固定按 HiF4 自然 block 宽度将
`frac(u)` 分成 64 个桶：`bucket=min(floor(64*frac(u)),63)`；把当前输出残差给出的邻码二阶近似代价用
`scatter_add` 累积到桶，再对 64 个桶做前缀和，直接取代价最小的边界 `tau=b/64`，并把父边界
`tau=0.5` 作为固定候选。相同最小值取最接近 0.5 的位置，再并列取较小阈值。这是一次固定的充分统计
求解，不生成 64 个候选版本，也不执行 64 次 hard forward。

每张卡都先在同一个冻结父状态上一次性求完整张边界表，然后只对整表候选做一次真实 hard
encode/decode 和最终输出复核；不按边界反复执行完整部署路径。

父边界 `tau=0.5` 始终是回退状态。没有 alpha、seed、阈值数量、学习率或轮数扫描。

## 3. 卡一：L-RB1 静态权重的有符号输出残差舍入边界

### 3.1 假设

根的 weight GPTQ 已经解决大量逐列局部误差，但它仍使用通用的最近舍入/局部 4x4 AdaRound。若某层
的 A@W 残差存在稳定偏向，同一 sign、同一 mantissa 区间内的一批权重可能应该更倾向 floor 或 ceil。
共享边界能表达这种系统偏向，同时远低于 AW8 的逐元素自由度。

### 3.2 精确变量

- 冻结当前根选中的 Linear `d/permutation/block transform/CAT`、activation state，以及最终部署
  `weight_params` 的 `scale_factor/lv2/lv3/sign/mant`。令其中 mantissa 整数码为 `c_parent`。
- 只学习 `tau_w[s,m]`：`s∈{-1,+1}`，`m∈{1,2,3,4,5,6}`，每层共 12 个标量。
- `m=0` 和饱和码 7 不改变；不产生零码插入，不改变 scale/hierarchy。
- 使用原始权重经过父 `d/permutation/block transform/CAT` 后的部署坐标值计算 `u=4|w|/D`。
  只有满足 `c_parent=clamp(round(u),0,7)` 的元素进入边界学习；被 AdaRound、GPTQ、FULL64、headroom
  或 JDRQ 改成其他码的元素保持 `c_parent`，不被通用阈值覆盖。由此 `tau=0.5` 按定义恢复最终父码，
  而不是假设最终父码来自普通最近舍入。
- 最终只保存新的静态 `weight_params`，动态 Linear API 不增加计算。

### 3.3 求解

对该层全部 4B calibration 数据，使用父动态激活的真实解码 `X_hat`，teacher 为原 NVFP4 解码后的
`Y = X W^T`，父输出残差定义为

`E = X_hat W_hat^T - Y`。

设共有 `F` 个 calibration case，第 `f` 个 case 的 token 数为 `n_f`、输出维为 `o`，固定使用
case 等权 MSE：`omega_f=1/(F*n_f*o)`。预计算

`H = sum_f omega_f X_hat_f^T X_hat_f`，`G = sum_f omega_f E_f^T X_hat_f`。

任一候选权重变化 `DeltaW` 的精确 A@W 损失变化为

`DeltaL = 2 <G, DeltaW> + tr(DeltaW H DeltaW^T)`。

按固定顺序 `sign=-1,+1`，每个 sign 内 `m=1..6` 处理一遍：

1. 只收集本节资格条件通过且属于该 sign/区间的元素；对每个可能边界计算
   `c_tau`，并令 `DeltaW=(c_tau-c_parent)*sign*D/4`；
2. 用 `2*G_ij*DeltaW_ij + H_jj*DeltaW_ij^2` 按 64 桶累计并做前缀和，确定唯一候选边界；
   12 个边界都使用同一份
   冻结父 `E/G/H` 求解，不接受一个边界后更新另一个边界的训练目标；
3. 合并 12 个边界形成唯一 `DeltaW_table`，只用上面的完整二次型计算一次整表真实 `DeltaL`；
   严格小于 0 才接受整表，否则 12 个边界全部恢复 0.5；
4. 不按 sign、区间或边界重复执行 A@W 完整前向，不做第二轮。

这不是 proxy loss：在固定 `X_hat` 下，二次型与实际 `||X_hat(W_hat+DeltaW)^T-Y||^2` 等价。

### 3.4 实现位置和记录

- 工作目录：`workbench/full_solution/linear-lrb1-residual-rounding/`。
- 在 Linear calibration 最终 `weight_params` 生成之后、返回之前执行；不修改归档候选。
- 记录每层 12 个边界、非 0.5 提案数、受影响元素数、整表 `table_accepted=0/1`、changed mantissa 数、
  精确 A@W `DeltaL`、最终 shard paired delta 和额外 calibration API 时间。
- 记录 eligible/ineligible 数量及 ineligible 的父码来源；`tau=0.5` 必须恢复最终父五字段和输出逐位
  一致，非 0.5 的合成阈值必须能改变预期 eligible mantissa，避免阈值参数实际未被使用。最终 state
  必须通过五字段合法检查。

### 3.5 结束条件

若全六 shard 所有层 `accepted=0` 或最终与根逐位相同，记 `NO_REACHABLE_GAIN`，关闭“静态权重共享
舍入边界”，不改成 per-row/per-block 阈值继续扫。只要形成合法、非等价完整候选，就归档一个版本并交
官方裁决；本地正负只作机制解释，不换算官方分。

## 4. 卡二：A-RB1 Q/K 联合的 softmax 输出舍入边界

### 4.1 假设

现有 Q/K 互逆缩放和旋转只改变送入量化器的坐标，最终仍由固定 `0.5` 舍入。Attention 的误差目标
是 `softmax(Q_hat K_hat^T)V_hat`，同样大小的 Q/K 元素误差对最终输出的影响并不相同。分别学习 Q、K
的少量共享边界，能直接利用这种非线性敏感度，而不再增加矩阵优化步数。

### 4.2 精确变量

- 保留根的 K-center、reciprocal scale、permutation、C76.4 rotation、offset/refine 和 V 路径。
- 学习 `tau_q[m]` 与 `tau_k[m]`，`m∈{1..6}`，每层共 12 个标量；正负号共享，降低过拟合。
- `tau[0]=0.5`，不重试零码插入；scale_factor/lv2/lv3 的候选规则不变。
- 动态 Q/K encoder 在每个 hierarchy 候选内使用学得阈值产生 mantissa，再按现有目标选择合法 hierarchy。
  参数必须从 `q_state/k_state` 经 `hif4_dynamic_quantize_q/k -> _nvfp4_to_hif4 -> _dense_to_hif4`
  传入 `_solve_exact_hierarchy`、offset/refine 和所有当前会重新生成 mantissa 的活动 Q/K 路径；默认
  `None` 等价于全 0.5。任何后续 refinement 不得静默用 `torch.round` 覆盖已学边界。state 只新增两条
  长度 7 的 CPU tensor；在线没有候选循环和矩阵求逆。

### 4.3 求解

以当前部署父 Q/K/V 和最终 Attention 输出残差为唯一冻结起点，同时求 Q、K 各 `m=1..6` 的完整边界表：

1. 对 calibration token 中落入区间 `m` 的元素，计算 floor/ceil 两个合法重建值；
2. 一次普通 backward 得到 `g=J^T(O_parent-O_ref)`。曲率使用一次固定 Hutchinson 对角估计：按
   `case_index` 和输出扁平索引用固定 seed `0` 生成 Rademacher 向量 `r∈{-1,+1}`，再做一次
   `v=J^T r`，定义 `h` 为各 calibration case 的 `v^2` 等权均值。该 `h` 是固定的一探针
   `diag(J^T J)` 估计，不写成精确 Jacobian 对角，也不搜索 seed 或探针数；
3. 用 `2*g*Delta+h*Delta^2` 计算改为邻码的近似代价，按固定 64 桶累计并做前缀和，解析选出
   `tau_role[m]`；12 个边界全部使用同一冻结父 `g/h`，不顺序更新残差；
4. 合并整张 Q/K 表，只重新执行一次真实 Q/K hard encode、causal Attention 和当前 V 路径。
   calibration case 各自先计算输出 MSE，再对 case 等权平均；整表均值严格下降才接受，否则 Q/K
   两张表整体恢复 0.5；
5. 不按 Q/K、mantissa 区间或 threshold 重复完整 Attention 路径，不做第二轮。

最终候选必须用完整部署路径重新生成 Q/K 五字段。不能用线性化代价冒充最终收益。

### 4.4 与历史实现的边界

- 不是 v171：v171 用统计矩匹配阈值，官方 `−348`；本卡由最终 softmax-output 残差解析学习六个非零区间。
- 不是 v190/v198/v199：这些版本学习 reciprocal transform，本卡不新增 transform。
- 不是 v203：v203 联合选择邻接 hierarchy 状态，本卡保持 hierarchy 搜索结构，只改变 mantissa
  floor/ceil 决策。
- 不是 v224/v225：没有 Cayley/rotation hard event，不增加事件槽和完整路径重复评估。

### 4.5 实现位置和结束条件

- 工作目录：`workbench/full_solution/attention-arb1-qk-rounding/`。
- 记录每层 12 个边界、attempted/accepted、Q/K changed mantissa、最终 calibration/holdout output delta、
  shard paired delta 和额外 API 时间。
- 必须分别记录**固定 hierarchy**与**随阈值重选 hierarchy**两种读数下五个字段各自的 changed count、
  解码输出差异和 Attention output 差异。只有重选后五字段全部与父逐位相同，或最终解码 Q/K 与
  Attention 输出全部逐位相同，才记 `ABSORBED_BY_HIERARCHY`；mantissa changed count 单独归零不能
  作为关闭理由，因为 scale_factor/lv2/lv3 仍可能产生有效变化。
- 增加两个直接 reachability control：全 0.5 表必须逐位恢复父；合成非 0.5 表必须在绕过
  calibration gate 时改变预期 Q/K 字段，并由计数证明 threshold 已穿过全部活动编码路径。
- 全六 shard `accepted=0` 或输出逐位相同则关闭“Q/K 共享输出舍入边界”；不继续拆 per-head、per-group、
  per-sign 表。形成合法非等价完整候选后只归档一个版本并交官方裁决。

## 5. 卡三：L-JRB1 A/W 双量化器的一轮联合边界坐标下降

### 5.1 为什么不是 L-RB1 的参数微调

L-RB1 只改变静态 W，激活误差保持不变；外部 `21071` 证据描述的是 A@W 拟合，可能需要同时改变
`Q(A)` 与 `Q(W)`，而不是在固定激活上再拟合 W。L-JRB1 增加的是第二个被部署的量化器自由度，目标是
联合乘积，不是给 L-RB1 增加更多 threshold。

### 5.2 变量和固定算法

- 始终从执行 L-JRB1 时已经确认的最高分完整根开始。若 L-RB1、A-RB1、v225 或其他候选此前已官方
  RETAINED，则使用更新后的完整根；否则才仍是 v202。不得因为 L-RB1 未晋级而忽略其他已晋级结果。
- activation 使用 sign 共享的 `tau_x[m]`，`m=1..6`；weight 使用 L-RB1 的
  `tau_w[s,m]`，共 18 个边界/层。
- 只做一次固定次序的 block coordinate descent：
  `在冻结父残差上一次性求 activation 6 个边界 -> 整表重新生成一次全部 X_hat/H -> 在新的固定
  X_hat 上一次性求 weight 12 个边界`。
- Linear 输出对 activation 是线性的，不使用 Hutchinson：对父 `Y_hat=X_hat W_hat^T`，精确使用
  `g_x=E W_hat` 与 `h_x[j]=sum_i W_hat[i,j]^2` 计算每个 activation 邻码的
  `2*g_x*DeltaX+h_x*DeltaX^2`。六个边界使用同一冻结父残差求解，合并后只重新量化一次 calibration
  activation，并用 case 等权的真实
  `||Q(A)Q(W)^T-AW^T||^2` 接受整张 activation 表或整体回退；不逐边界重跑动态量化。
- weight 阶段使用 L-RB1 的精确二次型。完成 weight 后不回到 activation，不增加第二轮。
- activation threshold 必须贯穿 v202 block-order wrapper、`_activation_gptq_quantize`、
  `_dense_to_hif4`、`_solve_exact_hierarchy` 及所有活动 refinement；默认 0.5 时保持父路径。动态 state
  只增加长度 7 的 activation threshold tensor；静态 weight 仍只返回合法五字段。

### 5.3 关键 control

- `tau_x=tau_w=0.5` 时六 API 与最终父五字段及输出逐位一致；另外用合成非 0.5 表证明 activation
  threshold 不会被 v202 wrapper、GPTQ 或后续 refinement 忽略/覆盖。
- activation-only、weight-only、joint 三个读数只用于归因，正式候选只有 joint 一个。
- 必须分别记录 activation 和 weight 的 changed mantissa；只有一侧变化不能写成“联合机制已验证”。
- 若 activation 分支因自适应 hierarchy 完全吸收，记清零翻码原因并关闭本卡，不拆粒度重试。

### 5.4 结束条件

形成合法非等价完整候选即归档一个版本并交官方；若两侧不能同时可达或联合输出回到父，记
`NO_JOINT_REACHABILITY`。不扫描交替轮数、符号拆分、行簇、block 专属表或模型/role 路由。

## 6. 执行顺序

| 顺序 | 工作 | 当前状态 | 完成后动作 |
|---:|---|---|---|
| 0 | 收口 v223/v224/v225/R3 事实并归档旧计划 | DONE | v224/v225 官方 TIMEOUT 已登记；进入 L-RB1 |
| 1 | L-RB1 静态权重有符号输出舍入边界 | DONE / `REJECTED` | 六 shard 全部负向（等权均值 `-4.10e-4`），机制可达但校准增益在噪声底；归档 v226，未提交官方；进入 A-RB1 |
| 2 | A-RB1 Q/K 联合 softmax 输出舍入边界 | DONE / `NO_EFFECT` | 六层全部被接受门回退、输出与根逐位相同（预测改进但 causal/non-causal 两条轨同时恶化 1.07×–2.97×）；不占版本号、未提交官方；进入 L-JRB1 |
| 3 | L-JRB1 A/W 双量化器联合边界 | READY | 归档一个代表版本或关闭该机制；三卡结束后重新分析 |

三张卡互不等待官方回传：某卡候选归档并提交后即可实现下一卡。官方正向且 `<300s` 才更新根；
后续卡始终从当时已经确认的最高分完整根构建，不建立 Linear 或 Attention 侧父。

## 7. 每张卡实际怎么执行

每张卡只走以下四步，不添加额外门禁：

1. **实现**：从完整根复制到对应 workbench；加入一个机制和固定配置，同时保留父回退。
2. **代码确认**：六 API 可独立导入，`tau=0.5` 父 control、五字段合法、finite、changed/accepted 计数。
3. **一次 4B 诊断**：先跑目标侧 shard0 排除接口/不可达错误，再跑目标侧固定六 shard一次；不跑 OOD、
   0.5B、GPT-2/opt、fresh timing，不因某个 split 或本地小负值增加调参版本。
4. **归档与官方**：非等价代表候选写入新的 `solutions/<version>/`，记录源码 SHA、配置、本地 JSON、
   API 时间和官方 `unregistered/NA`；官方回传后更新结果。无效机制只保存 workbench/result，不占版本号。

本地 paired delta 只回答机制有没有改动 hard output、错误集中在哪里；不换算官方分数，不设置本地时间门。
官方 `score` 和 `<300s` 是唯一晋级依据。同 SHA 或逐位等价候选不重复提交。

## 8. 明确停止的方向

- 不再做 rotation、Cayley event、Hadamard seed、GQA group 次数或 block size 邻域。
- 不再做 Q/K reciprocal scale 的 factor/step/window/矩阵秩邻域；只有源码能够证明为新的量化边界机制时
  才允许进入本计划。
- 不重试 Linear scalar/group gain、additive、零码插入、无结构逐码贪心、rank-8 残差基或 hierarchy
  邻码步进。
- 不为回收 v192 的 `+22` 单独做降时工程；其收益量级不足以解释当前缺口。
- 不修改 v222-v225 或任何已归档源码，不把本地微增益写成重大突破。

## 9. 三卡之后如何判断下一步

- 任一卡官方正向：切换完整根；只在同一算法的必要 bug 修复后复测，不扫邻域。其他卡继续从新根执行。
- 三卡均可达但官方无收益：说明“低维共享舍入边界”也不足以复现外部机制。下一计划不得继续拆
  per-head/per-block 表，而应要求绑定 `21071` 的源码/配置，或寻找新的合法表示自由度。
- 卡在本地逐位不变：先确认是自适应 hierarchy 吸收还是求解器没有尝试；修复实现错误只改 workbench，
  不改归档。确认算法确实执行后仍不变即关闭。
- 官方 TIMEOUT：只关闭该计算实现；但不通过减少边界数、token、fold 或轮数重试同一候选。

## 10. 算法依据

- SmoothQuant 说明等价通道缩放可以在 activation 与 weight 间迁移量化难度；当前根已充分使用这类
  变换，所以本计划不再把普通 SmoothQuant 当新方向：<https://arxiv.org/abs/2211.10438>。
- AWQ 说明激活统计可以指导少量、结构化的量化决策；本计划进一步用最终 A@W 残差而不是单纯幅度
  来学习共享舍入边界：<https://arxiv.org/abs/2306.00978>。
- SpinQuant 说明旋转的选择会显著影响量化效果；本仓库已经通过 v224/v225/R3 对该族完成实际裁决，
  因而把搜索自由度转向离散边界：<https://arxiv.org/abs/2405.16406>。
- AQLM 的关键启发是应在最终重建目标下联合选择离散码；本计划只借用“低维结构化离散优化”原则，
  不引入其加性码本，因为官方 HiF4 五字段不允许改变表示格式：<https://arxiv.org/abs/2401.06118>。
