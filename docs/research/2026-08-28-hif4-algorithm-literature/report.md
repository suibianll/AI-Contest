# HiF4 精度优化：算法理论、文献证据与可执行路线

> 调研日期：2026-08-28  
> 代码基线：根目录 `solution.py`，C38  
> 当前本地结果：Linear `0.5695`，causal Attention `0.4497`，CPU algorithm-stage 约 `99s`  
> 最新合规官方锚点：v025 / C21-C，`14437 / 166.6s`  
> 最终评测硬约束：运行时间严格小于 `300s`

## 1. 结论

现有实现还没有达到当前框架的算法上限。最关键的原因不是搜索候选太少，而是三个结构性缺口：

1. 当前所谓 `FULL64` 只取完整协方差矩阵的 **64×64 对角块**，每个 64 维块独立优化；它没有使用不同 64 维块之间的 `H_bj`，因此不等价于 QuIP/QuIP# 的 LDLQ 或 BlockLDLQ。
2. 当前量化时“用于决定 E2M1 编码的尺度”和“最终保存、用于反量化的 E6M2 尺度”仍基本绑定。SOAR、ScaleSearch、ScaleSweep 和 FOCUS 都表明，FP4 的主要剩余误差来自离散尺度与 E2M1 断层；临时放松编码尺度、再投影回完全相同的最终格式，可能显著改善重构误差。
3. 当前旋转以固定 4/8/16 Hadamard 和启发式排列为主。固定 H64 的失败只说明“整层随机大旋转”不稳定，并不能否定与 HiF4 的 4/8/64 层级严格对齐、由合法局部目标优化的结构化旋转。

建议优先级：

| 优先级 | 方向 | 预期潜力 | 合规性 | 主要代价 |
|---|---|---:|---|---|
| P0 | 跨 64 块的 BlockHiF4-LDLQ | 高 | 绿 | 校准计算量、内存 |
| P0 | 编码/解码解耦的层级尺度求解 | 高 | 绿 | 离散候选和交替优化 |
| P0 | Activation-only ScaleSearch / 临时编码尺度 | 中高 | 绿 | 在线路径需向量化 |
| P1 | 4/8/64 层级对齐的结构化旋转 | 中高 | 绿/黄，取决于目标 | 状态与优化稳定性 |
| P1 | 有下界的 anytime 层级离散求解 | 中 | 绿 | 求解器复杂度 |
| P1 | Attention 的精确 Q/K 平衡与 K 平移 | 中，主要改善 Attention | 绿 | GQA 共享、数值稳定性 |

若目标是 26000 分，不能继续把主要希望放在 offset、覆盖率和固定 seed 微调上。当前本地 Linear `0.5695` 只表示相对标准量化基线减少了约 56.95% 的 MSE；它与官方总分不存在已验证的稳定线性映射。历史外推只能说明 26000 需要结构级跃迁，不能证明某个本地阈值必然对应 26000。

## 2. 赛事合规边界

官方第一原则是：**不得以任何形式计算出 `A @ W`，再利用 `A @ W` 拟合、选择或反推出 `Q(A)`。** 下述路线必须服从这一原则，不能构造代数等价的绕行。

### 2.1 绿色：可以作为实现基础

- 用激活统计量 `H_A = E[A^T A]` 优化 `Q(W)`。被拟合的是权重，不是 `Q(A)`。
- 仅用 `A` 自身的数值分布、量化 MSE、鲁棒 MSE、分位数或饱和率选择 `Q(A)` 的尺度、中心和合法变换。
- 对 Linear 做严格等价坐标变换，如 `A' = A R`、`W' = R^{-1}W`，前提是激活侧参数不是由 Linear 输出或其代数等价物反推。
- Attention 专用的 Q/K 精确不变量、K 公共平移，以及赛事允许的真实 Attention 选择器；Linear 调用图不得访问 Attention 输出评分器。

### 2.2 黄色：必须获得明确规则确认

- 当前 `_ACTIVATION_QUADRATIC=True` 使用 `W^T W` 指导 `Q(A)`。它没有显式计算 `A@W`，但其目标与输出误差二次型紧密相关，可能被解释为用权重构造输出敏感代理来拟合 `Q(A)`。在获得官方书面确认前，应建立纯 activation-only 对照，后续核心算法不能依赖该机制。
- 同时由激活和权重统计训练激活侧旋转、尺度或状态，即使没有调用矩阵乘法，也要审查其目标是否实质上在拟合 Linear 输出。

### 2.3 红色：禁止

- 计算、缓存、局部计算、分块计算或重构 `A@W`，再用结果选择 `Q(A)`。
- 展开输出误差公式、利用交叉项或等价恒等式绕过显式 `A@W`，再拟合 `Q(A)`。
- 直接照搬 OmniQuant、SpinQuant、FlatQuant 等方法中的 task/output reconstruction loss 来训练 Linear 激活量化器。
- 通过官方隐藏集或反复官方分数反馈调参。

## 3. 当前实现的理论审计

当前代码已经具备 SmoothQuant 型对角缩放、排列、固定 signed Hadamard 4/8/16、层级尺度搜索、权重 Hessian 二次型、64 维精修、动态激活尺度搜索，以及 Attention 的 Smooth-QK、K midrange center、headwise permutation 和真实输出门控。

最大缺口是：`_full64_hessian_blocks` 只抽取

```text
H_00, H_11, ..., H_BB
```

没有向每个 64 块求解器提供 `H_01, H_02, ..., H_bj`。设权重量化误差 `E = Q(W)-W`，激活协方差 `H=E[A^T A]`，合法权重代理目标为：

```text
L_W = tr(E H E^T)
    = Σ_b tr(E_b H_bb E_b^T)
    + 2 Σ_{b<j} tr(E_b H_bj E_j^T)
```

当前求解器优化第一项，丢掉第二项。因此一个块的局部改进可能被跨块交叉项抵消。这正是 [QuIP](https://arxiv.org/abs/2307.13304) 的 LDLQ 和 [QuIP#](https://arxiv.org/abs/2402.04396) 的 BlockLDLQ 要处理的问题。

| 模块 | 当前方式 | 未覆盖的理论空间 |
|---|---|---|
| 权重尺度 | stored scale 邻近 offset + beam | 编码/解码尺度解耦；舍入边界候选 |
| 层级尺度 | 4/8/64 局部枚举与翻转 | 带下界的全局/anytime 搜索 |
| 旋转 | 固定小块 Hadamard、排列 | 层级对齐 butterfly/Givens；outlier-aware rotation |
| 激活 | 动态 offset 和权重 Gram 引导 | 纯 activation-only 尺度解耦和鲁棒目标 |
| Attention | Smooth-QK、midrange K center | 一般可逆 Q/K 平衡；量化感知 K 平移 |

## 4. P0：BlockHiF4-LDLQ

### 4.1 理论

QuIP 的顺序自适应舍入形式为：

```text
W_hat_k = Q(W_k + (W_<k - W_hat_<k) a_k)
```

反馈系数来自 Hessian 的 LDL 分解。QuIP# 将单列扩展为 `g` 维向量块。它不是简单扩大局部块，而是在量化当前块前，把先前块的误差按 `H` 的条件结构反馈给当前目标。

### 4.2 实现算法

以输入通道每 64 维为一个 HiF4 原子块：

1. 从校准激活计算完整 `H=A^T A/N`，加阻尼 `H_d=H+λ mean(diag(H))I`。
2. 首版把相邻两个 64 块组成 128 维 superblock，保留完整四个 `64×64` 子块。
3. 对 superblock 做 block LDL 分解，得到当前块对后续块的反馈矩阵。
4. 顺序处理 64 块：根据已量化块误差计算校正目标 `W_tilde_b`，调用现有 HiF4 64 维求解器，再传播实际误差。
5. 用完整 superblock 二次型评分，不再用各 `H_bb` loss 之和代替。
6. 128 维有效后扩到 256；最终可用 block-banded、按 `||H_bj||_F` 选 top-k 邻接块或完整 block LDL。

```python
H = damp(covariance(A))
for S in partition_channels(superblock_size=128):
    Hs = H[S, S]
    feedback = block_ldl_feedback(Hs, block_size=64)
    errors = []
    for b in range(num_blocks_in_S):
        target = W[S_b] + feedback_from(errors, feedback, b)
        q_b = solve_hif4_64(target, Hs[b, b],
                            scale_solver="decoupled")
        errors.append(target - dequantize(q_b))
```

实现时必须用 128 维合成例比较独立块解、顺序反馈解和完整二次型。只有 `tr(EHE^T)` 下降才算算法成立。分解每层一次并由所有输出行共享；研发阶段可完整计算，最终通过缓存、分块批处理和矩阵化传播压缩到 300 秒内。

## 5. P0：编码/解码解耦的层级尺度

E2M1 的可表示正幅值在 4 和 6 之间有大断层。[Four Over Six](https://arxiv.org/abs/2512.02010) 让块选择更适合 4 或 6 的缩放方式；[ScaleSearch](https://arxiv.org/abs/2605.12464) 和 [ScaleSweep](https://arxiv.org/abs/2606.07618) 表明最优 FP8 block scale 往往不是标准 amax scale。

[SOAR](https://arxiv.org/abs/2605.12245) 与 [FOCUS](https://arxiv.org/abs/2608.01847) 更重要的启示是：**决定临时编码的尺度不必等于最终保存的解码尺度。** 临时变量量化后丢弃，最终 HiF4 五字段和推理公式不变。

### 5.1 Decoupled Hierarchical Scale Search（DHSS）

最终重构保持：

```text
q_i = s_d · g2_i · g3_i · m_i
```

仅在决定 E2M1 值 `m_i` 时使用临时编码尺度：

```text
s_q,i = s_d · c_k
```

`c_k` 可按 4 或 8 元子组共享，完成后不保存。固定离散代码和层级后，记 `t` 为不含全局尺度的重构向量。普通 MSE 的连续最优尺度为：

```text
s* = <t,w> / <t,t>
```

Hessian 二次型的最优尺度为：

```text
s* = (t^T H w) / (t^T H t)
```

将 `s*` 投影到最近 E6M2 值及其相邻合法码，再用真实离散重构评分。

### 5.2 候选与交替优化

1. 由每个值跨越 E2M1 中点时的舍入边界反推 `s_q` 候选。
2. 加入标准 amax、当前 winner、闭式 `s*`、ScaleSearch 邻域和边界相邻 E6M2 值。
3. 去重并批量量化、评分。
4. 交替执行“代码分配 → `s_d` 闭式更新 → `g2/g3` 更新 → 临时 `c_k` 更新”2–3 轮。
5. 每轮只接受真实离散目标下降，并始终保留 incumbent；提前停止也返回合法五字段结果。

权重侧用完整 Hessian 目标；激活侧先只用 activation-only MSE/鲁棒 MSE，不把 `W^T W` 版本作为唯一实现。

## 6. P1：层级对齐结构化旋转

[QuaRot](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html) 证明严格等价 Hadamard 旋转可降低 outlier；[SpinQuant](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html) 说明学习旋转可明显优于随机旋转；[DuQuant](https://arxiv.org/abs/2406.01721) 和 [DuQuant++](https://arxiv.org/abs/2604.17789) 说明旋转应针对 outlier，并与 microscaling block 对齐。

推荐：

- 基本单元为 64 维，运算结构嵌套在 4/8/64 组内。
- 使用 6 层 butterfly Givens，或 `signed-Hadamard + permutation + diagonal scaling`，复杂度 `O(C log64)`。
- 保存角度/符号/排列，不保存每组一个稠密 64×64 矩阵。
- Identity 和当前 4/8/16 winner 始终保留。

合规目标只能是激活侧分位范围、峰均比、量化 MSE、QSUR、组间尺度离散度，以及权重侧 `tr((Q(W')-W')H'(Q(W')-W')^T)`。可复用 SpinQuant 的 Cayley/Stiefel 优化几何，但不能复制其 Linear task/output loss。

## 7. P1：有下界的 anytime 层级求解

当前 beam、top-ratio 和固定 sweep 无法说明距离最优还有多远。可将每个 64 元块的 `lv2/lv3`、stored scale 和 mantissa 建模为离散搜索：

1. 节点固定部分 8 元组层级状态。
2. 对未固定部分用连续尺度和未量化值构造乐观下界。
3. 按下界 best-first 展开，维护合法 incumbent。
4. 时间到返回 incumbent；若队列最小下界不优于 incumbent，则可证明该块已最优。
5. 仅对完整二次型贡献最大的块投入更多预算，预算由精度—耗时 Pareto 调度，不设僵硬的最小增益门。

该求解器还能测量当前启发式的最优差距，避免重复探索已饱和模块。

## 8. Attention 专项路线

### 8.1 量化感知 K 公共平移

对每个 head，把所有 token 的 key 减去相同通道向量 `c`：

```text
K' = K - 1 c^T
Q K'^T = Q K^T - (Q c) 1^T
```

每个 query 行只减去一个常数，softmax 完全不变，无需补偿状态。当前只有 midrange center；下一步应交替优化中心和真实 HiF4 量化：

1. 初始化 mean、median、midrange、trimmed mean。
2. 对每个候选执行真实层级 HiF4 量化。
3. 固定量化码后更新中心，使 `K-c` 到重构值的 MSE 最小。
4. 迭代 2–3 次，由真实 Attention A1 门控最终候选。

[SageAttention2](https://arxiv.org/abs/2411.10958) 提供了 smoothing 依据；需要额外修正项的 Q-centering 不能在当前接口直接照搬，K 公共平移则精确兼容。

### 8.2 一般可逆 Q/K 平衡

对任意可逆矩阵 `M`：

```text
Q' = Q M
K' = K M^{-T}
Q' K'^T = Q K^T
```

设带阻尼协方差 `A=Cov(Q)`、`B=Cov(K)`，可构造平衡两侧二阶尺度的矩阵：

```text
C = A^(1/2) B A^(1/2)
M = A^(-1/2) C^(1/4)
```

先用每 head 的 4×4 或 8×8 分块矩阵；GQA 下用关联 Q heads 的协方差平均，K 侧保持共享。做特征值截断、阻尼和条件数限制，保留 Identity。变换后分别运行 Q/K 的 DHSS，再用真实 causal 与 safety Attention 指标选择。

### 8.3 Attention 误差代理

一阶扰动可用于分配搜索预算和诊断，但不替代最终 A1 门控：

```text
δS = (δQ K^T + Q δK^T) / sqrt(d)
δO ≈ J_softmax(S)[δS] V + P δV
```

不能直接迁移的论文部分包括：与 HiF4 输出布局冲突的 K/V 量化轴、mixed precision sink、P 的量化、自定义 kernel，以及需要额外 attention 修正项的 Q-centering。

## 9. 联合算法蓝图

```text
Linear 权重：
  合法等价变换
    → 完整/带状 H
    → BlockHiF4-LDLQ
    → DHSS 64维块量化器
    → 五字段 HiF4

Linear 激活：
  同一等价变换
    → activation-only DHSS
    → 可选且经规则确认的二次型候选
    → 五字段 HiF4

Attention：
  K 公共平移 + Q/K 可逆平衡
    → Q/K/V 各自 DHSS
    → 真实 causal/safety Attention 门控
    → 五字段 HiF4
```

三个求解层共享 E2M1 boundary candidate 生成、E6M2 投影、层级状态枚举、批量真实重构和 Pareto 记录器。

## 10. 可执行实验顺序

研发阶段时间充裕，不给单个实验设置僵硬时限；最终候选必须严格小于 300 秒。

### A. 建立正确诊断

1. 固定 C38 SHA、输入 offsets 和评测环境。
2. 每个 Linear case 记录 activation-only MSE、weight MSE、完整 Hessian loss、仅对角 64-block loss、部署分数和耗时。
3. 对现有 `FULL64` 结果计算完整 `tr(EHE^T)`，量化丢弃跨块项的目标偏差。
4. 添加规则审计：Linear 激活选择调用图不得访问 output scorer；`_ACTIVATION_QUADRATIC` 保留黄色标记和纯 activation-only 回退。

### B. P0 单变量实验

1. **B1 / DHSS-W**：只替换权重 64 块尺度求解器。
2. **B2 / BlockLDLQ-128**：只加入相邻两块跨块反馈，块内仍用 C38 solver。
3. **B3**：联合 DHSS-W + BlockLDLQ-128。
4. **B4 / DHSS-A**：纯 activation-only 临时编码尺度，禁用 `W^T W` gate 做干净对照。
5. 保存完整精度—耗时点，保留非支配 Pareto 候选，不因单个分项微退化立即删除算法。

### C. 扩大结构自由度

1. BlockLDLQ 从 128 扩到 256，比较 adjacent、block-banded 和 top-k Hessian coupling。
2. 引入 4/8/64 butterfly rotation，先优化 operand-local flatness，再联合权重 Hessian loss。
3. 比较 DHSS 临时编码粒度 8 和 4；最终格式不变。
4. 对重要块启用 anytime branch-and-bound，测量启发式最优差距。

### D. Attention

1. **A-KCenter**：scale-aware K center 交替优化。
2. **A-QK4**：每 head 4×4 Q/K 平衡。
3. **A-QK8**：仅在 QK4 有稳定正信号后扩大。
4. Q/K/V 分别接入 DHSS，建立 causal/safety Attention Pareto 前沿。

### E. 最终压时与提交

1. 只对 Pareto 前沿候选做向量化、缓存 Hessian 分解、合并重构和减少 Python 循环。
2. 以真实端到端 `<300s` 为唯一硬时间判据。
3. 校准集用于选择；保留集只在冻结参数后使用一次，不能根据保留集或官方返回反复调参。
4. 每次提交绑定 SHA、官方分数和运行时间。

## 11. 26000 分的理论判断

不能从当前数据证明 Linear 达到某个本地分数就必然达到 26000，也不能把 Attention 和 Linear 提升简单相加。one-sided oracle 只说明误差预算存在，不是可实现上限。

若 26000 可达，最可信的路径需要四类互补收益：

1. **表示层**：DHSS/Four-over-Six 减少 E2M1 断层和离散尺度误差。
2. **优化层**：BlockLDLQ 使用当前被忽略的跨 64 块协方差。
3. **变换层**：层级对齐旋转降低局部组 outlier。
4. **双侧**：严格合规地同时改善 `Q(W)` 和 activation-only `Q(A)`；Attention 另走精确不变量路线。

单独调大 FULL64 覆盖率、增加固定 offsets 或随机 H64 seeds，理论上都不足以承担从 14437 到 26000 的主增量。先验证 B1/B2 能否产生新的增长斜率；若组合后仍只有千分位提升，应重新评估五字段 HiF4 表示的可达上限，而不是继续堆叠同类局部搜索。

## 12. 文献证据

| 日期 | 工作 | 可迁移结论 | 证据等级 |
|---|---|---|---|
| 2022-10 | [GPTQ](https://arxiv.org/abs/2210.17323) | Hessian-aware adaptive rounding | 高，ICLR 2023 |
| 2022-11 | [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) | 激活/权重等价对角缩放 | 高，ICML 2023 |
| 2023-06 | [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 激活统计用于权重显著性 | 高，MLSys 2024 |
| 2023-07 | [QuIP](https://arxiv.org/abs/2307.13304) | LDLQ 跨坐标反馈 | 中高，理论论文 |
| 2024-02 | [QuIP#](https://arxiv.org/abs/2402.04396) | BlockLDLQ 与 incoherence | 高，ICML 2024 |
| 2024-03 | [QuaRot](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html) | 精确旋转不变量 | 高，NeurIPS 2024 |
| 2024-05 | [SpinQuant](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html) | 学习旋转；目标需替换 | 高，ICLR 2025 |
| 2024-06 | [DuQuant](https://arxiv.org/abs/2406.01721) | outlier-aware rotation | 高，NeurIPS 2024 Oral |
| 2024-10 | [FlatQuant](https://arxiv.org/abs/2410.09426) | 快速可学习仿射变换 | 高，ICML 2025；目标不可直搬 |
| 2024-11 | [SageAttention2](https://arxiv.org/abs/2411.10958) | Attention smoothing | 高，ICML 2025 |
| 2025-01 | [RotateKV](https://arxiv.org/abs/2501.16383) | KV outlier-aware rotation | 中高，IJCAI 2025 |
| 2025-12 | [Four Over Six](https://arxiv.org/abs/2512.02010) | 针对 E2M1 4→6 断层缩放 | 中，预印本 |
| 2026-04 | [DuQuant++](https://arxiv.org/abs/2604.17789) | 旋转与 microscale 对齐 | 中，预印本 |
| 2026-05 | [SOAR](https://arxiv.org/abs/2605.12245) | 联合尺度、编码/解码解耦 | 中，预印本 |
| 2026-05 | [ScaleSearch](https://arxiv.org/abs/2605.12464) | 搜索邻近 FP8 scale bit patterns | 中，预印本 |
| 2026-05 | [ScaleSweep](https://arxiv.org/abs/2606.07618) | 离散尺度候选范围 | 中，预印本 |
| 2026-08 | [FOCUS](https://arxiv.org/abs/2608.01847) | coupled-relaxation、双粒度 | 中低，最新预印本 |

完整 20 篇论文清单见 `papers.csv`；20/20 PDF 已下载并完成文本检索，下载状态见 `paper_downloads.csv`。为避免给工程增加约 147 MB 二进制文件，校验后的本地 PDF/文本缓存不纳入资料包；日志中的 `saved_path_at_download` 是下载时的审计路径，论文可由 `download_url` 重取。

工程参考：[QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)、[SpinQuant](https://github.com/facebookresearch/SpinQuant)、[Four Over Six](https://github.com/mit-han-lab/fouroversix)、[SOAR](https://github.com/steven-bao1/SOAR)、[DuQuant++](https://github.com/Hsu1023/DuQuant++)、[NVIDIA NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)。

[NVIDIA Nemotron 3 Ultra 量化实践](https://developer.nvidia.com/blog/creating-the-nvidia-nemotron-3-ultra-nvfp4-checkpoint-with-nvidia-model-optimizer/) 显示 Four-over-Six 可降低中位重构 MSE，但也明确观察到更低局部 MSE 不总是等于更高任务精度。

## 13. 后续观察信号与资料完整性

需要继续观察：

1. FOCUS 与 SOAR 是否补齐稳定实现和消融。
2. ScaleSearch、ScaleSweep、Four Over Six 的同行评审和跨格式复现。
3. 官方对“使用 `W^T W` 拟合 `Q(A)`”的书面裁定。
4. C38 官方分数和端到端耗时。
5. B1 与 B2 是否产生新的增长斜率。

资料覆盖 arXiv、ICLR/OpenReview、ICML/PMLR、NeurIPS、MLSys、IJCAI；工程来源仅采用论文作者仓库、NVIDIA 官方文档/博客和框架官方文档。收集论文 20 篇，PDF 下载成功 20、失败 0；本地缓存完成文本检索后清理。元数据见 `papers.csv`、`technical_sources.csv` 和 `paper_downloads.csv`。

本报告给出的是研究优先级和实现算法，不是官方分数承诺。所有候选必须重新经过赛事规则审计、五字段格式检查、真实部署路径评测和 `<300s` 端到端时间验证。
