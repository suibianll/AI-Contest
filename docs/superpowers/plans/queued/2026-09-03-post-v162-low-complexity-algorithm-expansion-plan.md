# v162 侧向计划完成后的低复杂度算法扩展计划

> 状态：**QUEUED / INACTIVE**
>
> 创建：2026-09-03
>
> 激活条件：当前唯一活动计划
> [`2026-09-03-v162-official-side-isolation-optimization-plan.md`](../2026-09-03-v162-official-side-isolation-optimization-plan.md)
> 完成并移入 `docs/superpowers/archive/plans/`；随后在同一提交中把本文件移到
> `docs/superpowers/plans/` 根目录并更新计划入口。在此之前，本文件不授权实现、评测、分配
> 版本号或官方提交。

## 1. 触发背景

v165（standard Linear + v161 Attention，SHA
`033E85D5DAF1A820BACDB14F9E35183C485E8DD489D118899A1AE3CB491D8C1D`）官方
`timeout（>300s，无分数）`。同侧对照 v164 为 `13945 / 204s`，所以 Cross-Gram64 per-call
动态精化的官方增量成本下界约为 `>96s`。timeout 只否定该实现的时间可行性，不提供精度结论。

当前活动计划已经包含一次 rank-2 Gram 残差码本复杂度重构和 Linear rank-1 候选，本计划不与
它们并行，也不重复它们。本计划只在当前计划完整结束后接管尚未执行的新算法族。

## 2. 共同实验协议

1. 继续使用 v162 `1001 / 146s` 作为双标准官方零点；Linear 候选配 standard Attention，
   Attention 候选配 standard Linear。
2. 每个版本只实现一个固定数学机制；参数、rank、block 和聚合规则在读取 local holdout 与
   官方结果前固定。
3. 本地只硬检查单文件导入、六 API/state 合法、有限输出、机制可达和非目标侧逐位一致；本地
   精度分布只记录风险，不设置人为正向分数门槛。
4. 每个机制只提交一次官方。官方负向后不扫描 threshold、seed、rank、block、layer 或 role
   路由；timeout 只允许输出等价或同目标的纯复杂度重构一次。
5. 官方分数按当前侧向公式计算绝对贡献、相对 v160 侧增量和侧贡献提升率；官方时间只记录
   实测，不从本地换算。

## 3. Attention：只允许低在线复杂度

v165 之后，Attention 新机制不得在每次动态 API 中执行完整 `64×64` Gram contraction、多轮
coordinate sweep、Python 候选循环或随序列长度增长的候选搜索。首选校准期求解、在线单次
逐元素或逐 head 操作。

### A1：解析 logits 增益校正

目标是补偿量化 logits 的系统性收缩或膨胀。校准阶段用完整与量化 logits 解析计算每个 head
的最小二乘系数。两个 calibration folds 分别计算后，在 log 域取中位数，并固定向 1 收缩一半：

```text
gamma_f,h = clamp(Cov(L_h, Lq_h) / (Var(Lq_h) + eps), 0.5, 2.0)
gamma_h   = exp(0.5 * median_f(log(gamma_f,h)))
```

不搜索 clamp 或收缩系数。动态阶段固定在 Q/K 两侧各乘 `sqrt(gamma_h)`，只有一次广播乘法。
该机制与此前保持连续 QK 不变的 reciprocal scale 不同：它允许可控地校正量化后 logits。

复杂度：校准复用 attention scorer 的 logits，额外 `O(H T^2)` 归约；动态 `O(TD)`；状态
`O(H)`。

### A2：V 质心残差补偿

从 calibration folds 估计每个 KV head 的系统性 V 编码偏差：

```text
b_f,h = mean(V_h - dequant(Q(V_h)))
b_h   = 0.5 * median_f(b_f,h)
```

固定 `0.5` 收缩且不搜索。动态 V 编码前只做一次广播加法，目标以 attention output residual
解释，最终仍输出标准合法 HiF4 五字段。

复杂度：校准 `O(FTD)`；动态 `O(TD)`；状态 `O(H_k d_h)`。

### A3：K/V 非对称层级目标

固定格式不变，但不再让 K 与 V 共用同一种层级损失：K 使用 logit/channel sensitivity，V 使用
token attention mass 与输出残差的解析权重。候选集合固定为 parent 与一个解析规则，不做
长度、layer 或模型路由。

复杂度：校准 `O(FTD + FHT^2)`；动态维持一次标准 hierarchy encode 的 `O(TD)`，不得增加
多候选 sweep。

Attention 执行顺序固定为 `A1 -> A2 -> A3`；一个机制官方负向不阻止下一个不同机制，但不在
同族内调参。

## 4. Linear：优先改变离散编码几何

### L1：WUSH 与 CAT-64 公式等价审计

先只做公式和逐块数值审计，对比 WUSH 的数据感知非正交闭式变换与现有 CAT-64。若两者在
归一化、矩阵幂和变换顺序后数学等价或逐位等价，则记录 `DUPLICATE / NO VERSION`，不提交。
仅当 WUSH 产生不同的固定变换和合法部署输出时，才构造一个 WUSH-HiF4 侧向候选；不扫描
Hadamard seed 或 transform strength。

复杂度：每个 64-block 校准 `O(B^3)`；若替换现有 CAT 而非叠加，动态量级保持 `O(TDB)`。

### L2：HiF4 层级约束 Babai 解码

把 E6M2、lv2、lv3 与 mantissa 的共享合法组合视为 64-block 的层级格，在最终部署 Hessian
度量下执行固定顺序 nearest-plane/no-clipping 解码。它不是恢复旧 full64 坐标 sweep：只做一次
Cholesky/Babai 前向决策，不做邻域 beam、重复 sweep 或 fold 选优。

第一版固定 `B=64`、Hessian damping 为 block 平均对角线的 `1%`、顺序为 damped Hessian 对角线
降序，只修改静态 Weight 编码。Activation 保持 v160 Linear 父规则，避免把格解码带入在线
API。

复杂度：Weight 校准约 `O(R_o D B)`（Hessian 分解另计 `O((D/B)B^3)`）；在线零新增；状态不增。

### L3：合法 HiF4 Trellis/VQ

若 L2 不能表达共享层级的联合决策，使用固定宽度 trellis 对连续 4/8 元素微组联合选码，状态
只包含合法 HiF4 hierarchy，不引入新 decoder 或 side channel。第一版仍限定 Weight-only：
每个 64-block 按连续 4 元素形成 16 个 stage，beam width 固定为 8，不根据本地结果扩展。

复杂度：校准 `O(R_o D K S)`；在线零新增；输出/state 与标准 HiF4 相同。

### L4：Kronecker 可逆仿射变换

使用 `R = R1 tensor-product R2` 的结构化等价变换：`A'=AR`、`W'=WR^{-T}`。目标是在比
rank-1 更高的表达力下避免存储和执行完整 `D×D` 矩阵。首版对每个 64-block 固定使用
`R=R8 tensor-product R8`，不搜索因子形状或 rank。因现有 CAT/Householder 已覆盖大量坐标
变换，L4 只有在 L1--L3 没有官方正向时启动。

复杂度：校准取决于固定因子求解；动态约 `O(TD(s1+s2))`；状态 `O(s1^2+s2^2)`。

Linear 执行顺序固定为 `L1 审计 -> L2 -> L3 -> L4`。

## 5. 明确不进入实现的外部方法

- HBQ、AdaMX、RaZeR：改变数值格式、码字语义或官方 decoder；
- SVDQuant、ARCQuant、OSC：需要高精度残差支路、扩展通道或自定义 kernel；
- KronQ 输出侧变换：当前六 API 无法在矩阵乘法后执行逆变换；
- KVLinC 原始 adapter：需要 attention 内部 correction hook；
- mixed precision、sparse outlier side channel：违反固定统一 HiF4 输出契约。

这些方法只能提供理论启发，不能以论文名称直接建立候选。

## 6. 激活与结束条件

激活本计划时必须：

1. 当前活动计划已有最终状态并移入 archive；
2. 当前计划中 Linear、低秩 Gram Attention 及可能的组合官方结果均已登记，或已明确结束；
3. 将本文件从 `plans/queued/` 移到 `plans/` 根目录，状态改为 `ACTIVE`；
4. 更新 `plans/README.md`、`docs/current-solution-status.md` 和根 README，使本文件成为唯一活动
   计划；
5. 重新确认当时最佳可复现父版本与 `<300s` 官方余量，不能自动沿用今日假设。

本计划在所有 A/L 候选各获得一次官方裁决、明确因接口不可实施或被前序结果取消后完成。两侧
若产生新官方最好版本，再构造一次组合并实测交互和时间。

## 7. 文献依据

- WUSH：<https://arxiv.org/abs/2512.00956>
- GPTQ/Babai 几何：<https://arxiv.org/abs/2507.18553>
- QTIP trellis：<https://arxiv.org/abs/2406.11235>
- GPTVQ：<https://github.com/Qualcomm-AI-research/gptvq>
- FlatQuant：<https://arxiv.org/abs/2410.09426>
- AffineQuant：<https://arxiv.org/abs/2403.12544>
- H-Scale：<https://arxiv.org/abs/2608.28113>
- KIVI：<https://arxiv.org/abs/2402.02750>
- Quantized Keys Steal Attention：<https://arxiv.org/abs/2605.26266>
- KVLinC：<https://arxiv.org/abs/2510.05373>
