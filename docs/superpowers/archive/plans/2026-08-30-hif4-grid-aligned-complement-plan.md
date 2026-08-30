# HiF4 补充优化方案审查：合法尺度晶格、CAT 与 Qronos

> 日期：2026-08-30  
> 状态：**已审查，部分采纳并修正后并入主计划**  
> 定位：对 [`36,000 Accuracy-First 详细方案`](2026-08-30-hif4-accuracy-first-36000-plan.md) 的补充，不替代它。  
> 父版本：根目录 `solution.py`，clean BOAT + cross-fold Weight-HSDQ + Gram-hierarchy Activation-HSDQ + Attention deployed shortlist，SHA `5d1128cc79fef58154da2f600ec4b472ff95030e1f1e61b96593d06fd9aac94f`。  
> 合规边界：校准输出 `A@W` 只能优化离线 `Q(W)`；不得用于拟合、选择或反推在线 `Q(A)`，也不得写入 `activation_state`。  
> 阶段口径：当前是 accuracy-first 探索期，记录运行时间，但暂不以 400/420 秒作为算法淘汰条件。

---

## 0. 审查结论

原方案包含三个值得保留的方向，但其中两个需要重写，一个需要换到正确的优化阶段：

| 原提案 | 裁定 | 修正后的去向 |
|---|---|---|
| G1：按 NVFP4 `Δ` 对齐 HiF4 网格 | **原推导不适用于当前 BOAT 主路径；核心“搜索合法尺度晶格”可行** | 改为 GALS：对变换后的实际值生成/枚举合法 E6M2 scale，并与 A1 层级 HSDQ 联合 |
| G2：Four-over-six | **不能直接移植；诊断方法和有限 scale 裁决可借鉴** | 纳入 D0 组件 oracle 与 GALS 设计依据 |
| G3：CAT 几何平均初始化 | **可行，但原公式乘法顺序错误且缺少正则化** | 作为 A4 BOAT-2 的非正交初始化器/上界探针 |
| G4：Qronos 联合纠错 | **方向可行，但不属于 A6 激活拟合** | 移到 A5 FS-JDRQ / A3 权重求解器；冻结 activation state 后只更新 weight params |
| P0：立即提交当前根 | **不并入算法依赖链** | 仅保留为需用户单独授权的官方检查点 |
| “五模型全非负、<400s”门禁 | **与现阶段目标冲突且过严** | 改为 Qwen 主目标、异构模型软护栏、跨 fold 硬门；时间只记录 |

最重要的判断是：

> 当前可采纳的新增算法不是“原始 NVFP4 网格对齐”，而是“**变换后值域上的合法尺度晶格搜索**”。它能无损扩展当前 `±3` E6M2 邻域候选，但理论收益必须先用全 255 码 oracle 测量，不能预先假定很大。

---

## 1. 已核实的 HiF4 结构

评测器中一个 64 元素块的结构是 `(8, 2, 4)`：

```text
scale_factor : 无符号有限 E6M2 code，0..254，每 64 元素共享
scale_lv2    : {1, 2}，每 8 元素共享
scale_lv3    : {1, 2}，每 4 元素共享
mantissa     : {0, 0.25, ..., 1.75}
sign         : 每元素
denom        = scale_factor × scale_lv2 × scale_lv3
```

对一个 8 元素组内的两个 4 元素子组，设有效指数为 `(e₀,e₁)`，其中
`denom = s × 2ᵉ`。由于 lv2 共享，合法组合不是任意的 `{0,1,2}²`，而是：

```math
E₈ = {(0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2)}
```

特别地，`(0,2)` 和 `(2,0)` 不合法。因此不能先独立优化两个 4 元素组，再用“块级投票”拼起来；必须对共享 lv2 的 8 元素联合求解。

当前 `solution.py` 的事实是：

1. `_BASE_OFFSETS=(-3,-2,-1,0,1,2,3)`，顶层 scale 只搜索标准 `amax/7` code 附近 7 个 E6M2 code；
2. `_solve_hierarchy` 对固定 scale，在逐元素/importance 加权 MSE 下正确处理了上述 lv2/lv3 耦合；
3. 当传入 `gram64` 时，代码先用对角代理选择层级，再用 Gram 评价该层级；它没有在完整二次型下精确求最优层级。

所以确有两个尚未利用的自由度：

- 顶层 E6M2 scale 的完整有限码域；
- 非对角 Gram/Hessian 目标下的层级联合离散优化。

---

## 2. 为什么原始“按 NVFP4 Δ 对齐”不能直接成立

原始参考值在反量化后确实来自：

```math
xᵢ = qᵢ Δᵦ
qᵢ ∈ {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```

若直接编码 `x`，选择分母 `d` 后的 HiF4 重构为：

```math
kᵢ(d) = clip(round(4 |xᵢ| / d), 0, 7)
x̂ᵢ(d) = sign(xᵢ) × d × kᵢ(d) / 4
```

当 `d = 2Δᵦ` 时，`{0.5,1,1.5,2}Δᵦ` 的确都能精确落在均匀 mantissa 网格上。这个例子本身正确。

但当前 Linear 主路径不是直接编码 `x`。它先做等价变换：

```math
x′ = xT
W′ = T⁻¹W
x′W′ = xW
```

其中 `T` 已包含对角平衡、置换与 signed-Hadamard。Hadamard 后的单个分量形如：

```math
x′ⱼ = (1 / √d) Σᵢ hᵢⱼ dᵢ xᵢ
```

不同 `xᵢ` 可能使用不同 NVFP4 scale `Δᵦ`，对角因子 `dᵢ` 也不必是二的幂。因此一般不存在单个 `Δ` 使所有 `x′ⱼ ∈ E2M1 × Δ`。

由此得到两条约束：

- `denom/Δ` 命中率只适合 identity、纯置换，或可证明保留网格的特殊变换做归因诊断；
- 当前 BOAT 路径必须根据实际待编码的变换后值选择合法 scale，不能以原始 `Δ` 为主候选源。

---

## 3. 修正版 G1：GALS（Grid-Aware Legal Scale Search）

### 3.1 目标

对每个实际待编码的 64 元素块 `v`，扩展当前 `±3` 邻域搜索，寻找使真实部署目标最小的合法 E6M2 顶层 scale：

```math
s* = argmin[s ∈ S_E6M2] min[z ∈ Z_HiF4(s)] L(v,z)
```

其中 `S_E6M2` 是 evaluator 接受的 255 个有限无符号 scale code，`Z_HiF4(s)` 包含所有合法 lv2/lv3/mantissa/sign 配置。

### 3.2 GALS-O：全码域 oracle

对 sampled 64-block 枚举 code `c=0,...,254`，解码为 `s_c`，对每个 `s_c` 求合法层级解并精确评分：

```math
L_all255(v) = min[c ∈ {0,...,254}] L(v; s_c)

g_scale = (L_±3 - L_all255) / (L_±3 + ε)
```

`g_scale` 是“固定变换、固定目标、只扩大合法 scale/hierarchy 搜索”能获得多少收益的直接诊断。它用于测空间，不要求直接成为提交实现。

- validation 上 `g_scale` 很小：立即停止尺度方向；
- `g_scale` 明显且跨 fold 可迁移：继续压缩候选；
- train 大、validation 小：判定为代理失配，不继续增加候选。

2026-08-30 的只读 smoke test 给出了“可行但并非普遍大收益”的初步证据。样本为
Qwen 一层缓存、`v` role、当前 BOAT 后前 32 行，共 448 个 64-block：

| 侧别/目标 | 改善 block | 平均相对 gap | 总 loss gap | 最大单 block gap |
|---|---:|---:|---:|---:|
| weight，逐元素 MSE | 4 / 448 | 0.0393% | 未作为主指标统计 | 7.466% |
| activation，当前 `gram64` 目标 | 60 / 448 | 0.5771% | **0.6302%** | 19.321% |

这不是正式 panel 结果，也没有解决 Gram 下的完整 hierarchy beam；它只证明：

- 当前 `±3` 确实会漏掉少量真实合法收益，尤其是 activation/Gram 目标；
- 全局平均收益目前是亚百分比量级，不能据此宣称能显著抬高 Linear mean；
- 最合理的用法是先定位高-gap block，再把稀疏 GALS 候选接入 A1，而不是全局盲扫。

### 3.3 GALS-C：解析临界点候选

对变换后非零值 `|vᵢ|`、mantissa `m ∈ {0.25,...,1.75}`、有效指数 `e ∈ {0,1,2}`，生成：

```math
sᵢₘₑ = |vᵢ| / (m × 2ᵉ)
```

将每个 `sᵢₘₑ` 投影到最近合法 E6M2 code，并加入：

- 投影 code 的左右邻居；
- 标准 `amax/7` 的当前 `±3` 邻域；
- parent/incumbent code。

候选去重后，对每个共享 scale 做整 64 元素联合评分，而不是对 4 元素组投票。该候选集覆盖“某个真实值精确落到某个合法 mantissa 网格点”的位置，并以全码域 oracle 检查召回率。

### 3.4 层级求解

若目标是逐元素加权 MSE：

```math
L_diag = Σᵢ ωᵢ (vᵢ - v̂ᵢ)²
```

固定 scale 后，现有 `_solve_hierarchy` 可作为精确求解器。

若目标是 Gram/Hessian 二次型：

```math
e = v̂ - v
L_H = eᵀHe = Σᵢ Σⱼ eᵢ Hᵢⱼ eⱼ
```

不同位置通过 `Hᵢⱼ` 耦合。现有“先按对角 MSE 选层级，再用 `H` 评分”只是近似。修正路线是：

1. GALS 提供顶层 scale 候选；
2. A1 Progressive HSDQ 对 8 元素层级决策做 beam search；
3. 每扩展一个 8 元素组，用精确二次型的已定项和可计算下界排序；
4. 始终保留 incumbent，最终只接受真实目标下降的候选。

所以 GALS 与 A1 是组合关系：GALS 扩大 scale 候选，A1 解决非对角目标下的层级耦合。

### 3.5 理论上限与停止条件

GALS-O 给出的是“固定变换、固定目标、只改变 HiF4 scale/hierarchy”条件下的离散 oracle。它不是最终 36,000 上界，但可给出该分支的硬停止信号：

- `L_all255 ≈ L_±3`：继续设计 scale 启发式没有价值；
- `L_all255 ≪ L_±3` 但 GALS-C 追不上：问题是候选召回；
- GALS-C 追上局部 loss、跨 fold 却不提升：问题是目标代理；
- weight-float 或 activation-float oracle 仍有大差距：转向另一侧或 BOAT-2。

---

## 4. Four-over-six：只借鉴方法，不直接移植

Four-over-six 针对 NVFP4 E2M1 的 `4→6` 非均匀间隔，在有限 scale 方案间自适应选择。HiF4 的 mantissa 是均匀网格，没有同一个断层，因此不能照搬“最大值对齐 4 或 6”。

可采纳两点：

1. 用有限候选的真实离散重构 loss 做块级裁决；
2. 通过逐组件高精度 oracle 判断误差究竟来自 scale、低位码还是两侧共同量化。

NVIDIA Nemotron 3 Ultra 的公开技术报告给出的结果也提示：max-calibrated 4/6 在 49,152 个投影权重上中位相对重构 MSE 降低 16.4%；MSE calibration 的局部 MSE 降幅更大，却未稳定转化为下游精度。因此本赛题也不能把局部 MSE 降幅直接当成官方分收益。

---

## 5. 修正版 G3：CAT 作为 BOAT-2 初始化器

CAT 的价值在于给非正交等价变换一个由激活与权重二阶结构共同决定的解析初始化，而不是替代最终离散搜索。

令：

```math
A = Cov(X)
B = 权重侧二阶结构
A_ε = A + ε × tr(A) / d × I
B_ε = B + ε × tr(B) / d × I
C = A_ε^(1/2) B_ε A_ε^(1/2)
```

在列向量约定下，平衡变换为：

```math
T_CAT = C^(1/4) A_ε^(-1/2)
```

验证：

```math
T_CAT A_ε T_CATᵀ = C^(1/2)
T_CAT^(-T) B_ε T_CAT^(-1) = C^(1/2)
```

原文档写成 `A^(-1/2) C^(1/4)`，乘法顺序通常不可交换，会破坏上述等式，必须改正。实现时还要明确代码采用左乘还是右乘；若张量约定相反，应整体转置推导，而不是机械照抄。

CAT 原理论针对均匀整数 PTQ；HiF4 有共享 scale 和层级离散约束，所以这里的 `T_CAT` 只是：

- BOAT-2 的连续初始化候选；
- alignment 潜力的诊断 oracle；
- 后续离散可实现投影的起点。

只有在投影为合法、可逆、可部署的 `D/P/H/R` 结构后，且 cross-fold 真实目标改善，才准入主线。必须记录 condition number、逆变换误差和 continuous→legal retention。

---

## 6. 修正版 G4：Qronos 进入权重联合求解，不进入 A6

Qronos 使用原始激活 `X` 与量化激活 `X̃`，通过顺序 correction/diffusion 调整量化权重，使：

```math
X̃ Ŵ ≈ XW
```

这意味着它能补偿激活量化误差，但补偿动作发生在权重求解器中，而不是根据输出监督拟合在线 `Q(A)`。

本赛题内的合规版本应为：

1. 先仅用 activation-only 统计冻结 `activation_state`；
2. 计算部署侧 `X̃ = Q_A(X)`；
3. 允许 `(X,W,XW,X̃)` 只参与离线 `weight_params` 优化；
4. 不得把任何由 `XW` 选出的参数写回 `activation_state`；
5. 输出仍是合法 HiF4 五字段。

因此 G4 应并入 A5 FS-JDRQ，或作为 A3 权重求解器的误差扩散模块；不应放在 A6 Global Activation-HSDQ。

原始 Qronos 的逐标量舍入不直接适配 HiF4 的 4/8/64 共享结构。可执行改造是：

- 用合法 64-block 候选作为离散决策单元；
- GALS 产生 scale 候选，A1/HSDQ 产生 hierarchy/mantissa 候选；
- Qronos/Cholesky correction 决定块顺序和连续残差传播；
- 每步用部署侧 residual objective 接受或回退。

跨层漂移补偿不采纳，因为 evaluator 给每层的是真实未级联量化激活，不存在部署式跨层误差累积。

---

## 7. 官方提交与评测口径修正

“先提交当前根”可以是有价值的官方检查点，但不应写成所有算法实验的先决条件：

- 提交属于外部状态变更，需要用户单独授权；
- 当前 wall 记录为 414.03 秒，距离 420 秒只有约 5.97 秒，不能称为稳健余量；
- 用户当前明确要求暂不考虑时间，所以本轮不得用 `<400s` 淘汰高精度原型；
- 官方结果可以校验当前候选，但不能反复作为调参反馈，以免对隐藏集过拟合。

原文档还有两处事实需修正：

1. C66 的 Qwen panel 已记录为 **238.282409**，不是“待查”；
2. 本地—官方并非只有一个历史配对点，已有 C39/C41/C47/C66 等记录；但当前根超出历史本地 panel 的已观测范围，任何官方分外推仍不可靠。

本地 A/B 仍然能严格回答“同口径候选谁更好”。不能回答的是“本地提升会按固定比例兑换为多少官方分”。

---

## 8. 并入主计划后的执行顺序

```text
D0 误差账本与单侧 oracle
  ↓
D0-G 全 255 E6M2 scale-lattice oracle
  ├─ scale gap 小：停止 GALS，转 A2/A3/A4
  └─ scale gap 大：实现 GALS-C
                    ↓
A1 Progressive HSDQ + GALS 候选
  ↓
A2 FFN 扩张 / A3 跨块低秩 Hessian
  ↓
A4 BOAT-2 + 正则化 CAT 初始化
  ↓
A5 FS-JDRQ + HiF4-block Qronos correction
  ↓
A6 仅做 activation-only 全局 HSDQ
```

当前阶段的门禁为：

1. **硬门**：接口、五字段、可逆性和 Linear 激活侧无输出监督泄漏；
2. **硬门**：fold A 生成、fold B 选择、validation 终验，禁止同 fold 自证；
3. **主目标**：Qwen 同口径目标改善；
4. **软护栏**：异构模型不得出现无法解释的灾难性结构回退；不要求每个模型每次都严格非负；
5. **时间**：记录 six-API 与 wall，不在 accuracy-first 原型期淘汰候选；
6. **回退**：每个新求解器保留 incumbent，真实目标不降则不接受。

---

## 9. 风险登记

| 风险 | 影响 | 处理 |
|---|---|---|
| BOAT 后不再保留原始 E2M1 网格 | 原 G1 的 `denom/Δ` 诊断失真 | 对实际变换后值做 GALS；原始 Δ 仅用于 identity 归因 |
| lv2 每 8 元素共享 | 独立 4 元素对齐会生成非法/次优组合 | 对 8 元素合法指数集合联合求解 |
| Gram 非对角耦合 | 当前固定 scale hierarchy 解不再精确 | A1 beam/HSDQ 用真实二次型评分 |
| 全 255 code 很慢 | 不能直接作为最终在线候选器 | 只用于 sampled oracle，再蒸馏成解析候选 |
| CAT 矩阵病态 | 逆平方根放大噪声 | trace-scaled ridge、特征值截断、condition-number gate |
| CAT 连续解投影损失 | 理论 alignment 增益无法落到合法结构 | 单独记录 continuous→legal retention |
| Qronos 状态边界 | 输出监督可能污染 activation state | 先冻结 Q(A)，只更新 weight params，调用图审计 |
| 局部 loss 与官方分不单调 | 可能局部更优、泛化变差 | cross-fold、异构软护栏、官方仅作稀疏验证 |
| 当前时限余量小 | 后续提交可能超 420 秒 | accuracy-first 与 deployment-compression 两阶段分离 |

---

## 10. 明确不采纳的内容

- 不以原始 NVFP4 `Δ` 对齐作为当前 BOAT 主路径的核心候选规则；
- 不让两个 4 元素组独立选择后再“块级投票”；
- 不把原公式 `A^(-1/2)C^(1/4)` 直接用于 CAT；
- 不把 Qronos 放入 activation-only A6，也不让输出监督影响 `activation_state`；
- 不把“五模型全部非负”和“<400s”设为当前 accuracy-first 的硬门；
- 不把当前根提交写成算法开发的前置依赖；
- 不根据本地 panel 做线性官方分换算。

---

## 参考

- [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)
- [NVIDIA Nemotron 3 Ultra Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
- [CAT: Concentration-Alignment Transforms for Low-Bit Quantization](https://arxiv.org/abs/2603.04359)
- [Qronos: Correcting the Past by Shaping the Future…](https://arxiv.org/abs/2505.11695)
- 工程内部：[36,000 Accuracy-First 详细方案](2026-08-30-hif4-accuracy-first-36000-plan.md)、[当前主版本算法效果与评测状态](../../../current-solution-status.md)、[外部 v2.7 本地差距审计](../../../../logs/candidates/2026-08-29-external-hif4-gap-analysis.md)。
