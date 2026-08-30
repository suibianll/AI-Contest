# 2026 华为算法大赛 HiF4 量化赛题：问题重构与新算法方案

> 修订日期：2026-08-30
> 本版目的：跳出现有“局部重建 + 层层门控”的框架，从官方评分函数重新定义优化变量，给出能继续冲击 36000+ 的具体算法、实现顺序和收益判断。  
> 官方历史事实锚点：本地归档最高 `22557 / 217.2s`（C66）；外部 `youxilee/hif4` 为用户提供的 `24153 / 239s`；官方榜已有超过 `36000` 的方案。后两项分数按用户提供的同口径结果记录，外部源码审计版本为 `dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`（v2.7）。

> **当前根实现（2026-08-30）**：根目录 `solution.py` 已从 C86 实验集合重写为
> BOAT + cross-fold Weight-HSDQ + Gram-hierarchy Activation-HSDQ + 输出感知
> Attention shortlist 的单一路径。固定 Qwen2.5-0.5B 全 24 层、`seq=128`、
> `calib=2`、`test=4`、CPU 缓存实测：Linear mean `0.501558`、Attention mean
> `0.841829`、Qwen shaped panel `293.755106`、正式 API `382.153528s`。
> 这组数是本地相对排序指标，不是官方绝对分数；完整结果与角色归因见
> [`docs/current-solution-status.md`](docs/current-solution-status.md)。本文后续
> 的 JDRQ、FASA 等章节是理论/历史研究路线，除非明确写入根文件，否则不等于当前
> 已启用算法。

---

## 0. 执行结论

现有方法无法取得重大提升，不是因为 E6M2 offset 少搜了两档，也不是 FULL64 覆盖率不够，而是优化问题定义得仍然太保守：

1. **Linear 的真正目标不是让 `Q(W)` 接近 `W`，而是固定可部署的 `Q(A)` 后，直接求一个合法 HiF4 权重，使 `Q(A) @ Q(W).T` 接近 `A @ W.T`。** `Q(W)` 可以、也应当偏离原始 `W`，主动吸收激活量化的系统误差。
2. **离线校准允许使用 `A@W`。** 禁止的是用输出反推测试激活的逐元素 `Q(A)`，不是禁止用校准输出优化静态 `Q(W)`、选择可部署的固定变换、评估完整管道或优化 Attention 状态。
3. **当前误差分解漏掉了决定性耦合项。** 对 `E_A=Q(A)-A`、`E_W=Q(W)-W`，精确误差为

   $$Q(A)Q(W)^T-AW^T=E_AW^T+AE_W^T+E_AE_W^T.$$

   现框架主要分别压前两项，外部 v2.6 的优势恰恰来自开始利用第三项和交叉抵消；但它只改顶层 scale，仍远未求解完整问题。
4. **36000 不是“再涨一点”，而是另一个误差级别。** 若官方总分是 250 个 Linear + 200 个 Attention case 的百分制求和，则理论上限为 45000：

   | 方案 | 平均 case 得分 | `MSE_PLAYER / MSE_STD` |
   |---|---:|---:|
   | 当前 22557 | 0.501 | 0.499 |
   | 外部 24153 | 0.537 | 0.463 |
   | 榜单 36000 | 0.800 | 0.200 |

   从当前到 36000，需要把**当前剩余误差再压约 60%**；从外部实现到 36000，也要再压约 57%。千分位的 headroom、coverage、固定 offset 微调不可能完成。

新的主算法命名为 **JDRQ-HiF4（Joint Distilled Residual Quantization）**：

> 先冻结真实在线激活量化路径，离线得到 `Z=Q(A)` 和教师输出 `Y=A@W.T`；再通过岭回归得到能够补偿激活量化误差的连续权重目标；最后不再围绕原始 `W` 做局部舍入，而是在完整 HiF4 合法集合上，用输出残差、全 64 维 Gram 和 Gauss-Seidel/坐标下降直接求 `Q(W)`。

这是下一阶段唯一有希望产生“数量级提升”的 Linear 主线。Attention 则应独立改成**真实 softmax 输出目标下的结构化 Q/K 变换 + 策略级 Q/K/V 联合校准**，不能继续套 Linear 的局部重建思路。

---

## 1. 赛题的正确数学问题

### 1.1 数据与格式

源数据是 NVFP4，不是 FP32 原始模型：

- 元素：E2M1，值集 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`；
- 每 16 个元素共享一个 E4M3 scale；
- 参考真值由 NVFP4 反量化到 BF16/FP32 后得到。

目标是 HiF4：

- 每 64 个元素共享一个 E6M2 顶层 scale；
- 每 8 个元素一个 `scale_lv2 ∈ {1,2}`；
- 每 4 个元素一个 `scale_lv3 ∈ {1,2}`；
- 元素为 S1P2，即 `sign ∈ {-1,0,1}`、`mant ∈ {0,0.25,...,1.75}`；
- 总存储 4.5 bit/value。

合法值可写成

$$q_i=s_g\,2^{e^{(2)}_{g,j}+e^{(3)}_{g,l}}\frac{c_i}{4},\qquad c_i\in\{-7,\ldots,7\}.$$

### 1.2 官方评分等价目标

每个 case：

$$Score=1-\frac{MSE_{PLAYER}}{MSE_{STD}}.$$

因此最大化总分等价于最小化

$$\sum_c \frac{MSE_{PLAYER,c}}{MSE_{STD,c}}.$$

关键后果：

- 不是最小化张量重建误差；
- 不是最小化所有 layer 的 raw MSE 总和；
- 不是让每个 HiF4 参数尽量接近原值；
- 校准选型应尽量按最终部署量化器和最终算子输出排序；
- 多 case/多 fold 调参时应使用**标准基线归一化误差**，避免大能量层支配选择。

### 1.3 Linear 的真实自由度

给定校准激活 `X`、权重 `W`、固定在线量化策略 `Q_A(·;θ)`：

$$\min_{\theta,\;Q_W\in\mathcal H} \left\|XW^T-Q_A(X;\theta)Q_W^T\right\|_F^2,$$

其中 `𝓗` 是合法 HiF4 权重集合。

最容易被忽略的一点是：**题目没有要求 `Q_W≈W`。** 只要输出更准，静态 HiF4 权重偏离原始权重完全合理。传统 GPTQ 只是在 `Q_A(X)=X` 或忽略激活量化时的特例。

### 1.4 Attention 是另一个问题

$$\min_{\theta_Q,\theta_K,\theta_V}
\left\|Attn(Q,K,V)-Attn(Q_H,K_H,V_H)\right\|_F^2.$$

这里没有静态权重可以吸收误差，且 softmax 是强非线性。必须直接以真实 causal/non-causal Attention 输出做校准排序，不能用 Q/K/V 重建 MSE 替代。

---

## 2. A@W 的使用边界：应当积极利用，而不是自我禁用

赛事说明原文禁止的是：

> 计算 `A@W`，并利用 `A@W` 拟合反推出 `Q(A)`。

这不等于“离线不能计算 A@W”。建议采用下面的清晰数据流：

### 2.1 合法且应重点使用

- 计算校准教师输出 `Y=A@W.T`；
- 用 `Y` 优化、回归和离散搜索静态 `Q(W)`；
- 用 `Y` 比较完整、固定、可部署的 Smooth/Permutation/Butterfly 管道；
- 用 `Y` 选择岭回归强度、残差 sweep 数、候选池和 HiF4 权重参数；
- Attention 校准时计算真实 softmax 输出，选择固定 Q/K/V 状态；
- 所有输出残差、教师输出和拟合中间量在校准函数返回前释放。

### 2.2 不能做

- 为每个测试激活先算输出，再逐元素反解 `Q(A)`；
- 把 `Y`、输出残差或样本答案写入 `activation_state`；
- 在线激活量化阶段访问教师输出或原始权重输出；
- 记忆校准样本并在测试时做样本匹配。

### 2.3 推荐的审计不变量

1. 先构造并冻结 `activation_state`；
2. 固定状态后生成校准 `Z=Q_A(X)`；
3. `A@W` 之后只允许改变 `weight_params` 和离线标量超参；
4. JDRQ 开关前后，`activation_state` 序列化结果完全一致；
5. 在线 `hif4_dynamic_quantize_activation` 的代码路径不读取任何教师输出派生量。

这条边界既充分利用规则，也便于人工审核。

---

## 3. 现有框架为什么到达平台期

### 3.1 优化中心仍是原始 W，而不是最终输出

当前大部分流程仍是：

```text
W --局部重建/协方差加权--> HiF4(W)
X --局部重建/W Gram加权--> HiF4(X)
最后才用 A@W 做候选裁判
```

这只能在已有候选池里选一个较好者，无法产生“为了补偿 `Q(A)` 而主动偏移的权重”。

### 3.2 外部 v2.6 证明了交叉项价值，但仍只是弱解

外部实现固定 `Xq` 后，计算

```text
R = Xt @ Wt.T - Xq @ Wq.T
```

然后对每个 64-channel block、每个输出行，只枚举 5 个 E6M2 scale offset，并重新做面向原始 `W` 的层级求解；用

$$\Delta E=2a_b^T\delta+\delta^TG_b\delta$$

决定是否接受。它已带来外部记录中的 Linear 均值约 `+0.055`，说明方向正确。

但它仍有四个上限：

1. 候选围绕原始 `W` 生成，不围绕输出最优权重生成；
2. 主要改变顶层 scale，不能自由调整 mantissa、micro-exp、零点/符号；
3. 不先求连续最优补偿目标；
4. 只做局部 64-block 更新，没有多分辨率、跨 block 候选初始化和稳健回归。

JDRQ 不是复制 v2.6，而是把它推广成完整离散输出回归。

### 3.3 “误差独立”假设不成立

当前文档旧版写过“激活误差和权重误差可分开优化”。这只是一阶近似。W4A4 下 `E_AE_W^T` 不小，而且其符号可以被静态权重有意识地利用。外部 joint refine 的大增益已经直接否定“分开优化足够”的假设。

### 3.4 候选代理与落地量化器不一致

候选常按标准 HiF4、局部 MSE、少量 row 或简化 refine 排名，落地却使用 offset、importance、FULL64、动态 coverage。候选排序和最终评分目标不一致会：

- 拒绝真正能与最终量化器协同的变换；
- 选择在代理上好、在部署路径上差的变换；
- 促使代码不断增加安全阈值，进一步限制搜索空间。

### 3.5 过多硬门控，把搜索变成保守回退器

“每 fold 都不退化”“固定最小提升 0.5%/2%”“按模型宽度硬编码比例”适合保底，不适合冲上限。少量校准数据下，强候选可能在一个 fold 有轻微噪声，却在整体输出上显著更优。应改为连续的稳健目标和置信度，而不是大量 if/else 否决。

### 3.6 对 state 4096 节点的理解过于保守

赛事文本限制的是嵌套总节点数，不是明确规定 Tensor 只能有 4096 个元素。一个形如 `[num_blocks,64,64]` 的 CPU Tensor 通常仍是一个 state 节点。最终仍需用官方 `self_check.py` 验证大小和时间，但不应在没有证据时把完整 64×64 block metric 自动禁用。

---

## 4. 新主线：JDRQ-HiF4

### 4.1 总流程

```text
校准 X,W
  │
  ├─ 生成少量可部署变换 θ：D / P / signed-butterfly
  │
  ├─ 对每个 θ：
  │    1. 固定 activation_state(θ)
  │    2. 用最终在线量化器得到 Z = Q_A(X;θ)
  │    3. 计算教师输出 Y = X @ W.T
  │    4. 岭回归得到补偿激活误差的连续权重 W*
  │    5. W* 初始化 + 输出残差驱动的完整 HiF4 离散求解
  │    6. 交叉验证完整 Q(A)@Q(W).T
  │
  └─ 返回最优合法 weight_params + 与其配对的固定 activation_state
```

### 4.2 Step A：构造低自由度、可部署的变换族

设原始乘法为 `XW.T`。对可逆变换 `T`：

$$X_t=XT,\qquad W_t=WT^{-T},\qquad X_tW_t^T=XW^T.$$

按成本从低到高使用：

1. `T=I`；
2. SmoothQuant 对角矩阵 `D`，log2 scale 用 1/8 或 1/4 档小范围坐标搜索；
3. hierarchy-aware permutation，但分组目标改为 64/8/4 层级的联合代价，而不是单一 range 排序；
4. 4/8/16/32/64 signed Hadamard；
5. 两层稀疏 butterfly rotation，每层只做成对 2×2 旋转，角度取小离散集合；
6. 仅对高价值 down-projection 试更宽块和第二层 butterfly。

不要一开始上任意 64×64 稠密矩阵。它会增加在线 O(64K) 成本、过拟合和 state 体积。优先使用 `D + P + butterfly`，在线复杂度约 O(K log 64)。

### 4.3 Step B：用最终在线量化器生成教师输入 Z

对每个校准样本：

```python
Xt = transform_activation(X, state)
Z  = dequant_hif4(dynamic_quantize_activation(NVFP4_X, state))
```

必须调用真实动态路径，而不是简化的标准量化器。否则后面的权重补偿针对了错误的激活误差。

校准样本少时，把 token 行按交错区间切成 4 fold，例如 `row_id % 4`，避免只按两条序列硬切导致方差过大。

### 4.4 Step C：连续输出蒸馏权重

令：

- `Z ∈ R^(M×K)`：固定在线量化器产生的校准激活；
- `Y=XW.T ∈ R^(M×N)`：教师输出；
- `W0 ∈ R^(N×K)`：回归中心，候选同时取等价变换后的原始权重 `Wt` 和当前父版本解码后的合法权重 `QW_parent`；
- `R0=Y-ZW0.T`：当前输出残差。

求带岭正则的补偿：

$$\min_{\Delta W}\|R_0-Z\Delta W^T\|_F^2+\lambda\|\Delta W\|_F^2.$$

对偶形式避免 K×K 求逆：

$$\Delta W^T=Z^T(ZZ^T+\lambda I)^{-1}R_0.$$

最终连续目标：

$$W_*(\eta,\lambda)=W_0+\eta\Delta W,$$

其中 `η∈{0,0.25,0.5,0.75,1}`，`λ` 使用 `trace(ZZ.T)/M` 的倍数网格。M 通常远小于 K，求解 M×M 系统比 GPTQ 的 K×K 分解更便宜。

全自由度对偶解容量较大，只有 2×128 个校准 token 时可能过拟合，因此还要并行构造**层级结构化蒸馏目标**。先拟合 `X_t≈ZC`：

$$C_b=(Z_b^TZ_b+\lambda I)^{-1}Z_b^TX_{t,b},\qquad
W_{*,structured}=W_tC^T.$$

依次使用与 HiF4 层级对齐的 `C`：

1. diagonal，只有每通道一个 gain；
2. block-4；
3. block-8；
4. block-16；
5. block-64；
6. `block-64 + rank-r` 小残差，`r∈{4,8,16}`。

这些估计器自由度远低于完整 `N×K` 输出回归，更容易跨校准/测试泛化，也会把补偿集中在可由 HiF4 4/8/64 层级表达的方向上。最终候选池同时保留：`W0`、structured `W*`、full-dual `W*` 及二者的收缩混合。

这一步的意义不是直接返回 `W*`，而是生成**正确方向且不同容量的 HiF4 候选中心**。`η=0` 自动包含原始 W 路线，保证可回退。

为抑制过拟合：

- 用 3 个 fold 拟合 `(η,λ)`，1 个 fold 验证；
- 对 `ΔW` 做 per-output-row norm clipping；
- 对低奇异值方向使用更强 ridge；
- 最终选定超参后在全部校准 token 上重算一次。

### 4.5 Step D：多初始化合法 HiF4 候选

每个输出行、每个 64-channel group 至少生成以下初始化：

1. `Q_HiF4(W0)`；
2. `Q_HiF4(W*)`；
3. `Q_HiF4((1-η)W0+ηW*)` 的 2～3 个 η；
4. 每个初始化的 E6M2 邻域 `[-4,+6]`；
5. full code scan 的低成本下界筛选后保留 top-B scale；
6. 原始父版本 weight_params，作为绝对回退项。

候选生成必须面向 `W*` 和输出梯度，不能像外部 v2.6 一样只改变“如何重建 W0”。

### 4.6 Step E：输出残差驱动的完整离散求解

固定 `Z`，优化

$$\min_{Q_W\in\mathcal H}\|Y-ZQ_W^T\|_F^2.$$

维护当前残差

$$R=Y-ZQ_W^T.$$

对 block `b`：

$$G_b=Z_b^TZ_b,\qquad A_b=Z_b^TR.$$

若候选把旧块改为新块，令 `δ=Q_old-Q_new`，则每个输出行的精确增量为

$$\Delta L=2A_b^T\delta+\delta^TG_b\delta.$$

这使所有候选都能在不重算完整 matmul 的情况下精确打分。

每个 64 block 的局部搜索顺序：

1. **E6M2 beam**：保留 4～8 个顶层 scale；
2. **E1_8 toggle**：尝试翻转 8 个 lv2 bit，并对受影响 8 元素重求 code；
3. **E1_16 toggle**：尝试翻转 16 个 lv3 bit，并对受影响 4 元素重求 code；
4. **mantissa coordinate**：每个 signed code 尝试 `c±1`、`0`、必要时符号翻转；
5. **leaf joint move**：对高梯度的 4 元素 leaf 保留 top-2 联合方案；
6. **scale 再校准**：局部 code 稳定后再试相邻 E6M2；
7. 更新 `R`，进入下一 block；
8. 做 2～4 次 Gauss-Seidel sweep，直到校准 loss 不再下降。

优先级按可实现收益排序：

```text
连续 W* 初始化 > mantissa/exp 联合更新 > 顶层 scale 扩窗 > 更多 sweep
```

只扩大 offset 而不允许 mantissa 和 micro-exp 响应输出残差，仍会停在外部 v2.6 附近。

### 4.7 Step F：跨 64-block 的误差接力

无需显式存完整 K×K Hessian。Gauss-Seidel 每接受一个 block 后更新全局 `R`，后续 block 自然看到前面留下的输出误差，因此已隐式利用跨 block 相关性。

进一步可用两级顺序：

1. 按预计最大下降 `||A_b||² / trace(G_b)` 排序 block；
2. 先对 top 20% 难块做完整搜索；
3. 再对其余 block 做轻量 scale/mantissa 一轮；
4. 最后一轮只回访上一轮发生改变的 block 及其高相关邻块。

这比固定 FULL64 coverage 更精确，也把时间花在真正影响输出的 row-block 对上。

### 4.8 Step G：稳健选择，不再硬阈值堆叠

每个候选在 holdout fold 上计算

$$L_{robust}=mean(r_c)+\beta\,CVaR_{25\%}(r_c)+\gamma\,Var(r_c),
\qquad r_c=\frac{MSE_{player,c}}{MSE_{std,c}}.$$

推荐：

- 主排序：fold mean；
- 防灾：最差 25% fold 的 CVaR；
- 只把非法、非 finite、明显灾难回退设为硬拒绝；
- 保留父版本候选，最终用同一完整管道比较；
- 不再要求每个微小 fold 都严格正向。

---

## 5. 在线激活量化器也要升级，但不能越界

JDRQ 的最大收益来自离线 `Q(W)`，在线 `Q(A)` 仍应保持快速、固定、可审核。

### 5.1 Source-aware HiF4 搜索

NVFP4 的一个 HiF4 64-group 恰好由四个 16-block 组成。可直接利用四个源 E4M3 scale 和 E2M1 carrier：

- 用四个源 scale 的 log2 相位生成 E6M2 候选，不只围绕 amax；
- lv2/lv3 初值按 16-block scale 比例推导；
- 对重复 E2M1 pattern 使用小型向量化 lookup；
- 只对预测为 hard 的块扩展 offset/坐标搜索。

这样能同时提升精度和减少动态全扫描时间。

### 5.2 使用 W 导出的 64×64 block metric

在不使用 `A@W` 反推 `Q(A)` 的前提下，可以用静态权重 Gram 衡量激活重建误差：

$$\|(A-Q(A))W^T\|^2=(A-Q(A))(W^TW)(A-Q(A))^T.$$

建议 state 保存一个 Tensor：

```text
gram64: [K/64, 64, 64]
```

动态求解采用：

- 默认 4×4/8×8 block diagonal 快解；
- 只对 top hard block 用 full64 二次增量；
- full64 搜索 beam 2，最多一轮坐标下降；
- 若官方 state/check 或时间不允许，回退 low-rank + diagonal。

这比当前仅在窄层保存局部 Gram 更接近真实 Linear 输出代价。

### 5.3 不建议的高风险路线

不要在 state 中保存 `Q(W)^T W` 并在线直接最小化 `||AW^T-Q(A)Q(W)^T||`。虽然可以代数展开而不显式形成 `A@W`，但本质上仍是在输出域反推 `Q(A)`，容易触碰规则红线。将联合补偿集中在离线 `Q(W)` 足以获得主要收益，也更容易审核。

---

## 6. Attention 的独立重构方案

Attention 不能靠 JDRQ 权重吸收误差，必须从可部署不变量和真实 softmax 输出出发。

### 6.1 主目标必须是最终 Attention 输出

每个候选都用最终 Q/K/V HiF4 量化器计算：

```text
O_ref = softmax(QK.T / sqrt(d) + mask) V
O_hat = softmax(QhKh.T / sqrt(d) + mask) Vh
loss  = MSE(O_ref, O_hat)
```

同时评估 causal 与 non-causal（若官方两者均可能出现），按 fold mean + CVaR 排序。Q/K 重建 MSE、logit MSE和 Jacobian proxy只用于预筛，不能做最终裁判。

### 6.2 从正交 H 扩展到一般结构化可逆 T

对每个 KV head：

$$Q'=QT,\qquad K'=KT^{-T},\qquad Q'K'^T=QK^T.$$

GQA 中同一 KV head 对应的多个 Q head 共用一个 T。候选族：

```text
T = D1 · P · B1 · D2 · B2
```

- `D1/D2`：对角 reciprocal balance；
- `P`：head_dim 内 permutation；
- `B1/B2`：4/8/16/32/64 的 signed butterfly/Hadamard；
- 条件数限制在 4～8 内；
- state 只保存 scale、排列、block size、seed/角度索引。

现外部方案主要是单层正交 block-S；增加受控的非正交自由度，才可能继续显著降低 Q/K 同时量化的误差。

### 6.3 动态 K 平移

对每个 head，任意 token 无关的 `c`：

$$K'_t=K_t-c$$

只会给同一 query 的所有 logits 加同一个常数，softmax 完全不变。在线 K 量化可按当前样本计算：

- midpoint center；
- mean/trimmed-mean center；
- 按 64-group 对齐约束搜索 center 的低维候选。

最终用校准 Attention 输出选择 center policy，而不是固定某一种统计量。

### 6.4 学习量化策略，不拟合测试元素

为每个 head 学习低自由度 policy：

- Q/K/V 的 scale offset 子集；
- clipping/headroom 档位；
- hard-block refine ratio；
- Q/K reciprocal balance；
- 可选的轻微 logit temperature 校正；
- V 的 per-head / position-bucket refine 预算。

这些都是固定策略，不是逐测试样本用输出反推元素值。参数必须通过跨 fold 验证，特别是 temperature 和 V gain，防止只记住校准序列。

### 6.5 Softmax Jacobian 只做预算分配

一阶误差：

$$\delta O\approx P\delta V + (\mathrm{diag}(P)-PP^T)\,\delta S\,V,$$

$$\delta S=(\delta QK^T+Q\delta K^T)/\sqrt d.$$

可从校准集统计：

- 每个 head 的竞争度/熵；
- Q/K feature 的平均敏感度；
- V token 被关注的平方质量；
- causal 位置 bucket 的难度。

这些统计用于决定“哪些 block 多搜”，而不是替代真实 Attention 输出选型。当前 Q/K 小样本 full covariance 失败的原因，就是把不稳定 proxy 当成了最终目标。

### 6.6 Attention 的搜索顺序

1. 用 V 精确值，搜索 Q/K 的 `D + P + butterfly + K-center`；
2. 固定 Q/K 状态，用真实 `Qh/Kh` 搜索 V policy；
3. 固定 V policy，回访 Q/K 进行一次 coordinate update；
4. 搜索轻微 temperature/headroom；
5. 用完整最终量化器做两次 fold 复核；
6. 保留 identity/当前 A1 路径作为候选，而不是硬基线。

---

## 7. 具体实现阶段与实验矩阵

### Phase 0：重建正确基线与诊断量

必须先输出以下分解，否则后续无法判断瓶颈：

| 诊断 | 含义 |
|---|---|
| `L(X,W)` | 参考路径，自身误差应为 0，用于 sanity check |
| `L(X,QW)` | 只量化权重的误差 |
| `L(QX,W)` | 只量化激活的误差 |
| `L(QX,QW)` | 当前完整 W4A4 输出误差 |
| `L(QX,W*)` | 连续回归理论上限 |
| `L(QX,Q_HiF4(W*))` | 只做蒸馏初始化后的收益 |
| `L(QX,QW_JDRQ)` | 完整离散求解后的收益 |

最重要的新指标是：

$$Ceiling_{distill}=1-\frac{L(QX,W_*)}{L(QX,QW_{current})}.$$

若该 ceiling 在多模型主要算子上的中位数高于约 8%～10%，且 holdout 方向一致，JDRQ 离散求解值得继续；若连续 ceiling 本身普遍低，则应先改激活变换/量化器，而不是继续精修权重。30% 以上代表强信号，但不再把它设成唯一准入门槛。

#### 已完成的快速可行性诊断（2026-08-29）

使用冻结的 GPT-2 small 缓存、当前根量化路径、layer 0 做了两组只读诊断；ridge 以当前父版本解码后的合法 `QW_parent` 为中心：

| 层 | 当前完整测试误差 | ridge 后连续权重测试误差比 | 直接把连续权重送入现有 HiF4 量化器后的误差比 |
|---|---:|---:|---:|
| q，768→768 | 0.006707 | **0.799～0.817** | 0.992～0.999 |
| proj，3072→768 | 0.013959 | **0.956～0.970** | 0.995～1.000 |

结论很明确：

1. 固定当前 `Q(A)` 后，输出蒸馏确实有可泛化的连续改进空间，q 层约 18%～20%，proj 层约 3%～4%；
2. **朴素 `Q_HiF4(W*)` 几乎把连续收益全部丢掉**，说明仅实现 ridge + 现有量化器不会产生大分提升；
3. 真正的 P0 是“结构化蒸馏中心 + 输出残差驱动的 mantissa/micro-exp 离散求解”；
4. 固定当前 activation_state 的 ceiling 在不同算子差异很大，JDRQ 单独不足以解释 36000，必须同时改变变换和 `Z=Q(A)` 的质量。

这组结果只覆盖一个模型的两个层，不外推成官方分数，但它已足以否定“把 W* 直接重新量化就结束”的简化实现。

### Phase 1：JDRQ ceiling probe（最高优先级）

只做：

1. 固定当前 C66/C69 activation_state；
2. 生成真实 `Z=Q(A)`；
3. 对偶岭回归得到 `W*`；
4. 把 `W*` 送入现有 `_dense_to_hif4`；
5. 在 fold holdout 上与当前 `weight_params` 二选一。

这一步改动最小，用来验证“权重吸收激活误差”的连续上限和合法量化鸿沟；它还不算完整 JDRQ，也不应期待直接产生大分。不要在看清这两个量之前先实现复杂坐标下降。

### Phase 2：输出残差 full-code 局部求解

在 Phase 1 有明显正信号后加入：

- E6M2 beam；
- lv2/lv3 toggle；
- signed mantissa `±1` coordinate；
- 2 次 Gauss-Seidel；
- top row-block 20% 完整搜索。

对照组必须包括：

```text
current
external-style scale-only joint
W* initialization only
W* + mantissa coordinate
W* + full hierarchy + Gauss-Seidel
```

### Phase 3：完整管道候选重排

不再沿用旧的 proxy winner。对 identity、Smooth、P、H4/H8/H16/H32/H64：

1. 先用 `W* initialization only` 做低成本赛马；
2. top-2 候选进入完整 JDRQ；
3. 用 holdout 输出误差选择最终 transform + weight；
4. down-projection 单独给第二层 butterfly，其余层保持小候选池。

这叫 **successive halving**，能避免每个变换都跑完整求解。

### Phase 4：Activation gram64 / source-aware 动态量化

单独消融：

1. 仅 source-aware scale proposal；
2. 仅 gram64 hard-block refine；
3. 两者叠加；
4. 重新跑 JDRQ，因为更好的 Q(A) 会改变最优 Q(W)。

任何激活量化器升级后都必须重跑权重蒸馏，不能拿旧 `Q(W)` 直接比较。

### Phase 5：Attention 结构化搜索

按以下单变量递进：

1. 最终量化器 + 真实 causal/non-causal 输出统一选型；
2. per-head K center policy；
3. `D + single butterfly`；
4. `D + two-stage butterfly`；
5. V policy；
6. 一轮 QK/V 交替；
7. temperature/headroom 小网格。

---

## 8. 优化潜力与优先级

下面是机制级判断，不是承诺分数；正式数值必须由冻结 Qwen panel 和官方提交验证。不同模块收益重叠，不能直接相加。

| 优先级 | 方向 | 相对当前剩余误差潜力 | 官方分潜力粗估 | 置信度 | 原因 |
|---|---|---:|---:|---|---|
| P0 | structured/full-dual `W*` ceiling | Linear 连续上限已观测 4%～20%/层 | 诊断项，不单独计分 | 中高 | 已有本地实测；直接重新量化仅保留不到 1% |
| P0 | W* + 完整输出残差离散求解 | Linear 剩余误差降 8%～25% | +1200～3500 | 中 | 关键是让 mantissa/micro-exp 响应输出梯度 |
| P0 | 变换 T 与 JDRQ 交替优化 | Linear 再降 10%～30% | +1800～5000 | 中低 | 必须改变 Z，固定当前 Q(A) 的 ceiling 不够高 |
| P1 | 完整管道 successive-halving | Linear 再降 5%～15% | +600～2000 | 中 | 修复 proxy 与落地目标错配 |
| P1 | Attention 结构化 Q/K + 真实输出 | Attention 降 20%～45% | +1500～4000 | 中 | 外部 v2.3 单次结构升级已出现大增益 |
| P1 | V policy + QK/V 一轮交替 | Attention 再降 5%～20% | +500～1800 | 中低 | 无静态补偿，受校准泛化限制 |
| P2 | source-aware activation proposal | 动态重建降 3%～10% | +300～1200 | 中 | 同时可省时，为 JDRQ 提供更好 Z |
| P2 | gram64 hard-block 激活求解 | Linear 降 3%～12% | +400～1500 | 中低 | 受在线时间和 state 解释影响 |
| P3 | 更多 offset/coverage/固定阈值 | <3% | <500 | 高 | 已接近饱和，不能作为主线 |

建议分数里程碑：

1. **Phase 1 通过条件**：不是预设分数，而是至少多个模型/算子出现稳定 holdout ceiling；
2. **25k～29k**：完整离散 JDRQ 开始兑现连续 ceiling；
3. **29k～33k**：变换 T 与 JDRQ 交替优化、明显改善 `Z`；
4. **33k～36k**：Linear 和 Attention 两条结构主线都成功；
5. **36k+**：要求两侧同时接近各自上限，单靠 ridge 或 Linear 一侧不够。

若 Phase 1 的 `L(QX,W*)` ceiling 很高但量化 `W*` 后收益消失，瓶颈是 HiF4 离散求解；若 ceiling 很低，瓶颈是 activation transform/quantizer；这两个信号会直接决定下一步，不再靠盲目版本堆叠。

---

## 9. 时间与工程方案

### 9.1 420 秒内的预算原则

- `Y=A@W.T` 每 fold 只算一次；
- 岭回归用 M×M 对偶，不做 K×K 逆；
- `Z_b.T@Z_b`、`Z_b.T@R` 缓存并分块；
- row 分块 64/128，避免 `[candidate,N,block,64]` 爆内存；
- 先一阶 gain 排序，只精修 top row-block；
- candidate、offset、row 维向量化；
- top-2 transform 才跑完整 JDRQ；
- 校准可以较重，在线只保留结构化变换和小 beam；
- 先找精度信号，再压时间，不用旧 300 秒经验过早否决。

### 9.2 数值稳定性

- `ZZ.T` 用 FP32，ridge 下限与 trace 绑定；
- `torch.linalg.cholesky_ex` 失败则逐级加 damping；
- 对 `ΔW` 做 finite check 和 norm clip；
- 所有候选解码后再计算真实输出，不信任纯二次预测；
- mantissa 为 0 时 sign 强制归 0；
- 每次 block 更新后抽样核对增量公式与真实 loss 一致。

### 9.3 合规测试

新增至少这些回归：

1. JDRQ on/off 的 `activation_state` 字节一致；
2. 在线激活函数不读取教师输出派生对象；
3. 所有 HiF4 字段独立合法性校验；
4. 离散更新的预测 `ΔL` 与重算输出误差一致；
5. fold 行不重叠；
6. parent fallback 始终可选；
7. CPU state、无 NaN/Inf、嵌套深度和节点数通过官方 self-check；
8. 真实 250/200 流程总时间严格小于 420 秒。

---

## 10. 明确停止投入的方向

除非新诊断证明有新信号，否则暂停：

- 单纯把 FULL64 coverage 从 25% 调到 50%/100%；
- 继续扩大固定 E6M2 offset 池；
- 只改 headroom 百分比；
- 只增加 calibration row 数；
- 用更多硬阈值修补跨模型回退；
- 在旧 proxy 上继续搜索 CAT/Hadamard winner；
- 把旧组件机械叠加到 C69；
- 用本地 raw MSE 线性换算官方分；
- 根据官方单次分数反向硬编码模型/shape 分支。

这些方向可以作为 JDRQ 的子组件，但不能再担当主算法。

---

## 11. 下一轮最具体的执行清单

按顺序执行，不并行堆功能：

1. 在 evaluator 增加 `QX/W* ceiling` 七项误差分解；
2. 从当前冠军 C66，而不是实验性 C69/C70/C71，建立干净父版本；
3. 冻结 activation_state，缓存校准 `X、Z、Y`；
4. 实现 diagonal/block-4/8/16/64 structured regression，再加对偶 ridge；
5. 把各类 `W*` 送入现有 `_dense_to_hif4`，只作为 ceiling/量化鸿沟诊断，不期待它直接大幅提分；
6. 在 GPT-2 small、OPT、Qwen 三个冻结 panel 上报告 ceiling、train、holdout 和“连续→合法 HiF4”的损失；
7. 只要至少一类 structured target 有稳定 holdout ceiling，就实现 mantissa `±1` 输出残差坐标下降；
8. 再加 lv3/lv2 toggle 和 2 次 Gauss-Seidel；
9. 把 identity、当前变换、H16/H64 三个候选接入 successive-halving；
10. Linear 稳定后再启动 Attention 的 two-stage butterfly，不同时改两条主线；
11. 每个阶段记录精度—时间 Pareto，不以单个硬阈值提前结束机制探索；
12. 最终只提交通过合法性、合规不变量和 `<420s` 的版本。

最小 JDRQ 的伪代码：

```python
# calibration only
state = build_and_freeze_activation_state(X_calib, W)
Z = concat(dequant(dynamic_quantize(x_nvfp4, state)) for x in calib)
X = concat(dequant_nvfp4(x_nvfp4) for x in calib)
Y = X @ W.T

W0 = transformed_exact_weight(W, state)
R0 = Y - Z @ W0.T
Kdual = Z @ Z.T

best = current_weight_params
for lam in lambda_grid:
    C = solve(Kdual + lam * I, R0)       # [M, N]
    delta = C.T @ Z                      # [N, K]
    delta = clip_rows(delta)
    for eta in eta_grid:
        W_target = W0 + eta * delta
        params = dense_to_hif4(W_target)
        params = residual_discrete_refine(Y, Z, params)
        best = robust_holdout_select(best, params)

return {"weight_params": best, "activation_state": state}
```

---

## 12. 参考与证据

1. 赛事说明书：本仓库 `赛事说明书.txt`，包含 6 个接口、评分式、state 约束和 `A@W -> Q(A)` 禁令原文。
2. [HiFloat4 Format for Language Model Inference](https://arxiv.org/abs/2602.11287), 2026：HiF4 格式、三级 scale 和 HiGPTQ。
3. [MR-GPTQ: Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202), ICLR 2026：格式专用 scale search、static act-order、micro rotation。
4. GPTQ, ICLR 2023：Hessian 误差补偿。
5. QuIP/LDLQ：逐列误差反馈与二阶量化理论。
6. SmoothQuant, ICML 2023：激活/权重难度迁移。
7. QuaRot / SpinQuant：结构化旋转和学习式旋转。
8. AdaRound / BRECQ / QuantEase：输出重建与离散坐标下降。
9. [外部 `youxilee/hif4`](https://github.com/youxilee/hif4) v2.7：block-diagonal SmoothQuant、最终量化器 Attention 选型和 v2.6 X/W joint residual refine；它验证了方向，但不是 JDRQ 的上限。

---

## 最终判断

当前工程已经把“局部 HiF4 重建器”做得很深，继续在同一框架里增加 coverage、offset 和门控，只会得到越来越小且不稳定的增益。下一次算法跃迁必须来自重新定义 `Q(W)`：

> `Q(W)` 不是 W 的压缩副本，而是固定在线 `Q(A)` 下、为复现 `A@W` 教师输出而求得的合法 HiF4 离散回归器。

先用连续岭回归量化这个上限，再用完整层级离散残差搜索逼近它；同时把 Attention 改成真实 softmax 输出下的结构化策略优化。只有这两条路线，具备把剩余误差再压 50%～60%、接近 36000+ 的机制潜力。
