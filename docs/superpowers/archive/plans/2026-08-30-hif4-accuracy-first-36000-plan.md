# HiF4 36,000 Accuracy-First 详细优化方案

> 日期：2026-08-30  
> 状态：当前执行计划  
> 父版本：根目录 `solution.py`，clean BOAT + cross-fold Weight-HSDQ +
> Gram-hierarchy Activation-HSDQ + Attention deployed shortlist  
> 本阶段原则：**暂不以 420 秒或实现复杂度淘汰算法，只评估精度上限、泛化、
> HiF4 合法性和赛事合规性。** 时间压缩在算法方向证明后单独进行。  
> 官方合规边界：校准输出 `A@W` 只能优化离线 `Q(W)`；不得用于拟合、选择或
> 反推在线 `Q(A)`，也不得写入 `activation_state`。
> 补充审查：[`合法尺度晶格、CAT 与 Qronos 审查`](2026-08-30-hif4-grid-aligned-complement-plan.md)
> 已于 2026-08-30 部分采纳；其中原始 NVFP4 `Δ` 对齐被修正为变换后值域 GALS。

---

## 1. 当前基线与目标

固定 Qwen2.5-0.5B 全 24 层、`seq=128`、`calib=2`、`test=4`、`amax6`、CPU
缓存结果：

| 指标 | 当前值 |
|---|---:|
| Linear mean | 0.5015576125 |
| Attention mean | 0.8418285164 |
| Panel Linear | 125.389403 |
| Panel Attention | 168.365703 |
| Panel total | **293.755106** |
| native total | 417.862253 |

本地固定面板为：

```math
P=250g_L+200g_A
```

若以 `P=360` 作为 36,000 的本地诊断刻度，并保持当前 Attention：

```math
g_L^{360}
=\frac{360-200\times0.8418285164}{250}
=0.7665371869
```

当前 Linear 还差：

```math
\Delta g_L=0.7665371869-0.5015576125=0.2649795744
```

需要捕获当前 Linear 剩余误差的：

```math
\rho_L^{360}
=\frac{0.7665371869-0.5015576125}{1-0.5015576125}
=53.16\%
```

Linear `0.9` 是更强的迁移安全目标：

```math
\rho_L^{0.9}
=\frac{0.9-0.5015576125}{1-0.5015576125}
=79.94\%
```

若达到 `g_L=0.9` 且 Attention 不回退，本地 panel 为：

```math
P=250\times0.9+200\times0.8418285164=393.365703
```

所以本计划设置四级里程碑：

| 里程碑 | Linear mean | 当前 Attention 下的 panel | 含义 |
|---|---:|---:|---|
| M1 | 0.55 | 305.866 | 证明存在跨层结构增益 |
| M2 | 0.65 | 330.866 | 进入双侧联合改善区间 |
| M3 | 0.766537 | 360.000 | 达到本地 36,000 诊断刻度 |
| M4 | 0.90 | 393.366 | 为官方隐藏分布保留安全余量 |

以上均为本地研究刻度，不是官方分数承诺。

---

## 2. 当前瓶颈的定量归因

当前 Linear 分角色均值：

| q | k | v | o | fc_gate | fc_up | proj |
|---:|---:|---:|---:|---:|---:|---:|
| 0.616561 | 0.620526 | 0.563596 | 0.483463 | 0.375126 | 0.430255 | 0.421376 |

不能只修 `fc_gate/fc_up/proj`。即使这三个角色达到 `1.0`，其余角色保持当前值：

```math
g_L^{max(3)}
=\frac{0.616561+0.620526+0.563596+0.483463+3}{7}
=0.754878<0.766537
```

若 q/k/v 完全不变，则 o/gate/up/proj 四个角色平均必须达到：

```math
\bar g_{weak}
=\frac{7\times0.766537-(0.616561+0.620526+0.563596)}{4}
=0.891269
```

因此正确问题是：

```text
坐标系可量化性
  + 权重合法码域求解
  + 激活合法码域求解
  + 跨 64-block 相关性
  + 跨 fold 泛化
  + Attention 的剩余误差分担
```

而不是某一个角色增加更多 offset。

当前实现的主要结构缺口：

1. Weight-HSDQ 只在固定 hierarchy 中改 mantissa code；没有完整的
   E6M2/lv2/lv3/符号/mantissa 联合 beam。
2. Weight-HSDQ 每个 fold 只生成一个最终候选，没有保留渐进路径；完整 polish
   过拟合时，中间的更优合法候选也被丢弃。
3. `rows > 2 * channels` 的扩张 FFN 权重完全跳过 Weight-HSDQ，恰好覆盖最弱的
   `fc_gate/fc_up`。
4. Activation-HSDQ 只保存 64×64 block-diagonal Gram，缺少跨 64-block
   相关性。
5. BOAT 只搜索全层统一 diagonal alpha 和固定 signed-Hadamard；没有 blockwise
   balance、hierarchy-aware permutation 或低秩可逆变换。
6. Attention 已有真实部署输出复评，但 `v_state={}`，V 没有位置/概率敏感的
   refinement。

---

## 3. 总体算法路线

精度优先阶段按以下依赖顺序推进：

```text
D0 合法上限仪表盘
  -> D0-G 全 255 E6M2 scale-lattice oracle
  -> A0 GALS：变换后值域合法尺度晶格搜索（仅在 oracle gap 足够大时）
  -> A1 Progressive Cross-Fold HSDQ
  -> A2 Expansive-FFN Shrinkage HSDQ
  -> A3 LRH：跨 64-block 低秩 Hessian
  -> A4 BOAT-2：blockwise 可逆坐标变换
  -> A5 FS-JDRQ：冻结 Q(A) 后的稳健 Q(W) 联合重构
  -> A6 Global Activation-HSDQ
  -> B1 GQRB/FASA 扩展
  -> B2 PAWV：V 路径优化
  -> C0 全模型、全角色、宽形状综合选择
```

原则是先证明“合法求解器能兑现收益”，再增加更强连续目标；否则更复杂的
Hessian/JDRQ 只会产生无法投影到 HiF4 的连续上限。

---

## 4. D0：合法上限与误差仪表盘

### 4.1 目的

在修改主算法前，逐层回答三个问题：

1. 当前坐标系内，合法 HiF4 最多还能提升多少？
2. 损失发生在连续目标、合法投影还是跨 fold 泛化？
3. 哪些 role/layer/block 值得进入后续高成本算法？

### 4.2 覆盖矩阵

模型：

- Qwen2.5-0.5B：主模型；
- GPT-2 small/medium：MHA 与不同深度；
- OPT-125M：独立 q/k/v 和不同激活统计；
- Pythia-160M：fused QKV、RoPE；
- synthetic wide：hidden `4096/5120/6144`、FFN `out>in` 与 `out<in`、
  GQA `32:8/32:4`。

层：首层、中层、末层，后续扩展到全层。

角色：`q/k/v/o/fc_gate/fc_up/proj`。

fold：至少 `train-A / train-B / validation` 三份独立窗口。

### 4.3 每个 case 保存的指标

```text
parent deployed loss
weight-float / activation-quantized oracle
weight-quantized / activation-float oracle
both-float oracle
continuous ridge target loss
nearest legal hierarchy loss
progressive HSDQ loss
held-out HSDQ loss
cross-block LRH loss
accepted code count
changed block count
fold disagreement
local ±3 scale loss
all-255 legal E6M2 scale oracle loss
scale-lattice gap
continuous CAT alignment gain
CAT continuous-to-legal retention
role/layer/shape metadata
```

定义连续到合法的保留率：

```math
R_{cont\to legal}
=\frac{L_{parent}-L_{legal}}
{L_{parent}-L_{continuous}+\epsilon}
```

定义校准到验证的迁移率：

```math
R_{transfer}
=\frac{L_{val,parent}-L_{val,candidate}}
{L_{train,parent}-L_{train,candidate}+\epsilon}
```

### 4.4 产物

新增建议：

```text
evaluator/linear_oracle_dashboard.py
artifacts/oracle_dashboard/<candidate>-<model>.json
logs/evaluations/<date>-oracle-dashboard.md
```

D0 不改变 `solution.py`，只负责建立可证伪的上限地图。

### 4.5 D0-G：合法 E6M2 尺度晶格 oracle

当前 `_encode_rows` 只搜索标准 `amax/7` code 附近 `±3`。对 sampled 64-block，
枚举 evaluator 接受的全部 255 个有限无符号 E6M2 code，并在每个 code 下求合法
lv2/lv3/mantissa 解：

```math
L_{all255}(v)=
\min_{c\in\{0,\ldots,254\}}
\min_{z\in\mathcal Z_{HiF4}(c)}\mathcal L(v,z)
```

定义：

```math
g_{scale}=
\frac{L_{\pm3}-L_{all255}}
{L_{\pm3}+\epsilon}
```

`g_scale` 是固定变换、固定目标下仅扩大顶层 scale/hierarchy 搜索可获得的硬诊断。
若 validation 上很小，停止尺度搜索，直接进入 A2/A3/A4；若明显且跨 fold 可迁移，
才实现 A0 GALS。

E0-G 已执行（Qwen 一层、当前 BOAT 后前 32 行、完整 255-code，结果归档于
`logs/execution/2026-08-30-e0g-scale-oracle.md` 和
`artifacts/oracle_dashboard/e0g-qwen-layer1.json`）。结果显示：`fc_gate/fc_up/proj`
的总 gap 均低于 `0.1%`；`v` 的 weight-MSE gap 为 `0.0313%`，activation-`gram64`
总 gap 为 `0.6302%`，60/448 blocks 改善，最大单 block `19.321%`。因此 GALS
只保留为 `v` 高损 block 的候选插件，不再做全局实现；E1 只作为一次性诊断实验。

原始 `denom/Δ` 只在 identity、纯置换或可证明保留网格的特殊变换中记录归因。
当前 BOAT 含对角缩放和 signed-Hadamard，变换后元素通常不再属于单一
`E2M1×Δ` 集合，禁止以原始 `Δ` 命中率作为主算法依据。

### 4.6 A0：GALS 解析候选

对实际待编码的变换后非零值 `|v_i|`，枚举：

```math
s_{i,m,e}=\frac{|v_i|}{m2^e},\qquad
m\in\{1/4,2/4,\ldots,7/4\},\ e\in\{0,1,2\}
```

把 `s_{i,m,e}` 投影到最近合法 E6M2 code，并加入相邻 code、当前 `±3` 邻域和
incumbent。候选去重后必须按完整 64 元素块评分。每 8 元素共享 lv2，因此两个
4 元素子组的有效指数只能取：

```math
\mathcal E_8=
\{(0,0),(0,1),(1,0),(1,1),(1,2),(2,1),(2,2)\}
```

不得独立选两个 4 元素组后做“块级投票”。逐元素加权 MSE 可复用
`_solve_hierarchy`；Gram/Hessian 非对角目标必须交给 A1 beam 联合求解。

GALS-C 已在 sampled blocks 上追回 GALS-O 的全部增益，并在 v role 的两折与四个
validation 窗口保持同方向；但将它接入每行前 4 高损 block 的稀疏部署版后，
Qwen layer-1 panel 从 v100 的 `336.037091` 降至 `335.988995`，Linear mean
降至 `0.602878`，API 增加 `41.37s`。因此解析候选保留为 oracle 证据，部署版
归档，不进入主代码；若重启必须增加 role/state 标识再做 v-only 插件。

---

## 5. A1：Progressive Cross-Fold Hierarchical HSDQ

### 5.1 核心问题

当前 `_polish_weight` 在一个 calibration fold 上执行完整坐标扫描，只输出最终
candidate。历史 HSDQ-1 已证明这种做法会降低 calibration product loss，却在
独立 test 上回退。需要把“生成路径”和“选择路径”分离。

### 5.2 完整合法候选

每个 64 元素 block 的变量为：

```math
q=s_1\,s_2\,s_3\,m\,\sigma
```

其中：

- `s1`：E6M2 顶层 scale；
- `s2`：8 个 lv2；
- `s3`：16 个 lv3；
- `m`：64 个 `{0, 1/4, ..., 7/4}` mantissa；
- `σ`：64 个符号。

不再只固定 `s1/s2/s3` 后修改 mantissa，而是构造分层 beam：

```text
parent hierarchy
  -> top-scale candidates（当前邻域 + 通过门禁的 GALS-C）
  -> lv2 partial assignments
  -> lv3 partial assignments
  -> signed mantissa coordinate path
```

### 5.3 渐进候选路径

fold A 生成：

```text
C_A = {
  parent,
  hierarchy-only,
  accepted moves = 1,
  accepted moves = 4,
  accepted moves = 16,
  accepted moves = 32,
  accepted moves = 64,
  second-block checkpoints
}
```

fold B 只评价，不参与生成。然后交换 A/B。

稳健目标：

```math
J(c)=
\frac{L_A(c)+L_B(c)}2
+\beta\max[L_A(c),L_B(c)]
+\gamma|L_A(c)-L_B(c)|
+\lambda\|Q_W(c)-Q_W(parent)\|_{D_H}^2
```

初始研究网格：

```text
beta ∈ {0.25, 0.5, 1.0}
gamma ∈ {0.5, 1.0, 2.0}
lambda ∈ {0, 1e-4, 1e-3, 1e-2}
beam ∈ {2, 4, 8, 16}
active blocks ∈ {1, 2, 4, 8, all}
```

本阶段不按耗时删网格，但每次只改变一个结构变量，避免无法归因。

### 5.4 精确二次增量

固定 activation `Z`，权重输出误差为：

```math
L(Q_W)=\|Z(W-Q_W)^T\|_F^2
```

令 residual：

```math
R=Z(W-Q_W)^T
```

某输出行的合法变化为 `δ`，则：

```math
\Delta L
=-2\langle Z\delta^T,R_r\rangle
+\|Z\delta^T\|_2^2
```

候选比较全部用这个精确增量，不使用 representation MSE 代替部署 product loss。

### 5.5 双向准入

候选至少满足其一：

1. A 生成、B 验证为正，且反向存在同 hierarchy family 的正候选；
2. 三 fold 的 CVaR/均值混合目标优于 parent；
3. validation 改善且 fold disagreement 显著低于无正则 candidate。

parent 永远保留，不要求每一行都改变。

### 5.6 首轮实验范围

按以下顺序：

1. Qwen `fc_gate`：首/中/末三层；
2. Qwen `fc_up`；
3. Qwen `v`；
4. Qwen `proj/o`；
5. q/k；
6. 全层 Qwen；
7. 其他模型 guardrail。

### 5.7 继续条件

- legal realization 相对当前 solver 至少提升到 `20%`；
- validation product loss 与 panel 同方向；
- 全层 Linear mean 至少出现 `+0.01` 的结构增益，或明确识别出单 role
  `+0.03` 以上的可迁移增益；
- 不出现 OPT/Qwen 一正一灾难性回退。

如果训练目标大幅下降而 validation 不改善，停止扩大 beam，转 A4 BOAT-2。

---

## 6. A2：Expansive-FFN Shrinkage HSDQ

### 6.1 动机

当前代码对：

```text
rows > 2 * channels
```

直接跳过 Weight-HSDQ。Qwen 的 `fc_gate/fc_up` 因此缺少权重侧精修，而它们正是
最低分角色。直接对全部输出行完整 polish 又会产生过多自由度。

### 6.2 稀疏 row-block 选择

对每个输出行 `r` 和 block `b` 计算 held-out gain：

```math
v_{rb}
=\frac{L_{B,parent}^{(r)}-L_{B,candidate}^{(r)}}
{L_{B,parent}^{(r)}+\epsilon}
```

只接受：

```math
v_{rb}>0
```

且属于全矩阵 top percentile 的 row-block 对。研究网格：

```text
accepted row-block ratio ∈ {0.5%, 1%, 2%, 5%, 10%, 25%, 100%}
```

### 6.3 层级收缩

对于候选变化 `ΔW`，使用经验 Bayes 式收缩：

```math
\Delta W_{shrunk}=\eta_{rb}\Delta W,
\qquad
\eta_{rb}=\operatorname{clip}
\left(
\frac{g_{heldout}}{g_{train}+\epsilon},0,1
\right)
```

由于最终必须落在合法 HiF4，`η` 不直接插值数值，而是选择渐进 HSDQ 路径上
最接近该收缩幅度的合法 checkpoint。

### 6.4 验收

- `fc_gate/fc_up` 均值必须分别报告；
- 不允许只看全 Linear 平均掩盖某一扩张 FFN 灾难回退；
- 保留 parent、稀疏、全量三组对照；
- 统计 gain 随 row-block ratio 的完整曲线，不在本阶段考虑耗时。

---

## 7. A3：LRH——跨 64-block 低秩 Hessian

### 7.1 当前缺口

Activation-HSDQ 使用：

```math
G\approx\operatorname{blockdiag}(G_1,\ldots,G_B)
```

忽略不同 64-block 之间的相关性。Weight-HSDQ 虽通过全局 residual 间接看到其他
block，但 active block 数很少，没有显式建模全局低秩方向。

### 7.2 分解

对 activation Hessian：

```math
H=Z^TZ
```

使用：

```math
H\approx B+UU^T,
\qquad
B=\operatorname{blockdiag}(H_1,\ldots,H_B)
```

`U` 由 `H-B` 的 randomized eig/SVD 得到。研究 rank：

```text
r ∈ {4, 8, 16, 32, 64}
```

### 7.3 精确候选项

对跨块变化 `δ`：

```math
\delta^TH\delta
\approx
\sum_b\delta_b^TH_b\delta_b
+\|U^T\delta\|_2^2
```

低秩项允许用 `O(rn)` 增量更新，而不需要保存完整 `n×n` Hessian。

### 7.4 求解顺序

1. 用 block-diagonal HSDQ 产生每个 block 的 top-K 合法候选；
2. 计算候选的 `U^Tδ`；
3. 用 beam/动态规划组合多个 block；
4. 用精确 calibration product loss 终验；
5. 交叉 fold 选择。

组合 beam：

```text
per-block candidates ∈ {2, 4, 8}
global beam ∈ {4, 8, 16, 32, 64}
```

### 7.5 关键消融

必须区分：

- blockdiag HSDQ 增加同等 candidate 数；
- blockdiag + 随机 rank；
- blockdiag + 数据低秩 U；
- 完整 Hessian 小层 oracle。

只有数据低秩 U 在相同候选预算下优于 blockdiag，才能证明 LRH 捕获了真正的
跨块结构。

---

## 8. A4：BOAT-2——角色/块级双侧坐标变换

### 8.1 目标

当更强 HSDQ 仍无法把连续收益投影到合法 HiF4 时，说明需要改变坐标系本身。

对每个 64-channel block 使用：

```math
T_b=D_bP_bH_bR_b
```

其中：

- `D_b`：blockwise diagonal balance；
- `P_b`：hierarchy-aware permutation；
- `H_b`：4/8/16/32/64 signed-Hadamard；
- `R_b`：1–4 个 Householder reflector 或低秩可逆修正。

连续等价关系：

```math
X'=XT^{-1},
\qquad
W'=WT^T,
\qquad
X'W'^T=XW^T
```

### 8.2 blockwise balance

当前全层统一 `alpha` 改为每个 64-block：

```math
d_{bj}(\alpha_b)
=\left(
\frac{RMS(X_{bj})}{RMS(W_{:,bj})}
\right)^{\alpha_b}
```

```text
alpha_b ∈ {0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0}
```

使用两个 calibration fold 的 operand-local/Gram-safe 目标选择，不使用 `A@W`
选择 activation state。

### 8.3 hierarchy-aware permutation

HiF4 的 lv2/lv3 固定按 8/4 元素分组。构造 permutation，使：

- 极端值分散到不同 lv2/lv3 子组；
- 相近动态范围集中共享 scale；
- activation 与 weight 的高能通道不同时挤在一个 hierarchy group；
- permutation 对 X/W 同步应用，保持乘积等价。

候选：

```text
identity
sort by activation RMS
sort by weight RMS
sort by geometric mean RMS
interleave high/low
balanced bin packing into 8 lv2 groups
```

### 8.4 低秩可逆变换

以 covariance/CAT 解的主方向初始化 Householder：

```math
H(v)=I-2\frac{vv^T}{v^Tv}
```

候选 rank：`1/2/4/8`。本阶段可离线完整扫描，不因推理成本拒绝；但必须报告
条件数、逆变换误差和合法 HiF4 部署结果。

CAT 初始化必须使用正则化二阶矩：

```math
A_\epsilon=A+\epsilon\frac{\operatorname{tr}(A)}dI,
\qquad
B_\epsilon=B+\epsilon\frac{\operatorname{tr}(B)}dI
```

在列向量约定下令：

```math
C=A_\epsilon^{1/2}B_\epsilon A_\epsilon^{1/2},
\qquad
T_{CAT}=C^{1/4}A_\epsilon^{-1/2}
```

于是 `T A T^T` 与 `T^{-T} B T^{-1}` 同为 `C^{1/2}`。乘法顺序不可交换；若
代码使用行向量约定，必须整体转置推导。CAT 原理论不是为 HiF4 层级量化推导的，
所以这里只把它当作连续初始化和 alignment oracle；最终仍需投影为合法、可逆的
BOAT-2 结构并通过跨 fold 部署目标。

### 8.5 role-aware 策略

优先顺序：

1. `fc_gate/fc_up`：blockwise D + permutation；
2. `v/o/proj`：D + permutation + Householder；
3. q/k：只有跨模型一致时才扩展。

不得使用模型名称门控；最终策略只能依赖 shape、role、RMS、kurtosis、block
condition number 等可复现统计量。

---

## 9. A5：FS-JDRQ——冻结激活状态后的权重联合重构

### 9.1 合规顺序

```text
先仅用 operand-local/静态 W Gram 选择 BOAT 与 activation_state
  -> 冻结 activation_state
  -> 得到 Z=Q(A)
  -> 允许用 Y=A@W 优化离线 Q(W)
  -> 输出信息只进入 weight_params 候选选择
```

任何 `Y`、residual、target 或 candidate score 都不得进入 `activation_state`。

### 9.2 连续权重目标

固定 `Z=Q(A)`，教师输出：

```math
Y=AW^T
```

连续目标：

```math
\widetilde W^T
=(Z^TZ+\lambda I)^{-1}Z^TY
```

研究 ridge：

```text
lambda / trace(ZTZ) ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
```

### 9.3 不直接量化连续解

历史 JDRQ 失败说明“生成更好的连续 W”不足。FS-JDRQ 只负责提供多个低自由度
target：

```math
W_\eta=(1-\eta)W+\eta\widetilde W,
\qquad
\eta\in\{0,1/16,1/8,1/4,1/2,3/4,1\}
```

每个 `W_eta` 必须通过 A1/A3 的合法 HSDQ/LRH 投影，再进入跨 fold selector。

### 9.4 稳健选择

使用三 fold CVaR：

```math
J_{FS}(c)
=\operatorname{mean}_fL_f(c)
+\beta\max_fL_f(c)
+\gamma\operatorname{std}_fL_f(c)
```

保留 parent、不同 eta、不同 ridge、不同 HSDQ path checkpoint。禁止直接启用
无 parent 回退的多轮 residual Gauss--Seidel。

### 9.5 失败判据

若 continuous target 明显改善但所有合法投影都不迁移，问题归因到 HSDQ/BOAT，
不继续增加 JDRQ target 数。如果合法投影在单模型有效、异构模型灾难回退，则提高
fold/shape 正则，而不是增加模型名分支。

### 9.6 HiF4-block Qronos correction

Qronos 式纠错只进入冻结激活状态后的权重求解：先得到 `Z=Q(A)`，再用
`(A,W,AW^T,Z)` 调整离线 `weight_params`，目标是使 `Z Q(W)^T` 接近 `AW^T`。
任何 teacher output 或 residual 都不得写入 `activation_state`。

原始逐标量顺序舍入需改造成 HiF4 block 版本：GALS 生成顶层 scale 候选，
A1/HSDQ 生成合法 hierarchy/mantissa 候选，Qronos/Cholesky correction 只负责
块顺序与连续 residual diffusion。每步保留 parent，并用真实部署 residual 接受。
本赛题不采用跨层误差传播，因为 evaluator 为每层提供未级联量化的真实激活。

---

## 10. A6：Global Activation-HSDQ

### 10.1 合规目标

在线激活量化只能使用当前 activation、冻结变换和静态浮点权重信息。使用：

```math
G_W=W^TW
```

或其 blockdiag + low-rank 分解。禁止用 `Q(W)^TQ(W)` 的 cross-residual 拟合
Q(A)，也禁止使用 `A@W`。

### 10.2 全局目标

```math
L_A(q)=(q-x)^TG_W(q-x)
```

分解：

```math
G_W\approx B_W+U_WU_W^T
```

使用 A3 相同的 block candidate + global beam 组合，研究 rank `4–64`。

### 10.3 sample-local candidate

动态 activation 每一行独立生成：

- parent hierarchy；
- hierarchy beam；
- blockdiag HSDQ；
- blockdiag + low-rank global HSDQ。

最终以静态 `G_W` 二次型选择。状态仅保存 `B_W/U_W`、BOAT 参数和合法整数配置。

### 10.4 验收

- `tests/test_linear_compliance_guard.py` 全绿；
- state 序列化后不含 calibration output/residual；
- float-W Gram 与 quantized-W Gram 做明确隔离；
- q/k/v/o/FFN 分角色报告，尤其检查 activation 侧是否修复 `v/fc_gate`。

---

## 11. Attention 路线

Linear 是主线，但 Attention 从 `0.841829` 提升到 `0.90`，可以把达到 panel 360
所需的 Linear 从 `0.766537` 降到：

```math
g_L=\frac{360-200\times0.9}{250}=0.72
```

因此 Attention 仍有明确的目标分担价值。

### 11.1 B1：GQRB-2

当前 reciprocal RMS 是逐通道 diagonal。扩展到每个 GQA group 内的 2×2/4×4
可逆 block：

```math
Q'=QT,
\qquad
K'=KT^{-T}
```

候选从 Q/K covariance 的广义特征方向初始化；identity 永远保留。部署复评仍用
真实 causal Attention 输出，不恢复已失败的 softmax Fisher 非对角 Hessian。

### 11.2 B2：FASA shortlist 扩展（已执行）

保留当前两阶段结构：

```text
cheap proxy -> top-K -> full deployed output MSE
```

本阶段不考虑时间，可把 top-K 扩展到全部候选，以测量 shortlist 截断损失；候选族
包含 reciprocal balance、K-centering、shared Hadamard、GQRB block 和组合候选。
B1 已采用 parent top4 + GQRB top4 的 margin gate；B2 在此 shortlist 上加入
PAWV V refinement。rank-8 跨 token 版本 layer-1 panel `334.101693` 未通过门禁，
diag-only 版本全层 panel `293.797301`、API `392.42s`，已接受为 v100。

### 11.3 B3：PAWV（本轮已执行 B2 子集）

当前 V 直接 `_dense_to_hif4`，没有 refinement。由校准 Attention probability
`P` 计算 token 敏感度：

```math
w_t=\sum_{h,q}P_{hqt}^2
```

精确 V 误差：

```math
\|P(Q_V-V)\|_F^2
=\operatorname{tr}[(Q_V-V)^TP^TP(Q_V-V)]
```

首版依次测试：

1. `diag(P^TP)` 只用于选择需要 HSDQ 的 token rows；
2. position bucket / attention-sink 预算；
3. `diag + low-rank(P^TP)` 的跨 token HSDQ；
4. 与 Q/K candidate 交替选择 2–4 轮。

单纯给每行 loss 乘标量不会改变该行量化 argmin，所以 PAWV 必须作用于 refinement
预算、跨 token low-rank 项或 Q/K/V 联合选择，不能只增加一个无效 importance
数组。

---

## 12. 实验序列与停止规则

| 实验 | 唯一变量 | 首测范围 | 主要问题 | 执行状态 | 通过后下一步 |
|---|---|---|---|---|---|
| E0 | D0 dashboard | 4 模型×3 层×全 role | 上限在哪里 | 已补齐第 1 层五模型 oracle；三层全量仍非必要 | 记录跨模型 gap，停止全局 scale 扩张 |
| E0-G | all-255 scale-lattice oracle | Qwen gate/up/v sampled blocks | `±3` 是否漏掉大量合法 scale 收益 | 已完成；gap 不支持全局 GALS | 停止全局 scale 扩张 |
| E0-C | GALS 解析候选召回 | E0-G 高 gap blocks | 稀疏候选能否追回 oracle | 已执行：四角色召回 `1.0`；稀疏部署 layer-1 `335.988995`，回退 | 仅保留 oracle，停止部署版 |
| E1 | progressive full-hierarchy HSDQ | Qwen gate/up/v/proj | 强 solver 是否迁移 | 已执行并拒绝：panel `290.923906` | 归档，恢复 parent |
| E2 | expansive FFN shrinkage | Qwen gate/up | 能否解除 rows gate | 已执行并拒绝：panel `292.831952` | 停止该 row solver |
| E3 | LRH rank | v/gate/up/proj | 跨块是否重要 | 已执行并拒绝：true cross-block rank-8，全层 panel `292.426982` | 停止扩大 LRH |
| E4 | blockwise D/P | v/gate/up | 坐标系是否主瓶颈 | 已执行并拒绝：blockwise schedule 与 CAT/Householder 全组合均回退 | 仅保留 BOAT 基线 |
| E5 | Householder/CAT low-rank | o/v/proj | 正则化 alignment 初始化能否合法兑现 | 已执行并拒绝：full CAT/BOAT-2，全层 `283.159693` 且超时 | 不写入部署主线 |
| E6 | FS-JDRQ + block-Qronos | 全 Linear role | 冻结 Q(A) 后联合纠错能否合法兑现 | 已执行并拒绝：frozen-Q(A)/ridge/Qronos 持平但超时 `455.73s` | 停止 joint candidate |
| E7 | global activation LRH | 全 Linear role | 激活跨块上限 | 已执行并拒绝：Global LRH 全层 `282.616646` | 停止扩大 LRH |
| E8 | GQRB/PAWV | Attention | 把 A 提到 0.90+ | 已执行：B1 GQRB margin + B2 PAWV diag-only；当前 Attention mean `0.842039` | C0 已确认 |
| E9 | 全层五模型 + wide shape | 全部 | 泛化与最终组合 | 已执行：C0 五模型；Qwen panel `293.797301`，主 API `401.13s` | 取 v100 为当前最高版本 |
| E10 | 量化后权重 Gram 激活 Hessian | 全 Linear role | `W_qᵀW_q` 是否比浮点 `WᵀW` 更贴近部署输出 | 已执行并拒绝：layer-1 `336.562922`，full `290.226694`，API `470.58s` | 归档，不进入主线 |

停止规则：

1. **内部目标下降、validation 不变或回退**：停止扩大同类搜索，转坐标变换。
2. **Qwen 正向、OPT/Pythia 灾难回退**：增加 fold/shape 正则，不使用模型名门控。
3. **连续 oracle 高、合法 oracle 低**：优先 HSDQ/BOAT。
4. **合法 oracle 高、部署迁移低**：优先 cross-fold/shrinkage。
5. **所有合法 oracle 都低**：该坐标系已接近上限，必须转 BOAT-2。
6. **Attention 到 0.90 后边际很小**：计算和研究预算全部转回 Linear。

---

## 13. 统一验收指标

### 13.1 主指标

```text
Qwen shaped panel total
Qwen Linear mean
Qwen Attention mean
per-role Linear mean
remaining-error capture
```

候选对 parent 的剩余误差捕获率：

```math
capture_{local}
=\frac{P_{candidate}-P_{parent}}
{450-P_{parent}}
```

当前 parent 到 360 需要：

```math
\frac{360-293.755106}{450-293.755106}=42.40\%
```

### 13.2 泛化指标

- train fold mean/max/std；
- validation loss；
- fold disagreement；
- GPT-2/OPT/Pythia 同方向率；
- wide FFN/GQA shape 结果；
- role/layer 胜率和最差回退。

其他模型是软 guardrail，不要求全部逐 case 正向；但任何单模型灾难性回退必须
解释并通过统计量规则解决。

### 13.3 合法性与合规

- HiF4 五字段 shape/dtype/value 合法；
- 六个 API 不变；
- activation state 不含 output/residual/teacher target；
- dynamic Q(A) 不读取校准输出；
- 所有 transform 可逆且连续乘积误差接近浮点舍入误差；
- NaN/Inf/zero/极端 scale case 有 parent fallback。

### 13.4 本阶段明确不设的门

- 不以 420 秒淘汰；
- 不限制 state 大小；
- 不限制 beam/rank/candidate 数；
- 不因为 Python 实现慢而否定数学方向；
- 不把本地 panel 线性换算成官方绝对分数。

所有实验仍要记录时间和内存，只是不作为本阶段接受/拒绝条件，以便后续压缩。

---

## 14. 代码实施边界

### 14.1 建议新增的诊断代码

```text
evaluator/linear_oracle_dashboard.py
evaluator/hif4_candidate_replay.py
evaluator/e6m2_scale_lattice_oracle.py
tests/test_hsdq_progressive.py
tests/test_gals_hierarchy.py
tests/test_lrh_hessian.py
tests/test_boat_invariance.py
tests/test_cat_transform.py
tests/test_activation_state_provenance.py
tests/test_qronos_state_boundary.py
tests/test_pawv_objective.py
```

### 14.2 `solution.py` 预期接入点

| 机制 | 当前函数 | 新函数建议 |
|---|---|---|
| E6M2 oracle/GALS | `_encode_rows`、`_solve_hierarchy` | `_all_scale_oracle`、`_gals_scale_codes` |
| Progressive HSDQ | `_polish_weight` | `_generate_hsdq_path`、`_crossfold_select_path` |
| Expansive FFN | `_crossfold_weight_hsdq` | `_select_sparse_row_blocks` |
| LRH | `_gram64` | `_lowrank_cross_block_gram`、`_lrh_candidate_delta` |
| BOAT-2/CAT | `_choose_boat` | `_boat_block_candidates`、`_hierarchy_permutation`、`_cat_initializer` |
| FS-JDRQ/Qronos | weight calibration | `_ridge_weight_targets`、`_qronos_block_path`、`_robust_weight_pool` |
| Global A-HSDQ | `_refine_activation` | `_refine_activation_lrh` |
| GQRB | attention calibration | `_gqrb_block_candidates` |
| PAWV | `v_state={}` | `_build_pawv_state`、`_refine_v_pawv` |

研究阶段可以先把算法拆到 evaluator-side prototype 验证；只有机制通过全层和多模型
验证后，才合并进根 `solution.py`，避免重新形成包含大量 dormant branch 的主文件。

### 14.3 实验纪律

每个实验必须记录：

```text
parent SHA
unique change
mathematical target
candidate space
fold split
oracle result
legal result
deployed result
per-role result
heterogeneous guardrail
runtime/memory（只记录、不裁决）
compliance result
decision
next falsifiable experiment
```

被拒绝候选归档到 `solutions/`，不得以关闭 flag 的形式留在主代码。

---

## 15. 当前执行状态与下一步

E0-G 已完成并已归档；D0 多模型第 1 层 dashboard 已补齐，结论仍是 scale gap
亚百分比且没有跨模型统一 role。E0-C 解析候选在四角色上追回全 255-code oracle，但稀疏
部署版 layer-1 panel `335.988995`、Linear `0.602878`，较 v100 回退并增加
`41.37s`，因此仅保留为 oracle 证据，不写入主代码。
E1 已按计划实现并完成一层/全层门禁，结果已归档到
`logs/execution/2026-08-30-e1-progressive-hsdq.md`。它在一层样本上提升
`+2.591832` panel，但在 24 层 Qwen 上回退 `−2.831200` panel，Linear mean/gain 由
`0.501558` 降为 `0.490233`，且运行时间从 `382.15s` 增至 `693.21s`，超过 420s
门禁；因此 E1 明确拒绝，主代码恢复到 parent SHA256
`5D1128CC79FEF58154DA2F600EC4B472FF95030E1F1E61B96593D06FD9AAC94F`。

下一步不再把未经验证的算法留在主线。A2/A3 的 expansive FFN HSDQ、A4
blockwise BOAT-2 与 A5 joint-fold A@W 均已完成并归档；最高仍是 stable parent
panel `293.755106`。A5 单层虽达 `337.501045`，全层降至 `284.595177`，说明
仅扩大 calibration product 目标会破坏跨层迁移。随后首次实现真正跨 block 的
LRH-r8（最多 4 个 64-block、rank-8 off-block Hessian），单层 `334.245422`、
全层 `292.426982`，仍比 parent 低 `1.328124`，因此同样归档到
`logs/execution/2026-08-30-a3-lrh-r8.md`，不再扩大这条实现。

```text
当前动作：官方测评不可用，继续以固定 Qwen panel 作为本地门禁。完整 BOAT-2/CAT-Householder 组合已执行但全层 panel
`283.159693`、API `600.61s`，比 parent 低 `10.595413` 且超时，已归档；下一项
冻结 Q(A) ridge/Qronos 已执行：全层 panel 与 parent 持平但 API `455.73s` 超过
420s，已归档。Global Activation-LRH 也已执行（rank-8、10% energy），全层 panel
`282.616646`，较 parent 低 `11.138460`，已归档。B1 GQRB margin 先通过本地
门禁（panel `293.793700`、API `406.24s`），随后 B2 PAWV diag-only 也通过本地
门禁（panel `293.797301`、API `392.42s`，较 B1 `+0.003601`），当前根已切换到
v100；C0 五模型确认已完成，Qwen 主模型 panel 保持 `293.797301`、API
`401.13s`，四个软 guardrail 无精度灾难回退；gpt2-medium 仅出现 `492.64s`
的软 guardrail 时间超限。
所有候选必须满足 panel 不降、
Linear 不降、runtime ≤ 420s，失败即归档恢复 parent；每一步记录最高分版本和
提交号，不把未执行项标成已完成。
```

这一步的结论是：当前最高可信本地版本是 v100（B2 PAWV diag-only + B1 GQRB），
C0 已确认其五模型稳健性；追加 A7 量化后权重 Gram 仍在全层回退，因此仍不宣称已达到 36000。下一步取所有本地候选的最高
分版本，待官方恢复再建立兑换率；若继续研究，优先针对 Linear 的跨模型泛化和
gpt2-medium 运行时。所有已执行实验均保留 parent、fold loss、changed blocks
和完整运行时间，不删除历史证据。
