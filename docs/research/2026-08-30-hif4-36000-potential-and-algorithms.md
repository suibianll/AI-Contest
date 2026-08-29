# HiF4 Linear / Attention 优化潜力、36,000 分差距与具体算法

> 日期：2026-08-30  
> 适用工程：2026 华为算法大赛 NVFP4 → HiF4 赛题  
> 分析对象：当前根 `solution.py`、C66 官方冠军、C72–C75 JDRQ/Gram64 研究分支  
> 文档性质：算法设计与实验决策报告，不是官方分数承诺  
> 规则边界：输出必须为合法 HiF4 五字段；Linear 不允许用 `A@W` 拟合、选择或反推在线 `Q(A)`；官方总时间 `<420s`

---

## 1. 结论摘要

### 1.1 当前分数差距

已确认的新版官方评测由 250 个 Linear case 和 200 个 Attention case 组成。

| 方案 | 官方分数 | 距离 36,000 | 时间 |
|---|---:|---:|---:|
| v066 / C66，本仓库合规官方冠军 | 22,557 | 13,443 | 217.2s |
| `youxilee/hif4` 外部参考 | 24,153 | 11,847 | 239s |
| 当前 C75 研究版本 | 尚无官方分 | 尚不能确认 | 本地 Qwen API 约 180s |

当前 C75 的 Qwen 本地 shaped panel 约为 `242.505`，C66 为 `238.282`。按已有官方锚点做非正式回归只能得到约 23,000–23,300 分的粗略量级；由于本地数据和官方隐藏数据分布不同，该数值不能写成 Official Score。

### 1.2 核心判断

1. 36,000 分的绝对数学上限条件并未被排除；450 个 case 的数学满分是 45,000。
2. 从官方 22,557 到 36,000，需要消除当前剩余归一化 MSE 的约 **59.90%**。
3. 当前 C75 相对 C66 只消除了约 2% 的本地 proxy 剩余误差，增长斜率远不足以自然外推到 36,000。
4. Linear 约占当前本地代理剩余误差的 67%，是第一主战场；Attention 约占 33%，也不能继续保持冻结。
5. 当前最明显的结构性瓶颈不是连续优化目标不足，而是连续目标到合法 HiF4 五字段之间的离散兑现率很低。
6. 继续增加 coverage、offset、相同坐标下降轮数，已经进入千分位收益区；需要新的离散求解器、对齐变换和 Attention 联合目标。

### 1.3 推荐主线

按预期收益和依赖关系排序：

1. **HSDQ：HiF4 Structured Discrete Quantizer**——HiF4 层级动态规划、beam 与精确二次增量结合；
2. **LRH-GPTQ：低秩全 Hessian 残差 GPTQ**——在 block Gram 之外恢复跨 64-block 相关性；
3. **BOAT：Block Output-Alignment Transform**——64-block 集中度与权重/激活主方向联合对齐；
4. **FASA：Fisher-Aware Softmax Attention**——以真实 Attention 输出和 softmax Fisher 度量联合选择 Q/K/V；
5. **GQRB：GQA Reciprocal Balancing**——GQA 组内 Q/K 严格等价可逆平衡；
6. **PAWV：Position-Aware Weighted V quantization**——用位置、attention sink 和 `P^T P` 近似加权 V 误差；
7. **RABS：Residual-Aware Budget Scheduler**——按可消除损失/耗时分配 420 秒预算。

---

## 2. 评分公式与 36,000 分的精确含义

### 2.1 单 case 分数

对第 `i` 个 case，令：

- `M_i`：选手 HiF4 输出相对 NVFP4 参考输出的 MSE；
- `M_i_std`：标准 HiF4 输出相对同一参考输出的 MSE；
- `r_i = M_i / M_i_std`：选手误差相对标准算法的比例。

显示分数可以写为：

```math
s_i = 100\left(1-\frac{M_i}{M_i^{std}}\right)=100(1-r_i)
```

解释：

- `r_i = 1`：与标准 HiF4 相同，`s_i = 0`；
- `r_i = 0.2`：只剩标准算法 20% 的 MSE，`s_i = 80`；
- `r_i = 0`：无输出误差，`s_i = 100`；
- `r_i > 1`：比标准算法差，得负分。

### 2.2 总分

令：

```math
\bar r_L=\operatorname{mean}_{i\in Linear}r_i,
\qquad
\bar r_A=\operatorname{mean}_{i\in Attention}r_i
```

则总分为：

```math
S=100\left[250(1-\bar r_L)+200(1-\bar r_A)\right]
```

数学满分：

```math
S_{max}=100(250+200)=45000
```

36,000 分要求：

```math
250\bar r_L+200\bar r_A=90
```

也就是 450 个 case 的加权平均相对 MSE 为：

```math
\bar r=\frac{90}{450}=0.2
```

即平均消除标准 HiF4 的 80% MSE。

### 2.3 从当前官方冠军到目标

当前剩余误差质量：

```math
E_0=45000-22557=22443
```

目标还需增加：

```math
\Delta S=36000-22557=13443
```

必须消除当前剩余误差的比例：

```math
\rho=\frac{13443}{22443}=59.90\%
```

若某个新算法能消除当前剩余误差的比例为 `ρ`，理论总分关系为：

```math
S(\rho)=22557+\rho(45000-22557)
```

| 剩余误差消除率 `ρ` | 分数 |
|---:|---:|
| 0% | 22,557 |
| 10% | 24,801 |
| 25% | 28,168 |
| 50% | 33,779 |
| 60% | 36,023 |
| 100% | 45,000 |

这个表是后续所有优化的统一量纲。任何候选除了报告“提高多少分”，还应报告它消除了父版本多少比例的剩余误差。

---

## 3. 当前 Linear / Attention 误差预算

当前 C75 Qwen shaped panel：

```text
Linear panel    = 111.005949
Attention panel = 131.499409
Total           = 242.505358
```

对应平均 case gain：

```math
g_L=111.005949/250=0.444024
```

```math
g_A=131.499409/200=0.657497
```

剩余本地 proxy 误差：

```math
E_L=250(1-g_L)=138.994
```

```math
E_A=200(1-g_A)=68.501
```

因此：

```math
\frac{E_L}{E_L+E_A}=67.0\%,
\qquad
\frac{E_A}{E_L+E_A}=33.0\%
```

### 3.1 达到目标的组合条件

在同尺度下，36,000 对应 shaped panel `360`。

| 条件 | 需要的另一侧结果 | 含义 |
|---|---:|---|
| Attention 保持 `g_A=0.6575` | Linear `g_L=0.9140` | Linear 剩余 MSE 需下降 84.5% |
| Linear 与 Attention 均衡 | 两者均约 `0.80` | 两侧都必须结构性提升 |
| Attention 完美 `g_A=1` | Linear `g_L=0.64` | 即使 Attention 完美，Linear 仍需明显改善 |
| Linear 保持 `g_L=0.4440` | Attention `g_A=1.245` | 数学上不可能 |

结论：Linear 是主杠杆，但只优化 Linear 会把目标推到极端困难的 `0.914`；合理路线必须让 Attention 同时增长。

---

## 4. HiF4 合法码域与离散优化问题

一个 64 元素 HiF4 block 可以抽象为：

```math
q_i=s\cdot2^{b_{8,g(i)}+b_{4,h(i)}}\cdot c_i
```

其中：

- `s`：合法 E6M2 `scale_factor`；
- `b_8 ∈ {0,1}`：每 8 元素一个 lv2 micro exponent；
- `b_4 ∈ {0,1}`：每 4 元素一个 lv3 micro exponent；
- `c_i = sign_i × mant_i`；
- `sign_i ∈ {-1,0,1}`；
- `mant_i ∈ {0,0.25,...,1.75}`。

对一个 64 block，变量之间不是独立的：

1. 64 个元素共享 E6M2 scale；
2. 每 8 个元素共享 lv2；
3. 每 4 个元素共享 lv3；
4. mantissa 才是逐元素变量。

所以它不是普通的逐元素 nearest rounding，而是树形离散优化。

---

## 5. Linear 误差公式

### 5.1 双侧误差展开

令：

```math
\hat X=X+E_X,
\qquad
\hat W=W+E_W
```

则：

```math
\Delta Y=\hat X\hat W^T-XW^T
```

展开：

```math
\Delta Y=E_XW^T+XE_W^T+E_XE_W^T
```

三项分别是激活误差、权重误差和交互误差。最终损失：

```math
L=\frac{1}{N}\left\|E_XW^T+XE_W^T+E_XE_W^T\right\|_F^2
```

因此，单独最小化 `||E_X||²` 或 `||E_W||²` 不保证输出最优；还需要控制误差方向和两侧交互。

### 5.2 固定量化激活时的连续最优权重

冻结合法激活量化结果：

```math
Z=Q_A(X)
```

教师输出：

```math
Y=XW^T
```

带回归到原始权重的 ridge 目标：

```math
\min_B\|Y-ZB^T\|_F^2+\lambda\|B-W\|_F^2
```

正规方程：

```math
B^T=(Z^TZ+\lambda I)^{-1}(Z^TY+\lambda W^T)
```

无正则时的连续不可消除误差为：

```math
L_{cont}^*=\|(I-P_Z)Y\|_F^2
```

```math
P_Z=Z(Z^TZ)^\dagger Z^T
```

`P_Z` 是 `Z` 列空间的投影。该值是固定 `Q(A)` 条件下的连续上限，但 `B` 通常不是合法 HiF4。

### 5.3 离散二次目标

对一个输出行 `w`，定义：

```math
H=Z^TZ,
\qquad
b=Z^Ty
```

则合法权重行 `q` 的目标等价于：

```math
L(q)=q^THq-2b^Tq+const
```

若当前编码为 `q`，某个 64 block 变化为 `δ`，精确损失变化是：

```math
\Delta L=2g_b^T\delta+\delta^TH_{bb}\delta
```

其中：

```math
g=Hq-b
```

这条公式是 HSDQ、GPTQ、坐标下降、beam search 的统一评分公式；无需为每个候选重算完整 `Z@W`。

---

## 6. 算法一：HSDQ——HiF4 Structured Discrete Quantizer

### 6.1 目标

把连续 JDRQ target 更充分地投影到合法 HiF4，而不是只搜索当前 scale 附近的少量 offset 和一次 hierarchy toggle。

### 6.2 适用范围

- 第一阶段只用于 `out_features < in_features` 的 down-projection；
- 每个输出行优先处理残差最大的 1–4 个 64 block；
- 先用于离线权重，因为离线预算更宽、合规风险最低；
- 验证稳定后再把简化版本用于在线激活 hard block。

### 6.3 候选 scale 集

对每个 64 block 构造：

1. 当前 E6M2 code；
2. 标准 amax code 的 `±1,±2,±3,±4`；
3. 四个 NVFP4 source scale 的 median/q75/max 映射；
4. 连续 ridge target 的最优无约束 scale；
5. 父编码与连续 target 之间的几何插值 scale；
6. SOAR/FOCUS 风格的 relaxed quantization scale 对应的相邻合法 E6M2 code。

去重后保留不超过 12–20 个 scale。

### 6.4 树形 beam

对固定 scale `s`：

1. 从 8 个 lv2 group 逐组展开 `b8 ∈ {0,1}`；
2. 对每个 lv2 group 内的两个 lv3 subgroup 展开 `b4 ∈ {0,1}`；
3. 固定层级后，为每个元素选最优 signed mantissa；
4. 使用精确 `ΔL` 保留 top-B partial states；
5. 每完成一个 lv2 group，加入其与已决定坐标之间的 Hessian cross-term；
6. 所有组完成后执行 1–2 次 signed-mantissa coordinate polish。

### 6.5 伪代码

```text
function HSDQ_BLOCK(q, block, Hbb, gradient, target, scale_candidates, beam_width):
    best = parent block
    best_delta_loss = 0

    for scale in scale_candidates:
        beam = [{empty hierarchy, delta=zeros(64), score=0}]

        for lv2_group in 0..7:
            expanded = []
            for state in beam:
                for lv2_bit in {0, 1}:
                    local_states = solve_two_lv3_groups(
                        state, scale, lv2_bit, target, Hbb, gradient
                    )
                    expanded.extend(local_states)
            beam = keep_top_B_by_exact_partial_quadratic(expanded)

        for state in beam:
            state = mantissa_coordinate_polish(state, sweeps=1..2)
            delta_loss = 2*g^T*delta + delta^T*Hbb*delta
            if delta_loss < best_delta_loss:
                best = state
                best_delta_loss = delta_loss

    return best if exact_recheck_improves else parent
```

### 6.6 剪枝下界

对尚未决定的坐标集合 `U`，使用连续二次问题作为乐观下界：

```math
\min_{\delta_U}
2g_U^T\delta_U+\delta_U^TH_{UU}\delta_U
=-g_U^TH_{UU}^{\dagger}g_U
```

若：

```math
score_{partial}+lower\_bound_U \ge score_{best}
```

则该 beam state 不可能超过当前最好候选，可以安全剪枝。

### 6.7 复杂度

设：

- `K_s`：scale 候选数；
- `B`：beam width；
- `R`：处理的输出行数；
- `K_b`：每行处理 block 数。

64×64 Hessian 下近似复杂度：

```math
O(RK_bK_sB\cdot64^2)
```

建议首版 `K_s≤12, B=4, K_b≤2`，只对最高残差的输出行启用。

### 6.8 验收指标

1. 校准 product loss 不增；
2. validation fold product loss 不增或 robust mean 改善；
3. 对 Qwen/GPT-2 D0，至少兑现合法投影收益的 20%，当前约 5%–7%；
4. Qwen panel Linear 相对父版本有可复现正增量；
5. API 时间保持 `<420s`；
6. 输出五字段完全合法。

---

## 7. 算法二：LRH-GPTQ——低秩全 Hessian 残差求解

### 7.1 问题

当前 Gram64 主要使用：

```math
H\approx D=\operatorname{blockdiag}(H_1,...,H_B)
```

它忽略不同 64 block 之间的相关性。历史 cross64/LDLQ 直接使用相邻 128 维结构出现迁移回退，但这只否定了相邻固定配对的实现，不等于跨块结构不存在。

### 7.2 低秩模型

将 Hessian 写成：

```math
H=D+UU^T+R
```

其中：

- `D`：现有 64-block diagonal Gram；
- `U ∈ R^{d×r}`：跨块主方向，`r=4..16`；
- `R`：未建模残差。

`U` 可以从 `H-D` 的 randomized SVD、校准激活协方差主方向或 weight Gram 主方向得到。

### 7.3 精确候选增量

对一个只改变 block `b` 的 `δ_b`：

```math
\Delta L=
2g_b^T\delta_b+
\delta_b^TD_b\delta_b+
\|U_b^T\delta_b\|_2^2
```

修改后维护低秩残差状态：

```math
z_U \leftarrow z_U + U_b^T\delta_b
```

后续 block 的梯度增加：

```math
g_j \leftarrow g_j + U_j z_U
```

这样可以用 `O(64r)` 代价恢复全局相关性，而不是每次使用完整 `d×d` Hessian。

### 7.4 算法流程

```text
1. 计算 block diagonal D。
2. 对 H-D 做 randomized range finder，得到 rank-r 的 U。
3. 按当前精确残差损失选择 active blocks。
4. 对每个 active block 调用 HSDQ，评分中加入 ||U_b^T delta||^2。
5. 接受修改后更新全局 low-rank residual state。
6. 完成一轮后只回访发生过变化或梯度增幅最大的 block。
7. 在 validation fold 上用最终合法量化器复核。
```

### 7.5 风险控制

- `U` 必须只来自静态 activation covariance 或 weight Gram，不把 `A@W`、输出残差写入 `activation_state`；
- 权重离线优化可以使用输出 teacher，但它不能回流到在线激活量化策略；
- 首版只处理 down-projection，避免全形状迁移噪声；
- `r` 应通过解释方差和 validation fold 选取，不按模型名硬编码。

---

## 8. 算法三：BOAT——Block Output-Alignment Transform

### 8.1 动机

当前 SmoothQuant 和固定 Hadamard 主要降低 outlier/concentration。CAT 的 concentration-alignment 分析指出，量化误差还取决于权重与激活主方向是否对齐。随机旋转可能改善幅值，却破坏对齐，因此当前 R64/H64 的负结果不应被解释成“所有变换都无效”。

### 8.2 精确等价变换

对每个 64-channel block 选择可逆矩阵 `T`：

```math
X'=XT,
\qquad
W'=WT^{-T}
```

于是：

```math
X'W'^T=XT(T^{-1}W^T)=XW^T
```

未量化输出严格不变。

### 8.3 二阶平衡初始化

定义 block 激活协方差和权重 Gram：

```math
A=\mathbb E[X^TX]+\epsilon I
```

```math
B=W^TW+\epsilon I
```

令：

```math
C=A^{1/2}BA^{1/2}
```

选择：

```math
T=A^{-1/2}C^{1/4}
```

则变换后的两侧二阶矩均为：

```math
T^TAT=C^{1/2}
```

```math
T^{-1}BT^{-T}=C^{1/2}
```

这在连续二阶意义下把量化难度平衡到两侧。

### 8.4 可部署参数化

直接保存完整 64×64 `T` 在线开销较大。建议候选：

```math
T=D_s H_1 H_2 P
```

其中：

- `D_s`：对角 SmoothQuant scaling；
- `H_k=I-2u_ku_k^T/||u_k||^2`：1–4 个 Householder 反射；
- `P`：静态通道排列。

也可使用：

```math
T=D_s\prod_{k=1}^K G(i_k,j_k,\theta_k)
```

其中 `G` 是 Givens rotation。`K=8..32` 时比稠密矩阵便宜，并可直接从二阶平衡矩阵 `T` 做近似分解。

### 8.5 合规选型目标

影响 `Q(A)` 的变换不能用 `A@W` 输出监督选取。建议只使用：

```math
J_{operand}(T)
=
\alpha\,\mathcal E_A(Q(XT))
+(1-\alpha)\,\mathcal E_W(Q(WT^{-T}))
+\gamma\,J_{align}(T)
```

其中：

- `E_A`：激活自身合法 HiF4 重构误差；
- `E_W`：权重自身合法 HiF4 重构误差；
- `J_align`：变换后二阶矩的 alignment/concentration 指标；
- 不计算 `A@W`，不使用 Linear 输出误差选择 `T`。

变换冻结后，允许使用 `A@W` 只优化离线 `Q(W)`。

### 8.6 算法流程

```text
1. 每 64 channel 计算 A 和 B。
2. 以阻尼二阶平衡公式得到连续 T0。
3. 把 T0 投影成 diagonal + K 个 Householder/Givens + permutation。
4. 用 operand-only loss 比较 identity、Smooth、CAT-balance、低秩近似 T。
5. 通过 calibration folds 选择低自由度 winner。
6. 冻结 T 和 activation_state。
7. 重新生成所有量化激活 Z。
8. 对固定 Z 使用 HSDQ/LRH-GPTQ 优化 Q(W)。
```

### 8.7 预期与风险

- 预期潜力：高；它改变整个合法码字所处的坐标系，而不是继续在旧坐标系局部搜索；
- 主要风险：高自由度过拟合、在线变换时间、稠密 state 体积；
- 缓解：block64、低秩参数化、operand-only fold、identity parent 永远保留。

---

## 9. 算法四：FS-JDRQ——冻结状态的稳健联合重构

### 9.1 合规结构

Linear 必须严格采用：

```text
operand-only 选择 activation policy
        ↓ 冻结
生成固定合法 Q(A)=Z
        ↓
使用 A@W teacher 只优化离线 Q(W)
```

禁止：

```text
A@W → 选择 transform / activation offset / activation code
```

### 9.2 分布鲁棒目标

将校准 token 划分为 `F` 个窗口，定义每 fold 合法 product loss：

```math
L_f(q)=\|Y_f-Z_fq^T\|_F^2
```

稳健目标：

```math
L_{robust}(q)
=(1-\beta)\frac1F\sum_fL_f(q)
+\beta\max_fL_f(q)
```

或使用 CVaR：

```math
L_{CVaR,\tau}
=\frac{1}{|T_\tau|}\sum_{f\in T_\tau}L_f(q)
```

`T_τ` 是损失最高的 `τ` 比例窗口。

### 9.3 具体求解

1. parent 永远作为候选；
2. 连续 ridge 生成多个 `λ` target；
3. `η∈{0.25,0.5,0.75,1}` 控制 target 插值：

   ```math
   W_{target}=(1-\eta)W_{parent}+\eta W_{ridge}
   ```

4. 使用 HSDQ 把每个 target 投影到合法 HiF4；
5. calibration train folds 用于候选生成；
6. validation fold 只用于候选排序；
7. 选最小 robust loss，而不是要求每个 fold 都严格不退化；
8. 最后用全部 calibration rows 对 winner 做一次低自由度 polish。

### 9.4 为什么比当前 JDRQ 更可能兑现

当前诊断中连续上限很高，但 hierarchy 只兑现合法投影收益约 5%–7%。FS-JDRQ 不继续增加连续 target 数量，而是把预算转给 HSDQ 离散投影和跨 fold 稳健排序。

---

## 10. Attention 误差公式

### 10.1 一阶误差

定义：

```math
S=\frac{QK^T}{\sqrt d},
\qquad
P=softmax(S),
\qquad
O=PV
```

量化误差：

```math
\hat Q=Q+E_Q,
\quad
\hat K=K+E_K,
\quad
\hat V=V+E_V
```

logit 扰动：

```math
\Delta S=
\frac{E_QK^T+QE_K^T+E_QE_K^T}{\sqrt d}
```

softmax 单行 Jacobian：

```math
J_p=diag(p)-pp^T
```

一阶近似：

```math
\Delta P\approx J_p\Delta s
```

输出扰动：

```math
\Delta O\approx J_P\Delta S\,V+PE_V
```

因此 Q/K 应根据 softmax 和 V 的敏感度量化，V 应根据 `P^TP` 加权，而不是三者统一采用普通重构 MSE。

### 10.2 严格 Q/K 等价变换

对任意可逆 `M`：

```math
Q'=QM,
\qquad
K'=KM^{-T}
```

则：

```math
Q'K'^T=QK^T
```

### 10.3 K 公共平移

```math
K'=K-\mathbf1c^T
```

```math
QK'^T=QK^T-(Qc)\mathbf1^T
```

softmax 对每行常数平移不变：

```math
softmax(s-\alpha\mathbf1)=softmax(s)
```

所以 K 公共中心是严格安全的候选。

---

## 11. 算法五：FASA——Fisher-Aware Softmax Attention

### 11.1 目标

用真实 Attention 输出敏感度生成候选和排序 Q/K/V 状态，替代纯 Q/K reconstruction proxy。

### 11.2 Q/K Fisher metric

对每个 query row，定义：

```math
G_s=J_p^TVV^TJ_p
```

logit 误差的二阶代理：

```math
L_{QK}^{proxy}=\sum_t\Delta s_t^TG_{s,t}\Delta s_t
```

直接构造完整 `seq×seq` 矩阵较贵，可使用：

1. `G_s` 对角；
2. `diag + rank-r`；
3. 只保留 attention probability 最大的 top-k keys；
4. 按 head、causal distance 分桶累计。

### 11.3 候选族

每个 KV head/group 建立：

1. identity；
2. 当前 Smooth-QK；
3. K midrange/mean/scale-aware center；
4. diagonal reciprocal balance；
5. 4×4、8×8 block reciprocal balance；
6. 低秩 Householder reciprocal transform；
7. head/channel permutation；
8. source-aware scale proposals；
9. position/sink-aware refinement budget。

### 11.4 两阶段选择

```text
Stage 1: Fisher proxy
    在所有候选中快速保留 top 4–8。

Stage 2: deployed Attention
    用最终 hif4_dynamic_quantize_q/k/v 量化；
    计算真实 causal Attention 输出 MSE；
    同时计算 non-causal / heavy-tail safety MSE；
    用 robust fold loss 选 winner。
```

### 11.5 Q/K/V 交替

```text
1. 固定当前 V state，搜索 Q/K joint state。
2. 固定 Q/K winner，搜索 V importance/position policy。
3. 固定 V winner，只回访一次 Q/K。
4. 若第二次校准收益很小或 validation 不稳定，停止。
```

Attention 规则没有 Linear 的 `A@W -> Q(A)` 禁令，因此可以在校准阶段用真实 Attention 输出选择 q/k/v state，但必须保证 state 固定、在线阶段不访问未来 K/V 或测试输出。

---

## 12. 算法六：GQRB——GQA Reciprocal Balancing

### 12.1 GQA 结构

若一个 KV head 对应 `m` 个 Q heads，则先对这些 Q heads 的协方差求平均：

```math
A_Q=\frac1m\sum_{h=1}^m\mathbb E[Q_h^TQ_h]+\epsilon I
```

K 协方差：

```math
B_K=\mathbb E[K^TK]+\epsilon I
```

### 12.2 闭式初始化

```math
C=A_Q^{1/2}B_KA_Q^{1/2}
```

```math
M=A_Q^{-1/2}C^{1/4}
```

应用：

```math
Q_h'=Q_hM,
\qquad
K'=KM^{-T}
```

所有关联 Q heads 使用同一个 `M`，保证 GQA 点积严格不变。

### 12.3 部署限制

- 首版用 4×4 或 8×8 block diagonal `M`；
- 特征值裁剪到 `[λ_min,λ_max]`；
- 条件数限制，如 `κ(M)≤8`；
- 与 identity、diagonal balance 一起进入 FASA；
- 不用模型名门控，只按 `q_num_heads/kv_num_heads` 和 fold 结果选择。

---

## 13. 算法七：PAWV——Position-Aware Weighted V Quantization

### 13.1 精确 V 目标

固定 Q/K 后：

```math
L_V=\|P(\hat V-V)\|_F^2
```

令 `E_V=hat V-V`：

```math
L_V=tr(E_V^TP^TPE_V)
```

当前主要使用 head/channel importance，尚未充分利用 token 位置之间的差异。

### 13.2 对角近似

定义第 `t` 个 key/value token 的权重：

```math
w_t=(P^TP)_{tt}=\sum_qP_{qt}^2
```

则：

```math
L_V\approx\sum_tw_t\|e_{V,t}\|_2^2
```

### 13.3 状态设计

按下列维度统计 `w_t`：

- causal relative position bucket；
- sequence 前部 attention sink bucket；
- KV head；
- calibration window 的 mean/q75/max；
- causal 与 non-causal 两条安全轨。

在线时根据当前 `seq_len` 将每行映射到 bucket，仅改变 hard-block 排序、scale proposal 数和精修预算，不改变输出 shape。

### 13.4 低秩扩展

若对角近似上限不足，使用：

```math
P^TP\approx D+UU^T
```

对 top sensitive token rows 做 2–4 轮联合坐标更新。该步骤优先用于校准候选验证，不要一开始全部在线启用。

---

## 14. 算法八：DSHP——Decoupled Scale & Hierarchy Proposal

### 14.1 思路

最终反量化 scale 必须是合法 E6M2，但用于产生 rounding candidate 的连续 proposal 不需要被保存。对每个合法输出 scale `s`，额外搜索 rounding threshold coefficient `α`：

```math
c_i(\alpha,s)
=Round_{S1P2}\left(
\frac{x_i}{\alpha s2^{b_8+b_4}}
\right)
```

最终合法输出仍是：

```math
q_i=s2^{b_8+b_4}c_i
```

`α` 只改变候选码字的生成路径，不进入返回字段。

### 14.2 候选

```text
alpha ∈ {0.75, 0.875, 1.0, 1.125, 1.25}
scale code ∈ source-aware / amax / ridge target 邻域
```

DSHP 本身不扩大合法值集，因此不能替代 HSDQ；它的作用是更便宜地生成可能被 nearest rounding 漏掉的合法候选。

---

## 15. 算法九：RABS——Residual-Aware Budget Scheduler

### 15.1 动机

420 秒预算不应平均分配给所有层、行和 block。当前实验表明增大 Gram64 block 数已经饱和，但不同层的连续—合法 gap 差异很大。

### 15.2 价值函数

对候选任务 `j` 定义：

```math
V_j=
\frac{
L_{parent,j}-L_{relaxed,j}
}{
T_{estimated,j}+\epsilon
}
\cdot
G_{generalization,j}
```

其中：

- `L_parent-L_relaxed`：该任务的理论可消除损失；
- `T_estimated`：预计计算时间；
- `G_generalization∈[0,1]`：跨 fold 稳定系数。

### 15.3 successive halving

```text
Round 1: 32 rows, beam 2, 每行 1 block
Round 2: 64/128 rows, beam 4, 每行 2 blocks
Round 3: 256 rows, 完整合法部署路径
Final: 只对 winner 做 full calibration polish
```

每一轮淘汰低 `V_j` 任务，把预算集中到 down-proj、Attention 高敏感 head 和连续—合法 gap 最大的层。

---

## 16. 可搜集算法与本赛题适配判断

| 算法/论文方向 | 核心机制 | 本赛题可用部分 | 不能直接照搬的部分 | 优先级 |
|---|---|---|---|---:|
| GPTQ | Hessian、自适应舍入、误差反馈 | 合法 block 二次增量 | 原始 INT codebook | P0 |
| QuIP / QuIP# | incoherence、LDLQ、Hadamard、lattice | 误差反馈、理论分析 | E8 lattice 不合法 | P1 |
| SmoothQuant | 激活/权重对角迁移 | 已实现，可作 BOAT 子结构 | 单独继续扫 alpha 已趋饱和 | P2 |
| AWQ | activation-aware weight scaling | 权重重要性、通道保护 | 混合/额外 scale 不可直接返回 | P2 |
| QuaRot | 精确等价旋转 | Q/K、Linear block transform | 随机全局旋转本地曾回退 | P1 |
| SpinQuant | 学习等价旋转 | 低秩 Householder/Givens | 端到端任务微调不适用 | P1 |
| DuQuant / DuQuant++ | outlier-aware block rotation、排列 | HiF4 group=64 对齐旋转 | MXFP4 group=32 配方不能照搬 | P1 |
| OSTQuant | 正交+缩放、空间利用率 | BOAT 设计依据 | 高自由度训练需降维 | P1 |
| CAT | concentration + alignment | 直接针对当前缺口 | 原论文 uniform INT 假设需改 HiF4 | P0 |
| AffineQuant / FlatQuant | 可学习等价仿射变换 | block transform 候选 | 完整优化成本较高 | P1 |
| MR-GPTQ | microscaling-aware GPTQ | HSDQ/LRH-GPTQ 设计依据 | NVFP4 小 group 的结论不等同 HiF4 | P1 |
| Four Over Six | 双 scale 候选 | source-aware proposal | NVFP4 的 4/6 常数不应硬搬 | P2 |
| ScaleSearch | block scale 搜索 | DSHP、Attention scale proposal | 不能改变 HiF4 合法值集 | P1 |
| SOAR | 连续/离散 scale 解耦 | proposal scale、闭式初始化 | NVFP4 scale 层级不同 | P1 |
| ScaleSweep | scale 初始化搜索 | scale candidate ranking | 需要 HiF4 三级层级重写 | P2 |
| FOCUS | coupled relaxation、dual granularity | relaxed rounding proposal | 不能增加返回元数据 | P1 |
| SageAttention2 | Q smoothing、INT4 attention | Attention Q proxy/scale | 需要额外修正的路径不能照搬 | P1 |
| KIVI | K per-channel、V per-token | Q/K/V 不同策略的依据 | HiF4 block layout固定 | P2 |
| RotateKV | outlier rotation、sink-aware | GQRB、PAWV | KV cache 专用部署细节不同 | P1 |
| AQLM | codebook/beam/block reconstruction | beam 和 block tuning思想 | 自定义 codebook 不合法 | 研究参考 |
| SpQR | outlier 高精度旁路 | outlier 诊断 | 混合精度旁路不合法 | 不实施 |
| QAT | 量化感知训练 | 仅可借鉴 soft-to-hard loss | 赛题是转换/PTQ接口，不宜训练模型 | 不实施 |

主要资料：

- HiF4：[HiFloat4 Format for Language Model Inference](https://arxiv.org/abs/2602.11287)
- GPTQ：[Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- QuIP：[2-Bit Quantization with Guarantees](https://arxiv.org/abs/2307.13304)
- QuIP#：[Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)
- QuaRot：[Outlier-Free 4-Bit Inference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html)
- SpinQuant：[Learned Rotations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)
- OSTQuant：[Orthogonal and Scaling Transformations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html)
- CAT：[Concentration-Alignment Perspective](https://arxiv.org/abs/2603.04359)
- MR-GPTQ：[Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)
- DuQuant++：[Fine-grained Rotation for Microscaling FP4](https://arxiv.org/abs/2604.17789)
- Four Over Six：[Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)
- SOAR：[Scale Optimization for NVFP4](https://arxiv.org/abs/2605.12245)
- ScaleSearch：[Search Your Block Floating Point Scales](https://arxiv.org/abs/2605.12464)
- FOCUS：[Coupled-Relaxation and Dual-Granularity Scaling](https://arxiv.org/abs/2608.01847)
- SageAttention2：[INT4 Attention Smoothing](https://arxiv.org/abs/2411.10958)
- RotateKV：[Outlier-Aware Adaptive Rotations](https://arxiv.org/abs/2501.16383)
- KIVI：[Asymmetric 2-bit KV Quantization](https://arxiv.org/abs/2402.02750)

### 16.1 当前 `solution.py` 的具体接入点

| 新算法 | 现有函数/路径 | 建议新增 helper | 最小改动方式 |
|---|---|---|---|
| HSDQ | `_jdrq_refine_hierarchy_offsets`、`_jdrq_select_weight_candidate` | `_hsdq_scale_candidates`、`_hsdq_solve_block_beam`、`_hsdq_partial_lower_bound` | 保留当前 hierarchy 为 parent，只替换 candidate projection；flag 关闭时逐位不变 |
| LRH-GPTQ | `_full64_hessian_blocks`、JDRQ product loss | `_low_rank_cross_block_gram`、`_lrh_block_delta` | state 只保存 block Gram 与低秩 `U`；首版仅权重离线使用 |
| BOAT | SmoothQuant/CAT64 transform 选择路径 | `_boat_balanced_transform64`、`_factorize_householder`、`_apply_boat_transform` | identity、当前 CAT64、BOAT 三候选 operand-only 选择；变换冻结后重算 Q(W) |
| FS-JDRQ | `_jdrq_make_target`、`_jdrq_robust_product_loss` | `_jdrq_fold_losses`、`_jdrq_cvar_loss` | 保留现有 ridge target；把预算从 target 数转给 HSDQ 投影 |
| FASA | `_attention_candidate_metrics`、`_attention_deployed_mse` | `_attention_fisher_proxy`、`_fasa_shortlist`、`_fasa_alternating_select` | 当前 A1 deployed gate 作为最终裁判，Fisher 只做候选预筛 |
| GQRB | `_build_qk_states`、QK smooth/center 求解 | `_gqrb_group_covariances`、`_gqrb_balancing_matrix` | 先加入 4×4 block candidate；identity 永远保留 |
| PAWV | `_build_v_state`、`hif4_dynamic_quantize_v` | `_v_position_bucket_weights`、`_v_ptp_lowrank` | 首版只增加 row/bucket budget，不改 HiF4 codec |
| DSHP | `_standard_e6m2_scale`、source-aware proposal | `_decoupled_rounding_proposals` | 只生成候选，不新增返回字段 |
| RABS | calibration 统计和各 refine selector | `_candidate_value_per_second`、`_successive_halving_budget` | 开发阶段先在 evaluator 侧实现，策略稳定后再内联提交文件 |

### 16.2 建议 feature flags

```python
_HSDQ_ENABLED = False
_HSDQ_BEAM_WIDTH = 2
_HSDQ_MAX_BLOCKS_PER_ROW = 1
_HSDQ_SCALE_LIMIT = 12

_LRH_ENABLED = False
_LRH_RANK = 4

_BOAT_ENABLED = False
_BOAT_HOUSEHOLDER_RANK = 2

_FASA_ENABLED = False
_FASA_SHORTLIST = 4

_GQRB_ENABLED = False
_GQRB_BLOCK_SIZE = 4

_PAWV_ENABLED = False
_PAWV_POSITION_BUCKETS = 8
```

所有新 flag 必须满足：关闭时返回字段、state key、数值和候选选择均与父版本一致。

### 16.3 建议新增测试

```text
tests/test_hsdq.py
  - test_hsdq_returns_legal_hierarchy
  - test_hsdq_exact_quadratic_delta_matches_direct_product
  - test_hsdq_never_worsens_parent_on_selection_rows
  - test_hsdq_disabled_is_bit_exact

tests/test_lrh_gptq.py
  - test_low_rank_delta_matches_dense_hessian
  - test_rank_zero_matches_block_diagonal_parent

tests/test_boat.py
  - test_boat_full_precision_product_invariant
  - test_boat_state_has_no_product_or_residual
  - test_boat_disabled_is_bit_exact

tests/test_fasa.py
  - test_reciprocal_qk_transform_preserves_logits
  - test_k_common_shift_preserves_softmax
  - test_fisher_proxy_is_finite
  - test_fasa_states_and_params_are_legal

tests/test_pawv.py
  - test_v_diagonal_ptp_matches_direct_weighted_error
  - test_position_bucket_state_is_cpu_finite
```

### 16.4 HSDQ 最小可行版本的具体参数

首个实验不要一次实现全部设计，建议固定：

```text
role                 = proj only
shape gate           = out_features < in_features
calibration rows     = 256
scale candidates     = parent + standard offsets {-2,-1,0,1,2,3}
beam width           = 2
active blocks/row    = 1
mantissa polish      = 1 sweep
validation mix       = 0.35
parent candidate     = always retained
```

若这个最小版本不能提高当前 D0 hierarchy realization，则先检查 partial quadratic、层级展开顺序和 target/transform 坐标系，不应直接把 beam 扩到 8 或增加更多 scale。

---

## 17. 已进入饱和或需要谨慎的方向

最新本地 Qwen 结果：

| 方案 | Panel total | Linear | Attention |
|---|---:|---:|---:|
| C75 project-only Gram64 | 242.191 | 110.691 | 131.499 |
| wide Gram64 block=4 | 242.420 | 110.920 | 131.499 |
| wide Gram64 block=16 | 242.488 | 110.989 | 131.499 |
| wide hierarchy64 | 242.505 | 111.006 | 131.499 |
| hierarchy 后再 coordinate | 242.437 | 110.938 | 131.499 |

结论：

1. Gram64 从 4 block 增到 16 block 只增加约 `0.068` panel；
2. hierarchy beam 再增加约 `0.017`；
3. 额外 coordinate 反而回退约 `0.068`；
4. Attention 完全不变；
5. 继续扩大 coverage、offset、重复 sweep 不是主增量方向。

历史负实验只约束已测实现：

- R64/H64 随机或固定旋转回退，不否定对齐感知学习变换；
- cross64 Block-LDLQ 回退，不否定低秩全局 Hessian；
- activation quadratic16 回退，不否定 sample-local 或输出敏感度生成的 16/64 结构；
- V importance 候选无收益，不否定 `P^TP`、位置和 sink-aware V 目标；
- JDRQ continuous target 迁移失败，不否定更强的合法离散投影。

---

## 18. 理论上限的三个层次

### 18.1 绝对数学上限

```math
S_{absolute}=45000
```

只由 `MSE≥0` 得到，决策价值有限。

### 18.2 连续松弛上限

```math
M_i^{cont}
=\min_{\hat X,\hat W\in\mathbb R}
\|\hat X\hat W^T-XW^T\|_F^2
```

当前单层诊断：

| 层 | 连续解相对 parent 剩余损失 | 可消除比例 |
|---|---:|---:|
| Qwen layer-0 proj | 0.446% | 99.55% |
| GPT-2 layer-0 proj | 1.227% | 98.77% |

这证明连续目标空间没有枯竭，但不代表合法可达。

### 18.3 合法部署上限

```math
M_i^*=
\min_{
\substack{
Q(W),Q(A),Q(Q),Q(K),Q(V)\in\mathcal Q_{HiF4}\\
state\ compliant,\;time<420s
}}
M_i
```

```math
S^*=100\sum_i\left(1-\frac{M_i^*}{M_i^{std}}\right)
```

隐藏数据未知，因此当前无法严格求出 `S*`。下一步必须用多层、多 role、多 fold 的合法 oracle 估计区间，而不能用单层连续解外推官方总分。

### 18.4 当前离散兑现率

| 层 | 一次合法投影可消除 parent 损失 | 当前 hierarchy 可消除 | hierarchy / 合法投影收益 |
|---|---:|---:|---:|
| Qwen layer-0 proj | 29.90% | 1.42% | 4.75% |
| GPT-2 layer-0 proj | 44.63% | 3.17% | 7.11% |

“合法投影”仍不是严格最优，但当前 solver 只兑现其少量收益，说明 HSDQ 是最直接的高价值实验。

---

## 19. 分阶段实施顺序

### 阶段 D0：建立上限仪表盘

覆盖：

- 模型：Qwen、GPT-2、OPT、Pythia；
- role：q/k/v/o/fc/proj；
- 层：首层、中层、末层；
- fold：至少两个 calibration 窗口和一个 validation 窗口。

每项报告：

```text
parent loss
continuous ridge loss
legal nearest projection loss
HSDQ loss
validation loss
runtime
continuous→legal retention
legal→deployed realization
```

### 阶段 C76：HSDQ 权重版

1. 只接 Qwen down-proj；
2. top 1 block/row，beam=2；
3. 再扩到 top 2、beam=4；
4. 与当前 hierarchy 做同 target A/B；
5. 跨 Qwen/GPT-2 复核。

### 阶段 C77：LRH-GPTQ

1. rank 4；
2. rank 8；
3. 只比较相同 HSDQ budget；
4. 验证它解决的是跨块误差而非增加搜索量。

### 阶段 C78：BOAT

1. 二阶闭式 64-block `T` oracle；
2. diagonal + 1/2/4 Householder 近似；
3. operand-only fold 选择；
4. state 冻结后重新运行 HSDQ；
5. 对 `identity/Smooth/CAT-like` 做消融。

### 阶段 C79：FASA + GQRB

1. 保持 V state，搜索 Q/K reciprocal candidates；
2. 加 Fisher proxy；
3. deployed Attention 双 mask 终验；
4. MHA/GQA 分开报告，不使用模型名门控。

### 阶段 C80：PAWV

1. 先做对角 `w_t=sum_q P_qt²`；
2. 再加 position/sink buckets；
3. 最后才考虑 `diag+low-rank PTP`。

### 阶段 C81：RABS 与发布压缩

1. 汇总每个机制的 gain/time；
2. successive halving；
3. 移除无贡献候选；
4. 主模型完整 API `<420s`；
5. 最终官方提交前重新运行完整合法性和多模型 guardrail。

---

## 20. 每个候选的统一实验记录

```markdown
## Cxx / mechanism

- Parent SHA:
- Candidate SHA:
- Unique change:
- Compliance boundary:
- Mathematical target:
- Relaxed upper bound:
- Legal projection result:
- Deployed train-fold result:
- Deployed validation-fold result:
- Qwen Linear / Attention / total:
- Heterogeneous models:
- Runtime:
- State size / nodes:
- Legality tests:
- Remaining-error capture:
- Result: continue / revise / archive
- Exact conclusion:
- Next falsifiable experiment:
```

必须额外记录：

```math
capture=\frac{S_{candidate}-S_{parent}}{45000-S_{parent}}
```

本地 panel 则使用：

```math
capture_{local}=
\frac{P_{candidate}-P_{parent}}{450-P_{parent}}
```

---

## 21. 36,000 分可达性判断

### 21.1 已知事实

1. 绝对上限 45,000；36,000 在纯数学上可行。
2. 当前官方冠军到目标需要消除 59.90% 剩余误差。
3. 外部 24,153 分参考到目标仍需消除 56.83% 剩余误差。
4. 当前参数/coverage 调整只提供很小的剩余误差捕获率。
5. Linear 单层连续上限很高，但合法离散投影和部署泛化吞掉了绝大部分收益。
6. Attention 在 C66–C75 基本冻结，仍存在未充分利用的真实输出敏感度结构。

### 21.2 结论分级

| 目标区间 | 当前判断 |
|---|---|
| 24,000 左右 | 已被外部实现证明可达 |
| 26,000–30,000 | HSDQ/BOAT 有稳定跨模型收益时具有现实研究可能 |
| 30,000–35,000 | 需要 Linear 双侧联合改善和 Attention 同时出现结构性增益 |
| 36,000 | 需要约 60% 全局剩余误差捕获，是研究突破目标，不是调参目标 |

不能把上述区间理解成严格理论上限。真正的合法上限必须由 D0 多层 oracle、HSDQ 兑现率和官方提交共同校准。

---

## 22. 最终建议

下一项唯一优先实验应是：

> 在 Qwen layer-0 down-proj 上实现 HSDQ 的最小版本，固定当前 activation state 和 continuous target，只替换当前 hierarchy 投影器；用相同 calibration/validation folds 测量它能否把合法投影收益兑现率从约 4.75% 提高到至少 20%。

这个实验能最直接地区分两种情况：

1. **求解器不足**：HSDQ 明显降低合法 product loss并迁移到 validation；继续扩展 HSDQ/LRH-GPTQ。
2. **合法码域或泛化不足**：即使更强 HSDQ 也无法迁移；立即把主线转向 BOAT 改变坐标系，而不是继续增加离散搜索预算。

在 HSDQ 结论明确前，不建议继续扩大 Gram64 block 数、offset 数或相同 coordinate sweep。

---

## 23. 本地证据索引

- 官方锚点、时间限制和当前架构：`README.md`
- 当前唯一提交文件：`solution.py`
- C75 project-only Gram64：`solutions/20260829_v073_c75-source-aware-gram64_scoreNA_timeNA/result.md`
- Qwen JDRQ D0：`artifacts/jdrq/d0-qwen-proj.json`
- GPT-2 JDRQ D0：`artifacts/jdrq/d0-gpt2-proj-v3.json`
- 最新 wide hierarchy64：`logs/evaluations/2026-08-30-c75-wide-hierarchy64-b8-o9-qwen.md`
- 最新 wide JDRQ coordinate 回退：`logs/evaluations/2026-08-30-c75-wide-jdrq-b8-qwen.md`
- 文献调研底稿：`docs/research/2026-08-28-hif4-algorithm-literature/report.md`
