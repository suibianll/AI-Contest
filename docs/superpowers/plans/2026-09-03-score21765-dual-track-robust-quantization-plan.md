# 21765 目标：Attention 静态输出敏感量化 + Linear 鲁棒 A\@W 计划

> 状态：**COMPLETED / ALL LOCAL CANDIDATES REJECTED**
>
> 创建：2026-09-03
>
> 修订：2026-09-03（预执行审查修正：删除本地到官方的隐式分数换算、修正 C 的绝对归一化
> minimax 准则、把组合时间降级为只用于否决的非负增量估计，并固定外部搜索协议）
>
> 官方父版本：v160，`17532 / 232s`，归档源码 SHA256
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
>
> 当前官方榜首锚点：`21765 / 290s`（用户回传，源码未知）

## 1. 目标、差距与执行原则

当前可复现最好分数为 `17532`，距榜首：

```text
21765 - 17532 = 4233
```

v162/v163/v164 官方 2×2 校准得到：

```text
base = 1001
Linear(v160) - Linear(standard) = 3586
Attention(v160) - Attention(standard) = 12944
score interaction = 17532 - 4587 - 13945 + 1001 = 1
```

因此在 standard/v160 两个端点上，官方分数近似按侧可加；但 `12944:3586=3.61:1`
是当前已实现贡献比，不是公开评分权重，也不是本地 gain 到官方分的换算率。研发目标按
Attention `+3000~3300`、Linear `+900~1200` 分解，只作为资源规划，不用于预测单个候选。

量级注记：v161 只证明输出感知 Q/K 误差度量在本地与 GPT-2 上存在同号信号，不构成 A/B 的
收益上界。由于本地与官方发生过排序反转，当前没有可信证据把任一本地 mean 映射为官方分，
也不能预估 A+B+C 的官方增量区间或 M2 达成概率。`4233` 只用于说明必须寻找结构性增益；
所有转向判断只读取预注册的实际官方增量，不读取本地到官方的比例外推。

执行原则：

- 两侧从 v160 归档分别分支，Attention 实验冻结全部 Linear，Linear 实验冻结全部 Attention；

- 先 Attention、后 Linear，两个方向不并行调参；

- 每个数学假设只产生一个预注册候选，失败后关闭该机制，不扫描 alpha、seed、ridge、阈值、
  block size、候选数量或模型/layer/role/length 路由；

- 本地 proxy 只负责否定、机制归因和同机复杂度，不换算官方分数或时间；

- Qwen compact/default 共享数据假设，不视为两个独立泛化证据；GPT-2 与 OPT/Pythia 才是
  架构否决门禁；

- 动态 per-call Gram、多轮 refinement 和小张量 Python 循环已被 v161 官方 timeout 关闭；
  新增计算优先放在 calibration，动态路径保持 O(TD) 且不增加 sweep；

- 官方结果失败后不做邻域版本；只有死分支、state 未写回、device 错误或 case 身份不匹配等
  实现错误允许按原数学规则修复一次，并必须记录 attempted/accepted 计数。

## 2. 官方里程碑

| 里程碑 |       分数 | 判读                  |
| --- | -------: | ------------------- |
| M1  |  `18500` | 至少一个新机制达到约千分级官方增量   |
| M2  |  `20000` | 两侧或两个独立机制形成结构性可叠加收益 |
| M3  | `21766+` | 超过当前榜首              |

官方单机制增量的事后分类只用于决定其战略角色：`<=0` 关闭；`1~299` 为有效微模块但不是
冲榜主线；`300~999` 可作为组合模块；`>=1000` 才升级为冲击 M2 的新父版本。分类不能用于
反向修改候选参数。

## 3. 工作包 A：跨折收缩 Softmax-Fisher（第一优先）

### 3.1 理论依据

v161 的 Q/K 交叉 Gram64 per-call 精化在 Qwen default 得到 paired `+0.052502`、
`106+/14-`，GPT-2 `+0.0678` 同号并通过 D1，但官方 timeout。该证据说明输出感知的
Q/K 误差度量有精度信号，失败点是部署复杂度。A1 将该信号压缩为 calibration 编译的一维
静态 importance，不移植动态 sweep。

Attention 定义为：

```text
S = Q K^T / sqrt(d)
P = softmax(S)
O = P Q(V)
```

对 logit `S_ts` 的输出敏感度：

```text
J_ts = P_ts * (Q(V)_s - O_t)
```

在 v160 最终部署坐标中构造对角 Fisher：

```text
Fq_j = E[sum_ts ||J_ts||^2 * K_sj^2 / d]
Fk_j = E[sum_ts ||J_ts||^2 * Q_tj^2 / d]
```

它直接近似 Q/K 通道误差对最终 Attention 输出的二阶影响，不以 operand MSE 或 raw logits MSE
代替。V 必须先经过父版本动态量化，确保目标对应真实部署的 `Q(V)`。

### 3.2 跨折解析收缩

对五个 calibration fold 和 causal/non-causal mask 分别计算 Fisher。在每个 head 内：

```text
z_fj = log(F_fj + eps) - mean_j(log(F_fj + eps))
mu_j = clip(median_f(z_fj), -2, 2)
signal = Var_j(mu_j)
noise = median_f(Var_j(z_fj - mu_j))
rho = max(0, 1 - noise / (signal + eps))
I_new_j = normalize_head(I_parent_j * exp(rho * mu_j))
```

`rho` 由折间信噪比解析确定，不设置 blend 网格。折间不稳定时 `rho -> 0`，自然退回父版本；
稳定时才增强输出敏感通道。`c=2` 是固定的鲁棒性先验而非浮点溢出界：它把单通道相对父版本
的乘数限制在 `[e^-2,e^2]`，极端通道间最大比值为 `e^4≈54.6`，用于限制 Fisher 条件数对
量化选择的支配。该常数必须在 A0 前冻结，A0、holdout 或官方结果均不得触发调整；若此先验
失败，整个公式拒绝，不测试其他裁剪值。禁止直接开启根文件中旧 `_ATTN_FISHER_IMPORTANCE` 的
`3 blend × Q-only/K-only/QK` 九候选搜索和 calibration-output gate。

### 3.3 冻结与代码边界

唯一允许变化：Q/K state 中最终 `importance`。

以下逐字段冻结：multiplier、permutation、block smooth、rotation、Matrix-Smooth pair transform、
K center、offsets、refine 参数、V state 和全部 Linear state。动态 Q/K/V 仍调用父版本
`_nvfp4_to_hif4`，不增加矩阵乘、Gram、候选循环或 sweep。

### 3.4 复杂度

设 fold 数 `F=5`、query head 数 `H`、截断长度 `Tc<=128`、head dimension `d`：

```text
calibration time: O(F * H * Tc^2 * d)
temporary memory: O(H * Tc^2 + H * d)
stored state: O((Hq + Hkv) * d)
dynamic time: O(T * (Hq + Hkv) * d), 与 v160 相同
```

每折顺序处理并复用 output-selector 已构造的 logits/probability/V，禁止保存完整 1024-token
Fisher。Attention calibration 同机增量目标 `<=15s`；这是复杂度风险门禁，不是官方时间换算。

### 3.5 算法流程与停止条件

1. **A0 零 API 统计审计**：输出每层五折 Fisher 余弦一致性、causal/non-causal 一致性、
   `rho_Q/rho_K`、top-quartile 通道重合率和动态范围；只检查可估性，不据此改公式。
2. **A1 接口/control**：隔离导入、六 API 合法；Linear compact、V state 逐位一致；Q/K
   changed-channel 非零；动态调用图不变。
3. **A2 Qwen attention compact**：mean Δ `>0`、median Δ `>=0`、improved `>regressed`、
   worst Δ `>=-0.005`、QK-only 不恶化、V-only 精确不变、probability MSE 不恶化；任一失败
   即关闭 Softmax-Fisher，不试 blend/阈值。
4. **A3 Qwen attention default 120**：执行 D1（touch `>=50%`、improved `>regressed`、
   median `>=0`），并要求五个长度组 mean 非负、layer median 非负、worst-quartile 无集中
   深层回归、QK-only 不恶化、V-only 不变、probability MSE/KL 不恶化。
5. **A4 跨模型封存**：候选冻结后依次运行 GPT-2 和 OPT-125m/Pythia-160m；任一整体负向
   即 `model-specific / REJECTED`，禁止模型路由。
6. **A5 集成与官方**：完整六 API 调用图、单文件导入、SHA 和时间报告通过后只提交一次。

产物：一个未编号 workbench 源码、统计审计日志、parent/candidate JSON 与 report；只有通过
A0-A5 并准备官方提交时才分配版本号。

## 4. 工作包 B：低秩 Fisher 微块联合舍入（A1 成功后才启动）

A1 本地和跨模型均通过后，才能验证对角 Fisher 是否遗漏可迁移的误差相关性。若 A1 失败，
B 直接取消，不能复用一个已经失效的统计族。

对每个 HiF4 四元素微块，用固定 `H = D + u u^T` 近似输出 Fisher：`D` 为 A1 对角 importance；
`u` 是五折协方差经符号对齐、中位聚合和同型解析收缩得到的第一稳定方向。每个微块只比较：

1. parent 最近邻编码；
2. 分别把四个分量中的一个移到相邻合法码字。

共最多五个候选，按 `e^T H e` 一次向量化选择；禁止 16/256 组合枚举和第二轮坐标下降。

```text
per microblock: O(5 * 4)
total dynamic: O(TD), 常数高于 parent
state: diagonal + one rank-1 vector per head/block
```

在线实现硬约束（预注册）：五候选选择必须整批向量化，只允许 elementwise/reduction 类大张量
算子；禁止 per-microblock Python 循环、gather/topk/scatter 类小张量算子和任何迭代结构。
v161 官方 timeout 已证明该类算子在官方机的成本不可由本地外推，"动态 API 增量超过 20%
即停止"只是必要条件而非充分条件。失败隔离：B 作为独立候选在 A 的官方结果入账后单独提交；
B 官方 timeout 只损失一次提交机会，不回退 A 的已入账收益。

真实形状 smoke 中动态 API 增量超过 20% 即停止，不进入 default。其余门禁完全复用 A2-A5，
不得因 B 的候选更强而放宽负 case、长度尾部或跨模型条件。

## 5. 工作包 C：Linear 跨折 Minimax 部署 A\@W-GPTQ

Attention 完成一个候选的完整裁决后再启动 C。固定 v160 的所有 Linear 坐标变换、Activation
编码、Weight scale/lv2/lv3、Attention，仅允许在固定 hierarchy 下改变 Weight 相邻 HiF4 码字。

### 5.1 精确部署目标

对 calibration fold `f` 和一个 Weight row `w`：

`proxy-v2` 的 Linear API 每个 state 只提供两个独立 calibration window，因此这里的五折固定为：
先沿用 v160 的确定性等距采样、每个 window 最多保留 128 行，再按采样后行号 `mod 5` 交错分组，
最后把两个 window 中余数相同的行合并为同一 fold。该划分在读取张量值之前确定，五折都覆盖两个
文档；不得按 loss、长度或 role 重新分组。

```text
L_f(wq) = ||A_f w - Q(A_f) wq||_2^2
         = wq^T H_f wq - 2 b_f^T wq + c_f
H_f = Q(A_f)^T Q(A_f)
b_f = Q(A_f)^T A_f w
r_f = L_f / (||A_f w||_2^2 + eps)
```

这在固定 Activation 编码器后是精确二次型，包含 W 误差、A 误差及 interaction。优化采用
无权重的字典序 minimax：

```text
minimize (max_f r_f, median_f r_f, mean_f r_f)
```

先降低最坏折；最大值相同才比较 median，再比较 mean。禁止把所有折拼接成一个平均 Hessian，
也禁止根据结果选择 mean/median/CVaR 权重。

### 5.2 固定 64-block 单次码字更新

每个元素只允许 parent、相邻较小、相邻较大三个合法码字；E6M2 scale、hierarchy、变换和
Activation 不变。对码字变化 `delta`：

```text
Delta L_f = 2 * delta * g_fj + delta^2 * H_fjj
g_f <- g_f + delta * H_f[:, j]
```

逐元素选择准则（预注册，与 §5.1 的绝对归一化目标一致）：令

```text
E_f = ||A_f w||_2^2 + eps
r'_f(c) = r_f + Delta L_f(c) / E_f
```

其中 parent 候选的 `Delta L_f=0`。每个元素在三个合法码字中，按候选的绝对损失向量
`r'_f(c)` 对字典序 `(max_f r'_f, median_f r'_f, mean_f r'_f)` 取最小者；接受后同步更新
`L_f/r_f/g_f`。不得比较未归一化的 `Delta L_f`，也不引入新的聚合权重、折间投票或逐折贪心。

沿用 v159 已有固定通道顺序，只执行一次 sweep。整个 block 完成后用五折完整二次型复核；
只有字典序目标严格改善才接受，否则同时恢复 block 的 parent 码字及 sweep 前的 `L_f/r_f/g_f`。
必须记录 attempted blocks、accepted blocks、changed codes、各折 before/after、
W-only/Both/interaction 和各 role 接受率。

### 5.3 复杂度

设总输出行数 `R`、输入宽度 `D`、block `B=64`、fold `F=5`、每折 token 数 `N`：

```text
block Gram construction: O(F * N * D * B)
one-pass refinement: O(F * R * D * B)
temporary memory: O(F * B^2 + F * R_batch * B)
dynamic activation: 与 v160 完全相同
```

禁止完整 `D×D` 逆矩阵和第二轮 sweep。Weight calibration 同机 API 增量目标不超过 v160 的
20%，只作复杂度风险门禁。

### 5.4 验证漏斗

1. **C0 零 API 审计**：计算 parent 的五折最终部署误差、fold 方差、W-only/Both/interaction
   和 role 最坏折；不据此改变目标。
2. **C1 接口/control**：Attention state 逐字段一致；未接受 block 与 parent 逐位一致；
   attempted/accepted 非零；隔离导入通过。
3. **C2 Linear compact 56**：negative cases `=0`、median `>=0`、worst-quartile `>=0`、
   7 role mean 均非负、validation/test 同号、W-only 非负、Both 不依赖负 interaction。
4. **C3 Qwen Linear default 168**：improved `>regressed`、所有 role family 非负、收益不集中
   于单层/单 role、两个 split 同号、Attention control 逐位一致。
5. **C4 跨模型封存**：GPT-2 与 OPT/Pythia 总体同号，`fc/qkv/proj/o` 不出现系统性负 family；
   跨模型只否决不调参。
6. **C5 集成与官方**：通过完整调用图后只提交一次；官方 `<=0` 则关闭 minimax A\@W，
   不试其他 fold 聚合或码字邻域。

## 6. 组合规则

Attention 与 Linear 候选必须分别获得独立官方正向，才允许进入组合讨论。组合仅做一次本地完整
集成，检查六 API、state、单文件导入和时间；由于侧向校准的 score interaction 为 1，可把
两侧官方增量相加作为结构性预期，但不得自动执行官方 2×2，也不得把本地 delta 换算为官方分。

组合候选的官方时间预算使用非负增量估计，只作否决、不作通过证明。Attention 与 Linear 候选
各自官方回传后，忽略低于 v160 的表观加速，计算：

```text
T_nonnegative = 232
              + max(0, T_Attn_official - 232)
              + max(0, T_Linear_official - 232)
```

`T_nonnegative >285s` 时不提交组合，先做数学等价的实现复杂度审计；`<=285s` 仍只是必要条件，
不能证明官方 `<300s`。完整本地集成的 API delta 和算子类型审计作为辅助否决门禁，但不得换算
官方秒数。v163/v164 的 `-28s` 共享抵扣和榜首 `290s` 均不作为本候选可行性的证据。

若首轮 Attention + Linear 官方合计仍 `<1000`，说明当前两种目标修正不足以承担 4233 分差，
停止局部扩展并转向新的编码架构或外部高分方案分析。若合计 `>=1000`，以新官方父版本重新做
一次证据审计，再决定 M2 的第二个独立机制。

### 6.1 与本地调参隔离的外部搜索工作流

转向输入可在 A/C 裁决前准备，但必须与本地调参隔离。搜索范围冻结为截至 `2026-09-03` 已公开
的论文与对应官方源码，优先检查 KV-cache 量化、静态 rotation 和 outlier prevention；不得因
A/C 的中间结果临时扩展关键词或来源。每项机制统一按以下顺序登记：

1. 能否映射到六 API 和合法 CPU state；
2. 是否保持动态 `O(TD)`，且不引入 per-call Gram、迭代或小张量 Python 循环；
3. 是否兼容 HiF4 四元素码字与现有层级 scale；
4. 是否与已关闭的 full64/Householder/动态 Gram 族数学上不同；
5. 理论预期、额外状态、动态算子和主要失效模式。

先保存完整候选清单，再按上述布尔门禁筛除；每个机制族最多保留一个、总计最多三个实现提案，
并在读取任何本地 panel 前冻结排序和第一候选。该工作流只读文献与源码、不运行本地 panel、
不产生版本号；其产物只能为 §6 转向提供预注册输入，不能根据后续 holdout 或官方结果改排序。

## 7. 固定执行顺序与产物

1. 归档 v162/v163/v164 侧向校准计划并修正索引（已完成，commit `66b3336`）；
2. A0-A5：跨折收缩 Softmax-Fisher；
   并行（与本地调参隔离）：按固定协议准备外部机制清单（§6.1），不依赖 A/C 裁决；
3. A 官方强正向后才考虑 B；A 失败则跳过 B；
4. C0-C5：Linear 跨折 minimax 部署 A\@W；
5. 两侧分别官方正向后才做一次组合审计；
6. 每个实验保存不可覆盖的源码、SHA、JSON、Markdown report、API/wall、scope、control、
   attempted/accepted 和明确的 `RETAINED/REJECTED/ERROR/TIMEOUT`；未提交官方保持
   `unregistered/NA`；
7. 每次实质更新后 `git diff --check`、提交、push 并核验工作区。

## 8. 执行记录（2026-09-03）

- A0 reachability 通过：compact 四层中 Q `3/4`、K `4/4` state 发生变化，校准总计
  `10.576s`。

- A2 compact 被否决：相对 v160，Attention mean/median delta
  `-0.007813325/-0.004871463`，`1+/3-/0=`，worst `-0.027699490`；QK-only、probability
  MSE/KL 均恶化，V control 为 0。A 为 `REJECTED`，不进入 A3-A5、不做邻域调参。

- B 按依赖取消：A 未获得官方强正向，因此不实现、不运行。

- C0/C1 从 v160 干净实现：单个 `fc_gate [4864,896]` state attempted `68096`、accepted
  `65460`、changed codes `547226`、rollback 残留 0，Activation state 与 scale/lv2/lv3
  control 逐位一致。校准 `3.421s` 对 parent `2.160s`（`1.584×`）；`1.20×` 仅为工程风险
  目标，不作硬否决，候选继续进入 C2。

- C2 正式否决 C：Linear compact mean/median delta `-0.088775/-0.088583`，
  `4+/52-/0=`，worst `-0.216586`、worst-quartile mean `-0.164813`；七个 role mean、test、
  validation 和 W-only delta 全负。28 个 cross-holdout pair 为 `2` 对双正、`26` 对双负。
  C 为 `REJECTED`，不进入 C3-C5、不做 fold/Jacobi/coverage/邻域变体、不提交官方。

- A/B/C 均未产生官方候选；根 `solution.py` 未改，下一计划必须转向新的编码架构。
