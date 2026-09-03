# v180 后下一阶段计划：Linear 融合 rank-2 残差重分布，Attention 证据冻结

> 日期：2026-09-04
>
> 状态：**IMPLEMENTED — L-R2 已实现并归档为 v182（SHA `F3E39E99...A438`），
> 硬检查全部通过（reachability 全 1、vtu_cross_max ~1e-8、Attention 与 v180 逐位一致），
> 本地配对 v180 跨模型非负（Qwen default +0.000020、GPT-2 +0.001171、OPT +0.025632），
> 已请求官方评测（配额 3/10）；等待官方回传裁决。**
>
> 完整官方父：`v180 = 17597/242s`，源码 SHA256
> `2BA401228CACC49FADC7C78AC388616F490F3DC31CECA98F1DDE53C64EBF8AA3`。
>
> 独立侧向锚点：`P_L = v166 = 4590/226s`、`P_A = v168 = 14005/210s`、
> 双标准零点 `v162 = 1001/146s`。
>
> 当前榜首：`21765/290s`；差距 `4168`；v180 距 300s 硬限制余量 `58s`。
>
> 官方配额：已使用 `2/10`（v176、v180），剩余 8。本计划最多消耗 1 个新名额。

## 1. 计划结论

下一阶段只注册一个正式算法候选：

```text
L-R2：把 v166 的 rank-1 连续域等价残差重分布，扩展为一次融合的 rank-2 正交更新。
```

Attention 侧保持 v180 的 v168 A1 + D1，不创建新候选。原因不是“Attention 不重要”，而是
当前六 API 和五字段 state 下可表达的低成本自由度已经被覆盖：Q/K 的 scale、温度、折叠、
中心化、旋转、细粒度 gain、阈值和动态 Gram 均已有裁决；V 的 importance、bias、multiplier
和 per-token scale 也已分别证明无效、负向或不可表达。此时强行凑一个 Attention 版本只会
消耗配额并增加过拟合风险。

本计划遵循用户确定的裁决方式：本地 proxy 不设严格准确率门槛。只要 L-R2 的接口、合法
state、有限输出、机制 reachability、连续域不变量和 Attention control 正常，就提交一次
官方评测；是否提升完全由相对 v180 的官方分数决定。

## 2. 为什么选择 L-R2

### 2.1 已有官方规律

| 规律 | 直接证据 | 对本计划的含义 |
| --- | --- | --- |
| 两侧分数近似可加 | v160 端点残差 1；v175 interaction=0 | 只改 Linear，可直接用完整分数差归因 |
| 低自由度静态机制更稳 | v168 A1 `+60`、v166 rank-1 `+3`、v180 D1 `+3` | 沿官方正向解析结构扩展，不重新搜索大空间 |
| 高自由度 A@W 不泛化 | v138/v140/v147 官方反转；cross-fold minimax 本地系统负 | 不直接优化输出，不用 holdout 选参数 |
| 细粒度自由度常负 | C2 channel-group、D2 per-Q-head、C1 K-channel 均负 | 不做 per-channel/per-head/per-role 路由 |
| 码字与 full-rank 几何已闭 | Babai、Trellis、full64、Householder、Kron-CAT 均负 | 不改 HiF4 码本，不重启 block/rank/seed 网格 |
| 在线小矩阵操作有严重时间风险 | v161/v165 本地强正但官方 timeout | 动态只允许一次固定 rank-2 融合乘法，无候选循环/Gram |
| v180 Linear default 分解 | Both `0.636590`，W-only `−389.24`、A-only `−130.52`、interaction `+520.40` | 收益来自 W/A 成对坐标抵消；必须原子更新 W、A 和最终 Gram |

### 2.2 为什么 rank-2 仍未被已有失败覆盖

- v166 的 rank-1 机制在当前官方权重上获得了明确正分，因此其核心假设——残差主方向的
  连续域等价重分布可以改善 HiF4 落点——没有被否定。
- Householder、CAT、Kron-CAT 改的是整块坐标几何；L-R2 只在两个全局残差方向上做低秩
  近恒等更新，表达空间和动态复杂度都不同。
- Babai、Trellis、full64 和 cross-fold minimax 直接改码字或输出域目标；L-R2 不改解码器，
  不构造 `A@W`，只改变等价坐标后再运行父编码流程。
- rank-1 的系数、fold、方向估计已经固定。L-R2 不是扫描 rank=1/2/3/4，而是一次性检验
  “第二个与父方向正交的残差模式是否仍有官方增益”。无论结果正负，本计划都不继续 rank-3。
- v180 最新 168-case Linear 分解表明，任何只改 W 或只改 A 的解释都会严重失真；当前收益
  几乎完全来自双侧 interaction。这既支持继续研究成对等价变换，也意味着第二方向必须与
  第一方向严格正交，并在编码前同步更新 Weight、calibration Activation 和 Hessian。

### 2.3 为什么不继续 A1/D1 邻域

v168 A1 首次引入 logits gain 获得 `+60`；v180 只改变 Q/K 分配得到 `+3`；D2 将 gain
细化到 Q head 后本地负，C2 细化到 channel-group 后也负。这组成清晰的容量规律：

```text
per-KV-head shared gain 有效
→ 同一 gain 的 Q/K 分配只剩微增益
→ 再增加 head/channel 自由度开始回归
```

因此 alpha、Q-head、channel-group、模型/layer/length 路由全部停止。继续微调这些变量无法
合理解释 4168 分差，也违反一次机制一个预注册配置的防过拟合纪律。

## 3. Attention 冻结的理论边界

### 3.1 Q/K 自由度已经覆盖

当前 Q/K 在线编码允许的主要操作为 multiplier、permutation、rotation、pair transform、
centering、importance、offset/refine：

- multiplier 的乘积自由度由 A1 使用，Q/K 分配自由度由 D1 使用；
- per-Q-head、channel-group 细分由 D2/C2 否定；
- K outlier/channel equalization 由 C1 官方负向关闭；
- random/block rotation、Householder/QuaRot 对照由 C3 和历史实验否定；
- mantissa threshold 由 A4/v171 官方 `−348` 关闭；
- fixed offset/静态 scale 编译由 A3/v170 系统负关闭；
- dynamic Gram 精化虽本地强正，但 v161/v165 官方 timeout，且静态低秩重构 v167 负。

### 3.2 V 自由度为什么不能作为新候选

一个 token、一个 head 的 64 维恰好对应一个 HiF4 block：

- per-head importance 在块内是常数，只整体缩放目标，不能改变 argmin 码字；
- per-channel importance 会增加高自由度，并缺少可信的 attention-output 曲率泛化证据；
- per-token scale 需要在线独立表，五字段 state 无法保存；
- V bias 已由 v169/A2 在 Qwen 和 GPT-2 一致负向关闭；
- V multiplier 在 `_nvfp4_to_hif4` 中是编码前缩放，`_dequantize_hif4` 没有逆缩放。Q/K
  的这类变化经过 softmax 还能表现为温度校正，V 则会直接改变 `O=P@V` 的幅度，没有后续
  可补偿 API，因此不是连续域等价变换。

### 3.3 Attention 重新开启条件

本计划期间 Attention 只做 control。未来只有出现以下任一新证据才允许另建计划：

1. 官方规则增加可表达的解码后补偿、额外 state 字段或下游输出变换；
2. 新算法能证明在现有五字段内连续域等价，且不属于 multiplier/rotation/threshold/offset
   的已闭合数学域；
3. 能把 v161 的动态精化完整编译成无 per-call Gram、无候选循环、无模型/长度路由的规则，
   并且该规则不等价于 v167 已失败的静态低秩码本；
4. 新的官方单变量结果推翻当前某个关闭结论。

“找到新的论文名字”或“本地某几个 case 正向”本身不满足重新开启条件。

## 4. L-R2 数学定义

### 4.1 v166 rank-1 父机制

在最终 smooth/permutation/Hadamard 部署坐标中，v166 构造：

```text
R1 = I + u1 v1^T
v1^T u1 = 0
u1 = 0.25 d2
v1 = d1
```

其中 `d1,d2` 来自 activation/weight 基础 HiF4 量化残差的归一化协方差主方向。因为
`v1^T u1=0`：

```text
R1^-1  = I - u1 v1^T
R1^-T  = I - v1 u1^T
A1     = A R1 = A + (A u1) v1^T
W1     = W R1^-T = W - (W v1) u1^T
A1 W1^T = A W^T
```

### 4.2 rank-2 融合推广

保留父 rank-1 的 `u1,v1`，只在其正交补空间提取第二对方向：

```text
v2 = d3
u2 = 0.25 d4

U = [u1, u2]  in R^(D x 2)
V = [v1, v2]  in R^(D x 2)
V^T U = 0_(2x2)
R = I + U V^T
```

由 Woodbury 公式：

```text
(I + U V^T)^-1 = I - U (I + V^T U)^-1 V^T
```

而 `V^T U=0`，故：

```text
R^-1 = I - U V^T
```

部署变换为：

```text
A' = A + (A U) V^T
W' = W - (W V) U^T
```

连续域严格保持：

```text
A' W'^T = A R (W R^-T)^T = A R R^-1 W^T = A W^T
```

L-R2 不试图改变浮点模型，只改变 A/W 在 HiF4 离散格点上的位置。

### 4.3 Gram/Hessian 的精确更新

Linear 编码目标必须使用最终部署坐标的 Gram。若 `G=A^T A/N`：

```text
G' = R^T G R
   = G
   + V U^T G
   + G U V^T
   + V (U^T G U) V^T
```

`weight_group_gram`、importance、`h_inv` 和后续 GPTQ 都从 `G'` 重建。禁止使用变换前 Gram，
也禁止在权重量化完成后才追加 rank-2 state。

## 5. 第二残差方向的固定估计

### 5.1 残差算子

完全复用 v166 的定义。对每个固定 calibration fold：

```text
Ea = A_deploy - Q_parent(A_deploy)
Ew = W_deploy - Q_parent(W_deploy)

C(x) = Ea^T(Ea x)/||Ea||F^2 + Ew^T(Ew x)/||Ew||F^2
```

不显式构造 `D x D` 残差协方差，用 matvec 做 power iteration。

### 5.2 保留第一对、提取第二对

1. 先按 v166 原代码生成 `v1=rank1_v`、`u1=rank1_u`，不改变其 fold、符号或 median 逻辑；
2. 令 `b1=v1`，`b2=u1/||u1||`，构造正交补投影
   `P(x)=x-b1(b1^T x)-b2(b2^T x)`；
3. 在算子 `P C P` 上固定运行 128 次 power iteration 得到每 fold 的 `d3`；
4. 再从 `d3` deflate，固定 128 次得到 `d4`；
5. 每个方向先按最大绝对坐标定号，再与 fold 0 做内积符号对齐；
6. 对 folds 逐分量 median；随后只对新方向做 Gram-Schmidt：先投影掉 `v1/u1`，再令
   `d4` 投影掉 `d3`；
7. 固定 `v2=d3`、`u2=0.25*d4`；最后再投影一次，保证四个交叉内积接近 0。

固定项如下，不允许调参：

```text
rank = 2
coefficient = 0.25        # 继承 v166
power iterations = 128   # 继承 v166
folds = v166 原 even/odd folds
aggregation = component-wise median
```

不比较多个 rank、系数、seed、fold、聚合方式或方向配对。

## 6. 具体代码修改

### 6.1 基线与文件

- 从根 `solution.py` 复制；根当前与 v180 SHA 一致。
- 候选实现完成且 reachability 非零后，归档为
  `solutions/20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA/solution.py`。
- 在实现成功前不创建正式版本目录，不修改 v180 归档。

### 6.2 常量与 helper

只新增或替换以下最小结构：

```python
_WEIGHT_RESIDUAL_RANK = 2
_WEIGHT_RESIDUAL_COEFF = 0.25
_WEIGHT_RESIDUAL_POWER_ITERS = 128

def _rank2_residual_complement(
    act_residuals, weight_residual, parent_u, parent_v
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ...
```

v166 局部 `_rank1_top2` 可保留以确保第一列算法逐位不变。新 helper 只生成 `u2,v2` 和统计；
不为一次操作创建额外类、策略接口或通用 rank 框架。

### 6.3 Weight calibration 接线

在 `hif4_calibration_and_quantize_weight` 的 v166 rank-1 区块：

1. 完成父 `rank1_u/rank1_v`；
2. 调用 `_rank2_residual_complement` 得到第二对；
3. 组成 `U=[u1,u2]`、`V=[v1,v2]`；
4. 一次更新 `weight_smooth -= (weight_smooth @ V) @ U.T`；
5. 对全部 transformed calibration samples 一次更新
   `sample += (sample @ U) @ V.T`；
6. 用 rank-2 公式更新 `gram_full/h_x_smooth/weight_group_gram`；
7. 继续执行父 GPTQ、hierarchy、importance 和单次 encode；
8. activation state 保存 CPU float32 `residual_u:[D,2]`、`residual_v:[D,2]`。

### 6.4 Dynamic activation 接线

用一个融合 rank-2 运算替换 v166 的 rank-1 运算：

```python
u = activation_state["residual_u"].to(device=dense.device, dtype=dense.dtype)
v = activation_state["residual_v"].to(device=dense.device, dtype=dense.dtype)
dense = dense + (dense @ u) @ v.transpose(0, 1)
```

之后完全沿用父 HiF4 encoder。禁止：

- 两个 Python rank-1 循环；
- per-token/per-block rank 选择；
- 在线 Gram、SVD、QR、矩阵逆或候选比较；
- 因 layer/role/模型不同而关闭第二列。

## 7. 算法流程拆解

```text
输入：dense Weight W、calibration NVFP4 activations
  │
  ├─ 1. 执行 v180 原有 smooth / permutation / Hadamard，得到部署坐标 Wd, Ad
  │
  ├─ 2. 用父 HiF4 codec 得到 Ew = Wd-Q(Wd)、Ea = Ad-Q(Ad)
  │
  ├─ 3. 原 v166 流程得到 u1,v1
  │
  ├─ 4. 在 span(u1,v1) 正交补空间对残差算子做固定 power iteration
  │       └─ 各 fold 得到 d3,d4 → sign align → median → orthogonalize
  │
  ├─ 5. U=[u1,0.25d4]，V=[v1,d3]，核验 V^T U≈0
  │
  ├─ 6. Wd ← Wd-(WdV)U^T；Ad ← Ad+(AdU)V^T
  │
  ├─ 7. 精确更新最终坐标 Gram/Hessian
  │
  ├─ 8. 运行 v180 原 Linear GPTQ/HiF4 encode，输出合法 Weight 五字段
  │
  └─ 9. state 保存 U,V；dynamic activation 用一次融合 rank-2 更新后编码
```

## 8. 复杂度分析

设输入维为 `D`，Weight 行数为 `M`，calibration token 总数为 `N`，动态 token 数为 `T`，
固定 rank `r=2`，power iteration 次数 `K=128`。

### 8.1 校准复杂度

每次残差 matvec：

```text
x -> Ea^T(Ea x) + Ew^T(Ew x) = O((N+M)D)
```

新提取两个方向：

```text
O(2K(N+M)D)
```

其余增量：

```text
Weight rank-2 transform      O(MDr)
calibration A transform      O(NDr)
Gram exact update            O(D^2 r)
QR/orthogonal checks         O(Dr^2)
```

不新增 `O(D^3)` 分解，不显式建立新的全残差协方差。

### 8.2 在线复杂度

```text
A @ U          O(TDr)
(A @ U) @ V^T  O(TDr)
总增量          O(2TDr)，r=2 固定
```

与 dense 64x64/CAT 的 `O(TD64)` 相比，算术常数约为其 `2/64` 量级；与 v166 rank-1 相比
约增加一列投影。state 从两个 D 向量增至两个 `D x 2` 矩阵：

```text
state = 4D float32 = 16D bytes
```

### 8.3 官方时间风险

- v166 相对 v163 的官方时间从 202s 到 226s，rank-1 增量观测为 24s；该差异同时含校准和
  测量噪声，不能线性换算。
- v180 为 242s，剩余 58s。若第二列成本与第一列同阶，理论上存在余量，但本地 CUDA 对官方
  鲲鹏机的预测已失效。
- 第一版直接使用融合 `[D,2]` GEMM，避免复制 v161 的大量 per-call 小算子失败模式。
- 时间只随真实候选一并官方测量，不提交等价 time A/B。

## 9. 防过拟合设计

1. **不使用 A@W 选方向**：方向只来自 A/W 各自的基础量化残差算子。
2. **固定低秩**：只增加一个正交模式；rank=2 后无 rank 邻域。
3. **固定正则结构**：系数 0.25、128 次迭代、even/odd folds、median 全继承 v166。
4. **校准/验证分离**：参数只读 calibration；compact/default/GPT-2/OPT 只用于冻结后的描述。
5. **无选择性路由**：不按 layer、role、shape、模型、split 或本地收益开关第二列。
6. **单机制版本**：Attention 与 v180 逐位一致，Linear 除 rank-1→rank-2 外不变。
7. **官方负即关闭**：不通过改 c、rank、seed、fold、QR 顺序或方向配对重启。

## 10. 固定验证流程

### 10.1 父版本

1. 使用现有 v180 Attention baseline JSON；不重复运行。
2. v180 Linear default baseline 已固定为 `artifacts/official_eval/v180-linear-default.json`，
   不重复运行；compact baseline 若不存在，只运行一次并保存 immutable JSON/report。
3. 父与候选使用同一 cache、panel、device、evaluator commit。

### 10.2 硬检查

以下任一失败时修实现，不改数学规则；修复前不提交官方：

- 单文件脱离仓库导入六 API；
- `reference_hif4.py` state/五字段合法；
- 无 NaN/Inf；
- `u2/v2` 非零，第二列导致真实五字段或输出变化；
- `||V^T U||F` 足够接近 0；
- 连续域相对误差处于 float32 舍入量级；
- Attention state 和输出与 v180 逐位一致；
- 动态 API 无 Gram contraction、候选循环或未限制 Python loop。

这些是实现门禁，不是准确率门禁。

### 10.3 描述性本地评测

按顺序执行：

```powershell
.venv\Scripts\python.exe evaluator/official_eval.py --solution <v182.py> --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json <v180-linear-compact.json>

.venv\Scripts\python.exe evaluator/official_eval.py --solution <v182.py> --linear-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json <v180-linear-default.json>
```

然后固定运行 GPT-2，再运行 OPT-125m 或 Pythia-160m。必须记录但不以其正负取消首次官方
测量：

- mean、median、q25/q75、worst quartile、正/负/零 case；
- validation/test 同号率；
- 七个 role、各 layer、W-only/A-only/Both/interaction；
- d3/d4 跨 fold 余弦和残差谱能量；
- calibration、dynamic activation、总 API 和 wall time。

### 10.4 官方提交与裁决

硬检查通过后提交一个完整候选。令：

```text
S0 = 17597
G_L = S(v182) - S0
fixed_linear_ratio = G_L / 3586
closure = G_L / 4168
gap = 21765 - S(v182)
```

裁决仅看官方：

- `S(v182) > 17597` 且 `<300s`：`RETAINED`，v182 成为新完整父；
- `S(v182) <= 17597` 且 `<300s`：`REJECTED`，rank 扩展族关闭；
- `>300s`：`TIMEOUT`，不计算精度，rank 扩展族关闭；
- wrong answer/非法输出：`ERROR`，只允许修复实现 bug 后以相同数学算法重试。

任何正增益都保留，不设置 `+20/+50/+100` 的晋级门槛。分数幅度只用于后续资源判断。

## 11. 结果解释矩阵

| 官方结果 | 理论解释 | 后续动作 |
| --- | --- | --- |
| `G_L > 100` | 第二残差模式有明显独立容量 | 更新完整父；重新做误差谱审计，但不直接升 rank |
| `20 < G_L <= 100` | 中等有效，证明 rank-1 非偶然 | 更新完整父；停止本族，寻找正交新机制 |
| `0 < G_L <= 20` | 微增益，残差低秩族接近饱和 | 更新完整父；明确关闭 rank-3/系数扫描 |
| `G_L <= 0` | 第二模式不泛化或破坏离散落点 | REJECTED；rank 扩展族关闭 |
| timeout | 官方硬件不接受第二列动态成本 | TIMEOUT；不降 rank/缩 fold 重试 |

该表不预测结果，也不改变“任何官方正增益均保留”的规则。

## 12. 预期效果与边界

理论上，rank-2 扩大了 rank-1 能重分布的残差子空间；如果残差协方差的第三、第四方向在
calibration folds 和隐藏数据间稳定，它应比 v166 捕获更多 HiF4 格点误差。同时连续模型输出
严格不变，分布漂移风险显著低于直接 A@W 拟合。

但已有官方正增益只有 v168 `+60`、v166 `+3`、v180 `+3`，不能据此承诺数百或数千分。
本计划最多证明第二低秩模式是否存在；即使获得 100 分，距离榜首仍约 4068 分。若 L-R2
只有微增益或负向，结论应是现有 API 下的局部结构接近饱和，而不是继续添加 rank。

因此，本计划是一个有明确数学假设、明确官方判据和明确终止条件的单候选阶段，不把“继续
尝试”伪装成可达 21765 的保证。

## 13. 产物与记录

实现后至少生成：

- `solutions/20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA/solution.py`
- 同目录 `result.md`，记录父/候选 SHA、公式、复杂度、reachability 和结果
- `artifacts/official_eval/v182-compact-linear.json`
- `artifacts/official_eval/v182-linear-default.json`
- 对应 `logs/official_eval/*.md`
- `logs/execution/2026-09-04-v182-rank2-linear.md`

官方回传后同步 `AGENTS.md`、根 `README.md`、`docs/current-solution-status.md`、
`solutions/README.md` 和本计划。原始 JSON/report 不覆盖。

## 14. 完成条件

以下条件全部满足后归档本计划：

1. L-R2 获得 `DUPLICATE/ERROR/REJECTED/TIMEOUT/RETAINED` 中的最终状态；
2. 官方配额、完整父、榜首差距和时间余量已更新；
3. Attention 保持 v180 control，没有创建未授权的邻域候选；
4. 源码、SHA、JSON、report、execution log 和 result 完整；
5. `git diff --check`、commit、push、最终 `git status` 均完成。
