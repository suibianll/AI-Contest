# HiF4 唯一活跃优化计划 v4：L6 压缩跨 block 精度路线

> 状态：**ACTIVE**
> 建立日期：2026-08-31
> 适用根：`D:/工作内容/AI竞赛/solution.py`
> 当前精度 parent：v117 L6c full `G_64` hierarchy coordinate sweep
> 根 `solution.py` 规范 LF SHA256：`8746b8026495cb56a3dc1d622e463f89226b23e3206e2202bd468f45530d952c`
> 主目标：在合法 HiF4 五字段和 CPU static state 约束内，处理已测得的跨 64-channel
> block coupling，继续提升 Qwen full-layer `linear_mean`；Attention 只作回归检查。

## 1. 唯一执行规则

本文件是唯一可执行计划。执行优化时只读取本文件、根 `solution.py`、最新可复现
JSON/日志和官方规则；`docs/superpowers/archive/plans/` 只作历史证据，不产生顺序。

每个候选严格按以下顺序执行：

1. 合成测试：验证合法五字段、二次型增量与暴力重算一致、跨 block proposal 的
   PSD/finite fallback、原子层级写回、CPU state、state 节点 `<4096`；
2. 先跑 Qwen 五层 `{0,5,11,17,23}` × 七 role screen，至少两折 calibration；
3. 屏幕结果必须记录完整命令、cache/model/data revision、候选/parent SHA、每层和
   每 role 分数、API/wall 时间；
4. screen 有明确正向信号才运行一次 Qwen 24 层 full-layer；full 只以固定 Qwen
   `linear_mean`/panel 晋级，Attention 只防回归；
5. 每个成功、失败、无效、超时候选都先归档完整源码、JSON、日志和 README，再改
   本计划账本；拒绝候选不能留在根；
6. accuracy-first 阶段暂不以 420s 否决精度，但记录时间；所有 L6 方向完成后再
   开 C1 时间压缩，最终官方硬门仍为 `<420s`；
7. L6a–L6e 完成或有充分 `not actionable` 证据后，立即归档本计划并在同一提交
   创建下一份唯一 active 计划，不能在本文件追加新的“下一步”。

## 2. 基线、目标和停止量化

固定配置：Qwen2.5-0.5B、24 层、`seq=128`、`calib=2`、`test=4`、`amax6`、CPU、
只读 cache `artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`。

| 指标 | v117 |
|---|---:|
| Linear mean | `0.5095117268`（screen `0.53329460`） |
| Attention mean | `0.8420394885` |
| Qwen panel | `295.785829395641` |
| native total | `423.227671311092` |
| API time | `2019.475204s` |
| 规范 LF SHA | `8746b8026495cb56a3dc1d622e463f89226b23e3206e2202bd468f45530d952c` |

当前到 `linear_mean=0.9` 仍差 `0.3904882732`，需减少当前剩余归一化误差约
`79.61%`；L5e 证据表明固定 frame 的单侧理想臂最高仅 `0.8188905`，255-code
scale oracle 的总体余量远小于这个缺口。L6 不承诺 0.9，而是验证压缩跨 block
表示是否还能产生可泛化增益；连续两个候选无正向时立即停掉对应族。

## 3. 合规数学边界

权重 `W∈R^{m×d}`、激活 `A∈R^{n×d}`。允许的离线权重目标为

\[
H_A=A^TA,\qquad J_W(Q)=\operatorname{tr}[(Q-W)H_A(Q-W)^T].
\]

在线激活只能使用校准阶段冻结的静态权重统计。当前真实部署目标是

\[
G_q=W_q^TW_q=B+R,\qquad
J_A(E)=\operatorname{tr}(E G_qE^T),
\]

其中 `B=blockdiag_64(G_q)`，`R` 是跨 block 残差。L5e 测得 896 输入宽度的

\[
\rho_{off}=\frac{\|R\|_F}{\|G_q\|_F}
\]

平均为 weight `0.76125`、calibration activation `0.88382`，所以只用 `B` 会漏掉
大量耦合；但是不得把 `A@W` 输出、输出残差或 test/holdout 信息写入
`activation_state`。

所有 L6 proposal 必须满足：

- `R`/`H_A` 只作为 calibration/offline 的 operand-local 统计；
- state 只保存 CPU、finite、无 token/out_features 维度的静态压缩参数；
- 近似目标只生成 proposal，写回前重新解码合法 `scale_factor/scale_lv2/scale_lv3/sign/mant`，
  并用真实部署 `G_q` 逐行 gate；
- 禁止 joint residual `X_tW_t^T-X_qW_q^T`、模型/role/test 分支、全 dense Gram
  新增 state 和未经压缩的全 block beam。

## 4. L6 执行队列

### L6a：窄输入 rank 扩展（rank-16 global LRH）

**状态：done（2026-08-31；v115 accepted precision parent）。**

当前根只在 `d<=1024` 保存 rank-8 off-block factor `U_8`，用

\[
R\approx U_rU_r^T,
\qquad \nabla J=(B+U_rU_r^T)e
\]

生成离散 activation proposal，再用完整 `G_q` gate。先只把 `r=8` 提到 `r=16`，
保持 power steps、block cap、mix 和所有路径不变；若正向，再单独测试 `r=32`，不能
一次叠加。state 节点仍为一个 CPU tensor，合成测试要检查 `U_r` 的 shape、finite
和退化到 `B` 的 fallback。

执行结果：只把 `_ACT_GLOBAL_LRH_RANK` 从 8 改为 16；30 项定向回归通过，静态/运行时
compliance 均为 0 violations。screen Linear mean `0.53284175`，较 v111
`0.53188695` 增加 `+0.00095480`，触发 full-layer；v115 full Linear
`0.5090910148`、Attention `0.8420394885`、Qwen panel `295.6806514001`，较 v111
分别 `+0.0007927147`、`0`、`+0.1981786718`，API `716.482861s`、wall `748.372825s`。
候选已完整归档于 [`v115 L6a`](../../../solutions/20260831_v115_l6a-rank16-accepted_score295.680651_time716s/)，
证据为 [`screen JSON`](../../../artifacts/real_model_suite/l6a-rank16-stratified-qwen.json)
和 [`full JSON`](../../../artifacts/real_model_suite/v115-l6a-rank16-qwen-full.json)。
v115 成为新的 precision parent；时间仍只作探索记录，下一步为 L6b。

### L6b：宽输入 rank-4 cross-block factor

**状态：done（2026-08-31；v116 accepted precision parent）。**

针对 `d>1024` 的结构形状，增加一个受限 rank-4 随机 range factor：

\[
U_4=\operatorname{RangeIter}(W^TW-B),\quad
\widehat G=B+\lambda U_4U_4^T.
\]

只用于生成最多 4 个高损 block 的 proposal，最终仍以完整 `G_q` 行级 gate；不把
`A@W` 或输出误差放入 state。`d=4864,r=4` 的 factor 约 `4864×4×4≈76KB`，
作为一个 CPU tensor，不泄露 token/out_features；若根已有 full deployment Gram，
不得再复制第二份 dense 矩阵。

验收：先在 synthetic `d=2048/4096/4864/8192` 验证速度无关的数值正确性、state
节点和 fallback，再做 Qwen screen。只要 proj/宽层有负向就拒绝整候选；不允许靠
单一 role 的正向覆盖全局回退。

执行结果：新增 `d>1024,d<=8192` 的 rank-4 range factor，窄输入 rank-16 路径不变。
32 项定向回归通过，`guard_solution_file` 的 static/runtime violations 均为 0；
screen Linear mean `0.5330906465`，较 v115 screen `+0.00024890`，触发 full-layer。
v116 full Linear `0.5093045894`、Attention `0.8420394885`、Qwen panel
`295.7340450430`，较 v115 分别 `+0.0002135746`、`0`、`+0.0533936429`；唯一
正向角色为宽 `proj`（`0.4200260922→0.4215211142`），API `739.424609s`、wall
`771.865345s`。候选完整归档于 [`v116 L6b`](../../../solutions/20260831_v116_l6b-wide-rank4-accepted_score295.734045_time739s/)，
证据包括 [`synthetic`](../../../logs/execution/2026-08-31-l6b-wide-rank4-synthetic.md)、
[`screen JSON`](../../../artifacts/real_model_suite/l6b-wide-rank4-stratified-qwen.json)
和 [`full JSON`](../../../artifacts/real_model_suite/v116-l6b-wide-rank4-qwen-full.json)。
v116 成为新的 precision parent；时间仍只作探索记录，下一步为 L6c。

### L6c：完整 `G_64` 指导的层级坐标求解

**状态：done（2026-08-31；v117 accepted precision parent）。**

当前 `_solve_hierarchy` 在选择 `lv2/lv3` 时主要使用逐 4 元组平方误差，`gram64`
只在 offset 选择阶段介入。候选固定 scale 后，对一个 block 内的 8 个 8-channel
组做有界坐标更新：每次只重算一个合法 `lv2/lv3` + mantissa 原子组合，使用

\[
\Delta J=2e^TG\Delta q+\Delta q^TG\Delta q
\]

的精确增量；每 block 最多 1 sweep，最多处理当前既有的 block budget。weight 侧
只写 `weight_params`，activation 侧只以静态 `G_q` 生成 proposal，二者都要用部署
Gram gate。外部 4×4 group solver 已证明局部 `J_group` 下降不代表 `J_64` 下降，
因此本候选不得使用 group-only acceptance。

验收：合成暴力枚举必须逐字段一致；screen 至少超过 v111 且所有异常回退 parent；
若只改善局部 Gram、完整 `G_64` 不改善，标记 `not actionable`。

执行结果：固定 E6M2 scale，对每行最多 4 个高损 block 做 1 sweep 的 `lv2/lv3`
坐标更新；每次候选都重编码 hierarchy 并使用精确 `G_64` 增量，随后由完整部署
`G_q` 逐行 gate。33 项定向测试通过，静态/运行时 compliance 均为 0；screen
Linear mean `0.5332946034`，较 v116 screen `+0.0002039570`，7 个 role 均不降，
触发 full-layer。v117 full Linear `0.5095117268`、Attention `0.8420394885`、
Qwen panel `295.7858293956`，较 v116 分别 `+0.0002071374`、`0`、`+0.0517843527`；
API `2019.475204s`、wall `2051.884441s`，仅作 accuracy-first 探索记录。候选已归档于
[`v117 L6c`](../../../solutions/20260831_v117_l6c-g64-hierarchy-accepted_score295.785829_time2019s/)，
证据为 [`synthetic`](../../../logs/execution/2026-08-31-l6c-g64-hierarchy-synthetic.md)、
[`screen`](../../../logs/execution/2026-08-31-l6c-g64-hierarchy-stratified.md) 和
[`full`](../../../logs/execution/2026-08-31-v117-l6c-g64-hierarchy-qwen-full.md)。
v117 成为新的 precision parent；下一步执行 L6d。

### L6d：结构化跨 block factor（block-circulant / DCT 低秩）

**状态：pending。**

若 L6a–c 均无增益，测试不保存逐通道 `U` 的结构化近似：将 block 间 Gram 按相对
距离聚合为少量 `K_s∈R^{64×64}`，或用固定 DCT basis `V` 与每个 block 的小系数
近似

\[
R_{ij}\approx\sum_{s=1}^{S}c_{i-j,s}K_s,
\qquad S\le4.
\]

proposal 用结构化矩阵向量乘法，state 只保存 `S` 个 64×64 CPU 张量和系数；禁止
直接把所有 block-pair Gram 写入 state。先在合成 block-circulant、随机低秩和真实
校准统计上测近似误差，再跑 Qwen screen；若近似误差或 screen 回退，立即归档。

### L6e：压缩跨 block 表达 checkpoint

**状态：pending。**

汇总 L6a–d 的 screen/full 结果，重新计算 `ρ_off`、proposal recall、完整 `J_64`
下降和每个候选的 state 成本。若所有压缩表示都没有跨 fold 正向，归档 L6 计划并
记录“当前 state 接口下跨 block 压缩不可行动”；若某个方向正向，保留最高 Qwen
parent，另建下一计划做泛化/多模型审计。只有 checkpoint 后才能进入 C1 时间压缩。

## 5. 版本与候选账本

只登记固定 cache 的 full-layer accepted parent；screen、oracle、rejected candidate
在对应 execution log 和 `solutions/` 归档 README 中登记。新候选编号从 v115 开始
（v114 是已归档的外部 sampling screen）。

| 版本 | Linear | Attention | panel | API time | 状态 |
|---|---:|---:|---:|---:|---|
| v110 | 0.5073395278 | 0.8420394885 | 295.242780 | 701.90s | 前一精度 parent |
| v111 | 0.5082983001 | 0.8420394885 | 295.482473 | 726.09s | 历史 precision parent |
| v115 | 0.5090910148 | 0.8420394885 | 295.680651 | 716.48s | 前一 precision parent；L6a accepted |
| v116 | 0.5093045894 | 0.8420394885 | 295.734045 | 739.42s | 前一 precision parent；L6b accepted |
| **v117** | **0.5095117268** | **0.8420394885** | **295.785829** | **2019.48s** | **当前 precision parent；L6c accepted** |

## 6. 完成和换计划条件

L6a–L6e 完成后必须把本文件标记 `COMPLETED`，写明每个方向的结果和证据，移入
`docs/superpowers/archive/plans/`，更新 `docs/superpowers/plans/README.md`、根
README、`docs/current-solution-status.md`、算法清单和 `solutions/README.md`，并在
同一提交创建下一份唯一 active 计划。归档计划不可继续追加新的执行步骤。
