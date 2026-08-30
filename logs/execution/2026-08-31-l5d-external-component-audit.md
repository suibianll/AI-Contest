# L5d 外部 hif4 组件审计（2026-08-31）

## 范围与可复现信息

- 外部仓库：`https://github.com/youxilee/hif4.git`
- 审计版本：`dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`（remote `main`，浅克隆）
- 外部 `solution.py` LF SHA256：`47e8894dc49bac7745be3a3f8bbaac916a262c8cca391bf15411e920d05a0b7c`
- 当前根 v111 `solution.py` LF SHA256：`6b229081121c4a7edd69575c93dc01488be8f8b5e1479007522421e93e1adc57`
- 本地根：`D:/工作内容/AI竞赛/solution.py`
- 目标：只找 operand-local、可单变量验证且满足现有 state/compliance 的精度增益；不直接把外部整套实现叠加进根。

审计结论先行：codec、E6M2 rounding、NVFP4 dequant 和固定尺度层级的基础实现已经数值一致；外部 joint X/W residual 虽然在外部 changelog 中有增益，但在本地被 runtime guard 判定为跨残差，并且旧 C70 在 OPT/Qwen 回归，不能迁移。外部 H32/H64 block-S 也已有跨模型灾难性回归。当前唯一适合做单变量根候选的是 `_sample_rows` 的采样策略；外部 group-Gram hierarchy 是值得做同目标对照的研究方向，但不能仅凭外部代码直接替换。

## 逐组件差异

| 组件 | 外部 v2.7 | 当前根 v111 | 数值/合规证据 | 决策 |
|---|---|---|---|---|
| E6M2 encode/decode | BF16 `amax/7` 后 `torch.round`，同一 codebook | 同样的 BF16 scale 和 codebook | 随机审计 `e6m2_code_max_diff=0`、decode max diff `0.0`、standard-scale code diff `0` | `not actionable`，不迁移 |
| NVFP4 dequant | 调用标准 dequantizer 后转 FP32 | 同一 dequantizer | 随机审计 max diff `0.0` | `not actionable` |
| 固定尺度 hierarchy | `group_gram` 下按 4 元组求二次型，支持精确 lv2/lv3 | 先用未加权逐组平方选择 lv2/lv3；若有 `gram64`，再用完整 64×64 Gram 选 offset | 固定 scale 随机对照：逐字段 lv2/lv3/mantissa max diff `0`；loss max diff `9.31e-10`（浮点误差） | 基础层级不迁移；group-Gram 另做同目标实验 |
| 行采样 | `step=ceil(rows/limit)`，`x[::step][:limit]` | `linspace(0, rows-1, limit)` 后 round，覆盖首尾 | rows=100、limit=16：外部行号 `0,7,14,...,98`；根为均匀首尾分布 `0,7,13,20,...,99` | 形成 L5d 单变量 sampling candidate |
| block transform | 每 64 block Hadamard，可用 seed 生成平衡随机符号 | 全局确定性 signed-FWHT，加 L5a block-local permutation | Hadamard 正交误差 `0`；根 signed-Hadamard/permutation 已在 v111 screen 正增益 | 不直接拷贝随机 seed；避免扩大自由度 |
| joint X/W residual | 用 `R=X_tW_t^T-X_qW_q^T` 做 Gauss-Seidel 更新，再量化 W | 没有输出残差更新 | 旧 C70 本地 runtime guard：2 个 cross residual violation；C70 GPT2 +6.27 但 OPT −0.616、Qwen −6.992 | `not actionable`，保留为历史反例 |
| H32/H64 block-S | proj 专属 H32/H64 候选和 final quantizer ranking | 无此分支 | 旧 C71 OPT `−139.324449`，虽个别 GPT/Qwen gain，跨折不稳 | `not actionable`，需新表示/新 gate 才能重开 |
| state/device | 外部 state 含 `importance、gram、offsets、error_threshold、accept_margin` 等 CPU 张量 | 根 state 为 `smooth_inv、permutation、gram64、deployment_gram64` 等 CPU 张量 | 当前 compliance 要求 state CPU、禁止输出残差；外部 joint 路线不满足 | 不迁移 device/state 结构 |

## 关键数学核查

### Codec 与 scale

标准尺度是

\[
c=\operatorname{round}_{E6M2}\!\left(\operatorname{bf16}\left(\frac{\max_i|x_i|}{7}\right)\right).
\]

只要 BF16 舍入、E6M2 codebook 和 ties-to-even 的 `round` 相同，编码结果就相同；本次随机逐元素比较没有发现任何差异，因此不存在通过“换 codec/rounding”获得精度的隐藏空间。

### 两种层级目标不是同一个问题

外部 group-Gram 对每个 4 元组误差 \(e_g\) 最小化

\[
J_{group}=\sum_{b,a,c}e_{b,a,c}^{\mathsf T}G_{b,a,c}e_{b,a,c},
\qquad G_{b,a,c}\in\mathbb R^{4\times4}.
\]

根的部署目标是完整 64 通道 block 二次型

\[
J_{64}=\sum_b e_b^{\mathsf T}G_b e_b,
\qquad G_b\in\mathbb R^{64\times64}.
\]

外部 group solver 能利用 4 元组内相关性，但如果跨 4 元组的 \(G_b\) 非对角项较大，单独最小化 \(J_{group}\) 可能增加真实的 \(J_{64}\)。因此不能用“group loss 下降”替代最终 deployment Gram gate；必须在同一个目标上比较，并逐字段写回合法五字段。

### joint residual 的合规问题

外部 joint refine 的核心残差为

\[
R=X_tW_t^{\mathsf T}-X_qW_q^{\mathsf T}.
\]

对某一权重 block 的更新会出现激活误差与权重误差的乘积项，例如

\[
(X_t-X_q)(W_t-W_q)^{\mathsf T},
\]

这正是当前 guard 追踪的 cross residual。即使该目标在离线输出 MSE 上有效，也不能作为在线 `Q(A)` 的选择信号；并且历史 C70 已经证明跨模型不稳定，故本路线关闭。

## 最小数值对照

固定随机输入和合法 offsets 的审计结果：

- E6M2 code max diff：`0`
- E6M2 decode max diff：`0.0`
- standard scale code diff：`0`
- NVFP4 dequant max diff：`0.0`
- 固定 hierarchy 的 lv2/lv3/mantissa max diff：`0.0`
- 固定 hierarchy loss max diff：`9.313225746154785e-10`

这些差异均为零或 FP32 数值误差，不支持迁移 codec、dequant 或基础 hierarchy。

另做 8 个固定随机种子的同 offset 对照（16×128、32×128、16×256 三种形状）。外部
group-Gram solver 在自己的局部目标 `J_group` 上始终更低（平均比根的未加权 hierarchy
低约 4.6%–5.4%），但在根真正使用的完整 `J_64` 上平均高约 3.8%–4.3%，大多数种子
没有胜出。这验证了“局部 4×4 相关性下降不等于部署 64×64 输出误差下降”，因此当前
不把 group-Gram 直接迁移到根；若重开，必须先实现同一目标的全 block 求解或增加严格
的 `J_64` gate。

## L5d 执行决定

1. 保持根 v111，不引入外部 joint residual、H32/H64、随机 transform 或 device/state 结构。
2. 把外部 stride sampling 做成独立候选，只修改 `_sample_rows` 一处，按现行 screen→full→归档流程测试。
3. group-Gram hierarchy 暂不直接落地；先用相同 `J_group` 和相同 `J_64` 对照，若只改善局部目标而损害 `J_64`，记录为不可行动；若在两者都改善，再开独立候选。
4. 完成 sampling 与 group-Gram 对照后，L5d 才标记 done；所有候选必须归档完整源码、source SHA、JSON 和日志。
