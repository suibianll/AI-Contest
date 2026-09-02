# 官方分数归因与本地评测合理性（2026-09-01）

状态：**已完成第一轮归因；只读取已有评测产物，没有重新运行全量评测。**

这份记录回答三个问题：

1. 固定 Attention、只改 Linear 时，官方分数是否会响应；
2. 固定 Linear、只改 Attention 时，官方分数是否会响应；
3. 本地 `official-shape-v1` 的 Linear/Attention 均值能否直接推断官方权重。

## 数据口径

本地数据来自旧 `official-shape-v1` cache（Qwen2.5-0.5B、250 个 Linear case、200 个
Attention case；该协议已退役，JSON 隔离在 `artifacts/official_eval/legacy-v1/`）：

- [`v084 archive JSON`](../artifacts/official_eval/legacy-v1/archive-official-shape-v1.json)
- [`v086 idle-rerun JSON`](../artifacts/official_eval/legacy-v1/v086-idle-rerun-20260901-official-shape-v1.json)
- [`v138 rerun JSON`](../artifacts/official_eval/legacy-v1/v138-attention-static-v86-budget-rerun2-official-shape-v1.json)
- [`v139 JSON`](../artifacts/official_eval/legacy-v1/v139-linear-output-aware-gain-official-shape-v1.json)
- [`v140 JSON`](../artifacts/official_eval/legacy-v1/v140-linear-roab-pair-official-shape-v1.json)

官方分数和时间采用已确认的回传值：v84 为 `16517 / 252.563s`，v86 为 `16744 / 222.7s`，
v138 为 `15715 / 208s`，v139 为 `15716 / 202s`，v140 为 `15838 / 207s`。官方总分不能用
本地等权显示值替代。

## 现有对照结果

| 版本 | Local Linear mean | Local Attention mean | Local API total | Local wall | 官方总分 | 官方时间 |
|---|---:|---:|---:|---:|---:|---:|
| v84 | 0.4066682145 | 0.7181069892 | 279.191s | 300.848s | 16517 | 252.563s |
| v86 | 0.4066682145 | 0.7196960689 | 299.302s | 321.996s | 16744 | 222.7s |
| v138 | 0.5073195049 | 0.7159419612 | 187.935s | 210.855s | 15715 | 208s |
| v139 | 0.5072782560 | 0.7159419612 | 193.389s | 217.196s | 15716 | 202s |
| v140 | 0.5073546371 | 0.7159419612 | 205.365s | 229.337s | 15838 | 207s |

### 固定 Attention、修改 Linear

v138、v139、v140 的 Attention mean 为同一个浮点值 `0.7159419612310174`。因此它们构成
已有的固定-A 对照：

| 对照 | Δ Local Linear | Δ Local Attention | Δ Local 等权显示 `100(250L+200A)` | Δ 官方总分 | Δ 官方时间 |
|---|---:|---:|---:|---:|---:|
| v139 − v138 | −0.000041249 | 0 | −1.031 | **+1** | −6s |
| v140 − v138 | +0.000035132 | 0 | +0.878 | **+123** | −1s |
| v140 − v139 | +0.000076381 | 0 | +1.910 | **+122** | +5s |

这已经回答了“固定 Attention 修改 Linear 是否影响官方分数”：**会影响**。但本地 Linear
均值的 `10^-5` 级变化不能作为官方增益的线性刻度；v140 的官方增益 +123 远大于本地显示的
不到 1 分。

### 固定 Linear、修改 Attention

v84 与 v86 的 Local Linear mean 完全相同，Attention mean 增加 `+0.001589080`，官方分数
增加 **+227**。这说明 v86 Attention 的结构增益确实在官方隐藏 case 上有效，后续提交必须把
它作为冻结基线，不能再用 v138 的缩减 Attention 代替。

### 跨路线反转

v138 − v86 的本地差分为 `ΔL=+0.100651`、`ΔA=−0.003754`，本地等权显示增加 **+2441.2**；
但官方分数却下降 **−1029**（16744 → 15715）。这不是噪声级别的小反转，而是明确证明当前
本地总分不能给跨 Attention 家族排序。

## “Linear/Attention 权重”能否从现有分数求出

若错误地假设官方分数满足

\[
S_{official}=C+w_L L_{local}+w_A A_{local},
\]

则固定对照会给出以下边际系数（官方分数点 / Local mean 单位）：

| 对照 | 由该对照反推的系数 |
|---|---:|
| v84 → v86（Attention） | `w_A ≈ 142,850` |
| v138 → v139（Linear） | `w_L ≈ −24,243` |
| v139 → v140（Linear） | `w_L ≈ 1,597,252` |
| v138 → v140（Linear） | `w_L ≈ 3,501,054` |

同一 Linear 权重被反推出三个数量级不同、甚至符号相反的值。用全部五个差分做最小二乘，
得到 `w_L≈−4,887`、`w_A≈143,051`，但 v139/v140 两个固定-A 对照的残差仍约为 **122/123
分**。因此不存在可信的正的固定权重解；把本地均值乘以 250/200 也不能修复这个问题。

协议中的 **250/450 = 55.56% Linear case 数**、**200/450 = 44.44% Attention case 数**只是
case 数比例，不是官方分数贡献比例。官方每个隐藏 case 的基准 MSE、层形状和量化误差分布不同，
聚合后的总分不由两个公开均值这个两个充分统计量决定。

## 结论与执行约束

1. **Attention 基线：** v86 的 Attention 是唯一已有官方正向证据（+227），后续 Linear 实验
   必须冻结 v86 Attention；不再沿 v138–v145 的缩减 Attention 继续改。
2. **Linear 证据：** 固定-A 的 v138→v140 已证明官方对 Linear 改动敏感（+123），但本地
   `linear_mean` 只能用于同一公开 panel 的逐 case/逐 role 诊断，不能换算官方分数。
3. **本地评测定位：** 对代码正确性、相同 Attention 下的回归、role 误差和本机 API 成本仍然
   有效；对不同 Attention 家族的官方排序和官方时间换算不可靠。v86 本地 API 约 299s 而官方
   222.7s，v84→v86 本地 API 反而增加约 20s、官方却减少 29.9s，说明不存在统一换算比例。
4. **下一组实验：** 不重新扫旧参数。拿到官方 17816 的真实源码后，保留其 Attention，按
   “完整 Linear / 去 Weight GPTQ / 去 Activation GPTQ / 去变换 / 去层级 refine”做一次组件
   消融，并记录逐 role、逐 case 和 API 分解；每个微变体留在同一实验日志，不建立新归档目录。
5. 在官方提交前，只用本地评测检查实现是否破坏 v86 Attention 及 Linear 的逐 case 方向；
   官方总分和 `<300s` 仍以真实官方回传为准。
