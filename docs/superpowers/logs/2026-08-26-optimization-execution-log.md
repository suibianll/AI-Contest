# 优化方案执行日志

日期：2026-08-26
驱动文档：`docs/superpowers/specs/2026-08-26-current-codebase-optimization-direction.md`
评测配置：GPT-2 12层 / hidden 768 / heads 12×64 / seq 128 / calib 2 / test 2 / amax6 / CUDA
基线 B0：`solution_b0_tmp.py`（与开发起点 HEAD `solution.py` 内容一致）
B0 SHA256：`C3EC6101BF30BD42983D97664A27DEDFF13274234BE5B4C97B07C6276ACBB534`
B0 Git blob：`baf2270cefd8c052edec43c8195f931d58bac456`
当前 A1-only 本地 Champion SHA256：`310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`

## 状态总览

| 步骤 | 状态 | 日期 |
|---|---|---|
| 0 B0 本地门禁 | **已关闭**：源码、SHA、本地分项和统一 CUDA/CPU 时间已记录 | 2026-08-27 |
| 1 Telemetry | 完成 | 2026-08-26 |
| 2 causal/non-causal 与固定回归窗口 | 本地双轨与 offset 0/97/193/389 完成 | 2026-08-27 |
| 3 A1 真实输出选择器 + 终验门 | 完成，晋级本地 Champion C1 | 2026-08-27 |
| 4 A2 固定 H64 旋转 | 完成实验但未通过尾部门槛，默认关闭 | 2026-08-27 复核 |
| 5 A3 V bias-aware | 完成（判定不晋级，默认关闭） | 2026-08-26 |
| 6 L1 数据驱动 scale | 完成（判定不晋级，默认关闭） | 2026-08-26 |
| 7 本地归档复核 | A1-only 按综合本地证据晋级；官方状态 unavailable | 2026-08-27 |
| 8 E1 合成 Attention 安全评测器 | S0–S3 全部完成并 closed：安全轨入 plan，测试 14/14 通过 | 2026-08-27 |

## 累计配对评测（CUDA）

| 配置 | 指标 | B0 | A1+门 | A2 | A3 / L1（均关闭） |
|---|---|---:|---:|---:|---:|
| MHA | attention[causal] | 0.3785 | 0.4497 | 0.4524 | = A2 |
| MHA | attention[non-causal] | 0.4116 | 0.4942 | 0.5162 | = A2 |
| GQA(6) | attention[causal] | 0.3214 | 0.4066 | 0.4120 | = A2 |
| GQA(6) | attention[non-causal] | 0.3667 | 0.4949 | 0.4916 | = A2 |
| — | Linear 全项 | 基准 | 不变 | 不变 | 不变 |
| — | CUDA 算法阶段时间 | 18.57s | 20.57s | 21.2s | = A2 |

注：表中的 A2 是保留的历史实验结果。2026-08-27 发布复核发现 A2 虽提高
聚合均值，但 MHA causal 单层与 GQA non-causal 尾部超过 §10 门槛，因此
`_ATTN_H64 = False`。当前 `solution.py` 是 A1-only：MHA 0.4497/0.4942，
Linear 与 B0 完全一致。A3/L1 开启时的数值见各自步骤小节。

## 2026-08-27：切换为本地前向累积优化

- 当前无法运行官方评测，执行规则改由
  `docs/superpowers/plans/2026-08-27-local-progressive-hif4-optimization-plan.md`
  管理；不再等待官方结果或用本地数据估计官方绝对分数。
- A1 依据跨 offset、mask、mode 和 topology 的综合主效应晋级 C1 本地
  Champion。offset 193 的 GQA non-causal 单层 `0.3569→0.2880`
  （`-6.89pp`）完整保留为尾部债务和 C2 优化目标，不再触发整体退回 B0。

| 本地配对 | causal delta | non-causal delta | 结论 |
|---|---:|---:|---|
| MHA amax6 offset 0 | +7.12pp | +8.26pp | C1 胜 |
| MHA amax6 offset 97 | +5.66pp | +6.51pp | C1 胜 |
| MHA amax6 offset 193 | +5.03pp | +9.15pp | C1 胜 |
| MHA amax6 offset 389 | +6.56pp | +7.51pp | C1 胜 |
| MHA amax4 offset 0 | +5.91pp | +7.45pp | C1 胜 |
| MHA pow2 offset 0 | +3.99pp | +7.57pp | C1 胜 |
| GQA amax6 offset 0 | +8.52pp | +12.82pp | C1 胜 |
| GQA amax6 offset 193 | +8.36pp | +7.74pp | 均值胜；记录单层尾部 |

- 六组 MHA 平均增益为 causal `+5.71pp`、non-causal `+7.74pp`；两组
  GQA 平均增益为 causal `+8.44pp`、non-causal `+10.28pp`。
- CPU algorithm-stage：B0 `52.26s`、C1 `54.72s`，比率 `1.0471`；
  Linear 字段与分数保持 B0 等价。
- 下一候选固定为 C2 Segment-CVaR selector，只优化 A1 的跨 segment
  稳定性；不同时启用 A2/A3/L1。

## C2 预注册：Segment-CVaR Attention selector

- Candidate ID：`C2`
- Parent：`C1`，SHA256
  `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- 唯一机制：将每个 Attention calibration prefix 固定切为 4 个连续 token
  segment；候选选择和部署终验使用 `mean + 0.50×worst-quartile +
  0.25×cross-segment-std`，其余量化器、候选集合和动态路径不变。
- 开发数据：只使用 offset 0；offset 97/193/389 在候选定稿前不运行。
- 目标：相对 C1 改善跨 segment 稳定性和 GQA non-causal 尾部，同时保持
  MHA/GQA Attention 综合均值、Linear 等价和动态路径成本。
- 晋级门：目标综合均值至少 `+0.2pp`，win rate ≥70%，tail mean 不低于
  `-2pp` 或明确改善 C1 既有尾部，CPU time ratio ≤1.15；state/API/静态
  安全条件全部通过。
- 硬失败：综合均值下降、非目标 Linear 下降超过 `0.2pp`、数值/state/API
  非法、缺少 SHA 或结果归档。
- 开发结果：MHA causal/non-causal 相对 C1 为 `-3.42pp/+3.08pp`；GQA
  为 `-0.13pp/+0.59pp`；win rate 50%。MHA CUDA algorithm-stage
  `26.68s`，相对 C1 `20.57s` 为 `1.297×`。
- 诊断：独立切段重置 causal 历史上下文，并把终验动态调用放大约 4 倍；
  这不是需要的“只改变稳健统计”机制。
- 状态：`local-rejected`。开发门失败后未运行 offset 97/193/389 或 CPU；
  源码和结果归档为 v004，C1 保持本地 Champion。

## C2a 预注册：Query-Segment CVaR

- Candidate ID：`C2a`
- Parent：`C1`，SHA256
  `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- 唯一机制：在完整序列上只执行一次候选量化与 causal/non-causal
  Attention；保持完整 K/V 和 causal 历史，仅将输出 query 行切为 4 段计算
  `mean + 0.50×worst-quartile + 0.25×cross-segment-std`。
- 开发数据、晋级门和硬失败条件与 C2 相同；动态终验调用数必须恢复到 C1
  量级，MHA causal 不允许再出现 C2 的上下文重置损失。
- 状态：`in-progress`。

### C3 固定矩阵结论

- offset 0 amax6：Linear mean `0.5668→0.5779`（`+1.10pp`）；q/k/v/o/fc/proj
  六项分别 `+0.58/+1.11/+0.24/+0.96/+0.49/+3.24pp`。
- amax6 offset 97/193/389：Linear mean 分别约
  `+1.10/+0.87/+0.83pp`；Attention 与 C1 一致。
- amax4/pow2：Linear mean 分别约 `+1.23/+0.92pp`；36 个配置分项中
  35 个提升，唯一负项为 pow2 proj `-0.17pp`，未超过非目标容差。
- GQA offset 193 Attention 与 C1 完全一致（`0.4169/0.4928`）。
- CPU algorithm-stage `54.29s`，相对 C1 `54.72s` 比率 `0.992`；新增机制
  仅在 weight calibration，dynamic 时间不增加。
- 状态：`local-champion`，源码与完整本地结果归档为 v006。

## C4 预注册：8×8 coverage 10%

- Candidate ID：`C4`
- Parent：`C3`，SHA256
  `413B1C8F4FEE342F2E2A2AD73DE80D4E55237828BB56D4D89E647B5C6DF59AA2`
- 唯一变化：8×8 top-loss refinement coverage 从 5% 提到 10%，其余算法、
  sweeps、cap、Attention 和动态路径不变。
- 开发门：offset 0 Linear mean 相对 C3 至少 `+0.2pp`，Attention 不变，
  CPU ratio ≤1.15；通过后才运行固定矩阵。
- 状态：`planned`。

### C8 开发结论

- offset 0 Linear mean 相对 C5 约 `+0.090pp`；六项均正向，proj 最大
  `+0.23pp`；Attention 精确不变。
- CUDA algorithm-stage `23.55s`，高于 C5 开发运行 `20.81s`。
- 状态：`local-accepted-not-promoted`。低于 `+0.2pp`，不运行固定矩阵；
  C5 保持 Champion，v011 完成 8→16→64 尺度路线的归档闭环。

## C9 预注册：16×16 second sweep

- Candidate ID：`C9`
- Parent：`C5`，SHA256
  `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- 唯一变化：C5 的 16×16 coverage 保持 2%，坐标 sweep 从 1 提到 2；
  8×8、caps、Attention 和动态路径全部不变。
- 开发门：offset 0 Linear mean 相对 C5 `+0.2pp`，Attention 不变，CPU
  ratio ≤1.15；不通过即停止二阶 sweep/coverage 扩展。
- 状态：`planned`。

### C9 开发结论

- offset 0 Linear mean 相对 C5 约 `+0.025pp`；Attention 精确不变。
- CUDA algorithm-stage `22.35s`，8 项 release tests 通过。
- 状态：`local-accepted-not-promoted`。低于 `+0.2pp`，不运行固定矩阵；
  C5 保持 Champion，停止继续调整 8×8/16×16 sweep 与 coverage。

## C10 预注册：wide activation quadratic

- Candidate ID：`C10`
- Parent：`C5`，SHA256
  `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- 唯一变化：将 `_ACTIVATION_QUADRATIC_MAX_FEATURES` 从 1024 提到 4096，
  使 3072-wide FFN down-projection activation 使用既有 4×4 `W^T W` Gram；
  权重量化、Attention、offset、coverage 和 sweep 全部不变。
- 开发门：offset 0 proj 至少 `+0.5pp`、Linear mean 为正、其余五项及
  Attention 不下降；状态合法且 CUDA algorithm-stage ratio ≤1.15。
  通过后才运行固定矩阵和 CPU 计时。
- 状态：`planned`。

### C10 固定回归与晋级结论

- offset 0 `proj 0.5192→0.5246`（`+0.54pp`），其余五个 Linear 分项和
  Attention 不变；Linear mean `+0.090pp`。
- 固定矩阵 Linear mean 相对 C5：amax6 offset 97/193/389 分别
  `+0.107/+0.112/+0.163pp`，amax4/pow2 offset 0 分别
  `+0.065/+0.085pp`；六个配置全胜。
- GQA offset 193 Attention 精确保持 `0.4169/0.4928`。
- 同环境 CPU 配对：C10 `50.99s`、C5 `51.25s`，ratio `0.995`；只判定
  为时间持平，不宣称加速。9 项 release tests 通过。
- 状态：`local-champion`；root 保留 C10。

## C11 预注册：wide activation 8×8 residual

- Candidate ID：`C11`
- Parent：`C10`，SHA256
  `DD8587257299626718A24EB89013447DA9105E8884F391104A6B350607399E44`
- 唯一变化：仅对 `in_features > 1024` 的 activation state 保存 8×8
  `W^T W` Gram，并在既有 4×4 求解后对最高损失 2% 的完整 8-channel
  activation groups 做单 sweep `H·e` 增量更新；cap 4096。
- 开发门：offset 0 proj 至少 `+0.3pp`、Linear mean 为正、其余五项及
  Attention 不下降，CUDA algorithm-stage ratio ≤1.15；通过后运行固定矩阵。
- 状态：`planned`。

### C11 固定回归与晋级结论

- offset 0 `proj 0.5246→0.5277`（`+0.31pp`），Linear mean
  `+0.052pp`；其余五项与 Attention 不变。
- 固定矩阵六项全胜；offset 97/193/389、amax4、pow2 的 Linear mean
  增量分别为 `+0.002/+0.375/+0.370/+0.035/+0.068pp`。
- GQA offset 193 Attention 精确保持 `0.4169/0.4928`。
- 同环境 CPU 配对：C11 `60.02s`、C10 `58.93s`，ratio `1.019`。
- 状态：`local-champion`；root 保留 C11。

## C12 预注册：wide activation 16×16 residual

- Candidate ID：`C12`
- Parent：`C11`，SHA256
  `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- 唯一变化：在 C11 后仅为 wide activation 增加 16×16 `W^T W` Gram，
  对最高损失 1% 的完整 16-channel groups 做单 sweep，cap 2048；不改变
  4×4/8×8、权重或 Attention 路径。
- 开发门：offset 0 proj 至少 `+0.2pp`、Linear mean 为正、非目标分项和
  Attention 不下降、CUDA algorithm-stage ratio ≤1.15；通过后运行固定矩阵。
- 状态：`planned`。

### C12 开发结论

- offset 0 proj 相对 C11 `+0.07pp`，Linear mean `+0.012pp`；其余分项与
  Attention 不变。CUDA algorithm-stage `22.80s`，ratio `1.022`。
- 状态：`local-accepted-not-promoted`。低于 proj `+0.2pp` 开发门，不运行
  固定矩阵；C11 保持 Champion，activation group-size 扩展结束于 8×8。

## C13 预注册：all-width activation 8×8 residual

- Candidate ID：`C13`
- Parent：`C11`，SHA256
  `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- 唯一变化：把 C11 的 activation 8×8 eligibility 从 `in_features>=1025`
  放宽到所有 64-aligned Linear activations；coverage 2%、单 sweep、cap 4096
  均不变，不增加 16×16 路径。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`，六个分项均不下降，Attention
  不变，CUDA algorithm-stage ratio ≤1.15；通过后运行固定矩阵。
- 状态：`planned`。

### C13 固定回归结论

- offset 0 q/k/v/o/fc 分别 `+0.56/+1.12/+0.08/+0.66/+0.36pp`，
  Linear mean `+0.463pp`，proj 与 Attention 不变。
- 六个固定配置的 aggregate Linear mean 全胜，增量范围
  `+0.412pp~+0.517pp`；GQA Attention 精确不变。
- 安全门失败：amax4 offset 0 的 o `0.4208→0.4117`（`-0.91pp`）。
- 状态：`local-accepted-not-promoted`；CPU 不运行，C11 保持 Champion。

## C14 预注册：calibration-gated all-width activation 8×8

- Candidate ID：`C14`
- Parent：`C11`，SHA256
  `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- 唯一机制：对 `in_features<=1024` 的每层，在 calibration samples 上比较
  4×4 base 与 8×8 residual 的最终 output MSE；仅当平均 MSE 至少改善 0.05%
  且任一样本不退化超过 0.1% 时保存 gram8。wide C11 路径保持无条件启用。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`；固定矩阵六项均值为正，
  任一 Linear 分项不低于 C11 超过 `0.1pp`；Attention 不变；CUDA/CPU
  algorithm-stage ratio ≤1.15。
- 状态：`planned`。

### C14 固定回归与晋级结论

- offset 0 Linear mean `+0.450pp`；q/k/v/o/fc 分别
  `+0.56/+1.12/+0.06/+0.60/+0.36pp`，proj 与 Attention 不变。
- 固定矩阵六项全胜，Linear mean 增量 `+0.420~+0.723pp`；所有记录分项
  通过安全门。amax4 o 相对 C11 `+0.33pp`，C13 的 `-0.91pp` 已修复。
- GQA offset 193 Attention 精确保持 `0.4169/0.4928`。
- 同环境 CPU 配对：C14 `58.05s`、C11 `60.30s`，ratio `0.963`；按时间
  持平处理。10 项 release tests 通过。
- 状态：`local-champion`；root 保留 C14。

## C15 预注册：quantized-weight activation Gram

- Candidate ID：`C15`
- Parent：`C14`，SHA256
  `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- 唯一变化：activation 的 4×4/8×8 quadratic Gram 从
  `W_smooth^T W_smooth` 改为最终部署的 `W_hat^T W_hat`；C14 gate、coverage、
  sweep、weight 与 Attention 路径全部不变。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`；六个分项不下降，Attention
  不变，CUDA algorithm-stage ratio ≤1.15；通过后运行固定矩阵和 CPU 配对。
- 状态：`planned`。

### C15 开发结论

- offset 0 分项相对 C14 为 `-0.02/0.00/+0.01/+0.05/-0.01/-0.03pp`，
  Linear mean 在显示精度下无净变化；Attention 不变。
- CUDA algorithm-stage `25.62s`，高于 C14 `24.99s`。
- 状态：`local-accepted-not-promoted`；不运行固定矩阵，C14 保持 Champion。

## C16 预注册：gated activation 8×8 coverage 4%

- Candidate ID：`C16`
- Parent：`C14`，SHA256
  `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- 唯一变化：C14 的 activation 8×8 coverage 从 2% 提到 4%；calibration
  gate、单 sweep、cap 4096、Gram 来源以及其他路径全部不变。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`，六分项不下降、Attention
  不变、CUDA algorithm-stage ratio ≤1.15；不通过即结束 activation 8×8
  coverage 扩展。
- 状态：`planned`。

### C16 开发结论

- offset 0 q/k/v/o/fc/proj 全部正向，分别
  `+0.13/+0.19/+0.05/+0.25/+0.12/+0.15pp`；Linear mean
  `+0.148pp`，Attention 不变。
- CUDA algorithm-stage `24.78s`，不高于 C14 开发运行。
- 状态：`local-accepted-not-promoted`。低于 `+0.2pp`，不运行固定矩阵；
  C14 保持 Champion。

## C17 预注册：final gated activation 8×8 coverage 8%

- Candidate ID：`C17`
- Parent：`C14`，SHA256
  `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- 唯一变化：C14 的 activation 8×8 coverage 从 2% 提到 8%；其余 gate、
  sweep、cap、Gram 和路径不变。这是 coverage 路线最后一次有界检查。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`，六分项不下降、Attention
  不变、CUDA ratio ≤1.15；无论结果如何，本候选后关闭 coverage 调参。
- 状态：`planned`。

### C17 固定回归与晋级结论

- offset 0 六分项全部正向，Linear mean `+0.285pp`；Attention 不变。
- 固定矩阵六项全胜，Linear mean 增量 `+0.257~+0.302pp`；36 个记录的
  Linear component means 全部改善。
- GQA offset 193 Attention 精确保持 `0.4169/0.4928`。
- 同环境 CPU 配对：C17 `63.96s`、C14 `62.40s`，ratio `1.025`。
- 状态：`local-champion`；root 保留 C17。按预注册约定，固定 activation
  8×8 coverage 调参关闭在 8%。

## C18 预注册：block-local activation/weight-error cross term

- Candidate ID：`C18`
- Parent：`C17`，SHA256
  `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- 唯一机制：为 activation 8×8 coordinate objective 增加由
  `(W_hat-W_smooth)^T W_hat` 的 8×8 对角块产生的线性交叉项；C17 gate、
  coverage、sweep、cap、weight 与 Attention 路径保持不变。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`，六分项不下降、Attention
  不变、CUDA ratio ≤1.15；通过后运行固定矩阵和 CPU 配对。
- 状态：`planned`。

### C18 开发结论

- offset 0 q/k/v/o/fc/proj 全部正向，分别
  `+0.02/+0.06/+0.05/+0.25/+0.06/+0.02pp`；Linear mean
  `+0.077pp`，Attention 不变。
- CUDA algorithm-stage `25.19s`，相对 C17 ratio `1.023`；11 项测试通过。
- 状态：`local-accepted-not-promoted`。低于 `+0.2pp`，不运行固定矩阵；
  C17 保持 Champion。

## C19 预注册：cross-aware gain selection

- Candidate ID：`C19`
- Parent：`C17`，SHA256
  `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- 唯一机制：activation 8×8 候选的排序与坐标更新统一使用块局部交叉目标；
  排序分数使用 `max_i (H·e+b)_i²/H_ii` 的坐标收益上界。C17 gate、8%
  coverage、单 sweep、cap 4096 及其他路径不变。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`，六分项不下降、Attention
  不变、CUDA ratio ≤1.15；通过后运行固定矩阵和 CPU 配对。
- 状态：`planned`。

### C19 开发结论

- offset 0 六分项全部正向，Linear mean `+0.152pp`；约为 C18 增益的两倍，
  Attention 不变。
- CUDA algorithm-stage `25.25s`，相对 C17 ratio `1.025`；11 项测试通过。
- 状态：`local-accepted-not-promoted`。低于 `+0.2pp`，不运行固定矩阵；
  C17 保持 Champion。

## C20 预注册：exact discrete cross-gain selection

- Candidate ID：`C20`
- Parent：`C17`，SHA256
  `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- 唯一机制：activation 8×8 组选点按当前 scale hierarchy 与 HiF4 离散码本
  直接枚举每组可实现的最佳单坐标 objective decrease；更新继续使用相同交叉
  目标。C17 gate、8% coverage、单 sweep、cap 4096 不变。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`，六分项不下降、Attention
  不变、CUDA ratio ≤1.15；通过后运行固定矩阵和 CPU 配对。
- 状态：`planned`。

### C20 固定回归结论

- offset 0 六分项全正，Linear mean `+0.413pp`；Attention 不变，CUDA
  ratio `1.023`。
- amax6 offset 97/193/389 与 amax4 的 Linear mean 分别
  `+1.163/+0.433/+0.445/+0.503pp`。
- 安全门失败：pow2 Linear mean `-0.490pp`，其中 proj
  `0.4890→0.4303`（`-5.87pp`）；GQA Attention 仍不变。
- 状态：`local-accepted-not-promoted`；CPU 不运行，C17 保持 Champion。

## C21 预注册：all-width gated exact cross refinement

- Candidate ID：`C21`
- Parent：`C17`，SHA256
  `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- 唯一机制：保留 C17 pure 8×8 作为逐层 fallback；exact discrete cross
  refinement 仅在 calibration samples 的最终 output MSE 相对 pure 8×8
  平均改善至少 0.05%、且任一样本不退化超过 0.1% 时启用。该 gate 对所有
  宽度生效，包括 3072-wide proj。
- 开发门：offset 0 Linear mean 至少 `+0.2pp`；固定矩阵六项均值为正、任一
  Linear 分项不低于 C17 超过 `0.1pp`；Attention 不变；CUDA/CPU ratio ≤1.15。
- 状态：`planned`。

## E1 预注册：合成 Attention 安全评测器（S0）

- Candidate ID：`E1`（评测基础设施，非 solution 候选；不改 `solution.py`，
  不改 `evaluator/real_data_eval.py` 既有评分路径，只 import 复用其
  `causal_attention`/`score_attention` 与 `nvfp4_sim.nvfp4_encode`）。
- Parent 基线（HEAD `23d1cf7`，工作树 clean）：
  - `evaluator/real_data_eval.py` SHA256
    `749CE2F8C91C923548693C25E7CEA6B021644CB5BF7661576A7035EBF0CC1D9C`；
  - `evaluator/nvfp4_sim.py` SHA256
    `8F4A4AF1D41D2BFE386DB958163DBAE7357189FCB2D87208EC8EF699044CB8DB`；
  - `solution.py`（C21 本地 Champion）SHA256
    `40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`。
- 唯一机制：新增独立评测器 `evaluator/synthetic_attention_eval.py`，
  补齐官方合成 Attention 场景（saturated logits 等）的本地端到端精度
  诊断。评分公式与现行口径逐字一致：reference=NVFP4 反量化，
  standard=朴素 HiF4，candidate=solution 动态量化输出，
  `score=(mse_std−mse_player)/mse_std`，causal/non-causal 双轨。
- 动机：variantH 官方 `saturated_logits_h4_kv2_d64_s32`（seed 307）
  退化为 0.0000 而本地无法提前暴露；官方提交机会稀缺，需提交前本地预筛。

### 冻结的场景清单（8 类，生成参数在此固定）

所有分布以 `torch.manual_seed(seed)` 的确定性采样实现，元素独立同分布
`N(0,1)` 记为 `randn`：

| 场景名 | q | k | v |
|---|---|---|---|
| `balanced` | randn | randn | randn |
| `saturated_logits` | 4.0·randn | 4.0·randn | randn |
| `near_uniform` | 0.05·randn | 0.05·randn | randn |
| `v_outlier` | randn | randn | randn，通道 `j%20==7` 乘 50 |
| `qk_dynamic_imbalance` | randn，通道 `j%2==1` 乘 64 | randn | randn |
| `k_mean_shift` | randn | randn+5.0 | randn |
| `heavy_tail` | randn·m，`m=10 若 u<0.1 否则 1`（u~U(0,1) 逐元素） | 同 q 独立采样 | randn |
| `qk_correlated` | randn | 0.8·q[..., :k_dim]+0.6·randn | randn |

### 冻结的维度网格与命名

- 命名规则（对齐官方场景名）：`{scenario}_h{q_heads}_kv{kv_heads}_d{head_dim}_s{seq}`；
- topology：MHA `h4_kv4`、GQA `h4_kv2`；
- head_dim：64、128；seq：32、128；
- mask：causal、non-causal（每 case 同时报告两轨）；
- NVFP4 mode：`amax6`、`amax4`、`pow2`；
- seed：固定 `0/1/2`，禁止任何 seed 搜索；
- calib/test：各 2 batch，每 batch `[1, seq, D]`（mirror 真实评测器
  calib 2 / test 2 默认）；
- 总量：8×2×2×2×3=192 config × 3 seed = 576 case，每 case 输出
  causal/non-causal 两分。

### E1 验收门（S2 基线锚定，全部满足才验收）

1. 确定性：同一命令重跑两次，576 case 分数逐项完全一致（CPU）；
2. 全部分数 finite，无 NaN/Inf、无崩溃、无 state 非法（CPU、无梯度、
   五字段 shape/dtype 合法）；
3. 三方基线 B0（v002）/v013/C21 完整跑通；
4. 强校验：v013 与 C21 的 Attention 路径同源（C11–C21 只改 Linear），
   合成矩阵应逐 case 一致（容差 1e-6）；不一致即实现缺陷，先修复再验收；
5. 基线合理性：B0 相对朴素 HiF4 在 `balanced` 场景均值为正；其他场景
   若 B0 均值 ≤0 仅记录，不回调场景参数。

### 硬失败条件

- 任何 NaN/Inf、state/API 非法、崩溃；
- 复跑不一致（非确定性）；
- 评分公式偏离现行口径或引入 NumPy/file I/O；
- 为让结果好看而修改场景参数、维度网格或 seed（一经冻结即禁止）。

### 时间预算

单方案全矩阵 CPU 目标 ≤60 分钟；超时分批运行（CLI 过滤参数），不删减
场景。三方全矩阵一次会话硬上限 4 小时，超限拆批并记录。

### 后续候选安全轨（S3 生效，S2 锚定方差后按此写入 plan 修订）

- 含 Attention 改动的候选：合成全矩阵等权均值、`saturated_logits` 类
  均值相对父 Champion 各不低于 `-0.1pp`；任一单 case 不低于 `-2pp`
  （worst 完整记录，超限即硬失败）；
- 只含 Linear 改动的候选：合成 Attention 分数应与父逐 case 一致
  （容差 1e-6），否则视为回归信号须排查；
- 阈值若因基线方差需调整，须书面登记修订且不得与任何候选结果相关。

### 使用纪律

- 场景参数、网格、seed 自本登记起冻结；不用于已归档候选的追溯评级；
- 合成分数不外推官方绝对分数，只作追加安全轨，不替换现行晋级公式；
- 该评测器为本地诊断工具，不作为比赛提交物。
- 状态：`closed`（S0 预注册 → S1 实现 → S2 锚定 5/5 → S3 安全轨/测试）。

### E1 执行结论（S1 + S2，2026-08-27）

- S1：新增 `evaluator/synthetic_attention_eval.py`（约 380 行，torch-only，
  复用 `real_data_eval.load_solution/score_attention` 与
  `nvfp4_sim.nvfp4_encode`，评分口径与真实评测器逐字一致）。CLI 提供
  `--solution/--cases/--modes/--seeds` 过滤；每 case 输出
  `CASE {name} mode=… seed=… causal=… noncausal=… time=…`，结尾输出逐场景
  汇总与 worst case。state 校验含 CPU/strided/连续/无梯度/有限/叶子类型；
  动态路径额外校验五字段 `sign/mant/scale_lv3/scale_lv2/scale_factor`。
- S2 四次全矩阵运行（576 case × 4），原始输出存
  `artifacts/synthetic_attention/{c21_run1,c21_run2,b0_run,v013_run}.txt`
  （CONFIG 行自带 solution SHA256）：

  | 方案 | SHA256（文件内 CONFIG 行） | 用时 | 状态 |
  |---|---|---:|---|
  | B0=v002 归档 | `E126B23A7992E28FBB8E5973551B49AE40A930B76522265A6F36F641EB133A4B` | 73.7s | RESULT ok |
  | v013 归档 | `DD8587257299626718A24EB89013447DA9105E8884F391104A6B350607399E44` | 146.4s | RESULT ok |
  | C21=根 solution | `40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A` | 154.6s | RESULT ok |

  B0 直接使用 v002 归档文件在 CPU 上运行，无需 GPU 兼容补丁。

- 验收门核对（5/5 通过）：
  1. 确定性：C21 双跑剥离 time 字段后 576 行逐行一致（diff=0）；
  2. 全部分数 finite、无崩溃、state/五字段全部合法（4×RESULT ok）；
  3. 三方基线完整跑通；
  4. 强校验：v013 与 C21 的 576 个 CASE 分数行完全一致（0 diff，远严于
     1e-6 容差），确认 C11–C21 未触碰 Attention 路径，同时交叉验证了
     评测器实现正确性；
  5. B0 `balanced` 场景均值为正（causal 0.2452 / non-causal 0.2544）。

- B0 → C21 逐场景均值（72 case/场景，pp=百分点）：

  | 场景 | causal Δ | non-causal Δ | 备注 |
  |---|---:|---:|---|
  | balanced | +1.01 | +0.92 | |
  | saturated_logits | +2.92 | +2.93 | 官方踩坑场景，A1 为正增益 |
  | near_uniform | 0.00 | 0.00 | 分数与 B0 完全一致 → A1 门回退 B0 选择 |
  | v_outlier | +1.64 | +0.99 | |
  | qk_dynamic_imbalance | +4.08 | +5.01 | |
  | k_mean_shift | +43.41 | +42.91 | A1 主效应最大的合成场景 |
  | heavy_tail | +16.84 | +13.19 | 仍为负值（见下） |
  | qk_correlated | +0.25 | +0.09 | |
  | **OVERALL** | **+8.77** | **+8.26** | 0.1948→0.2824 / 0.2172→0.2997 |

- 关键发现：
  - `heavy_tail` 是唯一 C21 均值仍为负的场景（causal `-0.0998`、
    non-causal `-0.0444`，worst `-0.94/-1.08`，集中在 d128 + pow2/amax4）。
    这是 B0 继承属性（B0 为 `-0.2682/-0.1762`），C1/A1 改善约
    `+16.8/+13.2pp` 但未转正 → 登记为 Attention 尾部债务与下一候选目标。
    该发现不阻塞 C21 提交（C21 相对 B0 全场景正向）。
  - C21 的 worst non-causal case（`heavy_tail_h4_kv4_d128_s32` pow2 seed2，
    `-1.075382`）与 B0 同 case 数值逐位一致 → A1 终验门在该困难 case 上
    正确回退到 B0 选择。
  - 时间：单方案全矩阵 74–155s（预算 60 分钟），三方合计 <7 分钟
    （预算 4 小时）。

- S3（2026-08-27 完成）：
  - plan 修订：`2026-08-27-local-progressive-hif4-optimization-plan.md`
    新增 4.3 节（E1 合成矩阵冻结与 S2 锚定证据、heavy_tail 尾部债务
    登记）；5.2 晋级规则追加 E1 合成安全轨（Attention 改动候选：全矩阵
    等权均值与 saturated_logits 类均值 ≥ `-0.1pp`、单 case ≥ `-2pp`；
    Linear-only 候选：与父逐 case 一容差 1e-6）；5.3 硬失败追加越限条款；
  - 测试：`tests/test_release_candidate.py` 新增
    `test_synthetic_attention_states_and_params_are_legal`（E1 冻结矩阵
    代表子集：heavy_tail×2 / saturated_logits / v_outlier / balanced ×
    3 mode，校验 state 合法性与动态五字段）与
    `test_synthetic_case_generation_is_deterministic`（生成器确定性）；
    全套 14/14 通过（8.17s）。
- E1 状态更新：`local-accepted` → `closed`（S0–S3 全部完成）。
- heavy_tail 尾部债务进入下一 Attention 候选预注册时的问题定义。

### C7 开发结论

- offset 0 Linear mean 相对 C5 约 `+0.123pp`；q/k/v/o/fc/proj 六项均为
  正增益，proj 最大 `+0.22pp`；Attention 精确不变。
- CUDA algorithm-stage `21.99s`，9 项测试通过。
- 状态：`local-accepted-not-promoted`。未达到 `+0.2pp`，不运行固定矩阵；
  C5 保持 Champion，v010 归档 32×32 的明确正向边际。

## C8 预注册：严格受限 top-K 64×64 Linear 二阶

- Candidate ID：`C8`
- Parent：`C5`，SHA256
  `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- 唯一机制：保留 C5，只对最高损失 0.5% 的完整 64 通道 block 做一轮
  `H·e` 坐标更新，cap 1024；禁止全量 64×64 GPTQ或增加 sweep。
- 开发门：offset 0 Linear mean相对 C5 `+0.2pp`，Attention 不变，CPU
  ratio ≤1.15；不通过即结束 group-size 扩展。
- 状态：`planned`。

### C6 开发结论

- offset 0 Linear mean 相对 C5 约 `+0.063pp`；六项均小幅为正，Attention
  精确不变，CUDA algorithm-stage `20.64s`。
- 状态：`local-accepted-not-promoted`。边际低于 `+0.2pp`，不运行固定
  回归矩阵；C5 保持 Champion，v009 归档 16×16 coverage 饱和证据。

## C7 预注册：top-K 32×32 Linear 二阶

- Candidate ID：`C7`
- Parent：`C5`，SHA256
  `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- 唯一机制：保留 C5 的 8×8/16×16 winner，再对最高损失 1% 的连续
  32 通道组执行一轮 32×32 `H·e` 更新；cap 2048，scale/lv2/lv3 固定。
- 开发门：offset 0 Linear mean相对 C5 `+0.2pp`，Attention 不变，CPU
  ratio ≤1.15；通过才运行固定矩阵。
- 状态：`planned`。

### C5 固定矩阵结论

- offset 0 Linear mean 相对 C3 `+0.23pp`，六项全部提升；最大增益为
  proj `+0.45pp`、o `+0.35pp`。
- amax6 offset 97/193/389：Linear mean 分别约
  `+0.18/+0.46/+0.28pp`；amax4/pow2 分别 `+0.27/+0.27pp`。
- 六组固定配置的 36 个 Linear 分项全部提升；所有 Attention 与 C3 一致。
- CPU algorithm-stage `55.92s`，相对 C3 `54.29s` 比率 `1.030`；dynamic
  时间不变。
- 状态：`local-champion`，源码和结果归档为 v008。

## C6 预注册：16×16 coverage 4%

- Candidate ID：`C6`
- Parent：`C5`，SHA256
  `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- 唯一变化：16×16 top-loss coverage 从 2% 提到 4%；C3 的 8×8 5%、
  sweeps、caps 和全部其他机制不变。
- 开发门：offset 0 Linear mean 相对 C5 至少 `+0.2pp`，Attention 不变，
  CPU ratio ≤1.15；不足则结束 16×16 coverage 扩展。
- 状态：`planned`。

### C4 开发结论

- offset 0 Linear mean 相对 C3 约 `+0.092pp`；q/k/v/o 分别
  `+0.15/+0.15/+0.11/+0.14pp`，fc/proj 均为 `0.00pp`。
- Attention 精确保持 C3，CUDA algorithm-stage `20.02s`。
- 状态：`local-accepted-not-promoted`。正增益低于预注册 `+0.2pp` 门槛，
  不运行固定回归矩阵；C3 保持 Champion，v007 记录覆盖率饱和证据。

## C5 预注册：top-K 16×16 Linear 二阶

- Candidate ID：`C5`
- Parent：`C3`，SHA256
  `413B1C8F4FEE342F2E2A2AD73DE80D4E55237828BB56D4D89E647B5C6DF59AA2`
- 唯一机制：保留 C3 的 5% 8×8 refinement，再对最高损失 2% 的连续
  16 通道组执行一轮 16×16 `H·e` 坐标更新；scale/lv2/lv3 固定。
- 开发门：offset 0 Linear mean 相对 C3 至少 `+0.2pp`，Attention 不变，
  CUDA/CPU ratio ≤1.15；通过后运行固定矩阵。
- 状态：`planned`。

### C2a 开发结论

- offset 0：MHA causal/non-causal 相对 C1 为 `-0.53pp/-0.52pp`；GQA
  为 `0.00pp/-0.58pp`。
- CUDA algorithm-stage 为 MHA `19.65s`、GQA `19.96s`；动态调用恢复到
  C1 量级，证明完整上下文/query-row 分段解决了 C2 的工程错误。
- 精度综合均值没有超过父 Champion，因此不运行固定回归 offset 或 CPU。
- 状态：`local-rejected`，源码和结果归档为 v005；C1 继续作为 Champion。

## C21-C 预注册：Phase 0 合规基线（删除 Linear 输出监督）

- Candidate ID：`C21-C`（Compliance；对应 26000 计划 Phase 0，
  `docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`）
- Parent：`C21` / v024，SHA256
  `40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`
  （HEAD `d5d74b5` 工作树 clean，SHA 已重新计算核实）
- 唯一机制：删除 Linear 校准中的全部输出监督路径，不新增任何精度机制：
  1. 删除 `_linear_output_candidate_metrics` 及其全部调用；Block-Smooth
     候选评分改用既有 operand-local 的 `_linear_candidate_metrics`；
  2. 删除 `_activation8_gate_decisions` 的 Linear 输出评分；8×8 gate 改为
     activation-only 重构损失（base vs refined，均值改善 + 最差样本容差），
     删除未使用的 `_activation_quadratic8_is_safe` / `_activation_cross8_is_safe`；
  3. 删除 `group_cross8` / `cross8` state、`_ACTIVATION_QUADRATIC8_CROSS_*`
     与 `_ACTIVATION_QUADRATIC8_EXACT_DISCRETE_SELECTION` 开关及
     `_refine_weight_groups8` 的 cross 坐标更新（回到 C17 pure 8×8 形式）；
  4. activation_state 不再输出 `cross8` 字段，版本号升为 4。
- 评测矩阵（§4.6 验收）：pytest 全过；`real_data_eval.py` CUDA 开发评测；
  §10.2 完整固定回归 6 配置 × offsets 0/97/193/389 逐配置 delta 记录
  （作为后续所有候选 ROI 比较的强制基线）；§10.3 合成矩阵 576 case
  逐 case 与 C21 一致（容差 1e-6）；Attention 与 C21 逐 case 不变。
- 晋级门：不设分数门槛（预期低于 16043）；必须完整记录移除输出监督
  造成的真实分数变化；合规门禁（静态+运行时）全过；通过后归档为
  新的唯一合规 Champion，后续所有候选从 C21-C 派生。
- 时间预算：无新增机制，预期 CUDA algorithm-stage 不高于 C21；CPU ratio
  不设门（删除路径只会变快），如实记录。
- Holdout 台账：`holdout_runs_used=0 / remaining=3`（Phase 0 不消耗）。
- 状态：`local-champion`（Phase 0 验收全过，归档 v025；见 §4.6 验收记录）。

### C21-C §4.1 开发记录（2026-08-27）

- solution.py：删除 `_linear_output_candidate_metrics`、`group_cross8`/`cross8`
  state、`_ACTIVATION_QUADRATIC8_CROSS_*` 与
  `_ACTIVATION_QUADRATIC8_EXACT_DISCRETE_SELECTION` 开关及 cross 坐标更新；
  Block-Smooth 候选评分改用 operand-local `_linear_candidate_metrics`；
  8×8 gate 改为 activation-only 重构损失
  （`_activation8_refinement_is_safe`，base vs refined）；
  activation_state 版本号升为 4、不再输出 `cross8` 字段。
- tests/test_release_candidate.py：`test_release_flags_are_a1_only` 改为断言
  cross 开关已不存在；新增
  `test_activation8_refinement_gate_returns_a_decision` /
  `test_activation8_refinement_gate_rejects_regression`。
- 验证：`.venv` pytest 13/13 通过；静态合规测试
  （无 numpy / 文件 IO / 调试输出）通过。

### C21-C §4.2 开发记录（2026-08-27）

- 新建 `evaluator/reference_hif4.py`：从 C21 复制并冻结标准 HiF4
  codec（amax/7 BF16 中间、E6M2、lv2/lv3 阈值、mantissa、canonical zero），
  不含 offset/refinement，绝不调用候选 `_dense_to_hif4`。
- `evaluator/real_data_eval.py` 的 `std_hif4` 改用冻结 reference codec
  （标准分母与候选代码解耦；candidate 修改 `_dense_to_hif4` 不再可能
  影响 standard 输出）。
- 新建 `tests/test_reference_hif4.py`（5 用例）：与 solution 标准路径
  逐位一致（含非有限值/极值/批量前缀形状）、确定性、非法形状拒绝、
  canonical 零符号与冻结层级形状。
- 验证：pytest 18/18 通过。后续 candidate 变更导致的 standard 输出
  逐位不变由结构保证（§4.6 验收时再以固定回归分数复核）。

### C21-C §4.3 开发记录（2026-08-27）

- 新建 `evaluator/linear_error_decomposition.py`：operand-local 归因
  报告 `activation_local_error` / `activation_tail_cvar` /
  `weight_hessian_error`（`G=A^T A/N` 的 Hessian 形式，Q(W) 合法
  消费量）/ `weight_plain_error` / `transform_orthogonality_error`，
  外加 per-row activation error；工具绝不构造 reference Linear 输出、
  cross residual 或任何 `[tokens, out_features]` 张量。
- 新建 `tests/test_linear_error_decomposition.py`（6 用例）：零误差、
  手工对拍（含 trace 形式 Hessian）、tail CVaR、正交性检测（含 2I
  缩放）、输出形状只有标量/一维行向量、非法形状拒绝。
- 验证：pytest 24/24 通过。

### C21-C §4.4 开发记录（2026-08-27）

- 新建 `evaluator/linear_compliance_guard.py`：双层合规门禁。
  - 静态层（AST）：拒绝已知违规符号（`_linear_output_candidate_metrics`、
    `cross8` 家族等）以标识符/字符串形式回归；拒绝违规 state key
    （output/reference/residual/cross/target/label，仅限 dict 字面量键）；
    启发式标记激活/权重混合命名的收缩（方法调用接收者纳入操作数）。
  - 运行层（TorchDispatchMode 污点跟踪）：A/G/W/Wg/Ra/Rw 六种污点
    通过 ATen 算子传播；记录全部收缩的形状与污点。硬失败：激活残差×
    权重残差交叉收缩（改名 cross8）、tokens/out_features 维度泄漏进
    收缩输出、activation_state 张量携带 token/out_features 维度。
    `Wg`（weight-Gram 引导）标记区分权重数据与引导信号，避免把合法的
    权重 Gram 激活精修误判为交叉残差；`[K]` 通道统计（SmoothQuant
    scale）污点中和，落 review 人工确认而非 violations。
- 新建 `tests/test_linear_compliance_guard.py`（8 用例）：静态接受
  当前 solution / 拒绝违规符号与改名交叉收缩；运行时接受当前
  solution（含 review 通道统计确认）、拒绝 Linear 输出监督与交叉
  残差 state、放行合法 Hessian/Gram；`guard_solution_file` 全门禁。
- 修复静态层漏报：`a.mm(b)` 方法调用接收者不在 `node.args` 中，
  将其补入操作数后改名交叉收缩被正确标记。
- 验证：pytest 32/32 通过（含真实 solution.py 静态+运行时全门禁）。
  当前 C21-C solution：`violations=[]`，contraction 全部合法。

### C21-C §4.5 开发记录（2026-08-27）

- 新建 `evaluator/holdout_eval.py`：冻结最终 holdout（26000 计划 §4.5）。
  - 新固定文本 `HOLDOUT_TEXT`（约 30 句自然文本），与开发
    `real_data_eval.TEXT` 逐句无交集，从未用于任何候选调参；
  - 冻结配置 `HOLDOUT_CONFIG`：layers=12、seq=128、calib=2、
    **test=4（≥4 个 token windows）**、amax6、causal、models/gpt2；
  - `--freeze` 已执行：`evaluator/holdout_ledger.json` 记录
    `seed_hash=96dd4ed70a0597a0060fe696557d3a330af22e3d273e6676a501d7bfb4b589fc`
    （sha256(文本+冻结配置)）、budget=3、runs=[]；二次 freeze 幂等，
    不同 seed 拒绝覆盖；运行前校验 ledger seed_hash 一致（holdout
    冻结后不可修改）；
  - 运行时强制：每个候选（solution.py 的 sha256）至多跑一次最终
    holdout；总预算 3 次（`HOLDOUT_BUDGET`），超额/重复直接拒绝；
    `--reason` 必填并记入台账（holdout 仅限最终验收，禁止用于
    seed/threshold/coverage 搜索）；
  - 输出仅聚合量：linear_mean、attention_mean、algorithm_stage_seconds
    与台账余量，绝不打印逐层/逐组件数据。
- `real_data_eval.collect_real_data` 增加 `text` 参数（默认开发
  TEXT，行为不变），holdout 采集复用同一入口。
- 新建 `tests/test_holdout_eval.py`（7 用例）：文本与开发集逐句不
  交集、test≥4 windows、seed hash 确定性、freeze 幂等且不可覆盖、
  每候选一次强制、总预算强制、篡改 seed_hash 拒绝。
- 环境注记：系统 pytest 临时目录（`Temp/pytest-of-*`）ACL 受损，
  统一使用 `--basetemp=.tmp_pytest`（已加入 .gitignore）。
- 验证：pytest 39/39 通过。Phase 0 不消耗 holdout
  （`holdout_runs_used=0/remaining=3`）。

### C21-C §4.6 验收记录（2026-08-27，全部通过）

- pytest：39/39（reference codec / error decomposition / compliance
  guard / holdout ledger 全含）。
- CUDA 开发评测（amax6 offset 0 both）：Linear mean `0.5311`
  （q/k/v/o/fc/proj = 0.6008/0.5936/0.5940/0.5178/0.4749/0.4058）；
  Attention causal `0.4497` / non-causal `0.4942` 与 C21 逐位一致；
  algorithm-stage `24.03s` vs C21 `26.59s`（ratio 0.904，变快）。
- 固定回归矩阵（C21 同日同评测器重跑，6 配置与 v024 台账逐位一致，
  证明 reference codec 解耦不改变 standard 分母）：

  | Case | C21 | C21-C | Delta |
  |---|---:|---:|---:|
  | amax6 offset 0 | 0.5930 | 0.5311 | −6.19pp |
  | amax6 offset 97 | 0.5747 | 0.5148 | −5.99pp |
  | amax6 offset 193 | 0.5928 | 0.5319 | −6.09pp |
  | amax6 offset 389 | 0.5912 | 0.5235 | −6.77pp |
  | amax4 offset 0 | 0.4973 | 0.4663 | −3.10pp |
  | pow2 offset 0 | 0.5575 | 0.5454 | −1.21pp |

  6/6 配置下降即移除输出监督的真实代价（k −11.9pp、proj −13.0pp 最大，
  v 仅 −0.7pp）；该表为后续所有候选 ROI 的强制基线。各配置 Attention
  全部与 C21 逐位一致；GQA kv6 offset 193 causal `0.4169` /
  non-causal `0.4928` 逐位一致。
- 合成安全矩阵：576/576 case 与 C21 逐位一致（max abs delta `0`，
  容差 1e-6）；overall causal `0.282448` / non-causal `0.299711`；
  worst heavy_tail pow2 seed2 `−0.939570/−1.075382`（inherited）；
  两方案 `RESULT ok`。
- 合规门禁：静态 + 运行时全过（`violations=[]`）。
- 归档：`solutions/20260827_v025_c21c-compliance-baseline/`
  （result.md 含 holdout 台账 `0/3`、seed_hash `96dd4ed7…`）。
  源码 SHA256
  `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`。
- 决策：C21-C 为唯一合规 Champion；后续候选（C22 起）全部从 C21-C
  派生并通过同一合规门禁。

## C22 预注册：Linear R64 Incoherence Transform（2026-08-27）

- 候选 ID：`C22`；父版本：`C21-C`（v025，SHA256
  `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`）。
- 唯一机制（§5.1）：在 Linear transform candidate 中增加 64 维
  signed Hadamard `R64(seed) = S(seed) · H64`（butterfly FWHT 实现，
  动态路径禁止 dense [64,64] matmul）。不修改 Attention /
  refinement / scale offset / gate / coverage / sweep。
- 代码：仅 `solution.py`（新常量 `_LINEAR_R64*` 六项 + 新函数
  `_fwht_last_dim` / `_linear_r64_signs` / `_apply_linear_r64` /
  `_rank_r64_seeds` / `_select_r64_candidate`）与
  `tests/test_release_candidate.py`（§5.6 八项测试）。
- seed 选择两阶段（§5.4）：Stage A cheap rank（≤64 activation rows +
  ≤128 weight rows，标准 HiF4，激活硬重构损失 + H_A 加权权重损失，
  32 seeds 排序保留 top-4）；Stage B 部署验证（top-4 + identity，
  两折双向验证 + robust metric
  `max(ratio_A, ratio_W) + 0.10·max(0, tail−1)`，两个 fold
  activation-only 均不劣于 identity 且 robust metric 优于 identity），
  最终以全量数据 `_candidate_is_safe(0.005, 0.002)` 门控。
  Stage B 语义解释：使用 C21-C 部署的 operand-local 指标体系
  （与所有其他 transform 候选同一选择层），refinement 在胜出后
  单次运行——不引入任何 Linear output 监督。
- state（§5.5）：仅复用 `block_smooth_size=64` / `block_smooth_seed`，
  不保存 dense R64。
- 评测矩阵：开发集 cuda amax6 offset 0 both（对照 C21-C
  Linear 0.5311）；固定回归 6 配置 + GQA；576 合成 case；合规门禁。
- 晋级门（§5.7）：开发集 Linear mean ≥ +0.5pp、fc/proj/o 均值
  ≥ +1.0pp、任一分项 ≥ −0.1pp、Attention 逐 case 一致；固定矩阵
  6/6 为正、win rate ≥70%、tail mean ≥ −1pp、CPU stage ratio
  ≤1.12、官方时间 <205s。未达开发门 → 归档 rejected，停止 seed
  扩展，不直接实现双 Hadamard。

## C22 归档：REJECTED（2026-08-27，v026）

- 实现：按预注册完成（提交 e3bf1c6，SHA
  `8BF16F042C0AD45A8726A4E855FBEBF4B9E95E4E42289009626F5AF03306BD97`）。
  FWHT butterfly 与 dense H64 误差 <1e-5；§5.6 八项测试 + 合规门禁
  全过（pytest 47/47）。
- 开发评测（cuda amax6 offset 0 both）：Linear 六分项与 C21-C 逐位
  相同（mean 0.5311，0.00pp）——72/72 组件的 seed 选择全部回退
  parent（block_smooth_size=0）。Attention 逐位一致 ✓。时间
  algorithm-stage 36.62s vs 24.03s（ratio 1.52，超 ≤1.12 门）。
- 拒绝诊断（evaluator-side 临时脚本，已删除）：
  1. 两折门 288/288 全拒（4 top seeds × 72 组件）；
  2. L0 fc 逐折 ratio_A≈1.17–1.18、ratio_W≈1.05–1.06；
  3. 发现 sign 哈希对连续 seed 近似不变（`i·1103515245 +
     seed·214013 + 12345` 取 bit30，步进 214013 极少进位到 bit30，
     range(32) 实际≈1 个符号模式，seeds 21–26 指标逐位相同）；
  4. 分散 seed（`s·100003`）复测 9 个采样点（L0/L5/L11 × q/fc/proj）：
     9/9 最优 seed 两折 `max(ratio_A, ratio_W) > 1`（ratio_A 1.04–1.37，
     ratio_W 0.99–1.34）。拒绝是机制问题，非 seed 多样化不足。
- 决策：`rejected` per §5.7。停止 seed 扩展，不实现双 Hadamard。
  根目录 solution.py 默认 `_LINEAR_R64 = False`（行为与 C21-C 逐位
  一致，`test_linear_r64_disabled_matches_c21c` 验证）；归档
  `solutions/20260827_v026_c22-linear-r64-rejected_scoreNA_timeNA/`
  保存 flag=True 候选。Champion 仍为 C21-C（v025）。C23 从 C21-C
  构建。Holdout 未消耗（0/3）。
- 教训（新增）：sign 哈希 seed 步进 214013 对 bit30 几乎无扰动，
  任何 seed 搜索必须大间距；GPT-2 真实数据上 64 宽 Hadamard
  预混合对 HiF4 层级编码是净损伤（平滑/置换已处理离群通道）。

## C23 预注册：Full-64 Weight Schur/GPTQ（2026-08-27）

- 候选 ID：`C23`；父版本：`C21-C`（v025，SHA256
  `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`）。
  C22 已 rejected，按 §6 前置条件从 C21-C 构建，只保留已通过合规
  审计的 identity/R4/R8/R16 变换（R64 保持 `_LINEAR_R64=False`）。
- 唯一机制（§6.1）：只替换 weight 的 full-64 refinement；dynamic
  activation 仍使用父版本。
- 数学目标（§6.2）：每个 transformed weight row 的 64 元素 block，
  `H[64,64] = X_t^T X_t / N + damping·I`（完整 64×64，不截断），
  `loss(q)=(q−w)^T H (q−w)`；`damping=0.01·mean(diag(H))`，Cholesky
  失败依次 `0.03/0.1`，仍失败回退父版本。
- scale beam（§6.3）：`standard_code + {-2,-1,0,1,2,3}`，exact
  hierarchy proxy 排序保留 4 个；逐 block fallback（五字段合法 +
  full-H loss finite 且 ≤ 父版本，否则该 block 回退父参数）；
  固定回归必查 amax4/pow2 offset 0，任一 case Linear 相对指标
  不得低于父版本 2pp；扩展 offset 的 Attention 教训不归因于 Weight
  beam（§6.3 条 4）。
- 求解流程（§6.4–6.6）：`_solve_exact_hierarchy` 初始化 →
  Cholesky 逆因子 → `diag(H)` 降序 processing order → GPTQ
  sequential mantissa init（§6.5 伪代码）→ full-H 64-coordinate
  exact discrete descent（§6.6 状态 e/g 增量更新）→ 16 个 lv3
  toggle → 8 个 lv2 toggle → 再次 descent → 四 beam 择优。
- 代码结构（§6.7）：`_full64_hessian_blocks` /
  `_cholesky_inverse_factor` / `_gptq_initialize64` /
  `_coordinate_descent64` / `_hierarchy_toggle_refine64` /
  `_refine_weight_blocks64`。内存：weight rows 128 rows/chunk、
  禁止展开 [rows,blocks,beams,64,64]、H 每层共享、beam 顺序执行。
- 向量化硬性要求（§6.7）：禁止逐 block Python 循环，row chunk 内
  全部 blocks 合并为 [rows*blocks,64] 批量；descent 仅坐标维与
  beam 维循环；toggle 用布尔掩码批量；交付前批量路径 vs 逐 block
  参考实现数值一致（1e-6 探针）；micro-benchmark CPU/f32、
  rows≥2000、channels {768,3072}、chunk 128、预热 3 测 10 取中位，
  覆盖 `_refine_weight_blocks64` 生产调用与临时分配；≥10x 为诊断
  目标而非独立门，端到端 CPU ratio>1.15 不得晋级。
- 实施期修订（2026-08-27，任何 C23 评测运行之前登记；依据为
  C21-C 本地 CPU algorithm-stage 实测预算，非评测结果）：全量
  64-block 覆盖的 864 步/beam 批量求解在 CPU 上推算约 40s+，远超
  `ratio<=1.15` 的 +8s 预算。因此实现加入两个预算常量，语义仍为
  "逐 block fallback 到父参数"：
  1. `_WEIGHT_FULL64_MAX_RATIO`（默认 `0.25`）：每个 row-chunk 内按
     父版本 full-H loss 之和选取 top-`ceil(ratio*blocks)` 个 64 列
     进入 beam solve；未选列保持父参数（与 fallback 语义一致）。
     该 ranking key 只用 W 自身统计与 H（规则零白名单允许）。
  2. `_WEIGHT_FULL64_CHUNK_ROWS`：计划默认 128；因小 B 层（B=12）在
     128 行 chunk 下 Python/torch 调度开销主导（成本与覆盖率无关），
     生产默认调整为 1024（[1024, blocks, 64] float32 ≈ 3MB，仍满足
     "必须 chunk、不得全展开"的内存硬性要求）。micro-benchmark 按
     §6.7 规格在 chunk=128 测量，同时报告生产 1024 配置。
  两个常量在固定回归之前可依据 CPU ratio 门调整并在此登记；看过
  固定回归结果后不得再改（若需收窄须换 candidate ID 重新预注册）。
- 测试（§6.8）：`test_full64_hessian_extraction` /
  `test_gptq64_initialization_returns_legal_codes` /
  `test_coordinate_descent64_is_monotonic` /
  `test_hierarchy_toggle64_is_monotonic` /
  `test_weight64_final_loss_not_above_parent` /
  `test_weight64_chunking_is_exact` /
  `test_weight64_fallback_on_non_psd` /
  `test_weight64_deterministic`。
- 晋级门（§6.9）：Linear mean ≥ +2pp；fc/proj/o 平均 ≥ +3pp；
  固定矩阵 Weight full-H normalized error 下降 ≥20%；6/6 配置
  正向；CPU ratio ≤1.15；官方时间 <225s。若总 Linear 仍 <0.68，
  记录 26000 风险检查点，不进入 C24，先分析 weight residual。
- 评测矩阵与 holdout 纪律同 C22（开发集 cuda amax6 offset 0
  both 起步，通过后固定矩阵 6 配置 + GQA + 576 合成 case + 合规
  门禁；holdout 仅最终验收）。

### C23 实施与归档结果（2026-08-27，v027，rejected）

- 实现（flag=True 归档 SHA256
  `DD80CBBF43CD13D7AE6D5AD32399B91A64BB9EF49CA124DC4D526263F2766069`）：
  §6.7 六函数全批量向量化（`_full64_hessian_blocks` /
  `_cholesky_inverse_factor`（done 掩码 + 阶梯阻尼 0.01/0.03/0.1）/
  `_gptq_initialize64` / `_coordinate_descent64` /
  `_hierarchy_toggle_refine64` / `_refine_weight_blocks64`，全局列选择
  保证 chunk 无关性）。§6.8 八项测试 + 批量 vs 逐 block 1e-6 探针 +
  micro-benchmark（chunk 128，2048 行，768/3072 通道）全部通过，
  见 `tests/test_weight_full64.py`。
- 开发评测（cuda amax6 offset 0 both）：Linear mean `0.5311 → 0.5504`
  （+1.93pp；早前一次 +2.07pp，CUDA 归约抖动 ±0.2pp）；fc/proj/o 均值
  +1.64pp；Attention 逐位不变。
- 固定回归矩阵 6/6 正向（+1.93/+2.24/+2.19/+2.34/+2.66/+1.90pp，
  amax4/pow2 尾部检查通过）。
- 机制诊断（真实数据 instrumented）：full-H 总降幅 `20.95%`（门 ≥20%
  达标），替换率恰 25%；按组件 v 9.8% – k 37.7% 差异大。
- 合规门禁（flag=True 归档版）：static+runtime 全过，`violations=[]`，
  210 contractions，无 review 项。
- 时间（§10.4 规定口径，父子串行 CPU 同环境）：C21-C algorithm-stage
  `61.32s` vs C23 `95.17s`，ratio `1.55`（门 ≤1.15 未达）；CUDA 口径
  `24.03s → 32.77s`（ratio 1.37）为参考。推算官方时间 `~269s > 225s`
  （逼近 270s 硬上限）。CPU 口径下父子 Attention mean/min/max 逐位
  一致（0.4497 / 0.4944）。
- 决策：`rejected` per §6.9（6 门中 3 未达：Linear mean 踩线未过、
  fc/proj/o +1.64pp<3pp、CPU ratio 1.55>1.15；推算时间 ~269s>225s）。
  机制有效但绝对成本超预算。根目录 `_WEIGHT_FULL64 = False`（行为与
  C21-C 逐位一致，`test_weight_full64_disabled_matches_c21c` 验证）；
  归档 `solutions/20260827_v027_c23-full64-rejected_scoreNA_timeNA/`。
  Champion 仍为 C21-C（v025）。Holdout 未消耗（0/3）。
- 战略后果：C24 前置「C23 晋级」未满足，C25 前置 C24——计划 §7–§9
  链条全部受阻。后续任何推进必须换新 candidate ID 重新预注册（规则 3/4）。
  候选方向：收窄覆盖率（top-25%→10%，q/k/o 的 drop 33–38% 远高于均值）、
  分块协方差近似降 Cholesky 成本、或只对 fc/proj 宽层启用。

### Checkpoint B 归因分析（2026-08-27，C21-C，计划 §13 Checkpoint B / §6.9）

按 Checkpoint B 规定（Linear `0.5504 < 0.68`，C24 暂停），用评测器侧
operand 消融量化 C21-C 剩余 Linear 残差（0.5311）的归属。方法：从
activation_state 重构精确等价变换（smooth 1/d、置换、块 Hadamard，
变换重构 sanity `|T(A)@T(W)^T − A@W^T| <= 9.8e-4`，权重侧重构
relRMSE 0.113 = HiF4 权重量化误差本身），在同一变换坐标系内消融：
act-only = player 激活 × 精确变换权重；w-only = 精确变换激活 ×
player 权重。逐 case 分数后平均，与官方口径逐位对齐（mean `0.5312`，
分项 0.6008/0.5936/0.5940/0.5178/0.4749/0.4058）。诊断脚本为
evaluator 侧临时脚本，运行后已删除（§0.2：结果不回传 solution）。

```text
 comp   linear  act-only   w-only    act份额   w份额
    q   0.6008    0.8077   0.7921     48.2%    52.1%
    k   0.5936    0.7924   0.8005     51.1%    49.1%
    v   0.5940    0.8007   0.7923     49.1%    51.2%
    o   0.5178    0.7588   0.7551     50.0%    50.8%
   fc   0.4749    0.7453   0.7300     48.5%    51.4%
 proj   0.4058    0.6883   0.7184     52.5%    47.4%
 mean   0.5312    0.7655   0.7647     50.0%    50.2%
```

（份额 = 该侧单独误差 / full 误差；两侧份额之和 ≈ 100%，交叉项 ≈ 0，
即 full 误差 ≈ 激活侧误差 + 权重侧误差，精确可加。）

结论：

1. **两侧均衡 50/50**：C21-C 的剩余 Linear 残差在激活侧与权重侧各占
   一半，任何单侧候选的提升空间都被对侧封顶。
2. **单侧上限**：权重完美时 Linear 封顶 `0.7655`（act-only），
   激活完美时封顶 `0.7647`（w-only）。
3. **C24 绝对门在当前权重下不可达**：C24 晋级门要求 Linear `>=0.78`，
   而 C21-C 权重侧单侧上限 `0.7647 < 0.78`——即使激活求解器完美也
   无法通过 C24 自身的门。这从独立角度证实 Checkpoint B 暂停正确；
   任何未来 activation 候选必须与权重侧改进配对才可能突破 0.765。
4. 分项最弱单侧（改进优先级）：fc 权重侧（0.730）、proj 激活侧
   （0.688）、fc 激活侧（0.745）、o 双侧（0.756/0.755）。
5. 对 22k~25k 主目标（Linear 0.79~0.89）的含义：必须两侧同时各砍
   ~50%+ 残差；C23 类单侧权重改进（+1.9pp）量级远不够，需要机制级
   突破或两侧联合候选（新 candidate ID 预注册）。

### C21-C 最终 holdout 终验收（2026-08-27，第 1/3 次预算消耗）

- 触发：Checkpoint B 暂停后，对 Champion v025（sha `83AB4864…`）做
  官方提交前的最终验收；`--reason` 已记入台账。
- 冻结配置（amax6 causal offset 0，12 层，4 test 窗口，CPU）下：
  `linear_mean 0.523558`、`attention_mean 0.441898`、
  `algorithm_stage 69.29s`（calibration 53.26s / dynamic 9.94s）。
- 与开发集对照：Linear −0.75pp（0.5311→0.5236）、Attention causal
  −0.78pp（0.4497→0.4419），未见文本降幅 <1pp，无过拟合信号。
- 台账更新：`holdout_runs_used 1/3`，剩余 2 次；见
  `evaluator/holdout_ledger.json` 与 v025 归档 `result.md`。
- 含义：C21-C 获得提交就绪的完整证据链（合规门禁 + 固定矩阵 +
  576 合成 case + holdout 终验收），可随时官方提交；C23 的 holdout
  权利未消耗（v027 已 rejected，无需消耗）。

## C3 预注册：top-K 8×8 Linear 二阶

- Candidate ID：`C3`
- Parent：`C1`，SHA256
  `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- 唯一机制：在现有 4×4 Linear 二阶基础上，只对 calibration 损失最高的
  少量 8 通道组使用 8×8 Gram 与 `H·e` 增量更新；其余组保持 C1。
- 开发目标：优先提升 `fc/proj/o`，Linear mean 至少 `+0.2pp`，Attention
  相对 C1 不下降超过 `0.2pp`，CPU ratio ≤1.15。
- 开发数据：offset 0、amax6；先验证 Linear 分项和 Attention 双 mask，开发
  门通过后才运行固定回归矩阵。
- 状态：`in-progress`。

## 步骤 7（历史）：发布复核与计时修正（§1.1 / §7.3 / §10）

- 2026-08-27 更新：用户确认 v002/B0 官方结果为 **15313 分 @ 137s**，并明确
  可据此闭环 B0。该确认取代此前 `~15000 / ~140s` 近似值以及“撤回闭环”
  结论。官方上传文件未重新下载，因此不把本地 GPU-compatible derivative
  SHA256 冒充上传文件 SHA；v002 归档源码 SHA 与本地 derivative SHA 继续分列。
- 计时 wrapper 已修复 calibration 内部动态 API 的嵌套重复统计。CPU 配对
  采用串行运行、`--device cpu --attn-mask both` 和相同算法阶段边界：

| 版本 | algorithm-stage | calibration | dynamic | api-total |
|---|---:|---:|---:|---:|
| B0 | 52.26s | 42.79s | 4.85s | 47.64s |
| A1-only（当前） | 54.72s | 45.34s | 4.94s | 50.28s |

- A1/B0 algorithm-stage 比率为 `1.0471`。B0 官方时间现已精确为 `137s`；
  任何按本地比率计算的候选时间仍只作工程参考，不作为候选官方时间。
- CPU A1 分数为 causal `0.4497`、non-causal `0.4944`，与 CUDA
  `0.4497/0.4942` 一致，增益跨设备稳定。
- 新增 `tests/test_release_candidate.py`，覆盖 A1-only flags、无文件 I/O/
  调试输出、feature-off 字段级 B0 等价、GQA/head_dim 128 旋转不变量、
  state 合法性和计时嵌套去重；5 项测试通过，`git diff --check` 通过。
- 上述检查不裁决现有 Linear sampled Activation×Weight 是否符合官方规则，
  因此只能称为本地发布检查通过，不能宣称完整官方 AST 合规。
- B0 官方数据已按用户确认闭环。当前本地 Champion 决策不依赖该结果；
  未来如获得候选官方结果，仍只追加到对应候选归档，不覆盖本地实验记录。

## 步骤 1：Telemetry（§12.1）

- `evaluator/real_data_eval.py` verbose 模式新增 state 特征选择率输出
  （q/k/v 三侧的 multiplier/permutation/rotation/centering/importance 采纳率）。
- 仅从返回 state 只读字段推导，不触碰 solution 源码，符合 §12.1 合规边界。
- 当前 A1-only 观测（MHA）：rotation 0%；permutation、K centering 和 V
  importance 的选择率由评测器侧只读统计输出。历史 A2 实验的 rotation
  采纳率为 25%（3/12 层），不得再表述为当前候选状态。

## 步骤 3：A1 真实输出选择器 + 终验门（§5.1）

- selection 流程重构为 `_run_selection(use_a1)` 双轨闭包：A1 轨以真实 attention
  输出误差排序候选，proxy 轨精确复刻 B0 的 Q/K 重建 proxy 选择。
- 终验门：A1 winner vs B0 proxy winner 通过完整 `hif4_dynamic_quantize_q/k/v`
  部署路径在 calibration 前缀上重算真实输出误差（causal 主轨 + non-causal
  安全轨，V 部署路径固定），A1 无明确改善（≥0.5%）或安全轨退化（>2%）时
  逐层回退 B0 选择。
- 设计修正记录：初版门对比 winner vs identity，诊断（`artifacts/diag_gate.py`）
  证明 identity 在退化层并非正确回退目标（test 上同样差），改为对比 B0 proxy。
- 门在 L8/L10/L11 三次拒绝全部正确，退化层恢复 B0 数值，逐层全部 ≥ B0。
- MHA causal +7.1pp / non-causal +8.3pp；GQA causal +8.5pp / non-causal +12.8pp。

## 步骤 4：A2 固定 H64 旋转（§5.2）

- 组对齐 signed Hadamard(64)：`_attention_rotation_signs` 生成 [kv_heads, head_dim]
  确定性 signs，`_apply_attention_rotation` 旋转 head block；GQA 同组 Q heads 与
  K head 共享旋转，Q·K 点积严格不变（单元验证误差 ~1e-5）。
- 动态路径：`_nvfp4_to_hif4` 新增 `attention_rotation`/`rotation_num_heads` 参数，
  permutation 后施加；q/k API 从 state 读取；在线每张量一次 FWHT，V 不旋转。
- calibration：A1 终验门定稿后对 2 个 sign seed（0/1）做部署路径评估，复用同一
  门控且旋转额外要求 non-causal 均值不得变差（`safety_tolerance=0.0`）。
- 点积不变性验证：`artifacts/unit_rotation.py`（MHA 3.4e-5 / GQA 4.2e-6）。
- H64=off 消融精确复现 A1 数值，归因干净；采纳率 25%（3/12 层）。
- 诊断（`artifacts/diag_rotation.py`）：采纳层 calib 双轨均真实改善（ratio 0.89–0.97），
  GQA test 上 L1/L6 non-causal 退化 2.4/3.1pp 属 test 分布漂移（均值 −0.33pp，
  超过 §10 的单层 1pp 和非目标均值 0.2pp 门槛）。发布复核还发现 MHA
  causal L11 从 A1 的 0.3345 降至 0.3148（−1.97pp）。
- 判定修正：**不晋级**。`_ATTN_H64 = False`；实现和诊断保留，必须在 A1
  获得独立官方正结果后，才能作为单机制候选重新评估。

## 步骤 5：A3 V bias-aware（§5.3）

- 实现（接口不变、动态路径零成本，仅 calibration 侧）：
  - `_attention_head_square_mass` 扩展为返回 (E[A], E[A²]) 双矩
    （同一次 softmax，无额外开销）；
  - 三个 V importance 候选：当前 E[A²]（二阶矩对角项）、一阶矩 E[A]
    （attention 概率推导 head 权重）、E[A²]+E[A]²（对角 + 均值交叉项，
    均值误差抑制）；
  - v_state 构造重构为 `_build_v_state(importance)` 闭包；A2 定稿后对候选
    逐个用完整 `hif4_dynamic_quantize_v` 部署路径重算真实输出误差，复用
    A1 门控（causal ≥0.5% 改善 + 安全轨 ≤2% 退化）。
- 配对评测（候选开启）：MHA causal 0.4524→0.4518（−0.06pp）、GQA 持平；
  MHA L5 层 −0.8pp。未达 §10 的 +0.2pp 均值门槛。
- 判定：**不晋级**。`_V_IMPORTANCE_CANDIDATES = False`（实现保留，含判定
  注释），关闭后精确恢复 A2 数值，归因干净。
- 结论记录：head 级 importance 的矩选择对真实输出影响极小——印证 spec
  §5.3 的预判（head 内常量 importance 主要影响困难块排序，对块内离散解
  影响有限）。V 路径的后续增益空间不在 head 级权重。

## 步骤 6：L1 数据驱动 scale（§6.1）

- 实现（`_dense_to_hif4` 困难块精修段，offset 搜索 + 边缘扩展之后）：
  - 加权 LS 连续 scale：以当前五字段 winner 的 mantissa 为锚点求闭式 LS 解
    （importance / gram 对角为权重），经 ±1 相邻 E6M2 code 与精确层级求解
    交替更新；
  - 截尾分位数 scale：块内 |x| 的 0.90/0.95 分位映射到顶层 mantissa
    （截断离群值）；每个 base code 展开 3 个相邻 code，共 9 候选/块，
    全部经 `_solve_exact_hierarchy` 后逐块 improve-mask 回退。
- 诊断（`artifacts/diag_l1_weights.py`）：weights 触发 15–119 块/矩阵，
  plain MSE 仅改善 0.01–0.03%；配对评测 linear 均值 ≈ +0.04pp 且 o 单项
  −0.09pp；attention MHA causal −0.60pp / non-causal −0.85pp（逐块加权
  损失下降但 Q/K 经 softmax 非线性放大，proxy 错位——A1 终验门无法防御
  部署量化器本身的变化，两侧对比都在 L1 世界内）；dynamic 时间 3.5s→5.5s。
- 判定：**不晋级**。`_L1_DATA_DRIVEN_SCALE = False`（实现保留，含判定
  注释），关闭后 attention 精确恢复 A2 数值、时间回落 3.5s，归因干净。
- 结论记录：现有 offset 网格 + 边缘扩展已覆盖 E6M2 code 邻域（LS 锚点
  落在 winner ±1 code 内，增益边际）；困难块的 weighted 准则与 attention
  真实输出存在系统性错位，任何降低逐块损失的部署路径变更都需对"无变更
  基线"做真实输出门控（当前门只对比选择差异）。

## 风险与待办

- **当前本地 Champion**：`solution.py` 为 A1-only，本地 MHA causal +7.1pp、
  non-causal +8.3pp；GQA causal +8.5pp、non-causal +12.8pp。Linear 不变。
- **官方状态**：unavailable。不会阻塞本地优化和归档；未来结果只追加更新。
- 官方场景含合成 attention 场景（saturated logits 等，本地无对应评测器）；
  A1 的逐层门控以 B0 选择为回退目标、在当次 calibration 数据上判定，
  最坏情形≈B0 行为，但仍需公开合成集和最终 holdout 验证分布漂移。
- head_dim 128 已通过 synthetic GQA 不变量、state 和动态路径测试，但尚无
  对应真实模型精度评测。
- offset 97/193/389 已用于 A1 回归，不再称为匿名 holdout，也不得用于后续调参。
- 所有改动未提交 git。
- 下一步：① 归档 C1 本地源码与结果；② 预注册 C2；③ 实现并评测
  Segment-CVaR；④ 归档 C2 结论；⑤ 在最新本地 Champion 上进入 C3
  top-K 8×8 Linear 二阶。
