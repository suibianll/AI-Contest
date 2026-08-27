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

- 用户提供的官方结果仍只有 **~15000 分 @ ~140s**。该数据绑定到 v002
  归档的证据不完整，不能满足 §1.1 所要求的精确分数、精确时间和候选
  SHA256 绑定；此前“B0 门禁关闭”的结论撤回。
- 计时 wrapper 已修复 calibration 内部动态 API 的嵌套重复统计。CPU 配对
  采用串行运行、`--device cpu --attn-mask both` 和相同算法阶段边界：

| 版本 | algorithm-stage | calibration | dynamic | api-total |
|---|---:|---:|---:|---:|
| B0 | 52.26s | 42.79s | 4.85s | 47.64s |
| A1-only（当前） | 54.72s | 45.34s | 4.94s | 50.28s |

- A1/B0 algorithm-stage 比率为 `1.0471`。若仅用近似 140s 外推，结果约
  146.6s；因为 B0 官方数据不精确且未与当前 B0 SHA256 绑定，该数字只作
  工程参考，不作为正式 270s 门禁结论。
- CPU A1 分数为 causal `0.4497`、non-causal `0.4944`，与 CUDA
  `0.4497/0.4942` 一致，增益跨设备稳定。
- 新增 `tests/test_release_candidate.py`，覆盖 A1-only flags、无文件 I/O/
  调试输出、feature-off 字段级 B0 等价、GQA/head_dim 128 旋转不变量、
  state 合法性和计时嵌套去重；5 项测试通过，`git diff --check` 通过。
- 上述检查不裁决现有 Linear sampled Activation×Weight 是否符合官方规则，
  因此只能称为本地发布检查通过，不能宣称完整官方 AST 合规。
- 历史官方数据仍未完成精确 SHA 绑定，只作为历史来源信息。当前本地
  Champion 决策不依赖该数据，未来如获得官方结果则追加到候选归档。

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
