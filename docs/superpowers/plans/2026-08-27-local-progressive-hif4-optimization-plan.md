# HiF4 本地前向累积优化计划

日期：2026-08-27
状态：当前权威执行计划
替代范围：替代旧计划中“等待官方结果才归档”“任一层退化即整体回退 B0”和“官方时间预测作为本地阻塞项”的规则。

## 1. 目标与边界

当前无法运行官方评测，因此所有优化决策基于可复现的本地配对结果。历史官方分数只保留为历史信息，不参与当前候选排序，也不由本地指标外推。

优化采用**前向累积 Champion**：

1. 每个新候选只增加一个主要机制，并从当前本地 Champion 构建；
2. 候选失败时只判退该分支，当前 Champion 保持不变，不退回更早基线；
3. 某个配置或层的局部退化进入尾部指标和下一轮优化目标，不再自动否定跨配置稳定的综合增益；
4. 数值非法、接口/state 非法、崩溃或综合指标下降仍属于硬失败；
5. 本地评测结束立即归档，官方结果未来返回时只追加更新相同归档记录。

## 2. 当前基线与 Champion

| 编号 | 配置 | 角色 | 状态 |
|---|---|---|---|
| B0 | v002 GPU-compatible 基线 | 对照基线 | frozen |
| C1 | B0 + A1 真实 Attention 输出选择器 | 当前本地 Champion | local-champion |
| X-A2 | C1 + H64 | 历史实验分支 | local-rejected |
| X-A3 | C1/A2 + V bias-aware | 历史实验分支 | local-rejected |
| X-L1 | C1/A2 + data-driven scale | 历史实验分支 | local-rejected |

C1 晋级依据：

- 六组 MHA 配置的平均增益约为 causal `+5.71pp`、non-causal `+7.74pp`；
- 两组 GQA 配置的平均增益约为 causal `+8.44pp`、non-causal `+10.28pp`；
- Linear 与 B0 完全一致；
- CPU algorithm-stage 比 B0 增加约 `4.71%`；
- offset 193 的 GQA non-causal 单层 `-6.89pp` 是明确尾部债务，进入 C2 目标，但不覆盖以上综合主效应。

## 3. 候选预注册

任何代码修改前必须先在执行日志登记以下字段：

- candidate_id；
- parent_id 与 parent SHA256；
- 唯一机制；
- 假设和预期影响的指标；
- 固定评测矩阵；
- 晋级公式和硬失败条件；
- 允许的最大本地时间增量。

未预注册的试验只能记为 exploratory，不能晋级 Champion。

## 4. 固定本地评测矩阵

### 4.1 真实模型主矩阵

- GPT-2 12 层、seq 128、calib 2、test 2；
- token offset：开发窗口 `0`，固定回归窗口 `97/193/389`；
- Attention mask：causal 与 non-causal；
- topology：MHA 12/12、GQA 12/6；
- NVFP4 mode：`amax6`、`amax4`、`pow2`；
- CUDA 用于精度矩阵，CPU 用于最终配对时间。

offset `97/193/389` 已被观察，不再声称是盲测 holdout；它们只作为固定回归集，不允许据此搜索 seed、阈值或候选网格。

### 4.2 安全矩阵

- head_dim 64/128；
- MHA/GQA shape；
- saturated、near-uniform 和 outlier Attention logits；
- HiF4 五字段 shape/dtype/finite；
- state CPU、contiguous、dense-strided、无梯度；
- feature flag 关闭时与父 Champion 字段级等价；
- 无 solution 侧 telemetry、文件 I/O、网络或调试输出。

## 5. 本地综合晋级规则

### 5.1 指标

对每个测试 case 记录相对父 Champion 的 delta：

- `attention_mean_delta`：所有已注册 Attention case 的等权均值；
- `linear_mean_delta`：q/k/v/o/fc/proj 的等权均值；
- `win_rate`：delta 大于 0 的 case 比例；
- `tail_mean_delta`：最差 10% case 的均值；
- `worst_case_delta`：最差单 case，仅作风险和后续目标，不单独触发整体回退；
- `cpu_time_ratio`：候选/父 Champion 的 algorithm-stage 比率。

### 5.2 晋级

精度候选满足以下条件即可晋级：

- 目标均值相对父 Champion至少 `+0.2pp`；
- 非目标均值下降不超过 `0.2pp`；
- `win_rate >= 70%`；
- `tail_mean_delta` 不低于 `-2pp`，或相对父 Champion 的既有尾部有明确改善；
- CPU 单机制默认 `cpu_time_ratio <= 1.15`；
- 所有安全矩阵硬条件通过。

`worst_case_delta` 必须完整记录，但不再用一个层覆盖全部平均增益。若尾部明显为负，它自动成为下一候选的优先优化目标。

### 5.3 硬失败

- NaN/Inf、shape/dtype/state 非法；
- API 不兼容或动态路径崩溃；
- 目标综合均值下降；
- 非目标均值超过门槛；
- 超过时间门槛且没有预注册为时间换精度候选；
- 缺少源码快照、SHA 或完整结果记录。

## 6. 前向候选序列

### C1：A1 真实 Attention 输出选择器

状态：`local-champion`。不再回退 B0。已知 GQA non-causal 尾部作为 C2 的明确目标。

### C2：Segment-CVaR Attention 选择器

唯一变化：将 calibration token 切为多个固定 segment，在真实部署路径上同时计算均值、最差 25% 和跨 segment 方差，使用：

```text
robust_attention_objective =
    mean_error
    + 0.50 * worst_quartile_error
    + 0.25 * cross_segment_std
```

目标：相对 C1 改善 GQA non-causal 尾部，同时保持 MHA 主增益。不得同时启用 H64、V bias 或新 scale。

### C3：Linear top-K 8×8 二阶

状态：`local-champion`。沿用现有 4×4 二阶，在最高损失 5% 的 8 通道组上使用 `H·e` 增量更新；固定矩阵 Linear mean 提升 `+0.83pp` 至 `+1.23pp`，Attention 不变，CPU time ratio `0.992`。

目标：提升 `fc/proj/o`，Attention 相对 C2 不变，CPU 时间比不超过 1.15。

### C4：8×8 coverage 10%

只把 C3 的 top-loss coverage 从 5% 提高到 10%，验证收益是否仍随覆盖率增长；其余机制和 cap 不变。若边际收益不足 `+0.2pp`，保持 C3 并结束 coverage 扩展。组合后的时间精简顺延为后续独立候选。

### 暂缓

- A2 H64：聚合有增益但尾部和 GQA 安全轨不足，待 C2 稳定后重新立项；
- A3 head-level V importance：已证明主效应过小；
- L1 data-driven scale：已证明 Attention 代理错位且增加动态时间；
- 扩大固定 offset 网格、全量 64×64 GPTQ、learned butterfly：不进入 C2/C3。

## 7. 本地归档规范

每次完整评测后立即创建不可变归档，不等待官方结果。目录暂沿用：

```text
solutions/YYYYMMDD_vNNN_topic_scoreNA_timeNA/
  solution.py
  result.md
```

`result.md` 必须包含：

- candidate/parent 编号和 SHA256；
- 唯一机制及 flags；
- 完整本地配置；
- 父子配对分项、delta、逐层尾部和时间；
- `local-champion`、`local-accepted` 或 `local-rejected`；
- `official_status: pending/unavailable/recorded`；
- 明确结论和下一候选。

官方结果以后可用时：

1. 不覆盖本地结果；
2. 在相同 `result.md` 追加官方分数、时间、日期和提交 SHA；
3. 更新 `solutions/README.md` 的 Official 列和状态；
4. 如需重命名目录，在同一次变更中修正全部链接；
5. 本地与官方方向不一致时记录 proxy mismatch，不删除本地实验。

## 8. 执行纪律

- 同一时间只允许一个 in-progress candidate；
- 根 `solution.py` 始终对应当前本地 Champion 或正在评测的已预注册候选；
- 每轮必须先归档，再开始下一轮；
- 不以“继续回退”代替机制改进；
- 不因候选失败删除实现或历史结果；
- 所有结论必须能从归档中的 SHA、配置和指标复现。
