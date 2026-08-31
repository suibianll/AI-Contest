# v84、v98、v100 官方超时差异与统一运行时分析

日期：2026-08-31  
目的：解释为什么 v84 在官方评测中以 16517 / 252.563s 通过，而 v98/v100
在官方环境中超时；同时消除旧 v1 本地结果与活动 v2 结果混用造成的误判。

## 1. 结论先行

v84 与 v98/v100 的活动本地复评现在已经使用同一套 v2 配置，但官方行为仍然
不同，原因不是样例比例，而是 Attention 校准实现的序列长度复杂度不同：

1. v84 的代理搜索将每个 calibration sample 的候选评估限制为最多 128 行，
   真实 Attention 输出门限制为最多 256 行；只有一次 V head-mass 统计使用完整
   序列。
2. v98/v100 在每个候选上直接使用完整的 Q/K/V calibration sample。每个候选都
   要重新构造 QK^T、softmax 和 P@V；v100 还增加 PAWV 的完整 P^T P 对角统计和
   逐 token-row V refinement。
3. Attention 的核心矩阵计算是 O(T^2 d)。官方长度
   [10,128,512,1024,1024] 的 token-pair 工作量是本地两个 T=128 样本的
   约 72.5 倍；再乘以 v98/v100 的多候选全序列重算，官方时间会反转本地
   seq=128 下的排序。

因此，v84 的官方通过是“长序列运行时结构较好”的结果；v98/v100 的官方超时
是“本地固定短序列没有暴露多候选全矩阵成本”的结果。v100 的变长 PAWV 修复
只消除了 shape mismatch，不会消除超时复杂度。

## 2. 统一证据矩阵

### 2.1 官方结果

| 版本 | 官方分数 | 官方时间 | 官方结论 |
|---|---:|---:|---|
| v84 / C84 | 16517（新权重） | 252.563s | 通过，距离 300s 还剩 47.437s |
| v98 / B1 GQRB | — | >300s | timeout |
| v100 / B2 PAWV | — | >300s | 原始线先出现 Attention WA；变长修复线继续 timeout |

v84 的官方记录：[v84 官方结果](2026-08-31-v84-official-result.md)。

### 2.2 本地 v84 的两条记录必须分开

此前存在两个名为 v84 的本地文件：

| 记录 | profile | 设备 | 样例 | Linear | Attention | API / Wall |
|---|---|---|---|---:|---:|---:|
| 历史记录 | sampled-means-v1 | CPU/CPU | 224L/32A | 0.477266 | 0.709020 | 422.615 / 433.624s |
| 活动可比记录 | sampled-means-v2 | CUDA/CUDA | 112L/96A | 0.489389 | 0.739172 | 239.910 / 243.017s |
| 活动 v2 独立复测 | sampled-means-v2 | CUDA/CUDA | 112L/96A | 0.489389 | 0.739172 | 234.361 / 237.418s |

v84 的 422.615s 来自历史文件
[v84-sampled-means-qwen.json](../../artifacts/real_model_suite/v84-sampled-means-qwen.json)，
不能与 v98/v100 的 v2 时间比较。统一复评使用的 v84 记录来自
[official-anchors-sampled-means-v2.json](../../artifacts/real_model_suite/official-anchors-sampled-means-v2.json)，
独立复测保存在
[v84-sampled-means-v2-cuda.json](../../artifacts/real_model_suite/v84-sampled-means-v2-cuda.json)。

### 2.3 活动 v2 的同配置本地结果

所有版本均使用：Qwen2.5-0.5B、固定 cache
seq=128 / calib=2 / test=4 / layers=all、amax6、seed 20260831、
device=cuda、algorithm_device=cuda、同一 NVFP4 codec、同一
112 Linear + 96 Attention 构成。

| 版本 | Linear mean | Attention mean | API(s) | Wall(s) | Calibration(s) | Dynamic(s) |
|---|---:|---:|---:|---:|---:|---:|
| v84 | 0.489389 | 0.739172 | 239.910 | 243.017 | 221.593 | 18.317 |
| v84 独立复测 | 0.489389 | 0.739172 | 234.361 | 237.418 | 216.427 | 17.934 |
| v98 | 0.516969 | 0.842022 | 169.000 | 172.369 | 145.724 | 23.277 |
| v100 | 0.516969 | 0.842024 | 176.158 | 179.574 | 148.314 | 27.844 |

API 分解显示了本地短序列下的反直觉现象：

| 版本 | Weight calibration | Attention calibration | Activation | Q | K | V |
|---|---:|---:|---:|---:|---:|---:|
| v84 | 198.448 | 17.979 | 15.767 | 0.788 | 0.718 | 0.661 |
| v98 | 61.885 | 83.838 | 6.982 | 7.735 | 7.842 | 0.717 |
| v100 | 61.898 | 86.416 | 6.998 | 7.711 | 7.962 | 5.173 |

v84 的本地总时间较高，主要是 Linear 的 JDRQ/层级搜索较重；但它的 Attention
校准只有约 18 秒。v98/v100 的 Linear 更快，却把大量时间花在 Attention 候选
重算上。官方长序列会放大后者，而不会按同样方式放大 v84 的 Linear 成本。

## 3. 三个版本的 Attention 代码差异

### 3.1 v84：有明确的长度预算

v84 的关键常量为：

    _ATTN_STATS_TOKENS = 4096
    _ATTN_EVAL_TOKENS = 128
    _ATTN_A1_MAX_TOKENS = 256

在 hif4_calibration_attention 中：

    q_samples.append(_sample_rows(q, _ATTN_EVAL_TOKENS))
    k_samples.append(_sample_rows(k, _ATTN_EVAL_TOKENS))
    v_samples.append(_sample_rows(v_dense, _ATTN_EVAL_TOKENS))
    prefix = min(int(q.shape[0]), _ATTN_A1_MAX_TOKENS)

因此：

- 代理 Q/K 重建与候选搜索最多处理 128 行；
- A1 真实输出门最多处理 256 行；
- 统计量最多线性使用 4096 行；
- 完整序列只用于一次 v_head_square_mass 统计，不在每个候选上重复运行。

v84 的代理 attention_candidate_metrics 在没有 A1 context 时只计算加权重建误差，
不调用完整 Attention；只有 A1 context 走真实输出路径，而且输入已经被截断到
256 行。

### 3.2 v98：每个候选都跑完整 Attention

v98 在 hif4_calibration_attention 中直接保存所有解码后的 Q/K/V，并先计算：

    references = [_attention_forward(q, k, v) for ...]

之后 attention_candidate_score 对每个候选执行：

    q_params = _dense_to_hif4(q)
    k_params = _dense_to_hif4(k)
    q_hat = _dequantize_hif4(q_params)
    k_hat = _dequantize_hif4(k_params)
    output = _attention_forward(q_hat, k_hat, v_hat)

这里的 q/k 是完整 calibration sample，没有 128/256 行上限。

候选数量按源码固定网格估算：

- Stage 1：4 个 alpha × 2 个 center = 8；
- Stage 2：3 个旋转尺寸 × 2 个 seed = 6；
- GQRB：None 加两种宽度各 4 个角度、1 个 covariance、1 个 Hadamard，共 13；
- 最后再对 4 个基础候选和 4 个 GQRB 候选做 refine。

所以一次 Attention calibration 大约有 8+6+13+8=35 次候选评分，每次评分
遍历所有 calibration sample，并重新执行完整 QK^T 与 P@V。

### 3.3 v100：v98 成本之上再加 PAWV

v100 保留 v98 的完整候选扫描，同时增加：

    probability = softmax(Q @ K.T / sqrt(d))
    diagonal = probability.square().sum(dim=1).mean(dim=0)

_build_pawv_metric 按序列长度分组后，虽然已经修复了不同长度相加时的
shape mismatch，但每个 calibration sample 仍然要完整计算一次 QK^T。

随后 _refine_v 对每个 token row、64-channel block 和 15 个离散 level 枚举候选。
它的主要复杂度是 O(T·C·L)，而 PAWV metric 的主要矩阵成本仍是 O(T^2 d)。

## 4. 复杂度推导

设 Q 头数为 H_q，每个 head 的维度为 d，序列长度为 T。GQA 复制 KV 后，
Attention logits 为：

    L = QK^T / sqrt(d)
    Q ∈ R^(H_q × T × d), K ∈ R^(H_q × T × d)

矩阵乘法、softmax 和 P@V 的时间量级均受 H_q T^2 d 控制，显存中的
logits/probability 也受 H_q T^2 控制：

    Cost_attn(T) = Theta(H_q T^2 d)

本地 v2 的两个校准样本均为 128 token：

    W_local = 2 × 128^2 = 32,768

官方 mini calibration 长度为 [10,128,512,1024,1024]：

    W_official = 10^2 + 128^2 + 512^2 + 1024^2 + 1024^2
                = 2,375,780

    W_official / W_local ≈ 72.51

这只是一次 QK 矩阵的比例。v98/v100 还要乘以约 35 个候选评分；v100 还要
额外乘以 PAWV metric。v84 的候选评分把 T 替换为：

    T_proxy = min(T, 128)
    T_A1    = min(T, 256)

因此在 T=1024 时，v84 的候选搜索矩阵最多为 256^2，而 v98/v100 仍为 1024^2。
仅按平方项计算：

    1024^2 / 128^2 = 64
    1024^2 / 256^2 = 16

这解释了为什么 v84 在本地 T=128 下不一定最快，但在官方长序列环境中更容易
保持在 300 秒以内。

## 5. 为什么“样例比例相同”仍不能预测官方时间

活动 v2 已经把 Linear/Attention 构成调整到 112/96，这解决了组件比例过度
偏向 Linear 的问题，但没有解决以下独立问题：

1. 绝对 case 数不同：本地只执行抽样 case，官方执行 250L+200A；
2. 校准长度不同：本地固定 T=128，官方包含 512/1024 长序列；
3. 候选内部是否重复全矩阵：v84 对候选截断，v98/v100 不截断；
4. 硬件和后端不同：本地 v2 是 CUDA，官方是指定评测平台；
5. 官方时间是端到端 wall time：不能把本地 API 累计时间当成官方计时。

因此，112/96 只保证评分样例构成接近官方，不能保证运行时压力接近官方。
对 v84/v98/v100 来说，第 3 项是决定性差异。

## 6. v100 的 WA 与 timeout 是两个连续问题

v100 原始实现先遇到 PAWV 变长问题：如果第一个样本是 10 token，metric
固定为 [10,10]，第二个样本为 128 token 时会执行不合法的
[10,10] += [128,128]，导致 Attention wrong answer。

按长度分组后的修复使其能处理 [10,128,512,1024,1024]，但只是把 metric 变成：

    row_diagonal["10"]
    row_diagonal["128"]
    row_diagonal["512"]
    row_diagonal["1024"]

它仍然需要为每个长度计算完整 attention probability，并且 v100 的候选搜索
仍然对完整 Q/K/V 重复执行。因此结果从“早期 WA”变成“可以继续运行但超时”，
不是修复失败，而是暴露了第二个独立瓶颈。

## 7. 最终判断

### v84 为什么通过

- Attention 候选评估是固定预算，长序列只影响少量统计；
- 代理评分最多 128 行，A1 输出门最多 256 行；
- 没有 v100 的 PAWV full token-row refinement；
- 官方时间 252.563s，在新 <300s 限制下仍有约 47.4 秒余量。

### v98 为什么超时

- 完整 calibration Q/K/V 不截断；
- 约 35 个候选对每个完整样本重复执行 Attention；
- 官方 1024-token 样本使单次矩阵成本按 T^2 放大；
- 本地 v2 的 169s 只覆盖 T=128，无法暴露该成本。

### v100 为什么超时

- 继承 v98 的完整多候选 Attention 搜索；
- 额外执行 PAWV P^T P 对角统计；
- 额外执行 V 的 token-row 坐标 refinement；
- 变长修复解决的是正确性，不是复杂度；
- 本地 v2 的 176s 仍是固定 T=128 的短序列结果。

## 8. 后续评估和优化规则

1. sampled-means-v2 只用于 Linear/Attention 平均误差和同配置 A/B 排名。
2. 官方时间预测必须增加独立 runtime-stress profile，至少覆盖
   [10,128,512,1024,1024] 和所有 24 个 Attention layer。
3. 任何 Attention 候选必须记录：完整矩阵次数、候选数、最大 token 数、
   PAWV 次数和每阶段 wall time。
4. v84/v98/v100 的时间比较应使用同一 v2 结果表；旧 v1 的 v84 422.615s
   只能保留为 legacy，不得混入当前排名。
5. 真正要让 v98/v100 进入可提交区间，优先级是：
   - 把候选输出评分改为 128/256 行截断或分块近似；
   - PAWV 只保留按长度的低成本统计，避免对每个候选重新构造 dense P；
   - 对长序列使用 chunked/block-sparse attention probability，避免完整 T×T 常驻；
   - 在保持 Attention 均值不明显下降的前提下，缩减候选网格和 refine 次数。

## 9. 复现入口

- [v84 活动 v2 独立复测 JSON](../../artifacts/real_model_suite/v84-sampled-means-v2-cuda.json)
- [v84 活动 v2 独立复测报告](2026-08-31-v84-sampled-means-v2-cuda.md)
- [v98 活动 v2 复评](../../artifacts/real_model_suite/v98-official-timeout-sampled-means-v2.json)
- [v100 活动 v2 复评](../../artifacts/real_model_suite/v100-pawv-fixed-sampled-means-v2.json)
- [统一归档复评摘要](2026-08-31-official-archive-recheck-v2.md)
- [v84 官方结果](2026-08-31-v84-official-result.md)

