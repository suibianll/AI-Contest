# 当前主版本：算法与评测状态

更新：2026-09-01。根目录 [`solution.py`](../solution.py) 是当前已验证且满足本地 API `<300s` 代理的活动版本；归档源码只读。
本页只描述当前状态，历史算法的逐项证据见 [`algorithm-inventory-and-directions.md`](algorithm-inventory-and-directions.md)
和 [`archive-implementation-audit.md`](archive-implementation-audit.md)。

## 评测口径

唯一活动评测器是 [`evaluator/official_eval.py`](../evaluator/official_eval.py)，协议
`official-shape-v1`。它固定 Qwen2.5-0.5B 的 24 层、250 Linear + 200 Attention case，
Attention calibration 长度 `[10,128,512,1024,1024]`，并在每个 API 边界独立校验参数和 state。
`Linear calibration=2` 是公开本地数据包的明确假设，因为赛事说明书没有给出 Linear 折数；
该假设不能冒充官方隐藏数据。

每个 case 的分数为

\[
s=(MSE_{STD}-MSE_{PLAYER})/MSE_{STD},
\]

报告主指标是 250 个 Linear case 的算术平均 `linear_mean` 和 200 个 Attention case 的
算术平均 `attention_mean`。`total_sum` 是 450 个 case 分数的和；`equal_weight_45000_scale`
只是乘 100 的等权显示。官方最近减少 Linear 权重且未公布系数，因此不能从本地值拟合官方
绝对分。

时间分为六 API 的 `api_total_seconds` 和含本地评分开销的 `wall_seconds`；二者仅能做同机、
同 cache、同协议 A/B，不能直接推断鲲鹏 920B 的 `<300s` 结论。

## 活动根版本

根代码目前是 v132（固定预算 Attention + 输出监督 W + Q(W)-Gram），SHA256 为
`7C4884A710F17F44904E8C1C8EA1AC89667711A5B0162497AEAC4D5DAD389F3E`。实现保留：

1. BOAT 对角平衡与 signed-Hadamard 等价变换，满足 `X'W'^T=XW^T`；
2. Cross-fold Weight-HSDQ，在 64 通道块上使用校准激活统计做合法 HiF4 离散搜索；
3. Gram-hierarchy Activation-HSDQ，使用最终部署 Q(W) 的静态 Gram 做有限预算激活候选筛选；
4. Expansive-FFN CAT balance，仅在 `rows>channels` 的形状启用并失败回退；
5. `Q(A)`/`A@W` 输出监督的分块权重精修；
6. Attention 的 reciprocal balance、K-centering、rotation、GQRB shortlist，候选在
   128-token proxy/256-token shortlist 上评分，动态 Q/K 使用 2 sweep；dense PAWV 已移除。

已从根代码裁剪并仅保留在归档中的方向：Global Activation-LRH、final deployed-Gram
row gate、GALS、block-local permutation、L6 rank/factor 系列和 C1 refresh/rank 系列。
它们的 full-layer 结果普遍超过 300 秒或没有稳定的跨折收益，不是当前发布门禁。

## 新协议复测结果

批量结果写入 [`artifacts/official_eval/archive-official-shape-v1.json`](../artifacts/official_eval/archive-official-shape-v1.json)，
报告写入 [`logs/official_eval/archive-official-shape-v1.md`](../logs/official_eval/archive-official-shape-v1.md)。
表中本地数值必须来自同一 cache 和同一设备；`error` 是该归档在本机 CUDA 上的真实运行错误，
不能用零分替代。

| 版本 | Linear mean | Attention mean | API total(s) | Wall(s) | 官方裁决 |
|---|---:|---:|---:|---:|---|
| v001 | 0.315499 | 0.449961 | 36.868 | 59.582 | pass |
| v002 | — | — | — | — | 本机 device-mix error |
| v013 | 0.437667 | 0.639206 | 71.433 | 95.076 | pass |
| v024 | 0.450075 | 0.639206 | 76.148 | 99.192 | pass |
| v025 | 0.387146 | 0.639206 | 77.815 | 101.034 | pass |
| v030 | 0.265321 | 0.639206 | 93.005 | 116.100 | pass |
| v031 | 0.374651 | 0.639206 | 87.008 | 109.642 | pass |
| v032 | 0.356338 | 0.639206 | 143.127 | 165.348 | pass |
| v034 | 0.374651 | 0.639206 | 87.730 | 109.926 | pass |
| v051 | 0.355097 | 0.639206 | 163.484 | 185.502 | pass |
| v066 | 0.357919 | 0.639242 | 165.801 | 187.881 | pass |
| v072 | 0.365587 | 0.639242 | 165.616 | 187.545 | pass |
| v074 | 0.372739 | 0.639242 | 180.526 | 202.576 | pass |
| v084 | 0.406668 | 0.718107 | 279.191 | 300.848 | pass（新权重） |
| v098 | 0.465655 | 0.833573 | 406.681 | 429.285 | timeout |
| v100 | 0.465655 | 0.833617 | 417.747 | 439.896 | Attention WA/timeout |
| v107 | 0.469211 | 0.833617 | 436.719 | 459.727 | Attention WA |
| v121 | 0.472198 | 0.833617 | 3404.369 | 3429.645 | timeout |
| v127 | 0.465655 | 0.833617 | 416.465 | 439.617 | 新协议 root 基线 |
| v128 | 0.465655 | 0.833573 | 405.851 | 428.122 | PAWV 移除 |
| v129 | 0.465655 | 0.837789 | 310.732 | 332.557 | 固定预算，仍超代理 |
| v130 | 0.465655 | 0.836579 | 248.363 | 270.606 | 固定预算 sweep1 |
| v131 | 0.471837 | 0.836579 | 295.437 | 317.607 | 输出监督 W |
| v132 | **0.473131** | **0.834256** | **285.929** | 306.940 | **当前根，API<300** |

本轮统一复测中，旧归档**最高本地等权显示**为 v121：`linear_mean=0.472197763`、
`attention_mean=0.833617251`、`equal_weight_45000_scale=28477.289`，但 API/Wall 均远超
300 s，且官方历史裁决为 timeout；它不是可提交版本。当前根 v132 的新协议单轮结果为
Linear `0.473131`、Attention `0.834256`、API `285.929s`、wall `306.940s`；API 总和是
本计划采用的本地官方时间代理。以上数值均来自同一 cache、同一协议和同一 CUDA 设备；
完整字段以 JSON 为准。

官方历史锚点（独立于本地代理）：v74 `22750 / 239.387s`（旧权重，通过）；v84
`16517 / 252.563s`（新权重，通过）；**v86 `16744 / 222.7s`（新权重，通过，当前
官方最佳：分数比 v84 高 `+227`、时间比 v84 快 `29.863s`）**；v98/v121 timeout；
v100/v107 Attention WA。

v86 尤其重要：它是**唯一在官方 300s 限制内通过、且改动了 Attention 路径**
（C86 Q/K 共享 block-Hadamard）的候选，证明 Attention 侧改动本身不必然导致
超时，与 v098/v100/v121 的 B1 GQRB / B2 PAWV 路径形成对照。方向含义：官方时间
风险来自"per-seq_len 分组 + Python 循环"型 Attention 机制（PAWV/GQRB），而非
Attention 改动本身。

## 下一步

当前 active 计划是 [`2026-09-01-hif4-linear-0.8-under-300s-plan.md`](superpowers/plans/2026-09-01-hif4-linear-0.8-under-300s-plan.md)。它只允许在
`official-shape-v1` 上比较 Linear/Attention 均值和 API 时间；当前根 v132 已完成完整 450-case
复测。任何新版本必须保留源码 SHA、统一命名归档和可复现 JSON。

## 归档规则

- `solutions/` 只存版本源码和 `result.md`，不覆盖、不重写历史；本地结果不得填入 Official 字段。
- 活动计划目录只保留一份计划，完成或废止的文件移动到 `docs/superpowers/archive/plans/`。
- 活动评测输出只写 `artifacts/official_eval/`、`logs/official_eval/`；旧
  `artifacts/real_model_suite/` 和旧 evaluator 不再读取。
