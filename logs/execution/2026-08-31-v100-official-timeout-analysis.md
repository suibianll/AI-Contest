# v100 官方超时（>300s）根因分析

> 日期：2026-08-31
> 触发：用户确认 v100（PAWV 变长修复线）官方评测超时，运行时间超过 300s 上限。
> 此前 v100 原始版的官方结果是 Attention `wrong answer`（见
> [`v100 WA 边界审计`](2026-08-31-v100-official-wa-boundary-audit.md)）；修复
> 变长 bug 后完整跑通评测流程，暴露了真实时间成本。两个官方结果并存不矛盾：
> WA 会在早期 attention case 上提前判定，timeout 才反映完整流程耗时。

## 1. 本地实测分解（sampled-means-v1，seed 20260831，CPU）

数据来源：`artifacts/real_model_suite/v84-sampled-means-qwen.json`、
`v100-pawv-fixed-sampled.json`。本地计时只分两个桶（calibration/dynamic），
API 调用次数按 API 计：

| 候选 | calibration (s) | dynamic (s) | API 合计 (s) | calib 调用 | dynamic 调用 |
|---|---:|---:|---:|---:|---:|
| v84（官方 252.563s 通过） | 298.10 | 124.51 | 422.61 | 64（56 weight + 8 attn） | 320（224 act + 96 q/k/v） |
| v100-pawv-fixed（官方 >300s timeout） | 93.96 | 56.29 | 150.25 | 64 | 320 |

本地每 call 均值：v84 calib `4.658s`/dyn `0.389s`；v100 calib `1.468s`/dyn
`0.176s`。v84 的时间大头在 weight calibration（gram64 sweep5，vectorized），
v100 的边际成本在 attention PAWV/GQRB 路径。

## 2. 官方运行结构与本地抽样的三重错配

官方环境为 **Kunpeng 920B（ARM CPU）**，端到端限时 300s，panel 为
**250 Linear + 200 Attention（450 cases）**。本地 sampled-means-v1 只跑
8/24 层、4 窗口、224 Linear + 32 Attention、固定 seq128。三类错配：

| 工作类型 | 本地 sampled | 官方全量 | 低估倍数 |
|---|---|---|---:|
| Linear weight calib 调用 | 56（8 层×7 role） | ~168（24 层×7 role） | ×3.0 |
| Attention calib 调用 | 8 | ~24 | ×3.0 |
| Linear dynamic 调用 | 224 | ~250 | ×1.12 |
| Attention dynamic 调用（q/k/v） | 96（32 case×3） | ~600（200 case×3） | ×6.25 |
| **Attention case 占比** | 32/256 = 12.5% | 200/450 = 44.4% | ×3.56 |
| Attention dynamic 调用占比 | 96/320 = 30% | ~600/850 = 70.6% | ×2.35 |

叠加硬件差异：官方 Kunpeng 920B 对 vectorized numpy 权重校准类工作相对本地
x86 更快（v84 总比值 0.60），但 PAWV keyed-diagonal 的 per-seq_len-group
Python 循环、变长输入分组在官方端没有同等奖励；官方 attention 输入为变长
（PAWV 变长修复正是为此），本地固定 seq128 从不触发分组开销。

## 3. 定量验证：均匀比值模型无法解释，attention 侧必须单独归因

用 v84 官方结果拟合"均匀每-call 成本 × 官方结构"模型（官方结构 = 192 calib
call + 250 Linear dyn + ~600 attn dyn）：

- v84 本地等价官方结构：`192×4.658 + 850×0.389 ≈ 1225s` → 拟合官方/本地
  机器因子 `r ≈ 252.563/1225 ≈ 0.206`。
- 用同一 r 预测 v100：`0.206 × (192×1.468 + 850×0.176) ≈ 88.9s` → 应轻松
  通过。**与实际 timeout 矛盾。**

反推：v100 linear 侧（168 calib + 250 dyn，本地等价约 291s）按 r=0.206 官方
约 60s；要超过 300s，attention 侧（24 attn calib + ~600 attn dyn，本地等价
约 141s）官方须 ≥240s，即 attention 操作的官方/本地因子 `r_A ≥ 1.7`
（官方端比本地还慢 70%+）。对 Python 循环 + 变长分组 + 官方 200 case 放大的
PAWV 路径，这在方向上完全合理；精确值需评测器 per-API 计时（已补，见 §5）。

## 4. 官方证据链：超时根因是 PAWV/GQRB 机制，不是 Attention 改动本身

| 候选 | 官方结果 | 本地 sampled API | 备注 |
|---|---|---:|---|
| v74（旧权重链） | 239.387s 通过 | — | 旧权重基准 |
| v84（无 PAWV，GQA rotation） | **252.563s 通过** | 422.6s | 新权重基准锚点 |
| **v86（= v84 + C86 Q/K 共享 block-Hadamard）** | **222.7s 通过，16744** | 313.6s（旧协议 Qwen） | **Attention 侧改动但官方反而快 29.863s** |
| v098（B1 GQRB） | timeout | 219.0s | 同为 attention 重路径 |
| v100（B2 PAWV，原始） | Attention WA | — | WA 短路，未测得完整时间 |
| v100（pawv-fixed） | **timeout（>300s）** | 150.25s | 本轮用户确认 |
| v107（+L3） | Attention WA | 241.5s | WA 短路 |
| v121（+C1 线） | timeout | 832.9s | 本地最慢 |

**2026-09-01 更新（关键反例）**：v86 在 v84 基础上做了 Attention 侧改动
（C86：Q/K 共享 head-local block-Hadamard，block 4/8/16 + 终选器），官方结果
`16744 / 222.7s` —— 分数 `+227`、时间比 v84 **快 29.863s**。它证明"改动
Attention 路径"本身不会导致超时，反例成立。

因此结论需收紧为：**超时根因不是 Attention 改动，而是 B1 GQRB / B2 PAWV 这类
引入 per-seq_len 分组 + Python 循环、且官方端无向量化奖励的机制**。C86 属于
静态候选搜索（calibration 期一次性选出 block size 与 sign，online 无分组循环），
成本结构与 v84 同源，所以官方时间与 v84 同量级甚至更低。

本地 sampled 时间对 PAWV/GQRB 类候选系统性低估（attention 权重低估 2.3–3.6 倍），
本地时间越低越危险；但对 v84/v86 这类 linear/calibration-heavy 候选，本地时间是
保守上界。

## 5. 已落地的改进：评测器 per-API 计时

`evaluator/real_data_eval.py::instrument_solution` 新增
`stats["seconds"]`（每 API 顶层调用秒数），`real_model_suite.py` 的
`timing` 输出新增 `api_seconds` 字段。此后每次 sampled 运行可直接得到
attention（calibration_attention + q/k/v）与 linear（weight + activation）
时间分解，用三分量投影官方时间：

```
T_official ≈ r_cal·(192/64)·C_local + r_L·(250/224)·D_L + r_A·(600/96)·D_A
```

其中 `C_local/D_L/D_A` 由 `api_seconds` 直接读出；`r_cal/r_L` 可用 v84 锚点
拟合，`r_A` 可用官方通过的 attention 路径校准。

**2026-09-01 校准点补充**：官方通过锚点现增至 v84（`252.563s`）与 v86
（`222.7s`）。v86 = v84 + C86，Linear 侧完全不变，因此两者官方时间差
`−29.863s` 可直接归因于 Attention 侧改动：C86 用静态候选搜索替代了部分
online attention 工作，官方端净收益为正。这给出 `r_A` 的一个可用上界区间——
C86 型（静态搜索、无 per-seq 分组）机制在官方端不慢于本地；PAWV/GQRB 型
（动态分组、Python 循环）仍无官方通过数据点，`r_A` 只能给下界 `≥1.7`。

测试：`tests/test_real_model_suite.py` 15 passed；`instrument_solution`
冒烟验证 `seconds` 与 calibration/dynamic 桶一致。

## 6. 提交纪律更新

1. 提交线回到 **v84 链**（官方已验证的 attention 路径）；Linear 优化移植时
   保持 attention 调用闭包语义等价（WA 审计纪律第 2/3 条继续有效）。
2. PAWV/GQRB attention 机制在拿到官方通过数据点前一律视为
   official-time-infeasible，不再进入提交冻结候选。（2026-09-01：v86 证明
   C86 型静态 Attention 改动可通过，该限制只针对 PAWV/GQRB 动态分组机制。）
3. 预算红线的构成修正：`≤150s` 总红线只对 linear/calibration-heavy
   候选保守有效；attention-heavy 候选（`api_seconds` 中 attention 占比高者）
   须把 attention 分量单独压到官方通过路径的对应水平，或直接放弃该路径。
