# A2 深层 K 结构诊断（第 0 步，零 API）— 无一致病因，跳过 A2

日期：2026-09-03
状态：**DONE / SKIPPED（未找到跨层一致解析病因，不构造 A2 候选）**

## 1. 方法

零 API 诊断（`workbench/a2_k_diagnosis.py`）：同一 dense cache
（`qwen2.5-0.5b-proxy-v2.pt`）中 24 层 K 投影权重 (128,896) 与全部 5 折校准 K dense
(T,128) 的结构特征，对 `v160-a2-attn-default-candidate.json` 的 per-layer K-only
gain（120 case 按层聚合）做 Spearman 秩相关。

特征定义：

- `ch_spread`：K dense 通道 RMS 的 max/median（通道 outlier / 能量集中度）；
- `head_ratio`：两个 KV head 平均 RMS 之比（head 间 scale 失衡）；
- `within`：head 内通道 max/median 对 head 平均（head 内失衡）；
- `w_spread`：K 权重行 RMS 的 max/median。

## 2. 结果（关键行）

| layer | K-only | ch_spread | head_ratio | within | w_spread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | −66.52 | 4.22 | 1.174 | 3.958 | 2.05 |
| 23 | **−200.93** | 5.44 | 1.450 | 4.376 | 1.79 |
| 1 | −3.17 | 73.03 | **4.350** | 40.665 | 1.55 |
| 8 | −3.59 | **111.51** | 5.655 | 56.939 | 2.10 |
| 12 | −46.93 | 4.77 | 1.188 | 4.581 | 1.80 |

秩相关（K-only gain，越负越差）：`ch_spread +0.627`、`within +0.593`、
`head_ratio +0.178`、`w_spread −0.022`。深层 21–23 的 `ch_spread` 均值 `5.07` 低于
其余层 `16.16`。

## 3. 解读

- **所有通道结构特征与 K 误差的关联方向都与其作为"病因"的假设相反**：outlier/集中度
  越高的层 K-only gain 越好（+0.63），深层反而能量分布更均匀；
- `head_ratio` 只在 layer 23 突出（1.45），但 layer 1/8 的 head 失衡（4.35/5.66）大得
  多而 K-only 仅 `−3.2/−3.6`，不构成解释；
- 深层 K-only 负增益更可能来自参考能量分母效应：深层 attention 分布更 sharp，K 误差
  对 logits 的相对影响被放大——这是 `softmax(QK^T)V` 的算术性质，不是可用全层统一
  解析规则修正的权重/通道结构。

按活动计划 A2 第 0 步判据（"仅当找到跨层一致且跨模型可验证的解析病因时构造候选"），
**判定无一致病因，跳过 A2**。诊断脚本保留于 `workbench/a2_k_diagnosis.py`。

## 4. 计划状态推进

A1（组内扩展）REJECTED、A2 无病因跳过、A3 前置条件（A1/A2 至少一个通过 D1）不满足
不启动。活动计划《Attention 解析式宽域实验》三条候选全部关闭，按其 §5 解释表：
Attention 解析宽域族在当前父版本 v160 上饱和，计划归档，等待外部材料或新机制。
