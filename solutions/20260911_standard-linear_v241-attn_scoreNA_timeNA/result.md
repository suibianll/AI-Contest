# 标准 Linear + v241 Attention（VK 核 per-Q-head）——官方超时

> 状态：**官方 `TIMEOUT (>300s)`，REJECTED，无分数**（用户 2026-09-11 回传）。
>
> Attention 侧父版本：v241 完整候选归档，SHA
> `8D364B3D0FF11858AE2BD2AD7F8BE6AA3210E922037206E77E96816C1DB186CD`
> （完整包亦 `TIMEOUT`，无分数）。
>
> Linear 侧构造来源：v162 标准 Linear 追加段，SHA
> `3619bfeb0e017555bd8fe31410888f80950a0128e3a2b94f643bd3f20b30bfc8`；
> 追加尾块 SHA `cb265612adc030d8d2f301c184d8290cf1b11dc2904da8f94d68084777be3a58`。
>
> 对照锚点：标准 Linear + **v195** Attention 官方 **14426 / 243s**；
> 基线 标准 Linear + **R3** 官方 **14405 / 238s**。
>
> 候选 SHA256：`C86C972A75B26BC5F7E0A6EAFC2CE9F65183BE434BDB831F434D2D6749A45B6A`（537267 B）
>
> 官方结果：`score NA / time >300s / TIMEOUT`

## 1. 唯一实验目的

VK 核（相对位置核加权选 V 的 HiF4 码）**至今没有任何官方分数**：完整包的两次提交
（v241 per-Q-head、v242 per-KV-group）都超时。本探针换用仓库既有的**侧隔离通道**
给同一机制定价——标准 Linear codec + 候选 Attention，包更小，只回答
"该 Attention 机制本身值多少官方分"。

**官方没有返回分数，因此该问题仍未回答。** VK 机制的官方分为空。

## 2. 构造方式

v241 候选归档原文件全文（末尾 `rstrip`）+ 固定的标准 Linear 尾块：

- `hif4_calibration_and_quantize_weight` → 标准编码（忽略校准样本）
- `hif4_dynamic_quantize_activation` → 标准 codec

尾块取自 v162 R1 标记之后的全部内容，**只 shadow 两个 Linear API**；v241 的 Attention
代码路径完全未动。构造式与 v195 / v234 探针逐位同型（前缀逐字节相同，固定余数 6475 B）。

## 3. 本地验证（提交前）

| 项目 | 结果 |
|---|---|
| 脱离仓库单文件导入 + 六 API | OK（无仓库依赖 import；六 API 位置参数个数 3/3/4/5/5/5） |
| Attention 侧 shard0（基线 v195 探针） | delta_mean `+0.003940`，`11/1/0`，L1 `0.004163`，`reasonableness_issues: 0` |
| Linear 侧 API 调用 | `0` calls（单侧运行不调用另一侧 API） |
| SHA 唯一性 | 在全部 300 个归档 `solution.py` + 根中唯一 |

本地读数不外推官方、不设提交门。

## 4. 官方结果判读

命中 `TIMEOUT` 行：**不重试该 SHA，不计算任何相对 14426 的官方增量，不写秒数预测。**

对照基线 v195 的 `243s`，本探针 `>300s` 意味着 VK（per-Q-head）在侧隔离口径上相对
v195 的 Attention **至少多花约 57s**（相对 R3 基线 `238s` 至少约 62s）。
这是**硬限给出的不等式下界，不是一次测量**；**不由此推出任何降时预算**
（AGENTS §2：官方秒数带未量化的波动，"余量/需要省 N 秒"不构成研发预算或优先级依据）。

## 5. 与本地成本读数的一致性（仅陈述）

本地 shard0 的 V API 实测：本探针 `7.094s / 12 calls`（`0.591 s/次`），
同批的 v242 探针 `1.100s / 12 calls`（`0.092 s/次`），比值约 **6.4×**。
超时发生在成本高 6.4× 的那个粒度上，方向一致。
本地是 GPU，官方判题是鲲鹏 920B（CPU），**不据此换算**。

## 6. 未做与已关闭

- 未做官方分数归因——没有分数。
- 未做 Linear 侧（本探针的 Linear 就是标准 codec，`0` 调用）。
- **本实现关闭**：不重提交同 SHA，不缩核半径/窗口/候选数重试。
- **VK 机制本身未关闭**：v242（per-KV-group）是另一个实现，其侧隔离探针尚未回传，
  不受本结果约束（计划 §7）。

## 7. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\eval.py --solution solutions\20260911_standard-linear_v241-attn_scoreNA_timeNA\solution.py `
  --baseline-solution solutions\20260908_standard-linear_v195-attn_scoreNA_timeNA\solution.py `
  --attention-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt `
  --calibration-cache-mode auto --algorithm-device cuda `
  --output-dir artifacts\proxy_v3\side-vk241-shard0-20260911
```

完整过程见[构建记录](../../logs/execution/2026-09-11-vk-side-isolation-probes.md)与
[超时登记](../../logs/execution/2026-09-11-side-vk241-attn-official-timeout.md)。
