# 标准 Linear + v242 Attention（VK 核 per-KV-group）——官方 `14098 / 250s`

> 状态：**官方 `14098 / 250s`**（用户 2026-09-11 回传）。
>
> Attention 侧父版本：v242 完整候选归档，SHA
> `b11ba4f2f56259dd1fcd2d9f6229afb3c35035d067890c0e53e6dde9fb1e9227`
> （完整包 `TIMEOUT`，无分数）。
>
> Linear 侧构造来源：v162 标准 Linear 追加段，SHA
> `3619bfeb0e017555bd8fe31410888f80950a0128e3a2b94f643bd3f20b30bfc8`；
> 追加尾块 SHA `cb265612adc030d8d2f301c184d8290cf1b11dc2904da8f94d68084777be3a58`。
>
> 对照锚点：标准 Linear + **R3** Attention 官方 **14405 / 238s**；
> 标准 Linear + **v195** Attention 官方 **14426 / 243s**。
>
> 候选 SHA256：`61CC48878275E881C72DB37D9439AD4C197487BB4504F9EE9B71CFB8334782C4`（537466 B）
>
> 官方结果：`score 14098 / time 250s`

## 1. 唯一实验目的

VK 核（相对位置核加权选 V 的 HiF4 码）**此前没有任何官方分数**：完整包的两次提交
（v241 per-Q-head、v242 per-KV-group）都 TIMEOUT，姊妹探针
[v241（per-Q-head）](../20260911_standard-linear_v241-attn_scoreNA_timeNA/result.md)
的侧隔离提交也 TIMEOUT。本探针换用**更便宜的粒度**（per-KV-group，4 核）走侧隔离通道，
目的是给同一机制定价——**这是 VK 机制唯一还能拿到官方分的通道。**

**拿到了。答案是负向。**

## 2. 构造方式

v242 候选归档原文件全文（末尾 `rstrip`）+ 固定的标准 Linear 尾块：

- `hif4_calibration_and_quantize_weight` → 标准编码（忽略校准样本）
- `hif4_dynamic_quantize_activation` → 标准 codec

尾块取自 v162 R1 标记之后的全部内容，**只 shadow 两个 Linear API**；v242 的 Attention
代码路径完全未动。构造式与 v195 / v234 探针逐位同型（前缀逐字节相同，固定余数 6475 B）。

## 3. 官方结果

| 对照 | 分数 | 时间 |
|---|---|---|
| **本探针（标准 Linear + VK per-KV-group）** | **`14098`** | **`250s`** |
| 标准 Linear + R3（基线） | `14405` | `238s` |
| 标准 Linear + v195 | `14426` | `243s` |
| **Δ vs R3 基线** | **`−307`** | +12s |
| **Δ vs v195** | **`−328`** | +7s |

单变量口径：与基线共用同一个标准 Linear codec（逐字节同一实现），唯一差异是 Attention 侧。
**`−307` 是该 Attention 机制的官方效应——不是"值 0 分"，是"倒扣 307 分"（−2.13%）。**

这是侧隔离通道上**幅度最大的一次读数，也是唯一为负的一次**（此前全部落在 `0 … +50`）。

## 4. 本地读数与官方的方向相反

| 口径 | 读数 |
|---|---|
| 本地 4B 面板 shard0 | `+0.005159`（`12/0/0`，L1 `0.005159`） |
| 本地 4B 面板（姊妹 v241，完整六 shard） | `+0.004547`，**六片全正** |
| **官方侧隔离** | **`−307`** |

**方向相反，且幅度差两个量级。** VK 卡正是以"本地六片全正"为依据推进到官方提交的。

**不解释成因**——本地与官方在数据/参考/聚合口径上的差异有多种可能，没有测量支持任何解释。

## 5. 关闭

- **本机制关闭**：V 侧相对位置核加权选 V 码，含两个已测粒度（per-Q-head、per-KV-group）。
  共享版是代数重参数化（同一最小化点），不构成两个不同机制。
- **不重提交同 SHA，不缩核半径/窗口/候选数/粒度重试。**
- **该机制的降本工作线到此为止**：`250s` 证明这条线在时间上确实有效（v241 侧隔离 `>300s`
  → 本次 `250s`，本地 V API 成本 `0.591 → 0.092 s/次`），**但进限也救不了 −307**。
- **不推广**为"V 侧码分配整族无效"或"位置信息不能用于选 V 码"：被证伪的是这一**具体形式**，
  后续尝试需要**新机制**，不是调参。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\eval.py --solution solutions\20260911_standard-linear_v242-attn_scoreNA_timeNA\solution.py `
  --baseline-solution solutions\20260908_standard-linear_v195-attn_scoreNA_timeNA\solution.py `
  --attention-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt `
  --calibration-cache-mode auto --algorithm-device cuda `
  --output-dir artifacts\proxy_v3\side-vk242-shard0-20260911
```

完整过程见[构建记录](../../logs/execution/2026-09-11-vk-side-isolation-probes.md)与
[官方回传登记](../../logs/execution/2026-09-11-vk242-side-isolation-official.md)。
