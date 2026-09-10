# v241 / v242 Attention VK 核完整包官方超时（2026-09-11）

用户原话：“v241、v242全都超时了”。

这两张卡是**同一机制的两种粒度**，各自独立提交，现在各自 TIMEOUT。

- 类型：完整六 API —— **v237 完整根 + VK 核**（用相对位置核加权选择 V 的 HiF4 码，代替均匀核）。
- 父根：**v237 Linear L-TF2**，SHA `ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554`，
  516697 B，官方 **18518 / 289 s**，余量 **11 s**。
- v241 归档 SHA256：`8d364b3d0ff11858ae2bd2ad7f8be6aa3210e922037206e77e96816c1db186cd`（530793 B）。
- v242 归档 SHA256：`b11ba4f2f56259dd1fcd2d9f6229afb3c35035d067890c0e53e6dde9fb1e9227`（530992 B）。
- 官方结果：**两张均 TIMEOUT（>300s），REJECTED**，精确秒数/分数未知（记 null，不写预测）。

## 独立复核（本记录自己跑的机器检查，不采信 result.md 自述）

- **纯追加**：两候选的前 516697 B 与 v237 根 `cmp` **逐字节相同**；v241 余 +14096 B，v242 余 +14295 B。
- **两候选只差核粒度**：`diff(v241, v242)` 共 **30 行**，全部落在核拟合段——
  `totals` 由 `q_heads`(16) 改为 `kv_heads`(4)、索引改 `h // group`、内层 `for j in range(heads_per_group)`
  循环消失。与 result.md 声称的"唯一差别是核粒度"一致。

## 两张卡的差别只在成本，不在机制

| | v241 | v242 |
|---|---|---|
| 核的粒度 | 每 Q head 一个（16） | 每 KV 组一个（4） |
| 优化目标 | `Σ_g Σ_{h∈g} Σ_t (Σ_k w_h[t−k]·δ)²` | 同式，组内 `w_h` 相同 |
| 每轮带状运算 | 16 次 | **4 次** |
| V API 单次成本 Δ | +0.0666 s | **+0.0117 s** |
| 本地六 shard 均值 | +0.004547（六片全正） | +0.004602 |
| 官方 | **TIMEOUT** | **TIMEOUT** |

v242 的共享不是精度折让而是**重参数化**：组内 4 个 head 共享 `w_g` 时目标化为
`4·Σ_t (Σ_k w_g[t−k]δ)²`，**最小化点与逐 head 版相同**，只是每轮带状运算 4→1。
这个论证是代数上成立的，本地读数也印证了（两版面板均值差 6e-5）。

**所以本次证伪的不是"共享会掉精度"——那个没掉。证伪的是"成本降下来就能落地"。**

## 关键：v242 的"落进余量"有三个支点，只有形式那一条是可靠的

v242 被定位为"**当前可提交形态**"的依据是这条算术：

```
V API Δ +0.0117 s/次  ×  250 个 attention 用例  ≈  2.9 s   <   根余量 11 s
```

三个支点逐条核：

**支点一 · 一 case 一次 V 调用 —— 结构上成立。**
评测器把 `hif4_dynamic_quantize_v` 放在 `for case in (pack.attention_cases …)` 循环体内，
每次 case 恰好加一次计数（`evaluator/official_eval.py:2689`、`:2705`、`:2709`）。
所以"调用次数 = attention case 数"这个**形式**是对的。

**支点二 · case 数 = 250 —— 有出处，但出处互相矛盾，且都不是仓库内的实测读数。**
仓库里同时躺着两组数字：

- **250 Linear + 200 Attention** —— `docs/research/2026-08-30-hif4-36000-potential-and-algorithms.md:25`、
  `logs/execution/2026-08-29-optimization-log.md:8`、`docs/evaluation-attribution-2026-09-01.md:84`
- **50 Linear + 250 Attention** —— 用户报告，`docs/superpowers/archive/plans/21071-evidence-driven-research.md:9`、
  `logs/execution/2026-09-07-user-21071-mechanism-evidence.md:5`（标注 `EXTERNAL_USER_CONFIRMED / SOURCE_UNBOUND`）

"250"取自后者。**它在仓库内没有实测出处，并与更早记录的 200 Attention 直接冲突。**
但按 200 算，v242 的预期成本是 2.3 s，结论不变（仍"落进余量"）——
所以**这一条不是本次超时的解释**，它只是那条算术里一个无法核验的支点。

**支点三 · +0.0117 s/次 —— 在 GPU 上量的，官方判题是 CPU。这才是最可疑的一条。**
计时脚本设 `device = cuda`（`workbench/full_solution/attention-v-kernel/verify.py:43`，
输出头 `device=cuda`，见 `verify.out:1` / `verify2.out:1` / `verify3.out:1`），
而官方判题机是**鲲鹏 920B（CPU）**——v241/v242 的 result.md 自己就写着这一点，却仍拿 GPU 秒数去算 CPU 的账。
更关键的是**计时口径**：每层 3 次取 **min**、再对 6 层取均值（`verify.py:113-123`、`:156-160`），
是单次调用的粗测，**不是 A-CT1 那种带同字节 sham null 的配对口径**。
本仓库自己的纪律早就写着：**"按假设调用次数累加的 2.56s、3.2s 均不是官方 API 成本"**
（`docs/optimization-round-summary-2026-09-10.md:49`）。

### 兄弟卡在同一天给出了相反的做法

**同一天（2026-09-10）归档**的 v240 把同一件事登记为缺失事实，并据此拒绝一切投影：

> The number of dynamic calls in the official evaluation is not recorded anywhere in this repository.
> With 50 the card costs about 5.8 s officially; with a full-model pass per sample it saves far more.
> **The sign is therefore undetermined locally, and it is registered as a missing fact rather than assumed.**
> —— `solutions/20260910_v240_linear-lmc1-compiled-metric_scoreNA_timeNA/official-result.json`，字段 `missing_fact`

并写死 `local_projection_seconds: null`。

### 这次能确定什么、不能确定什么

**能确定**：支撑"v242 是可提交形态"的那条算术，三个支点里只有第一条（调用形式）可靠；
其余两条——case 数与单次成本——都不是仓库内可核验的实测，且单次成本是从 GPU 外推到 CPU 判题机的。
**官方结果没有证实这条算术。** 这是本次回传可复用的部分。

**不能确定**：真实调用次数、真实单次成本、超限幅度。官方只回了超时，不给数字；
**不反推调用次数、不反推官方计时方差、不推导本地→官方换算系数。**

**也不等于**："成本模型错了"或"VK 机制无效"。降本本身是做成了的——
共享核是代数上的重参数化（最小化点相同），本地面板也印证了（两版均值差 6e-5）。

## 家族账：本仓库第四次"降本后仍超时"

| 候选 | 机制 | 父根余量 | 官方 |
|---|---|---|---|
| v236 | A-GR1 原实现 | 9 s | TIMEOUT |
| v238 | A-CT1（A-GR1 降本，输出逐位等价） | 9 s | TIMEOUT |
| **v241** | VK 核，per-Q-head（16） | 11 s | **TIMEOUT** |
| **v242** | VK 核，per-KV-group（4），**成本 ÷5.7** | 11 s | **TIMEOUT** |

已发生两次"把重复/冗余计算去掉后再交一次"（v238 的 gate 去重、v242 的共享核降本），
**两次都没能把包拉进 300 s**。本记录只登记现象，**不由此推出"降本方向整族耗尽"**——
两次降本的量级都远小于余量缺口，它们不足以否证一个量级更大的降本。

## 顺带核出的两条（供后续卡参考，不改本记录结论）

**一 · v241 的 result.md 有一处自相矛盾，两候选的归档文档都未登记核半径。**
v241 `result.md:25` 写"**带宽 15 的带状卷积**"，但两张候选**实际发出的常量都是 `_VK_RADIUS = 7`**
（`solutions/20260911_v241_.../solution.py:12609` 与 `.../v242_.../solution.py:12609`），
而同文件 `result.md:63-64` 又写着"半径已从 15 降到 7"。即**同一个文件里两处互相打架**，
读者据前一处在评估一个并不存在的实现。核半径 15→7 的降本是在诊断侧量的
（`vk3-r7.out`：−0.9859%，n=12，range [−2.1296%, +0.6889%]；对照半径 15 的 `vk3-rule.out`：−1.1744%，11/12 窗改善）。
**本记录不修改旧产物**，只在此登记该缺陷。

**二 · 机制的余量上限仓库已有读数。** 同一诊断里：

| 加权 | 留出窗 V 侧收益 |
|---|---|
| 均匀核（父） | — |
| **核（可部署）** | **−1.4951%** |
| **真实 A（每 head T×T，不可部署）** | **−8.8159%** |

即**可部署的核只拿到真实 A 加权所给空间的约 1/6**。这是机制侧（不是成本侧）的读数，
按计划 §5 的边界，它**不证明**该方向已耗尽；登记在此只为后续卡不必重测。

## 不做的事

- 不缩窗 / 减候选 / 降核半径重试；不重复提交同一 SHA。
- 不写本地秒数预测，不从一次超时反推官方计时方差或本地→官方换算系数。
- 不改写 v241/v242 的本地面板读数（六片全正、幂等复核成立），也不把它们外推官方。
- 不据此关闭 VK 机制族：按活动计划 §7，**停止条件只作用于具体实现**。

## 根

未切换：当前完整根仍是 **v237，18518 / 289 s**，回退根仍是 v233（18428/288s）。
v241/v242 均为 v237 的后代（含 L-TF2），与 v233/v235/v238/v239/v240 不可相加。
