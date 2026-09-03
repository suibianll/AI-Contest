# v165 候选：标准 Linear + v161 Attention（精化信号官方测量实验）

> 状态：**CANDIDATE — 本地验证通过（两侧逐位一致），等待一次官方提交**
>
> Attention 侧父版本：v161 归档，SHA `27EEE4710B0170384A17E2F3E9AB87B3437E7B224883150D70BEBF8A5FB11848`
> （官方 timeout，无分数；本地 attention default mean `0.794856`）
>
> Linear 侧构造来源：v164 归档的标准 Linear 追加段，SHA
> `896B4ACA9F9F0C55D91C439E628B59D0B04D3BD77E23AA6F17144B0D665793D7`
>
> 对照锚点：v164（标准 Linear + v160 Attention）官方 **13945 / 204s**
>
> 候选 SHA256：`033E85D5DAF1A820BACDB14F9E35183C485E8DD489D118899A1AE3CB491D8C1D`
>
> 官方结果：`unregistered / NA`

## 1. 唯一实验目的

测量 v161 因 timeout 从未获得的**官方 Attention 分数**，检验 Gram64 per-call 精化的
本地增益（`0.742354 → 0.794856`，paired +0.0525、106+/14−、GPT-2 同号）是否迁移官方。
v165 与 v164 的 Linear 侧逐位一致，唯一差异是 Attention 侧多了精化，因此
`S(v165) − 13945` 是精化效果的**单变量官方测量**。

本候选为纯测量实验：即使全部成功（比例外推 ~+900），S ≈ 14800，仍远低于 v160 的
17532，不能也不得作为提交解使用；per-call 动态部署族无论结果如何保持关闭。

## 2. 构造方式

v161 归档原文件（10465 行，含就地修改的 Gram64 精化 Attention，零改动）+ 末尾追加
v164 的标准 encode 辅助（`_ref_solve_standard_hierarchy` / `_ref_encode_standard_hif4`）
与两个 Linear API 的模块级重定义（Python 后定义覆盖前定义）：

- `hif4_calibration_and_quantize_weight` → 忽略校准样本，标准编码；
- `hif4_dynamic_quantize_activation` → 标准 codec。

v161 的 Attention 代码路径完全未动。

## 3. 本地验证

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API（脱离仓库） | OK |
| linear-only default 168 | linear_mean **0.000000000**（168 case gain 全 0）；与 v164 linear JSON 配对 `0/0/168` 全不变 |
| attention-only default 120 | attention_mean **0.7948561184142177**；与 v161 JSON（`s1-gram-refine-attn-default.json`）配对 **max Δgain = 0.0、max Δmse = 0.0**，gain sum `95.38273420970611` 相等，120/120 case 身份匹配 |
| API total | attention-only 86.523s（v161 为 85.995s）/ linear-only 1.839s |

证据：`artifacts/official_eval/sidecal-v165-attn-default.json`、
`sidecal-v165-linear-default.json`（`logs/official_eval/` 对应 report）。

## 4. 时间预算（timeout 风险已知并接受）

```text
T(v165) ≈ T(v164) + X = 204 + X    （X = 精化的官方时间增量，本地 ~27s）

已知：v161 = 232 + X > 300  →  X > 68s（官方/本地成本比 > 2.5×）
通过：204 + X < 300         →  X < 96s（成本比 < 3.6×）
```

通过条件为成本比落在 `(2.5×, 3.6×)` 的 28s 窗口内；v161 的 >300s 无上界，timeout
可能性真实存在（估计三到五成）。timeout 则不重试、不缩 sweeps（预注册）。

## 5. 官方提交与解释（预注册）

一次官方提交。判读基线为 v164 = 13945：

| 观测 | 判读 | 后续 |
| --- | --- | --- |
| S − 13945 ≥ +300 | 精化信号官方实质迁移（比例外推预期 ~+900） | 开设“Gram64 校准期静态化”新工作包（回收该信号且不依赖 per-call 动态路径） |
| 0 < S − 13945 < +300 | 弱迁移 | 记录，静态化优先级降低 |
| S − 13945 ≤ 0 | 无迁移 / 本地-官方反转 | 永久关闭 v161 信号问题 |
| timeout（无分数） | 精化成本比 ≥ 3.6× | 无重试；继续 21765 计划工作包 C |

次级读数：Δ_A(v161) = S − 1001（对照 v160 的 12944）。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v165_standard-linear_v161-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-gram-refine-attn-default.json --output artifacts\official_eval\sidecal-v165-attn-default.json --report logs\official_eval\sidecal-v165-attn-default.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v165_standard-linear_v161-attn_scoreNA_timeNA\solution.py --linear-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\sidecal-v164-linear-default.json --output artifacts\official_eval\sidecal-v165-linear-default.json --report logs\official_eval\sidecal-v165-linear-default.md
```
