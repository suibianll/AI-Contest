# v165：标准 Linear + v161 Attention（官方超时）

> 状态：**TIMEOUT — 官方 `>300s`，无分数（用户 2026-09-03 回传）**
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
> 官方结果：`score NA / time >300s / TIMEOUT`

## 1. 唯一实验目的

测量 v161 因 timeout 从未获得的**官方 Attention 分数**，检验 Gram64 per-call 精化的
本地增益（`0.742354 → 0.794856`，paired +0.0525、106+/14−、GPT-2 同号）是否迁移官方。
v165 与 v164 的 Linear 侧逐位一致，唯一差异是 Attention 侧多了精化；若官方返回分数，
`S(v165) − 13945` 本可成为精化效果的单变量官方测量。实际因 timeout 无法计算该差值。

本候选为纯测量实验。官方没有返回分数，因此不能判断本地正向是否迁移，也不计算相对 v164
的官方 Attention 增量或优化比例。

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

官方实际 timeout。相对 v164 的 `204s` 对照，隔离增加的精化使总时间跨过 `300s`，所以在
官方运行稳定的已知条件下，额外官方成本至少约 `>96s`，对应本地约 `27s` 增量的成本倍率
`>3.6×`。这只是下界；官方完整耗时没有返回。

## 5. 官方提交与解释（预注册，判读已由侧向隔离计划取代）

> 2026-09-03 更新：本节初版判读（`+300` 人为门槛、timeout 不重试、继续 21765 计划
> 工作包 C）已被
> [`2026-09-03-v162-official-side-isolation-optimization-plan.md`](../../docs/superpowers/plans/2026-09-03-v162-official-side-isolation-optimization-plan.md)
> §7.1 取代：21765 计划 A/B/C 已全部本地 REJECTED；新计划不设人为准确率门槛，timeout
> 允许一次保持 Gram 目标的低复杂度重构。以下为现行判读。

一次官方提交已经完成。判读基线为 v164 = 13945：

| 观测 | 判读 | 后续 |
| --- | --- | --- |
| S > 13945 | Gram 目标获得官方正证据 | 按侧向计划 §3.2 登记 `C_A/G_A/P_A/R_A`；是否成为 Attention 主线由绝对增量与官方时间决定 |
| S ≤ 13945 | Gram 动态精化官方负向 | 关闭该数学目标，更换机制；不调 sweeps/阈值/邻域 |
| timeout（无分数） | 精度结论未知 | 允许一次保持 Gram 目标不变的低复杂度重构（侧向计划 §7.2 低秩 Gram 残差码本） |

次级读数：Δ_A(v161) = S − 1001（对照 v160 的 12944）；不用本地 gain 比例预测官方分数。

实际命中 timeout 行：不重试该 SHA，不计算上述次级读数；当前活动计划继续一次低秩 Gram
残差码本复杂度重构。更广的低复杂度算法扩展另存于排队计划，须等当前活动计划完成后才激活。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v165_standard-linear_v161-attn_scoreNA_timeout\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-gram-refine-attn-default.json --output artifacts\official_eval\sidecal-v165-attn-default.json --report logs\official_eval\sidecal-v165-attn-default.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v165_standard-linear_v161-attn_scoreNA_timeout\solution.py --linear-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\sidecal-v164-linear-default.json --output artifacts\official_eval\sidecal-v165-linear-default.json --report logs\official_eval\sidecal-v165-linear-default.md
```
