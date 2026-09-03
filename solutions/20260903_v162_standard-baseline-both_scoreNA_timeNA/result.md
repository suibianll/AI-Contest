# v162 候选：全标准 HiF4 基线（两侧比重校准实验锚点）

> 状态：**MEASURED — 官方锚点** **`1001 / 146s`（2026-09-03 用户回传，实验角色完成）**
>
> 父版本：无（独立最小实现；实验父上下文为 v160，SHA
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`，官方 `17532 / 232s`）
>
> 候选 SHA256：`56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`
>
> 官方结果：**1001 / 146s**

## 1. 唯一算法内容

六个 API 全部执行标准 HiF4 codec：NVFP4 输入 → 官方 BF16 中间解码 → 标准
amax/7 E6M2 scale + 8 配置 MSE-optimal lv2/lv3 层级 + 3-bit mantissa + canonical
zero sign。无校准统计、无搜索、无 state 内容（空 dict）。与
`evaluator/reference_hif4.py` 的 `encode_standard_hif4(dequantize_nvfp4(...))`
链逐位一致。

## 2. 实验角色

官方两侧分数比重校准（[`已归档计划`](../../docs/superpowers/archive/plans/2026-09-03-official-side-weight-calibration-plan-superseded.md)）的
**标准基线锚点**：官方回传 `S(v162)` 定义 Δ\_L = `S(v163)−S(v162)`（Linear 优化贡献）
与 Δ\_A = `S(v164)−S(v162)`（Attention 优化贡献）的分母。

## 3. 本地验证

| 项目                              | 结果                                                             |
| ------------------------------- | -------------------------------------------------------------- |
| 隔离导入 + 六 API                    | OK（`workbench/side_weight_unit_check.py`）                      |
| 标准 codec vs reference（CPU+CUDA） | 逐位一致                                                           |
| 完整 default 168+120              | linear\_mean **0.0**、attention\_mean **0.0**（全部 case gain = 0） |
| API total                       | **2.627s**（v160 为 290.7s）                                      |

证据：`artifacts/official_eval/sidecal-v162-both-default.json`、
`logs/official_eval/sidecal-v162-both-default.md`。

## 4. 时间预算

本地 API 2.6s，为全部归档版本最低；官方时间风险可忽略。

## 5. 官方提交与解释（预注册）

一次官方提交。预期官方分数为"标准行为"的官方值：若官方 STD 也是标准 HiF4，则
`S(v162)` 应为官方基础分（可能为 0）；若 `S(v162) > 0`，说明官方存在非零基础分或
STD 定义不同，`S(v162)` 只作为锚点记录，不反推公式。判读表见活动计划 §3。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v162_standard-baseline-both_scoreNA_timeNA\solution.py --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --output artifacts\official_eval\sidecal-v162-both-default.json --report logs\official_eval\sidecal-v162-both-default.md
```
