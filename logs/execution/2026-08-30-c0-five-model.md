# C0 五模型确认 — v100 PAWV diag-only

日期：2026-08-30  
状态：`confirmed-local`；官方评测不可用，主门禁仍为固定 Qwen shaped panel。

## 目的与口径

C0 不引入新算法，只检查已通过 Qwen 主门禁的 v100 在五个真实模型上的泛化。
五个模型均使用同一套 WikiText-2 缓存（`seq=128`、`calib=2`、`test=4`、全层、
`amax6`、CPU），评分先按每个 native case 的

\[
s=(\mathrm{MSE}_{std}-\mathrm{MSE}_{player})/\mathrm{MSE}_{std}
\]

计算，再保留每个模型的 Linear/Attention 均值。Qwen 主排序使用

\[
P=250\,\bar s_L+200\,\bar s_A,
\]

其他四个模型只做软 guardrail；不能把不同层数的 native total 相加作为排名分。

## 结果

| 模型 | native total | panel total | Linear mean | Attention mean | API time | 420s | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | --- |
| gpt2-small | `182.256323` | `256.254385` | `0.529628` | `0.619236` | `196.975s` | ✓ | 通过 |
| gpt2-medium | `333.803554` | `239.397733` | `0.480028` | `0.596954` | `492.641s` | ✗ | 软 guardrail；时间超限 |
| opt-125m | `153.584527` | `218.474840` | `0.443643` | `0.537821` | `192.776s` | ✓ | 通过 |
| pythia-160m | `193.632429` | `310.098006` | `0.522846` | `0.896932` | `193.423s` | ✓ | 通过 |
| **qwen2.5-0.5b** | **`417.882506`** | **`293.797301`** | **`0.501558`** | **`0.842039`** | **`401.131s`** | **✓** | **主门禁通过** |

Qwen 的 C0 复测 panel 与 v100 单模型结果一致（`293.797301`），本次 API 为
`401.130873s`，仍低于官方 `420s` 限制。五模型 aggregate 诊断为 panel
`263.604453`（guardrail 均值 `256.056241`），native total 合计
`1281.159340`；二者只用于检测跨模型回退，不用于官方分数换算。

## 裁决与归档

- v100（B2 PAWV diag-only + B1 GQRB）在主模型和全部五模型上完整运行，Qwen
  主 panel、Linear mean 均保持最高；C0 确认通过，不改变根目录代码。
- GPT-2 medium 的 `492.641s` 只影响软 guardrail 时间，不否决 Qwen 主模型的
  `401.131s`；若官方对所有模型分别限时，应另行做模型级运行时优化。
- 源码 SHA256（规范 LF 内容）：
  `617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`。
- 完整原始证据：
  `artifacts/real_model_suite/c0-b2-pawv-five-model.json` 与
  `logs/evaluations/2026-08-30-c0-b2-pawv-five-model.md`；版本归档在
  `solutions/20260830_v101_c0-five-model-confirmed_score293.797301_time401s/`。
