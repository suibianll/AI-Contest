# 多模型真实模型评估器

## 目标

`evaluator/real_model_suite.py` 是开发阶段的评估器，用于回答两个问题：

1. 候选算法在不同模型结构和真实语言模型激活上是否仍然有效；
2. 本地指标能否复现已有官方锚点的相对排序。

它不是官方分数的替代品，也不把官方分数回灌到候选算法中。

## 合规边界

候选 `solution.py` 只收到 NVFP4 编码的权重、校准激活和正式 API 参数。评估器在候选返回量化参数之后，才计算 evaluator-side reference output 误差，用于打分；输出、输出残差和官方分数不会进入候选的 calibration state。

因此候选实现仍必须遵守赛事第一原则：不能通过任何形式构造 `A @ W`，再用它拟合或选择 `Q(A)`。本评估器中的输出乘法是评测参考，不是候选校准路径。

## 固定数据

- 数据集：`Salesforce/wikitext` 的 `wikitext-2-raw-v1`；revision 固定为 `b08601e04326c79dfdd32d625aee71d232d685c3`。
- calibration 窗口来自 train，test 窗口来自 validation。
- 每个窗口只来自一个文档，长度固定为 128 token；不循环重复、不跨文档拼接、不重叠。
- 代码在模型前向前检查 calibration/test 的 source-document 和 token-range 不重叠，并在 JSON 记录 parquet SHA256。

数据文件放在 `data/wikitext-2-raw-v1/`，不入库。中国大陆网络环境下可从 `hf-mirror.com` 下载固定 revision；如果大文件速度不稳定，模型文件可使用 ModelScope 的同名官方模型。

## 模型与适配器

默认矩阵包含：

| 模型 | 结构 | 注意事项 |
|---|---|---|
| GPT-2 small | GPT-2 / MHA | Conv1D fused QKV |
| GPT-2 medium | GPT-2 / MHA | 全 24 层 |
| OPT-125M | OPT / MHA | q/k/v 独立投影 |
| Pythia-160M | GPT-NeoX / MHA | fused QKV + RoPE |
| Qwen2.5-0.5B | Qwen2 / GQA | 14 Q heads、2 KV heads、RoPE、SwiGLU |

每个适配器都从真实 `AutoModelForCausalLM` 前向 hook 中获取 Linear 输入和 Q/K/V；NeoX/Qwen 的 Q/K 还会应用该模型实际 forward 使用的 rotary position embedding。不能用 GPT-2 的 module name 或无 RoPE 的 synthetic attention 代替这些路径。

## 运行

先安装评估器依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r evaluator\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

完整矩阵（默认所有本地模型和 C21/C38/C39/C40 锚点）：

```powershell
.\.venv\Scripts\python.exe -u evaluator\real_model_suite.py `
  --device cuda --algorithm-device cuda `
  --seq 128 --calib 2 --test 4 `
  --output artifacts/real_model_suite/latest.json `
  --report docs/real-model-evaluator-calibration-2026-08-28.md
```

`--algorithm-device` 默认跟随 `--device`。前向捕获先落 CPU，候选阶段再按该参数回搬；这样既不长期占用模型显存，又不会把候选算法错误地切到 CPU。每完成一个候选，评估器都会写 `*.partial.json`，中断后至少保留已完成结果。

调试适配器时可以使用 `--layers 1 --calib 1 --test 1`，但这个配置只能做接口冒烟，不能用于候选排名或官方拟合。

## 指标口径

- `linear.global_gain`：所有 Linear 输出元素按 evaluator reference MSE 加权的相对改善。
- `linear.macro_gain`：每个 Linear layer/role 等权的相对改善。
- `linear_component_macro_gain`：先聚合 q/k/v/o/fc/proj，再对五类等权；Qwen 的 gate/up 共同归入 fc，避免 FFN 双计数。
- `attention_causal.global_gain`：真实 Q/K/V（含 RoPE/GQA 适配）进入 causal attention reference 后的相对改善。
- `algorithm_stage_seconds`：候选正式 API 阶段的本地计时，必须小于 300 秒才通过本地预筛；官方端到端计时仍以赛事结果为准。

评估器同时输出 Pearson、Spearman、pairwise rank agreement、OLS R² 和 leave-one-out MAE。四个官方锚点太少，OLS 只用于诊断，不能当作分数兑换公式。

## 当前矩阵结论

完整运行结果见 [real-model-evaluator-calibration-2026-08-28.md](real-model-evaluator-calibration-2026-08-28.md) 和 `artifacts/real_model_suite/20260828_full.json`：

- 5 个模型、4 个候选全部加载成功；20 个组合的候选阶段均小于 300 秒。
- C39 的 linear-global 在 5 个模型上都高于 C40，复现了官方 `14613 > 14432` 的关键方向。
- C38 在 GPT-2 small/medium、OPT、Qwen 上低于 C21，复现了官方 `14092 < 14437` 的失败方向；Pythia 的模型内排序不同，说明单模型 proxy 不够稳。
- 多模型 `linear-macro` 与四个官方分数的 Pearson 为 `0.9131`，pairwise rank agreement 为 `5/6`；但 `linear-global` Pearson 只有 `0.4534`，且 C21/C39 仍未正确排序。
- 因而当前评估器适合做“淘汰明显跨模型退化候选”和机制诊断，不适合预测官方绝对分数或承诺某个本地分数对应 26000 分。

下一步应增加真实语料的跨文档 fold（仍禁止重复和泄漏），并在不使用官方反馈调参的前提下，观察 C21/C39 排序是否在多个语料 fold 上稳定；同时把 Linear 与 Attention 分量、原始 MSE 权重和最差层分开报告。
