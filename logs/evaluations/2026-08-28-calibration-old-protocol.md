# 历史多模型评估报告（旧评分协议，禁止用于候选排序）

> 本报告由 2026-08-28 的旧协议生成，使用候选私有反量化、causal Attention、global/component 聚合和绝对分拟合诊断，不符合当前官方逐 case 求和流程。仅保留为历史记录；当前排序请使用 `real_model_suite.py` 生成的 `official_flow_total`。

运行时间：2026-08-28 16:06:32（配置 mode=amax6，seq=128，calib=2，test=4）

本报告只用于检查本地评估器是否能复现已有官方候选的相对方向。官方分数没有进入候选校准状态，也没有传给 `solution.py`。评估器内部的输出矩阵乘法只在候选返回量化结果之后，用作固定参考误差。

## 数据与模型完整性

- 数据集：`Salesforce/wikitext` / `wikitext-2-raw-v1` / revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- calibration 来自 train，test 来自 validation；每个窗口来自一个文档，禁止环形重复、窗口重叠和跨 split 文档复用。
- 模型状态：

| 模型 | 状态 | 层数 | hidden | heads / kv-heads | 说明 |
|---|---|---:|---:|---:|---|
| gpt2-small | loaded | 12 | 768 | 12 / 12 | gpt2 |
| gpt2-medium | loaded | 24 | 1024 | 16 / 16 | gpt2 |
| opt-125m | loaded | 12 | 768 | 12 / 12 | opt |
| pythia-160m | loaded | 12 | 768 | 12 / 12 | gpt_neox |
| qwen2.5-0.5b | loaded | 24 | 896 | 14 / 2 | qwen2 |

## 候选在各模型上的结果

`linear-global` 按 evaluator reference MSE 的元素数加权；`component-macro` 先按 q/k/v/o/fc/proj 聚合再平均，避免 Qwen 的 gate/up 两个投影重复放大 FFN；`attention-causal` 使用真实模型的 Q/K/V（含模型自身 RoPE/GQA 适配）。

| 模型 | 候选 | linear-global | component-macro | attention-causal | algorithm-stage(s) | <300s |
|---|---|---:|---:|---:|---:|---|
| gpt2-small | c21 | 0.463855 | 0.486730 | 0.444413 | 27.735 | True |
| gpt2-small | c38 | 0.451126 | 0.473983 | 0.444413 | 36.736 | True |
| gpt2-small | c39 | 0.452724 | 0.479154 | 0.444413 | 30.486 | True |
| gpt2-small | c40 | 0.435772 | 0.466644 | 0.444413 | 49.095 | True |
| gpt2-medium | c21 | 0.403608 | 0.433111 | 0.423611 | 62.894 | True |
| gpt2-medium | c38 | 0.366423 | 0.389038 | 0.423611 | 72.499 | True |
| gpt2-medium | c39 | 0.395807 | 0.428025 | 0.423611 | 66.839 | True |
| gpt2-medium | c40 | 0.371660 | 0.412289 | 0.423611 | 118.601 | True |
| opt-125m | c21 | 0.525003 | 0.409567 | 0.417941 | 26.761 | True |
| opt-125m | c38 | 0.509214 | 0.368881 | 0.417941 | 32.463 | True |
| opt-125m | c39 | 0.524647 | 0.402527 | 0.417941 | 29.644 | True |
| opt-125m | c40 | 0.522136 | 0.389619 | 0.417941 | 47.899 | True |
| pythia-160m | c21 | 0.840717 | 0.606678 | 0.875291 | 27.907 | True |
| pythia-160m | c38 | 0.881722 | 0.582378 | 0.875291 | 32.960 | True |
| pythia-160m | c39 | 0.840696 | 0.603059 | 0.875291 | 29.989 | True |
| pythia-160m | c40 | 0.840591 | 0.595679 | 0.875291 | 48.760 | True |
| qwen2.5-0.5b | c21 | 0.417263 | 0.309965 | 0.661443 | 65.104 | True |
| qwen2.5-0.5b | c38 | 0.354523 | 0.246450 | 0.661443 | 81.931 | True |
| qwen2.5-0.5b | c39 | 0.400634 | 0.306551 | 0.661443 | 81.602 | True |
| qwen2.5-0.5b | c40 | 0.373892 | 0.301059 | 0.661443 | 141.975 | True |

## 与官方锚点的拟合诊断

官方锚点：C21=14437、C38=14092、C39=14613、C40=14432。下表先对已加载模型取均值，再与四个官方分数计算相关性；样本只有四个，不能据此拟合可靠的绝对分数换算公式。

| 本地特征 | Pearson | Spearman | pairwise rank agreement | OLS R² | leave-one-out MAE |
|---|---:|---:|---:|---:|---:|
| linear_global_gain | 0.4534 | 0.6000 | 0.6667 | 0.2056 | 309.95 |
| linear_macro_gain | 0.9131 | 0.8000 | 0.8333 | 0.8338 | 160.90 |
| component_macro_gain | 0.8754 | 0.8000 | 0.8333 | 0.7663 | 207.97 |
| attention_causal_global_gain | nan | nan | nan | 0.0000 | 201.00 |
| attention_causal_macro_gain | nan | nan | nan | 0.0000 | 201.00 |

### C39 / C40 排序

- `gpt2-small`：C39 linear-global=0.452724，C40=0.435772，C39>C40：`True`。
- `gpt2-medium`：C39 linear-global=0.395807，C40=0.371660，C39>C40：`True`。
- `opt-125m`：C39 linear-global=0.524647，C40=0.522136，C39>C40：`True`。
- `pythia-160m`：C39 linear-global=0.840696，C40=0.840591，C39>C40：`True`。
- `qwen2.5-0.5b`：C39 linear-global=0.400634，C40=0.373892，C39>C40：`True`。
- `aggregate`：C39 linear-global=0.522902，C40=0.508810，C39>C40：`True`。

## 解释与使用边界

1. 只有当多个模型、多个特征同时保持方向，并且至少复现 C39 高于 C40 的已知官方排序时，才把本地分数当作候选筛选信号。
2. 如果某个特征只在 GPT-2-small 上有效，或 C38/C40 的排序反转，应优先检查数据分割、架构适配和聚合口径，不应继续调候选阈值。
3. `synthetic_attention_eval.py` 不由本套件调用；它只能做接口/性质测试，不能用于候选排名。
4. algorithm-stage 是候选 API 的开发计时，官方端到端计时仍以赛事评测为准；报告中的 `<300s` 只是本地硬约束预筛。
