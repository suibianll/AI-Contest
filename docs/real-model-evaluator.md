# 多模型真实模型评估器

> **当前协议 v4（2026-08-31）**：日常评测使用 `sampled-means-v1`，默认 Qwen、
> 固定 seed 的分层 layer/window 样本；主结果只有 `Linear mean` 和 `Attention mean`。
> 本文中旧 panel/native/official-flow 段落仅作历史兼容说明，不能与 v4 主表混用。

## 目标

`evaluator/real_model_suite.py` 是开发阶段的评估器，用于回答两个问题：

1. 候选算法在真实语言模型激活上两个组件的平均 gain；
2. 在固定抽样计划下对候选做快速、可复现的 A/B 比较。

它不是官方分数的替代品，也不把官方分数回灌到候选算法中。原始逐 case
`official_flow_total` 和 `panel_score` 仍保留在 JSON，专门用于兼容回溯；v4
报告主指标是 `mean_scores.linear_mean` 与 `mean_scores.attention_mean`。

## 合规边界

候选 `solution.py` 只收到 NVFP4 编码的权重、校准激活和正式 API 参数，并且只要求赛事规定的六个函数。NVFP4/HiF4 反量化、标准 HiF4 编码、state 校验和 HiF4 参数合法性校验全部由评估器独立实现，不调用候选的私有 `_dequantize_*` 或 `_dense_to_hif4`。评估器在候选返回量化参数之后，才计算 evaluator-side reference output 误差；输出、输出残差和官方分数不会进入候选的 calibration state。

官方边界不是禁止所有离线 `A @ W`：在 `hif4_calibration_and_quantize_weight`
中，候选可以用自己的校准 `A`、`W` 计算输出目标来优化离线权重量化器
`Q(W)`。禁止的是把该输出、量化输出或输出残差用于拟合、选择或反推在线
激活量化器 `Q(A)`，或把它写入 `activation_state`。本评估器的输出乘法是
评测参考；合规守卫另外审计候选是否把离线输出数据流送入激活状态。

## 固定数据

- 数据集：`Salesforce/wikitext` 的 `wikitext-2-raw-v1`；revision 固定为 `b08601e04326c79dfdd32d625aee71d232d685c3`。
- calibration 窗口来自 train，test 窗口来自 validation。
- 每个窗口只来自一个文档，长度固定为 128 token；不循环重复、不跨文档拼接、不重叠。
- 代码在模型前向前检查 calibration/test 的 source-document 和 token-range 不重叠，并在 JSON 记录 parquet SHA256。

数据文件放在 `data/wikitext-2-raw-v1/`，不入库。中国大陆网络环境下可从 `hf-mirror.com` 下载固定 revision；如果大文件速度不稳定，模型文件可使用 ModelScope 的同名官方模型。

## 官方评测集（2026-08-29 修订；2026-08-31 再次修订）

官方评测面板现在包含 **250 个 Linear case 与 200 个 Attention case**，官方
分数按全部 case 的百分制 `case_score` 求和；本地不能看到隐藏 case，也不能
把本地均值换算成官方绝对分数。官方时间限制已于 **2026-08-31 收紧为 300s（5
分钟）**，且官方不再限制任何 `A@W` 拟合用法，只限制端到端时间；只有官方平台
实测才能最终确认。**2026-08-31 晚官方再次更换评分权重：减少 Linear 样例的
权重**，官方总分据此大幅下降；旧权重与新权重总分不可互相换算，本地一律不
复制 case、不拟合官方绝对分。

本地 v4 默认采用 `--evaluation-profile sampled-means-v1`：Qwen2.5-0.5B、
固定 seed 分层抽取 8 层、保留全部 7 role、4 个 validation window 和全部
calibration window，主结果只有抽样 Linear/Attention 的算术均值。需要跨模型
时显式传入 `--models`，各模型独立报告均值，不按层数相加。

已确认的官方锚点：v031/C39-FW `21864 / 161.3s`、v034/C41b
`21864 / 159.4s`、v051/C47b `22451 / 234s`、v066/C66
`22557 / 217.2s`、v072/C74 `22662 / 226s`、v074/C75
`22750 / 239.387s`（以上均为**旧评分权重**口径）；**v84/C84 `16517 / 252.563s`
为新评分权重（减少 Linear 权重）下第一个确认的官方通过锚点，< 300s**；外部
[`youxilee/hif4`](https://github.com/youxilee/hif4) 报告 `24153 / 239s`（旧权重），
仅用于参考，不作为本地评测器的候选输入。任何官方分数都不进入候选校准或锚点
拟合。

## v4 快速运行与当前配对结果（2026-08-31）

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --evaluation-profile sampled-means-v1 `
  --sample-layers 8 --sample-test-windows 4 --sample-seed 20260831 `
  --device cpu --algorithm-device cpu --cache-mode read `
  --solution solution.py --candidate-name active `
  --output artifacts\real_model_suite\active-sampled.json `
  --report logs\execution\active-sampled.md
```

该 profile 固定 `224 Linear + 32 Attention` case，报告表只读取
`results[*].mean_scores.linear_mean`、`attention_mean`、`timing.local_api_total_seconds`
和 `timing.wall_seconds`。当前 v127 与 v74 共用同一 sample plan：

| 候选 | Linear mean | Attention mean | API(s) | Wall(s) |
|---|---:|---:|---:|---:|
| v74 | 0.440305 | 0.671106 | 218.619 | 229.485 |
| v127 | **0.509408** | **0.828395** | **151.136** | **161.840** |

完整校准、官方锚点和时间解释见
[`2026-08-31-local-metric-calibration.md`](../logs/execution/2026-08-31-local-metric-calibration.md)。

## 旧 full-layer/panel 结果（legacy，仅兼容）

根目录 `solution.py` 是重写后的 clean Gram-hierarchy 实现，不是新的官方提交。
以下结果来自固定缓存 `clean-gram-hierarchy-full`：Qwen2.5-0.5B 全 24 层，
`seq=128`、`calib=2`、`test=4`、`amax6`、CPU、`cache-mode=read`。

| 指标 | 当前根 | 旧 C86 归档 | 变化 |
|---|---:|---:|---:|
| Linear native mean | 0.501558 | 0.477821 | +0.023737 |
| Attention native mean | 0.841829 | 0.739264 | +0.102565 |
| Qwen panel total | **293.755106** | 267.307909 | **+26.447197（+9.89%）** |
| official-flow native total | 417.862253 | 392.064774 | +25.797479 |
| formal API time | **382.153528s** | 313.577669s | +68.575859s |
| wall time | 414.025852s | — | 超过最新 300s；仅 legacy 记录 |

报告：[Markdown](../logs/evaluations/clean-gram-hierarchy-full.md)，
[JSON](../artifacts/real_model_suite/clean-gram-hierarchy-full.json)。当前根的
`official_score`/`official_time` 为空；`panel_score` 只用于历史相对排序，不能换算
官方绝对分数。Linear 仍是主要优化缺口：mean 为 `0.501558`，到 `0.9` 还差
`0.398442`（当前剩余误差的 `79.94%`，即 250-case panel 的 `99.6106` 分）。

## 外部实现的本地最高基准

外部 `youxilee/hif4` v2.7（提交
`dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`）在同一固定缓存上采用 CPU/CPU
复测。五模型逐项结果和 CUDA device mismatch 诊断见
[`外部差距审计`](../logs/candidates/2026-08-29-external-hif4-gap-analysis.md)。
其中 Qwen2.5-0.5B 的 native `369.527269` 是最高单模型结果；经过本评测器
固定 `250 * Linear_mean + 200 * Attention_mean` 投影后，Qwen panel
`250.327102` 是最高的同口径本地比较线。五模型 raw sum `1085.743597` 只用于
结构性 guardrail，不能作为最高分、不能与官方 `24153` 做线性换算。

当前根相对外部最高基准：Qwen native `417.862253 - 369.527269 =
48.334984`（`+13.08%`），Qwen panel `293.755106 - 250.327102 =
43.428004`（`+17.35%`）。因此评测脚本和后续实验报告必须把外部 Qwen panel
作为第一比较线，把 Qwen native 作为第二诊断线，并单独列出官方
`24153 / 239s`，三者不可混为一个分数。

官方另提供了两个测试用例；目前根据样例统计特征判断其接近千问 30B，
但用例文件和完整形状清单尚未进入本地仓库。它们应作为后续候选的独立
宽层/GQA 压力测试，不能被拿来反向调节官方分数或逐层门限。

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

完整矩阵（默认所有本地模型和当前修订面板锚点 C39/C41b/C47b/C66）：

```powershell
.\.venv\Scripts\python.exe -u evaluator\real_model_suite.py `
  --device cuda --algorithm-device cuda `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --seq 128 --calib 2 --test 4 `
  --output artifacts/real_model_suite/latest.json `
  --report logs/evaluations/official-flow-latest.md
```

`--algorithm-device` 默认跟随 `--device`。前向捕获先落 CPU，候选阶段再按该参数回搬；这样既不长期占用模型显存，又不会把候选算法错误地切到 CPU。每完成一个候选，评估器都会写 `*.partial.json`，中断后至少保留已完成结果。

调试适配器时可以使用 `--layers 1 --calib 1 --test 1`，但这个配置只能做接口冒烟，不能用于候选排名或官方拟合。若只评估 GPT-2 等单模型，应显式设置 `--primary-model gpt2-small`；评测器会在 JSON 中记录主模型回退提示。

## 持久化真实模型数据

模型前向本身是评估中最慢、也最依赖本地模型文件的一步。评估器支持把一次真实前向产生的完整输入快照保存到 `artifacts/real_model_suite/cache/`，后续候选评测直接从快照读取，不再加载 tokenizer/model，不执行模型 forward，也不访问网络。快照包含：

- 固定 WikiText 窗口及 token ids；
- 每层各 Linear 的真实权重和真实输入激活；
- Attention 评分所需的真实 Q/K/V（包含 NeoX/Qwen 实际 RoPE 处理和 GQA 形状）；
- 模型结构、源 revision、数据集 revision、parquet SHA256、层数和样本配置。

首次采集建议单独执行，候选列表可以为空，因为 `--capture-only` 会在采集完成后停止：

```powershell
.\.venv\Scripts\python.exe -u evaluator\real_model_suite.py `
  --device cuda --cache-mode write --capture-only `
  --seq 128 --calib 2 --test 4
```

后续只读缓存评测：

```powershell
.\.venv\Scripts\python.exe -u evaluator\real_model_suite.py `
  --device cpu --algorithm-device cuda --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\cache-read.json `
  --report logs\evaluations\cache-read.md
```

`--cache-mode` 的语义如下：

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 有效快照直接读取；缺失或过期时执行一次模型前向并写入快照 |
| `read` | 只读快照；缺失、schema/版本/配置/形状校验失败时立即报错，绝不回退到模型加载 |
| `write` | 始终执行模型前向并刷新对应快照 |
| `off` | 不读取、不写入快照，保持一次性评测行为 |

缓存文件名编码了模型、序列长度、calibration/test 数量、层数和 schema 版本。读取前会校验这些字段、固定数据集 revision、模型 family/source revision、窗口防泄漏约束、CPU 张量结构以及 NaN/Inf；因此不能把不同语料、不同层数或不同模型的快照误当成同一评测。缓存属于本地生成资产，已加入 `.gitignore`，不会提交到仓库；需要刷新时显式使用 `--cache-mode write`。

缓存只改变数据供给方式，不改变合规边界：候选仍只收到原有 NVFP4 权重/激活和正式 API 参数。缓存中的 Q/K/V 只供 evaluator 在候选返回量化状态后计算 reference attention 误差。候选可以在离线权重校准中自行使用 `A @ W` 优化 `Q(W)`，但任何候选都不能利用该输出拟合或选择 `Q(A)`，也不能把它放入 `activation_state`。

## 自定义候选与官方 Champion 配对排序

根 `solution.py` 可以直接使用同一缓存模型面板评测：

```powershell
.\.venv\Scripts\python.exe -u evaluator\real_model_suite.py `
  --candidates c39 `
  --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --cache-mode read --device cpu --algorithm-device cuda `
  --output artifacts\real_model_suite\active.json `
  --report logs\evaluations\active.md
```

v4 报告的唯一排序字段是 `mean_scores.linear_mean` 和
`mean_scores.attention_mean`；`sample_plan` 必须完全一致才能横比。旧报告中的
`local_primary_panel_order`、`local_official_flow_order` 和 panel/native 字段仅
作历史诊断。官方锚点拟合不进入评测器，独立结果见校准日志。

## 评分口径（v4 主结果；旧字段兼容保留）

每个测试 case 独立计算：

```text
case_score = (MSE_STD - MSE_PLAYER) / MSE_STD
```

- `mean_scores.linear_mean`：抽样 Linear case 的 `case_score` 算术平均。
- `mean_scores.attention_mean`：抽样 Attention case 的 `case_score` 算术平均。
- `mean_scores.*_percent`：对应均值乘 100，仅为显示便利。
- `official_flow_score` 与 `panel_score`：旧版 raw sum/fixed-panel 字段，只留作
  兼容回溯，不写入 v4 报告主表。
- `timing.local_api_total_seconds`：本地设备六 API 的 calibration+dynamic 累计；
  `timing.wall_seconds` 还包含调度和报告开销。两者都不是 official time。
- 本地 state/HiF4 参数非法、API 异常、结果缺失或非 finite 才令
  `local_result_valid=false`；本地 API 超过 300s 不再伪装成官方 timeout。

赛事说明只写明判题器会加载“标准 HiF4 量化函数”，没有附上该函数源码。当前 `reference_hif4.py` 使用工程历史中独立实现的 amax/7 E6M2 与八种合法 lv2/lv3 配置最小 MSE 解；每次报告记录其 SHA256。收到官方标准函数后必须逐位替换并提升评分协议版本，旧协议结果不得与新协议绝对混算。

`linear.global_gain`、`linear.macro_gain` 等全局误差归因仍可在 JSON 查看，但不参与
v4 主结果。官方分数/时间校准单独记录，不能在评测器内部用 OLS 伪造绝对分数。

## 当前协议验证

协议冒烟结果见 [2026-08-28-official-flow-smoke.md](../logs/evaluations/2026-08-28-official-flow-smoke.md)：

- 独立 codec、合法性校验、六 API 接口、逐 case 求和和 non-causal Attention 已完整跑通。
- GPT-2-small 上 C21-C total=`151.078193`，C39 total=`150.313301`，仍与官方 `C39>C21-C` 反序。
- 这说明评分流程问题已经修复，但当前真实语料 case 分布仍不能模拟官方隐藏数据；不得通过绝对分回归或负权重修改该结论。

下一步校准对象是 case 分布而不是分数换算公式：增加冻结的跨文档 fold、形状和 NVFP4 分布覆盖，然后只看官方锚点的 pairwise 排列是否稳定改善。任何 case 权重必须在观察新候选官方结果前按赛事数据生成逻辑固定，不能针对单个候选调权。
