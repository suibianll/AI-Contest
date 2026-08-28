# 多模型真实模型评估器

## 目标

`evaluator/real_model_suite.py` 是开发阶段的评估器，用于回答两个问题：

1. 候选算法在不同模型结构和真实语言模型激活上是否仍然有效；
2. 按官方逐 case 求和流程得到的本地排列能否复现已有官方排列。

它不是官方分数的替代品，也不把官方分数回灌到候选算法中。

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
  --report logs/evaluations/official-flow-latest.md
```

`--algorithm-device` 默认跟随 `--device`。前向捕获先落 CPU，候选阶段再按该参数回搬；这样既不长期占用模型显存，又不会把候选算法错误地切到 CPU。每完成一个候选，评估器都会写 `*.partial.json`，中断后至少保留已完成结果。

调试适配器时可以使用 `--layers 1 --calib 1 --test 1`，但这个配置只能做接口冒烟，不能用于候选排名或官方拟合。

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
  --cache-mode read --device cpu --algorithm-device cuda `
  --output artifacts\real_model_suite\active.json `
  --report logs\evaluations\active.md
```

报告中的 `local_official_flow_order` 是唯一主排序。官方锚点只用于计算 Spearman 和 pairwise rank agreement，不用于回归、换算或预测官方绝对分数。旧的 OLS 校准器及冻结系数已经从活跃工程删除。

## 官方流程评分口径

每个测试 case 独立计算：

```text
case_score = (MSE_STD - MSE_PLAYER) / MSE_STD
```

- `official_flow_score.linear`：所有 Linear 测试 case 的 `case_score` 直接求和。
- `official_flow_score.attention`：所有 Attention 测试 case 的 `case_score` 直接求和；当前按任务书未注明 causal mask 的 `Attn(Q,K,V)` 使用 non-causal 路径。
- `official_flow_score.total`：Linear sum 与 Attention sum 之和，是唯一主排序分。
- `official_api_total_seconds`：单个模型代理的一次完整六 API 调用耗时；每个代理都必须严格 `<300s`。多模型代理是独立诊断运行，不能把它们的时间相加冒充一次官方提交。
- 任一 state/HiF4 参数非法、API 异常、结果缺失或面板不完整，`valid_submission=false`。

赛事说明只写明判题器会加载“标准 HiF4 量化函数”，没有附上该函数源码。当前 `reference_hif4.py` 使用工程历史中独立实现的 amax/7 E6M2 与八种合法 lv2/lv3 配置最小 MSE 解；每次报告记录其 SHA256。收到官方标准函数后必须逐位替换并提升评分协议版本，旧协议结果不得与新协议绝对混算。

`linear.global_gain`、`linear.macro_gain`、组件均值和 Attention global 指标继续输出，但只做误差归因。它们不得参与候选晋级。评估器只输出 Pearson、Spearman 和 pairwise rank agreement 的排序审计，不再执行 OLS。

## 当前协议验证

协议冒烟结果见 [2026-08-28-official-flow-smoke.md](../logs/evaluations/2026-08-28-official-flow-smoke.md)：

- 独立 codec、合法性校验、六 API 接口、逐 case 求和和 non-causal Attention 已完整跑通。
- GPT-2-small 上 C21-C total=`151.078193`，C39 total=`150.313301`，仍与官方 `C39>C21-C` 反序。
- 这说明评分流程问题已经修复，但当前真实语料 case 分布仍不能模拟官方隐藏数据；不得通过绝对分回归或负权重修改该结论。

下一步校准对象是 case 分布而不是分数换算公式：增加冻结的跨文档 fold、形状和 NVFP4 分布覆盖，然后只看官方锚点的 pairwise 排列是否稳定改善。任何 case 权重必须在观察新候选官方结果前按赛事数据生成逻辑固定，不能针对单个候选调权。
