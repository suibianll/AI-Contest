# HiF4 量化竞赛工程

华为 2026 算法竞赛 NVFP4 → HiF4 赛道的开发工作区。根目录
`solution.py` 是唯一活跃、可提交的算法文件；历史候选保存在
`solutions/`，不会被运行时引用。

英文版：[README_EN.md](README_EN.md)

## 当前状态

- 官方评测集已更新为 **250 个 Linear case + 200 个 Attention case**；逐
  case 求和的分数和端到端时间都会高于旧口径，旧分数不能与新分数直接比较。
- 当前归档候选中，修订评测集下的合规官方冠军为 v066 / C66，
  `22557 / 217.2s`；此前 v051 / C47b 为 `22451 / 234s`，v031 / C39-FW
  与 v034 / C41b 均为 `21864`，时间分别为 `161.3s` 与 `159.4s`。
- 外部参考：[`youxilee/hif4`](https://github.com/youxilee/hif4) 当前公开代码据
  用户提供的同口径官方结果为 `24153 / 239s`；未导入本仓库。用未修改的 v2.7
  源码在本地 CPU 代理上复测得到五模型诊断合计 `1085.743597`，其中
  `qwen2.5-0.5b=369.527269` 最高；该代理分不能与官方绝对分换算，CUDA 路径
  还存在外部代码的设备混用问题。C66 与官方外部结果相差 `1596` 分、`21.8s`。
- 历史 v024 得分为 `16043 / 173.8s`，但其 Linear 输出监督路径把输出信息
  用于激活侧选择；这类 `A@W -> Q(A)` 用法仍不合规，因此不作为后续合规父版本。
- 当前根 `solution.py` 为 v080/C80 full gram64 + C76.4 GQA rotation；
  Qwen 本地 native `386.903134`、panel `265.372589`，四模型结果与实现
  细节见 [`solutions/README.md`](solutions/README.md)。
- 当前根源码 SHA256：
  `62EC3DB74933986886D01751E5307E58DDC8F4007E56D9A484C239F74AE69813`。
- 旧版本地评测器（单模型 dev 与 frozen holdout）曾因 calibration/test
  文本重叠不能可靠排序合规候选，相关代码（`real_data_eval.py`、
  `holdout_eval.py`、`cap_oracle.py`）已于 2026-08-28 移除；诊断结论见
  [C40 官方结果与评测器诊断](logs/candidates/C40-official-evaluator-diagnosis.md)，
  历史代码可从 git 历史恢复。
- 当前唯一活跃评测器为 `real_model_suite.py`：默认用 Qwen2.5-0.5B 作主模型，
  将冻结语料上的 Linear/Attention 平均 case gain 投影到官方的 250/200 面板；
  其他模型只作软 guardrail。`official_flow_total` 原始逐 case 求和仍保留作诊断，
  但不再按模型层数直接累加主排序。评测仍不能模拟官方隐藏数据分布，只用于
  A/B 排序。

本地时间和本地分数仅用于候选比较，不冒充官方结果。任何官方结果都应与
实际提交 SHA、分数和时间一起归档。

## 本地评测是否能反映官方方向

使用已冻结的五模型结果，新的 Qwen 主面板与官方锚点给出相同的相对顺序：

| 候选 | 官方分数 | Qwen panel（本地相对分） |
| --- | ---: | ---: |
| C39 | 21864 | 230.096230 |
| C41b | 21864 | 230.096230 |
| C47b | 22451 | 237.541351 |
| C66 | 22557 | 238.282409 |

官方与本地均为 `C39 = C41b < C47b < C66`；Qwen 主面板的 Spearman 为
`1.0000`，五模型 raw sum 为 `0.9487`。这只证明相对排序方向，不证明本地
分数可以线性换算成官方分数。外部 `youxilee/hif4` 的本地 Qwen panel 为
`250.327102`，方向上也高于 C66，但它不是本地候选锚点。

## 修订版官方评测锚点（2026-08-29）

以下结果按新版 `250 Linear + 200 Attention` 样例统计；前三项为用户确认的本
地归档提交结果，最后一项是外部仓库参考，不属于本仓库提交：

| 方案 | 分数 | 时间 | 备注 |
| --- | ---: | ---: | --- |
| v031 / C39-FW | 21864 | 161.3s | 合规归档锚点 |
| v034 / C41b | 21864 | 159.4s | 合规归档锚点 |
| v051 / C47b | 22451 | 234s | 此前本地官方冠军 |
| v066 / C66 | **22557** | **217.2s** | 当前本地官方冠军 |
| `youxilee/hif4` | **24153** | **239s** | 外部官方参考；v2.7 本地 CPU 代理 `1085.743597`，非官方换算 |

新版官方时间限制为 **7 分钟（420 秒）**。历史 `14613 / 159.2s`、
`14437 / 166.6s` 等数值属于旧评测集，仍保留作历史记录，不与上表直接混算。

## 官方硬约束

1. **离线校准可以使用 `A@W` 优化离线量化器，尤其是 `Q(W)`。** 但不得把
   `A@W`、其量化输出或输出残差用于拟合、选择或反推在线激活量化器 `Q(A)`，
   也不得将这类信息写入 `activation_state`。因此规则禁止的是输出监督驱动
   `Q(A)`，不是一律禁止离线权重量化目标中的 `A@W`。
2. 输出必须是合法 HiF4 五字段，API、state、shape、dtype 和设备必须符合要求。
3. 最终官方评测总时间必须严格小于 `420s`（7 分钟）。
4. 不使用 holdout 或官方分数反向调参。

除上述规则外，不设置固定的增益、coverage、beam、单组件非退化或中间时间门槛。
开发阶段允许完整扫描和超过 420 秒的诊断实验；发现精度信号后，再通过算法和实现
优化压入最终时间限制。

## 当前算法

当前根版本为 v080/C80，评测和优化优先级如下：

| 优先级 | 组件 | 当前机制 | 作用/状态 |
| --- | --- | --- | --- |
| 1 | Linear | SmoothQuant、通道排列、4/8/16 组二阶精修 | 主收益来源，优先用 Qwen panel 比较 |
| 2 | Linear | wide FFN `fc/proj` 的 FULL64 Hessian/GPTQ | 覆盖率 `0.25`，只更新离线 `Q(W)` |
| 3 | Linear | 动态激活 Gram-8 + all-shape Gram-64 | C80 full coverage (`ratio=1.0`, `max_blocks=128`)，只保留静态 CPU `WᵀW` state |
| 4 | Attention | Smooth-QK、K 居中、head permutation、GQA H16/H32/H64 rotation | C76.4 GQA-only；MHA 保持稳定路径 |
| 5 | 研究候选 | V importance、Q/K policy alternating、压缩覆盖预算 | 以 v080 为父版本，先测输出级上限再压入 420s |

优化决策只看同一冻结缓存上的相对增量：Qwen `primary_panel_score_total` 是主
指标，其他模型用于发现结构性回退。不得用官方分数反向调参，也不设置固定的
增益、coverage 或“每个模型必须正向”门槛；只有合规、合法性、非 finite 和
主模型 `<420s` 是硬条件。

### Linear

1. 按 NVFP4 scale 和 E2M1 载荷重建浮点参考。
2. 搜索 SmoothQuant 对角缩放、通道排列和 4/8/16 小块正交变换。
3. 生成标准 HiF4 参数，并执行 4/8/16 组二阶精修。
4. Weight FULL64 使用合法激活 Hessian：
   - scale beam 保留 4 路；
   - 仅处理 wide FFN `fc/proj` 层，覆盖率为 `0.25`；
   - 执行 GPTQ 初始化、一次 64 维坐标下降和层级 toggle；
   - 已删除无收益的第二轮坐标下降。
5. C40 相邻 128 维 Block-LDLQ 仅保留在历史归档中；当前根不启用该已被官方
   否定的路径。
6. 当前根 C80 将动态 activation Gram-64 的合法 refinement 覆盖设为
   `ratio=1.0/max_blocks=128`，并保留 Gram-8、source-aware proposal、
   sample-local HiF4 编码和已验证的 4/8 组精修。

### Attention

当前保留 A1 路径：Smooth-QK、K midrange 居中、headwise permutation、
MHA/GQA 对齐、真实 Attention 双 mask 安全选择，以及 GQA-only H16/H32/H64
head-local rotation。固定 H64、Segment-CVaR 和无收益的 V importance 候选
保持关闭。

## 开发原则

- 真实部署路径的配对分数是候选裁判；oracle 和局部损失只用于诊断、排序和解释。
- 不用任意百分比阈值在实现前否决候选，除非存在严格数学不可能证明。
- 允许同时删除冗余计算并重新分配预算；完成后再做消融归因。
- 保留完整的精度—时间 Pareto 曲线，不因单次负结果宣称整个赛事空间不可达。
- 小而稳定的正增量可以累计，不要求每个候选达到固定百分点。
- 失败实验照常归档，但失败结论只约束被实际测试的实现和配置。

## 工程结构

```text
solution.py                         唯一活跃提交文件
evaluator/real_model_suite.py       多模型真实语料评测、前向缓存与 Qwen 主面板排序
evaluator/reference_hif4.py         独立官方评分协议、标准基线与合法性校验
evaluator/nvfp4_sim.py              NVFP4 编码模拟
evaluator/real_data_eval.py         共享的候选加载/计时/评分工具与旧版单模型评测入口
evaluator/synthetic_attention_eval.py
                                    576-case Attention 安全矩阵（性质诊断，不参与排名）
evaluator/linear_compliance_guard.py
                                    Linear 合规静态/运行时检查
evaluator/linear_error_decomposition.py
                                    Linear 误差归因诊断
tests/                              发布、格式、合规和算法测试
solutions/                          不可变候选归档
artifacts/real_model_suite/         评测 JSON 结果；cache/ 为本地模型快照，不入库
logs/evaluations/                   评测运行报告（每次运行显式指定路径）
logs/candidates/                    候选官方结果与诊断报告
logs/execution/                     执行日志与校准记录
docs/real-model-evaluator.md        评估器使用说明
docs/research/                      文献调研
docs/superpowers/plans/             当前通用流程
docs/superpowers/specs/             设计与规范
docs/superpowers/archive/plans/     已失效优化计划，仅供历史查阅
```

## 运行评测

使用工程虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

### 推荐：Qwen 主评测（已有缓存）

这是日常比较候选的最短命令。它只使用 Qwen2.5-0.5B，主分固定投影为
250 Linear + 200 Attention；`--cache-mode read` 要求对应快照已经存在。
没有 CUDA 时使用下面的 CPU 命令，结果仍可用于相对排序：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --candidates c39 c41b c47b c66 `
  --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\qwen-panel-YYYYMMDD.json `
  --report logs\evaluations\qwen-panel-YYYYMMDD.md
```

有可用 CUDA 时，将上面两项改为 `--device cuda --algorithm-device cuda`。
若缓存不存在，先执行下方“采集缓存”命令；`read` 模式不会偷偷下载模型或
改用其他配置。

结果字段按下面方式读取：

| 字段 | 用途 |
| --- | --- |
| `results[*].panel_score.total` | 单模型固定面板分；Qwen 主模型使用它参与排序 |
| `official_ranking_audit.primary_panel_score_total` | 候选主排序特征 |
| `official_ranking_audit.guardrail_panel_mean_total` | 其他模型的软稳定性诊断 |
| `results[*].official_flow_score.total` | 旧版 native 逐 case 和，仅用于回溯 |
| `timing.official_api_total_seconds` | 单个模型六 API 代理耗时；主模型必须 `<420s` |

带 `--solution` 的命令在主模型非法、非 finite 或超时会返回退出码 `2`，但仍会
写出 JSON 和 Markdown，便于定位问题；只做锚点比较时不带 `--solution`。

CPU 全量 Qwen 评测可能接近或超过 420 秒，适合诊断；正式时间判断应使用 CUDA
或先用 `--layers 1 --calib 1 --test 1` 做接口冒烟，再运行完整配置。

先做不加载模型的环境检查：

```powershell
.\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
.\.venv\Scripts\python evaluator\real_model_suite.py --help
```

单模型快速评测（gpt2-small，优先读缓存）：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model gpt2-small `
  --device cpu --algorithm-device cpu --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-YYYYMMDD.json `
  --report logs\evaluations\quick-YYYYMMDD.md
```

GQA 示例（Qwen2.5-0.5B 自带 14Q/2KV 与 RoPE 适配）：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-qwen-YYYYMMDD.json `
  --report logs\evaluations\quick-qwen-YYYYMMDD.md
```

Attention 合成矩阵：

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

完整测试：

```powershell
.\.venv\Scripts\python -m pytest -q
```

完整套件还包含真实语料窗口和历史算法回归；若本机未安装 `transformers`，或
旧断言与当前 C69 实验开关不一致，相关测试会单独报告环境/历史债务。评测器发布
前至少执行上面的语法检查和下面的评测器回归（使用仓库内被忽略的临时目录），
再按本机环境补齐依赖并运行完整套件：

```powershell
.\.venv\Scripts\python -m pytest -q tests/test_real_model_suite.py --basetemp=.tmp_pytest\readme-verify
```

### 候选测试顺序与结果保存

每次实验只修改根目录 `solution.py`。先完成语法、合规和单模型真实路径测试，再进行多模型比较；不要直接修改 `solutions/` 中的历史源码。

1. **提交前的快速检查**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
   .\.venv\Scripts\python -m pytest -q
   ```

2. **冒烟测试当前根 `solution.py`**

   下面命令显式只跑 `gpt2-small`，用于快速确认输出格式、Linear、Attention
   和本地计时；它不是官方方向的主排序。需要比较候选时，请使用上面的 Qwen
   主评测命令，或把本命令的模型和 `--primary-model` 一并改成
   `qwen2.5-0.5b`：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --models gpt2-small --candidates c39 `
     --solution solution.py --candidate-name active `
     --panel-profile qwen-official --primary-model gpt2-small `
     --device cpu --algorithm-device cpu --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   完整候选比较的默认主排序使用 Qwen shaped panel：
   `panel_score.total = 250 * Linear_mean + 200 * Attention_mean`。推荐始终配对
   `--candidates c39 c41b c47b c66`；本地分数只用于 A/B 排序，不填入 Official
   Score；同时记录完整命令、源 case 数量、目标 250/200 面板、API 总时间和
   source SHA256。`official_flow_total` 仍写入 JSON，便于与旧报告回溯。

3. **一次性采集多模型真实前向数据**

   `real_model_suite.py` 默认覆盖 GPT-2 small/medium、OPT-125M、Pythia-160M、Qwen2.5-0.5B，并对已登记的当前修订面板锚点 C39/C41b/C47b/C66 进行比较。Qwen 是主模型，其余模型用于软 guardrail；先采集模型数据，避免每个候选重复执行模型前向：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --device cpu --algorithm-device cpu --cache-mode write --capture-only `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\cache-capture-YYYYMMDD.json `
     --report logs\evaluations\cache-capture-YYYYMMDD.md
   ```

   命令中的 `YYYYMMDD` 应替换为实际运行日期。机器有 CUDA 时可将两项 device
   同时改为 `cuda` 以缩短采集时间。快照保存在
   `artifacts/real_model_suite/cache/`，不提交到 Git；它包含真实模型权重、
   Linear 输入、真实 Q/K/V、token ids、模型/data revision 和窗口校验信息。

4. **只从缓存评测**

   缓存生成后，候选测试不再加载 tokenizer/model、不执行模型 forward，也不访问网络：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --candidates c39 c41b c47b c66 --solution solution.py --candidate-name active `
     --panel-profile qwen-official --primary-model qwen2.5-0.5b `
     --device cpu --algorithm-device cpu --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   `read` 模式遇到缺失、版本不一致、配置不一致、窗口泄漏或张量形状错误会直接失败，不会偷偷重新加载模型。`auto` 适合日常使用：有效缓存直接读取，缺失或过期时重新采集；`write` 强制刷新；`off` 不保存缓存。更换 seq、calib、test、层数、模型或固定数据集 revision 后，必须生成对应的新缓存。

5. **检查本地排列是否复现官方排列**

   评测器只用官方锚点做 Spearman 和 pairwise 排序审计，不拟合官方绝对分数。默认候选晋级比较同一冻结语料上的 Qwen `primary_panel_score_total`；其他模型的 panel 均值只用于发现结构性回退，不能覆盖主排序。`official_flow_total` 是兼容诊断字段。推荐用 `--candidates c39 c41b c47b c66` 与当前修订官方锚点配对运行。

   官方流程代理分为：

   ```text
   score(case) = (MSE_STD - MSE_PLAYER) / MSE_STD
   native official_flow_total = sum(all native Linear case scores) + sum(all native Attention case scores)
   qwen panel_score.total = 250 * mean(Linear case scores) + 200 * mean(Attention case scores)
   ```

   标准 NVFP4/HiF4 反量化、HiF4 参数校验和 state 校验全部由评测器独立完成；候选只需实现赛事规定的六个 API。评分器中的 `A@W` 只在候选返回量化结果后用于计算参考误差，不会作为输出传回候选；候选在离线 `hif4_calibration_and_quantize_weight` 中可以自行使用 `A@W` 优化 `Q(W)`，但不能让它进入 `activation_state` 或在线 `Q(A)` 选择。

   赛事说明未附官方“标准 HiF4 量化函数”源码；当前独立标准 codec 使用历史已审计实现并在每份报告记录 SHA256。取得官方函数后必须逐位替换并升级评分协议版本。

6. **确认时间约束**

   主模型代理的一次完整六 API 评测，其 `official_api_total_seconds` 必须严格小于官方硬限制 `420s`；等于 420 秒也判失败。多模型套件的时间只用于检查各代理，不把多个代理的时间相加冒充一次官方提交；软 guardrail 缺失不会否决 Qwen 主排序。缓存读取只省去模型前向时间，不能掩盖候选算法自身的超时；最终仍需以官方端到端评测确认。

### 候选归档步骤

一次实验无论成功、失败、未提交或官方超时都要归档，不能只保留“提升”的版本。归档前先固定根 `solution.py` 的字节和测试结果：

1. 分配下一个版本号，目录格式为 `solutions/YYYYMMDD_vNNN_topic_scoreSCORE_timeTIME/`。不知道官方结果时使用 `scoreNA_timeNA`；不要把本地分数或本地时间写进 Official Score/Time，也不要事后覆盖原始记录。
2. 将根文件复制为归档快照，根文件继续作为唯一活跃提交文件：

   ```powershell
   New-Item -ItemType Directory -Path solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA
   Copy-Item -LiteralPath solution.py `
     -Destination solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   Get-FileHash -Algorithm SHA256 solution.py
   Get-FileHash -Algorithm SHA256 solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   ```

   两个 SHA256 必须完全相同；归档后的 `solution.py` 不再修改。
3. 在同一目录创建 `result.md`，至少记录：日期、版本/父版本、唯一算法变化、假设、完整测试命令和配置、各 Linear/Attention/时间结果、缓存文件名与 dataset/model revision、active source SHA256、官方分数/时间、delta、状态、结论和下一步。缓存未入库时，`result.md` 还要注明“缓存需按 README 重新采集”。

   推荐使用以下最小模板，并把 `NA` 保留为未知值：

   ```markdown
   # vNNN — topic

   - Date: YYYY-MM-DD
   - Parent: vNNN / commit
   - Change: one primary algorithm change
   - Hypothesis: why this change may improve accuracy
   - Test command: `完整命令`
   - Test config: model/data/cache/mode/layers/algorithm-device
   - Local official-flow Linear sum / cases: ...
   - Local official-flow Attention sum / cases: ...
   - Local official-flow total and paired ordering: ...
   - Local official API total runtime: ...
   - Cache: filename, schema, dataset revision, model revision
   - Source SHA256: `...`
   - Official score: NA
   - Official runtime: NA
   - Status: `local-rejected` / `local-accepted` / `official-compliant-champion`
   - Conclusion: evidence-based decision
   - Next direction: next falsifiable experiment
   ```

4. 更新 `solutions/README.md` 的比较表和必要的执行日志；官方结果返回后只追加官方 SHA、分数、时间和日期，不覆盖已有本地证据。官方提交文件必须与归档 SHA256 一致。
5. 检查归档和测试后提交：

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   .\.venv\Scripts\python -m pytest -q
   git add solution.py solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py `
     solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\result.md solutions\README.md `
     logs\execution\YYYYMMDD-experiment.md
   git commit -m "archive vNNN candidate"
   git push origin master
   ```

   若本次只更新评测器或文档，也要在提交说明中明确“不改变 active `solution.py`”。

## 记录位置

- 当前优化事实以根 `solution.py`、最新执行日志和可复现评测输出为准。
- 历史版本及其结论见 [solutions/README.md](solutions/README.md)。
- 最新执行记录见
  [2026-08-26-optimization-execution-log.md](logs/execution/2026-08-26-optimization-execution-log.md)。
- 候选归档流程见
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/plans/2026-08-26-solution-archive-workflow.md)。
- 多模型真实语料、缓存模式和合规边界见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 官方流程逐 case 求和、独立 codec/校验和排序审计见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 旧优化计划已移入 `docs/superpowers/archive/plans/`，不再作为后续执行依据。
