# HiF4 量化竞赛工程

华为 2026 算法竞赛 NVFP4 → HiF4 赛道的开发工作区。根目录
`solution.py` 是唯一活跃、可提交的算法文件；历史候选保存在
`solutions/`，不会被运行时引用。

英文版：[README_EN.md](README_EN.md)

## 当前状态

- 最新合规官方冠军：v031 / C39-FW，`14613 / 159.2s`；v025 / C21-C
  `14437 / 166.6s` 保留为次级锚点。
- 历史 v024 得分为 `16043 / 173.8s`，但包含官方后来明确禁止的
  Linear 输出监督路径，不作为后续合规父版本。
- 当前根 `solution.py` 是已冻结的 C40 robust Block-LDLQ：本地 Linear
  `0.5393`、Attention causal `0.4497`，官方 `14432 / 216.667s`；较 C39
  下降 181 分并增加 57.467 秒，因此已拒绝且不得作为后续父版本。
- 当前源码 SHA256：
  `D24BC94F513907CBE97B43865973D1498133D8B9264FAF12661836FF65AAB656`。
- 当前本地评测器已判定不能可靠排序合规候选；dev 与 frozen holdout 都存在
  循环文本造成的 calibration/test 重叠。详见
  [C40 官方结果与评测器诊断](docs/C40-official-evaluator-diagnosis.md)。

本地时间和本地分数仅用于候选比较，不冒充官方结果。任何官方结果都应与
实际提交 SHA、分数和时间一起归档。

## 官方硬约束

1. 不得显式或隐式计算 `A@W`，并利用其输出拟合、选择或反推 `Q(A)`。
2. 输出必须是合法 HiF4 五字段，API、state、shape、dtype 和设备必须符合要求。
3. 最终官方评测总时间必须严格小于 `300s`。
4. 不使用 holdout 或官方分数反向调参。

除上述规则外，不设置固定的增益、coverage、beam、单组件非退化或中间时间门槛。
开发阶段允许完整扫描和超过 300 秒的诊断实验；发现精度信号后，再通过算法和实现
优化压入最终时间限制。

## 当前算法

### Linear

1. 按 NVFP4 scale 和 E2M1 载荷重建浮点参考。
2. 搜索 SmoothQuant 对角缩放、通道排列和 4/8/16 小块正交变换。
3. 生成标准 HiF4 参数，并执行 4/8/16 组二阶精修。
4. Weight FULL64 使用合法激活 Hessian：
   - scale beam 保留 4 路；
   - 仅处理 wide FFN `fc/proj` 层，覆盖率为 `0.25`；
   - 执行 GPTQ 初始化、一次 64 维坐标下降和层级 toggle；
   - 已删除无收益的第二轮坐标下降。
5. 当前根文件还启用了 C40 相邻 128 维 Block-LDLQ 条件重求解；该机制官方
   已失败，只保留用于归档复现，不代表 Champion 算法。
6. Activation 动态路径使用 sample-local HiF4 编码和当前已验证的 4/8 组精修。

### Attention

当前保留 A1 路径：Smooth-QK、K midrange 居中、headwise permutation、
MHA/GQA 对齐和真实 Attention 双 mask 安全选择。固定 H64、Segment-CVaR 和
无收益的 V importance 候选保持关闭。

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
evaluator/real_data_eval.py         真实 GPT-2 配对评测
evaluator/synthetic_attention_eval.py
                                    576-case Attention 安全矩阵
evaluator/real_model_suite.py       多模型真实语料评测与前向缓存
evaluator/official_score_calibration.py
                                    冻结的官方分数拟合与预测
evaluator/cap_oracle.py             固定坐标误差空间诊断
evaluator/linear_compliance_guard.py
                                    Linear 合规静态/运行时检查
evaluator/holdout_eval.py           受预算保护的 holdout 评测
tests/                              发布、格式、合规和算法测试
solutions/                          不可变候选归档
artifacts/real_model_suite/cache/   本地真实模型快照，不入库
docs/superpowers/logs/              执行日志和校准记录
docs/superpowers/plans/             当前通用流程
docs/superpowers/archive/plans/     已失效优化计划，仅供历史查阅
```

## 运行评测

使用工程虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

真实模型评测：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model models/gpt2 --device cuda
```

GQA 示例：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model models/gpt2 --device cuda --kv-heads 6
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

### 候选测试顺序与结果保存

每次实验只修改根目录 `solution.py`。先完成语法、合规和单模型真实路径测试，再进行多模型比较；不要直接修改 `solutions/` 中的历史源码。

1. **提交前的快速检查**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_data_eval.py evaluator\real_model_suite.py
   .\.venv\Scripts\python -m pytest -q
   ```

2. **测试当前根 `solution.py`**

   这一步走真实 GPT-2 前向和正式候选 API，确认输出格式、Linear、Attention 和本地时间：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_data_eval.py `
     --solution solution.py --model models\gpt2 --device cuda `
     --layers 12 --seq 128 --calib 2 --test 2 `
     --mode amax6 --attn-mask causal
   ```

   输出中的本地分数只用于 A/B 比较，不填入 Official Score；同时记录完整命令、各 Linear 组件、Attention causal、运行时间和 source SHA256。

3. **一次性采集多模型真实前向数据**

   `real_model_suite.py` 默认覆盖 GPT-2 small/medium、OPT-125M、Pythia-160M、Qwen2.5-0.5B，并对已登记的 C21/C38/C39/C40 锚点进行比较。先采集模型数据，避免每个候选重复执行模型前向：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --device cuda --algorithm-device cuda --cache-mode write --capture-only `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\cache-capture-YYYYMMDD.json `
     --report docs\real-model-evaluator-cache-capture-YYYYMMDD.md
   ```

   命令中的 `YYYYMMDD` 应替换为实际运行日期。快照保存在 `artifacts/real_model_suite/cache/`，不提交到 Git；它包含真实模型权重、Linear 输入、真实 Q/K/V、token ids、模型/data revision 和窗口校验信息。

4. **只从缓存评测**

   缓存生成后，候选测试不再加载 tokenizer/model、不执行模型 forward，也不访问网络：

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --solution solution.py --candidate-name active `
     --device cpu --algorithm-device cuda --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report docs\real-model-evaluator-active-YYYYMMDD.md
   ```

   `read` 模式遇到缺失、版本不一致、配置不一致、窗口泄漏或张量形状错误会直接失败，不会偷偷重新加载模型。`auto` 适合日常使用：有效缓存直接读取，缺失或过期时重新采集；`write` 强制刷新；`off` 不保存缓存。更换 seq、calib、test、层数、模型或固定数据集 revision 后，必须生成对应的新缓存。

5. **使用冻结校准预测官方分数**

   先用固定官方锚点矩阵生成版本化校准文件；已有 v0 时无需重复拟合：

   ```powershell
   .\.venv\Scripts\python -u evaluator\official_score_calibration.py fit `
     --input artifacts\real_model_suite\20260828_full.json `
     --output artifacts\real_model_suite\official_score_calibration_v0.json `
     --feature linear_macro_gain

   .\.venv\Scripts\python -u evaluator\official_score_calibration.py predict `
     --calibration artifacts\real_model_suite\official_score_calibration_v0.json `
     --input artifacts\real_model_suite\active-YYYYMMDD.json `
     --output artifacts\real_model_suite\active-YYYYMMDD.official-prediction.json
   ```

   当前 v0 使用四个官方锚点，状态为 `diagnostic`。预测输出必须与留一 MAE、外推标记、5 模型逐项结果一起归档，不能把预测值写成 Official Score。完整说明见 [official-score-calibration.md](docs/official-score-calibration.md)。

6. **确认时间约束**

   候选 API 的 `algorithm_stage_seconds` 必须小于官方硬限制 `300s`。缓存读取只省去模型前向时间，不能掩盖候选算法自身的超时；最终仍需以官方端到端评测确认。

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
   - Local Linear q/k/v/o/fc/proj: ...
   - Local Attention causal: ...
   - Local runtime: ...
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
     docs\superpowers\logs\YYYYMMDD-experiment.md
   git commit -m "archive vNNN candidate"
   git push origin master
   ```

   若本次只更新评测器或文档，也要在提交说明中明确“不改变 active `solution.py`”。

## 记录位置

- 当前优化事实以根 `solution.py`、最新执行日志和可复现评测输出为准。
- 历史版本及其结论见 [solutions/README.md](solutions/README.md)。
- 最新执行记录见
  [2026-08-26-optimization-execution-log.md](docs/superpowers/logs/2026-08-26-optimization-execution-log.md)。
- 候选归档流程见
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/plans/2026-08-26-solution-archive-workflow.md)。
- 多模型真实语料、缓存模式和合规边界见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 本地指标到官方分数的冻结拟合与预测见
  [official-score-calibration.md](docs/official-score-calibration.md)。
- 旧优化计划已移入 `docs/superpowers/archive/plans/`，不再作为后续执行依据。
