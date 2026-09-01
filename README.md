# HiF4 优化实验仓库（官方对齐版）

更新时间：2026-09-01。当前仓库只认一套本地评测协议：
[`evaluator/official_eval.py`](evaluator/official_eval.py)。旧的
`real_model_suite.py`、`sampled-means-v1/v2` 和旧 JSON 不再用于排名、时间判断或调参。

## 当前结论

- 根目录 [`solution.py`](solution.py) 是活动研究版本；`solutions/` 下的 `solution.py` 是只读归档。
- 已知官方面板为 **250 Linear + 200 Attention**，总运行时间要求严格小于 **300 s**。
  官方最近减少了 Linear 评分权重，但没有公开两项新权重，因此本地不能从代理分数换算
  官方绝对分。
- 官方历史锚点：v74 `22750 / 239.387 s`（旧权重），v84 `16517 / 252.563 s`、
  v86 **`16744 / 222.7 s`**（新权重，通过；v86 为新权重下分数最高且最快的官方通过点）。
  v98/v121 为 timeout，v100/v107 为 Attention 相关失败；v128 fixed-attn-budget 已由用户
  确认官方 timeout（`>300 s`，分数未返回）；v129 fixed-attn-budget-sweep1 也已确认官方
  timeout（`>300 s`，分数未返回）；v130 输出监督 W 版本同样已确认官方 timeout
  （`>300 s`，分数未返回）；v131 Q(W)-Gram 版本也已确认官方 timeout（`>300 s`，
  分数未返回）。v129–v131 均使用高复杂度 Attention 路径，不能把 timeout 单独归因于
  Linear 改动。
- 本地报告只回答两个问题：固定公开数据上的 `linear_mean`、`attention_mean` 如何变化，
  以及六个官方 API 在同一台机器上的耗时如何变化。官方分数和鲲鹏 920B 时间只保留为
  独立历史字段，绝不伪造映射。
- v134 在相同 cache 上两次完整空闲复测得到 Linear `0.5073195`、Attention `0.8342565`，
  API `289.042/289.832 s`；该版本加入 L2 输出监督激活 cross64。由于 v130 官方 timeout，
  本地 API `<300s` 不能作为官方保证。
- 根文件当前仍是 v140 实验代码，但它相对 v138 只有本地 Linear `+0.000035`，没有官方结果，
  不再称为当前最优；下一实现基线恢复为原样 v86。
- v138 的官方结果现已更正为 **`15715 分 / 208 s`，通过 300 s 限制**；其本地复测数字仍仅作
  代理记录，不能与官方分数混用。
- v139 的官方结果为 **`15716 分 / 202 s`，通过 300 s 限制**；它比 v138 高 1 分，但二者
  都比 v86 低约 1029 分，v138–v145 路线因此关闭。
- v141–v145 的 rank-4 选列 BDLR-JAQ（含锚点冻结、仅动态激活和两档阻尼）均已完整复测，
  Linear `0.281760/0.282559/0.361154/0.506418/0.506256`，均低于 v140；该方向已关闭，
  源码目录已删除，仅保留评测 JSON 和执行日志。下一步不再调 BDLR 参数，而是先建立合法
  Joint oracle，再研究零空间误差整形、子空间嵌入联合舍入和乘积保持的新表示。
- 2026-09-01 归档复测已完成 18 个有官方记录的候选：本地最高返回结果为 v121
  (`0.472197763 / 0.833617251`)，但 API `3404.369 s`、官方 timeout；v002 的本机
  CUDA/CPU device-mix 错误被原样记录。完整明细只看
  [`archive-official-shape-v1.json`](artifacts/official_eval/archive-official-shape-v1.json)。

## 唯一协议：`official-shape-v1`

评测器将官方已知的接口、形状、合法性和调用结构集中在一个文件中：

| 项目 | 固定值 |
|---|---|
| 模型 | Qwen2.5-0.5B，24 个 Transformer block |
| 数据 | 固定 revision 的 Salesforce/WikiText-2-raw-v1；train 只做 calibration，validation 只做 test |
| Attention calibration | **`[10, 128, 512, 1024, 1024]`**，每个 Q/K/V 样本保持自己的序列长度 |
| Linear calibration | 使用公开本地数据包的前两折；赛事说明书未公开 Linear 折数，假设在元数据中明确记录 |
| Test windows | 9 个互不重复的 validation 文档窗口，每个 128 token |
| 用例 | 从 24 层、7 类 Linear role 和 9 个窗口中稳定选取 250 个 tuple；Attention 选取 200 个 tuple，禁止重复 |
| API | 六个赛事接口，顺序和参数形状与 `赛事说明书.txt` 一致 |
| 参数校验 | 独立校验 E6M2、`scale_lv2/lv3`、sign、mant、state 深度/节点数和 CPU tensor 规则 |
| 标准基线 | `evaluator/reference_hif4.py` 的固定标准 codec；候选代码不能改变分母 |

每个测试用例的公开公式为

\[
s_i=\frac{\operatorname{MSE}_{\rm STD,i}-\operatorname{MSE}_{\rm PLAYER,i}}
          {\operatorname{MSE}_{\rm STD,i}},
\qquad
L=\frac1{250}\sum_{i=1}^{250}s_i,
\qquad
A=\frac1{200}\sum_{j=1}^{200}s_j.
\]

JSON 的 `score.linear_mean` 和 `score.attention_mean` 是唯一主指标；
`score.total_sum` 是 450 个 case 分数的和；`equal_weight_45000_scale` 只是把这个和乘
100 的等权显示，不是新版官方分数。

计时同时保存：

- `timing.api_total_seconds`：六个候选 API 调用耗时之和，是最接近赛事“量化函数执行时间”
  定义的本机代理；
- `timing.wall_seconds`：从第一次校准到最后一次动态量化的墙钟时间，包含本地评分和调度；
- `timing.api_calls`：实际调用次数。两种本地秒数都不能直接换算为鲲鹏官方秒数。

## 运行方式

首次采集固定公开数据包（需要本地模型和数据）：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode write --capture-device cuda --algorithm-device cuda `
  --output artifacts\official_eval\capture.json `
  --report logs\official_eval\capture.md
```

只读缓存、批量复测所有已有官方结果的归档版本：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --archive `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\archive-official-shape-v1.json `
  --report logs\official_eval\archive-official-shape-v1.md
```

只测一个候选：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --solution solutions\20260830_v084_c84-gram64-sweep5_scoreNA_timeNA\solution.py `
  --name v084 --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\v084.json --report logs\official_eval\v084.md
```

`--cache-mode read` 缺少或不符合协议的快照会直接失败，不会悄悄切换数据或形状。
缓存必须记录模型、数据 revision、五个 Attention 长度、SHA256 和权重原生
`[out_features, in_features]` 布局。没有 CUDA 时可以把 `--algorithm-device` 改为 `cpu`，
但 CPU 秒数只适合 CPU 内部 A/B。

## 官方结果归档与本地复测

| 版本 | 官方分数 | 官方时间 | 官方裁决 |
|---|---:|---:|---|
| v001 | 10250 | 127 s | pass |
| v002 | 15313 | 137 s | pass |
| v013 | 15799 | 144 s | pass |
| v024 | 16043 | 173.8 s | pass |
| v025 | 14437 | 166.6 s | pass |
| v030 | 14092 | 170.57 s | pass |
| v031 | 21864 | 161.3 s | pass |
| v032 | 14432 | 216.667 s | pass |
| v034 | 21864 | 159.4 s | pass |
| v051 | 22451 | 234 s | pass |
| v066 | 22557 | 217.2 s | pass |
| v072 | 22662 | 226 s | pass |
| v074 | 22750 | 239.387 s | pass |
| v084 | 16517 | 252.563 s | pass（新权重） |
| v086 | 16744 | 222.7 s | pass（新权重，当前最佳） |
| v098 | — | >300 s | timeout |
| v100 | — | >300 s | 原始 Attention WA；修复线仍 timeout |
| v107 | — | — | Attention WA |
| v121 | — | >300 s | timeout |
| v128 | — | >300 s | timeout（官方，用户确认） |
| v129 | — | >300 s | timeout（官方，用户确认） |
| v130 | — | >300 s | timeout（官方，用户确认） |
| v131 | — | >300 s | timeout（官方，用户确认） |
| v138 | 15715 | 208 s | pass（官方，用户报告） |
| v139 | 15716 | 202 s | pass（官方，用户报告） |

统一复测生成的文件只能放在 `artifacts/official_eval/` 和 `logs/official_eval/`；
结果表以 `archive-official-shape-v1.json` 为准。旧 `artifacts/real_model_suite/`
结果不再读取、不再更新，旧 evaluator 源码统一放在
`evaluator/archive/legacy-20260901/` 作为历史证据。

## 归档、计划和清理规则

1. 计划目录只保留一个活动计划：`docs/superpowers/plans/`；完成或废止的计划移动到
   `docs/superpowers/archive/plans/`，执行时只读取活动文件。
2. 参数/阻尼/rank/seed 等内部试验只使用一个工作副本和汇总日志；只有新数学算法、官方提交
   或一个代表性失败实现才分配版本。最终目录名必须标注 `retained/rejected/timeout`，未知官方
   值写 `scoreNA_timeNA`，不能把本地数值填入 Official 字段。
3. `result.md` 记录唯一算法变化、父版本、命令、协议、数据/模型 revision、Linear/Attention
   均值、API/Wall、源 SHA、官方分数/时间和状态。官方回传只追加，不覆盖本地证据。
4. 评测输出与源码分离：活动结果只写 `artifacts/official_eval/`；旧结果清理时可以删除
   `artifacts/real_model_suite/` 的 JSON/MD。对已明确拒绝的微版本，可在保留 `artifacts/official_eval/`
   JSON 和 `logs/` 执行日志后删除 `solutions/` 源码，以控制归档规模；通过版本和代表性结构源码保留，
   版本号全局唯一。
5. 任何排序都以同一 `official-shape-v1`、同一 cache、同一设备为前提；不得混用旧
   `sampled-means-v1/v2` panel，也不得用官方分数反向调参。

## 算法文档

- [`docs/current-solution-status.md`](docs/current-solution-status.md)：官方基线、实验根状态和失败路线。
- [`docs/algorithm-inventory-and-directions.md`](docs/algorithm-inventory-and-directions.md)：
  已实现算法、归档版本和未验证方向。
- [`docs/archive-implementation-audit.md`](docs/archive-implementation-audit.md)：归档实现审计。
- [`docs/superpowers/plans/`](docs/superpowers/plans/)：唯一活动优化计划。
- [`赛事说明书.txt`](赛事说明书.txt)：官方接口和评分定义原文。

运行单元测试：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

提交前至少执行 `py_compile`、官方协议测试和 `git diff --check`，并将新 JSON/报告与
源码 SHA 一起提交。缓存（数 GB）不进入 Git。
