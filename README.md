# HiF4 优化实验仓库（官方对齐版）

> **最新官方进展（2026-09-03）**：v160 官方结果为 **17532/232s**；v162/v163/v164 两侧校准
> 分别为 **1001/146s、4587/202s、13945/204s**，端点可加性残差仅 1 分。用户确认的榜首为
> **21765/290s**，当前差 **4233 分**。v165（standard Linear + v161 Attention）已官方
> **timeout（>300s，无分数）**，确认 Cross-Gram64 per-call 动态精化超出时间预算。侧向
> 隔离计划已收官归档（v167 低秩 Gram 码本本地 REJECTED；v166 rank-1 Linear 已官方提交
> 待回传）。当前按
> [`低复杂度算法扩展计划`](docs/superpowers/plans/2026-09-03-post-v162-low-complexity-algorithm-expansion-plan.md)
> **Attention 优先**推进（A1 解析 logits 增益校正 → A2 V 偏差质心 → A3 静态 scale 编译
> → A4 矩匹配阈值，随后 Linear L1-L4）；候选仍从 v162 双标准零点单侧构造，本地只保留
> 合法性、可达性和 control 检查，官方差分按计划 §3.3 登记。

更新时间：2026-09-03。当前仓库只认一套本地评测协议：
[`evaluator/official_eval.py`](evaluator/official_eval.py)。旧的
`real_model_suite.py`、`sampled-means-v1/v2` 和旧 JSON 不再用于排名、时间判断或调参。

## 当前结论

- 根目录 [`solution.py`](solution.py) 与 v159 归档均为 GPU 修复和等价复用后的 SHA256
  `13C9CF0BFCF2277F0828D8CC1A18A8F7414DB183F3E27DD898D52597ACC5EC79`。17532 绑定原始
  SHA `0508045A...4242`，当前归档尚未官方复测；v158 `16861/223s` 仍是时间完整的安全父版本。
- 已知官方面板曾使用 **250 Linear + 200 Attention**，总运行时间要求严格小于 **300 s**。
  本地 `proxy-v2` 不再人为限制分数比例，默认使用固定分层的真实 W/A panel；官方最近
  减少了 Linear 评分权重但没有公开新权重，因此本地不能从代理分数换算官方绝对分。
- 官方锚点：v84 `16517 / 252.563 s`、v86 `16744 / 222.7 s`、v158
  **`16861 / 223s`**（从 v86 只修改 Attention Q/K，当前仓库内最高可复现通过点）。
  **v74 在当前官方评测集为 `14561 / 188.9s`**
  （2026-09-02 回传），其旧权重时期的 `22750 / 239.387s` 已失效、差 `−8189` 分，
  **不再是安全基线或归档冠军**，详见
  [`v74 官方结果更正`](logs/execution/2026-09-02-v74-official-result-correction.md)。
  其它旧权重数字（v66 `22557`、v72 `22662`、外部 `24153`）同样只作历史证据。
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
- v140 相对 v138 只有本地 Linear `+0.000035`，官方回传为 **`15838 分 / 207 s`**，虽通过
  时间限制但低于 v86，因此已标记为 rejected。当前根文件是 v140 Linear + v86 Attention 的
  可读直接合并，并加入一次固定 A3 更新。
- v138 的官方结果现已更正为 **`15715 分 / 208 s`，通过 300 s 限制**；其本地复测数字仍仅作
  代理记录，不能与官方分数混用。
- v139 的官方结果为 **`15716 分 / 202 s`，通过 300 s 限制**；它比 v138 高 1 分，但二者
  都比 v86 低约 1029 分，v138–v145 路线因此关闭。
- v158 相对 v86 仅增加解析式 Attention Matrix-Smooth，官方 `+117` 且时间仅 `+0.3s`，已成为
  新基线；后续 Linear 实验冻结 v158 Attention。v138/v139/v140 的 Attention 本地均值完全相同，固定 Attention 修改 Linear 的官方差分为
  `+1/+123`；v84/v86 的 Linear 均值完全相同而官方 Attention 差分为 `+227`。因此后续 Linear
  必须冻结 v86 Attention，本地均值不能反推出官方权重。完整计算见
  [`官方分数归因记录`](docs/evaluation-attribution-2026-09-01.md)。
- v147 官方结果为 **`16579 / 211s`**：时间通过但低于 v86 的 16744，已标记 rejected。
  原始 pre-A3 本地 JSON 为 Linear `0.5073546371`、Attention `0.7196960689`、API
  `222.227s`；后来被写入归档的 A3 单文件本地 JSON 为 Linear `0.5100503237`、API
  `300.351s`。两份源码 SHA 均未被确认为官方提交 SHA，原始 JSON 不改写。
- v148 按计划实现一次双侧 Weight–Activation 交替，Linear 提升到 `0.5097287173`，但 Weight
  calibration 达 `291.582s`、API 总计 `369.038s`，已标记 rejected；这证明重复完整 block
  oracle 不满足时间目标，下一步必须做结构化复用而非继续调参。
- v141–v145 的 rank-4 选列 BDLR-JAQ（含锚点冻结、仅动态激活和两档阻尼）均已完整复测，
  Linear `0.281760/0.282559/0.361154/0.506418/0.506256`，均低于 v140；该方向已关闭，
  源码目录已删除，仅保留评测 JSON 和执行日志。后续 v151–v154 已完成 pre-A3 role 控制与
  fc decoupled encoder 验证：v152 为 mixed，v153/v154 明确回归，均已拒绝。L3-D0 teacher
  已完成但结论为 `margin_exists_but_not_compile_safe`：layer 3 / fold 128 的 exact output
  margin 对 `fc_gate/fc_up` 为 `-0.094751/-0.112680`，不创建 v155。当前下一步是只跑最坏层的
  cross-fold feature/decision stability 快探针；不通过就转 L2，且不再调 `s_q/s_d`、CAT、ROAB
  或 offset。
- 首个 L2 2×2 analytic pair-balance local-only probe 已拒绝：fc focus 配对 `0/16/0`、均值
  `-0.314079`，说明朴素矩阵平衡破坏静态 Weight code。后续 L2 必须直接使用部署输出 metric
  做约束，不能重复同类无约束变换。规范 D0 约 `597.7s`，日常 layer-3 fast probe 约 `10.35s`。
- 评测混乱的根因和修复已写入 [`artifacts/official_eval/README.md`](artifacts/official_eval/README.md)：
  只有同 cache 的 `default-panel` 可做本地 proxy 排名；effect/replay、full stress、smoke、
  GPT-2/hif4 和旧 v1 都是诊断，任何本地结果都不等价于官方分数或官方时间。
- 2026-09-01 归档复测已完成 18 个有官方记录的候选：本地最高返回结果为 v121
  (`0.472197763 / 0.833617251`)，但 API `3404.369 s`、官方 timeout；v002 的本机
  CUDA/CPU device-mix 错误被原样记录。完整明细只看
  [`archive-official-shape-v1.json`](artifacts/official_eval/legacy-v1/archive-official-shape-v1.json)（历史 v1 证据，已隔离到 `legacy-v1/`）。

## 当前协议：`proxy-v2`（旧 `official-shape-v1` 仅作历史诊断）

评测器将官方已知的接口、形状、合法性和调用结构集中在一个文件中。`proxy-v2` 是诚实的
同机趋势代理，不声称复制隐藏官方数据或鲲鹏硬件；旧 `official-shape-v1` cache 不能被新协议
静默读取。

| 项目 | 固定值 |
|---|---|
| 模型 | Qwen2.5-0.5B，24 个 Transformer block（本地 proxy 结构假设；说明书未公开指定模型） |
| 数据 | 固定 revision 的 Salesforce/WikiText-2-raw-v1；train 做 calibration，validation/test 交替 holdout |
| Attention calibration | **`[10, 128, 512, 1024, 1024]`**，每个 Q/K/V 样本保持自己的序列长度 |
| Linear calibration | default audit 每个 layer/role 使用前两折；compact 使用 128/512 两折并只建立选中 state |
| Test windows | 12 个互不重复的 validation/test 文档窗口，长度按 `[10,128,512,1024,1024,10,128,512,1024,1024,128,512]` 轮换 |
| 用例 | 默认 168 Linear + 120 Attention 仅作低频 audit；日常 `--compact-panel` 为 28 个 Weight state + 56 个跨 validation/test Linear case（4 个纵深层×7 role×2 holdout）；`--effect-panel` 保留完整校准图专项诊断；`--full-cases` 才展开 stress |
| API | 六个赛事接口，顺序和参数形状与 `赛事说明书.txt` 一致 |
| 参数校验 | 独立校验 E6M2、`scale_lv2/lv3`、sign、mant、state 深度/节点数和 CPU tensor 规则 |
| 标准基线 | `evaluator/reference_hif4.py` 的固定标准 codec；候选代码不能改变分母 |

每个测试用例的公开公式为

\[
s_i=\frac{\operatorname{MSE}_{\rm STD,i}-\operatorname{MSE}_{\rm PLAYER,i}}
          {\operatorname{MSE}_{\rm STD,i}},
\qquad
L=\frac1{N_L}\sum_{i=1}^{N_L}s_i,
\qquad
A=\frac1{N_A}\sum_{j=1}^{N_A}s_j.
\]

JSON 的 `score.linear_mean` 和 `score.attention_mean` 是分场景主指标；
`score.overall_mean` 是全部真实 case 的未加权均值；`score.total_sum` 只表示本次实际 case 的
和，不跨不同数量的运行比较。没有任何 Linear:Attention 人为比例或官方分数拟合。

校准调用保持官方状态图：Qwen 默认 168 次 Weight calibration、24 次 Attention
calibration；每个动态 case 只调用一次相应动态 API。报告中的 `trend_diagnostics` 会在同一
官方权重 cohort 的已知版本之间做 pairwise 顺序检查；它只标记反转，绝不把官方分数反向拟合
进候选分数。

默认评测还输出误差源分解，不参与主分数：Linear 用 `E00/E10/E01/E11` 分离 W-only、A-only、
W+A 以及交互项，并按 role/layer/shape/length/split 聚合；Attention 用 Q-only、K-only、
V-only、QK-only、QKV 控制臂，同时报告 logits MSE、softmax probability MSE、KL 和
layer/length 聚合。逐 case 结果在 JSON 的 `case_scores`，聚合结果在 `decomposition`；
这些控制臂复用已产生的候选输出，不增加候选 API 调用。仅在快速 smoke 时使用
`--no-decomposition`。

机制迭代不再用两个总均值手工相减。先把父版本运行一次并保存 JSON，候选用同一 cache、
同一 `--compact-panel` 和 `--baseline-json` 逐 case 配对。compact 额外输出 median、q25、
worst-quartile、负 case、MSE ratio 与 validation/test cross-holdout 同号率/gap；JSON 的
`paired_effect` 会分别给出：
目标 role/family、未改动 control、每个 role/layer/shape/split/length 的 signed delta，改善/回归/
不变 case 数，W/A 或 Q/K/V 控制臂变化，最好/最坏 case，以及六 API 同机时间差。配对会校验
case identity、标准 codec MSE 和 reference energy；任一不一致就拒绝比较，不把不同 panel
误当成算法效果。`consistent_improvement`、`consistent_regression`、`mixed`、`no_effect`
只是符号描述，不是新的人为晋级阈值。

说明书只规定六个 API 和 Linear/Attention 数据组织，没有公开 Qwen、层数、GQA 或 RoPE。
要检查模型结构假设，可运行独立的
[`evaluator/cross_model_eval.py`](evaluator/cross_model_eval.py)：它在本地 GPT-2 的真实 fused
QKV、12×64 MHA、绝对位置编码和单一 GELU FFN 上评测，使用独立
`cross-model-probe-v1` cache，不修改 Qwen proxy 分数或官方趋势审计。GPT-2 结果是结构压力
测试，不是官方分数；当前 v86/v147/v140 的 GPT-2 顺序为 `v140 > v147 > v86`，与官方
`v86 > v147 > v140` 完全相反，详见 `docs/current-solution-status.md` 的跨模型小节。

另外按要求直接运行了外部仓库 [youxilee/hif4](https://github.com/youxilee/hif4) 的
`real_data_eval.py`，对 v84/v86/v140/v147 使用相同 GPT-2 12 层、`amax6 / seq128 /
calib2 / test2 / current` 配置。其结果（Linear mean / Attention）依次为
`v84 0.586733/0.4477`、`v86 0.586733/0.4727`、`v140 0.599617/0.4661`、
`v147 0.599617/0.4713`；完整逐 role 表、源码 SHA 和限制见
[`外部复测日志`](logs/execution/2026-09-01-hif4-external-gpt2-v84-v86-v140-v147.md)。
该脚本的标准 codec 由候选私有实现提供，不能替换本地主评测器或官方趋势判断。
逐 role 归因显示 v140 相对 v86 的主要回归在 `fc`（12/12 层为负），其次是 `proj`（混合但有
严重层级异常）；静态 q/k/v 稳定改善、o 近中性。因此下一轮冻结 q/k/v/o，先处理 proj，再
重做保留 BOAT 的 fc 编码/scale。完整消融和限制见
[`外部 role 归因日志`](logs/execution/2026-09-01-hif4-external-role-attribution-v140-v86.md)。

本地主评测器现在还会在 `--archive` 的同一 cache 中自动生成跨候选 static Linear role 差分：
`q/k/v`、`o`、合并后的 `fc` 和 `proj` 均报告平均 Δ、正负 case 数及最差层；这与候选内部的
W/A 四臂分解互补，不增加六个 API 调用。ROAB/BOAT/CAT 这类私有机制仍须用 local-only 变体
或外部 hif4 副本消融，不能伪装成官方 API 评测。

计时同时保存：

- `timing.api_total_seconds`：六个候选 API 调用耗时之和，是最接近赛事“量化函数执行时间”
  定义的本机代理；
- `timing.wall_seconds`：从第一次校准到最后一次动态量化的墙钟时间，包含本地评分和调度；
- `timing.api_calls`：实际调用次数。两种本地秒数都不能直接换算为鲲鹏官方秒数。

## 运行方式

首次采集固定公开数据包（需要本地模型和数据）：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode write --capture-device cuda --algorithm-device cuda `
  --output artifacts\official_eval\capture.json `
  --report logs\official_eval\capture.md
```

只读缓存、批量复测所有已有官方结果的归档版本：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --archive `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\archive-proxy-v2.json `
  --report logs\official_eval\archive-proxy-v2.md
```

只测一个候选：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --solution solutions\20260830_v084_c84-gram64-sweep5_scoreNA_timeNA\solution.py `
  --name v084 --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\v084.json --report logs\official_eval\v084.md
```

快速机制迭代先建立一次父版本 effect baseline：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --solution <parent-solution.py> --name parent --effect-panel `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\parent-effect.json `
  --report logs\official_eval\parent-effect.md
```

随后每个候选只运行自己，直接与保存的父版本逐 case 配对；例如只改 fc：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --solution <candidate.py> --name candidate --effect-panel `
  --baseline-json artifacts\official_eval\parent-effect.json `
  --focus-linear-roles fc `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\candidate-effect.json `
  --report logs\official_eval\candidate-effect.md
```

已有两个相同 panel 的 JSON 可零 API 重放：

```powershell
.venv\Scripts\python.exe evaluator\official_eval.py `
  --baseline-json <parent.json> --candidate-json <candidate.json> `
  --focus-linear-roles fc --output <paired.json> --report <paired.md>
```

判读固定按以下顺序：先看 focus 的均值、median 和正负 case 是否同向；再看 control 是否保持
不变；再看 W-only/A-only/Both/interaction（Attention 则看 Q/K/V/QK/QKV）；最后检查最坏层、
长度桶和同机 API 增量。只有这种配对证据清楚后才值得跑默认 168+120 panel。总均值的小幅上升
但 focus 正负混合，只能说明结果不稳定，不能据此晋级。

`--cache-mode read` 缺少或不符合协议的快照会直接失败，不会悄悄切换数据或形状。
默认命令使用 168 Linear + 120 Attention 的固定分层 panel，覆盖所有 layer/role 和五个官方
Attention 长度；它保留真实 W/A，同时避免 12 个窗口与所有 layer/role 的笛卡尔展开。快速
算法迭代显式加 `--effect-panel`：Linear 选择 8 个覆盖模型首、中、末深度的层且每层保留全部
7 个 role，Attention 选择 5 个同时覆盖深度和公开长度的哨兵。校准调用仍保持完整 168 Weight +
24 Attention state，因此它缩短的是动态评分和分解，不会伪造一个按 case 重新校准的便宜协议。
只有做压力测试时才显式加 `--full-cases`；`--linear-cases/--attention-cases` 是按顺序截断的接口
smoke，尤其 `14/56` 不是纵深采样，不能再用于判断算法效果。不同 panel 的结果不能混排。
缓存必须记录模型、数据 revision、五个 Attention 长度、SHA256 和权重原生
`[out_features, in_features]` 布局。没有 CUDA 时可以把 `--algorithm-device` 改为 `cpu`，
但 CPU 秒数只适合 CPU 内部 A/B。

## 官方结果归档与本地复测

> **口径警告**：下表自上而下分为两个权重时期。**v001–v074 是旧权重分数，v084 起是新权重
> 分数，两段不可互相比较、不可换算、不可用于联合排序。** 旧权重数字只作审计证据。
> v074 另有当前评测集回传 `14561 / 188.9s`（2026-09-02），远低于其旧权重 `22750`。

### 旧权重时期（历史证据，不可用于当前排序）

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
| v074 | 22750（旧权重）→ **14561**（当前评测集回传） | 239.387 s → **188.9 s** | pass（**非安全基线**，低于 v84/v86） |

### 新权重时期（当前口径）

| 版本 | 官方分数 | 官方时间 | 官方裁决 |
|---|---:|---:|---|
| v084 | 16517 | 252.563 s | pass（新权重） |
| v086 | 16744 | 222.7 s | pass（新权重） |
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
| v140 | 15838 | 207 s | pass but rejected（官方，低于 v86） |
| v147 | 16579 | 211 s | pass but rejected（官方，低于 v86；提交 SHA 未确认） |
| v158 | 16861 | 223 s | pass（源码与时间完整的安全基线） |
| v159 | 17532 | — | 官方分数已报告，时间未知 |

统一复测生成的文件只能放在 `artifacts/official_eval/` 和 `logs/official_eval/`；
当前复测结果以 `proxy-v2` JSON 为准；历史 v1 结果表隔离在
`artifacts/official_eval/legacy-v1/`（不可迁移比较）。旧 `artifacts/real_model_suite/`
结果不再读取、不再更新，旧 evaluator 源码统一放在
`evaluator/archive/legacy-20260901/` 作为历史证据。

## 分数体系与归档对照（2026-09-01 归档整理）

仓库历史上存在多套**互不相通**的分数，任何排序、对比与结论必须先声明体系，严禁混用：

| 体系 | 来源 | 适用版本 | 状态 |
|---|---|---|---|
| 官方旧权重分数 | 官方回传（旧权重时期，panel 数次修订） | v001–v074 | 历史事实，仅存档 |
| 官方新权重分数 | 官方回传（历史上 250 Linear + 200 Attention） | v084/v086/v098/v100/v107/v121/v128–v131/v138–v140/v147 | 当前官方口径 |
| 本地协议分 | proxy-v2 分层真实 W/A 复测（`linear_mean`/`attention_mean`/`overall_mean`） | 当前活动候选 | 仅同机 A/B，不换算官方分 |
| 旧协议分（已废弃） | real_model_suite / sampled-means-v1/v2 / oracle dashboard | v000–v127 时期 | 已全部归档，禁止再用于排序或调参 |

注意：`solutions/` 目录名中的数字字段**不是统一口径**——v001–v032 的 `score`/`official`
字段为官方分；v034–v086 多数为 `scoreNA_timeNA`（官方分见上表）；v087 之后目录名中的
`score29x` / `screen0.53x` 是**本地协议分数**，不代表官方结果。2026-09-01 归档整理中
v031 目录名已从旧面板 `official14613` 更正为官方 `21864`，v125 screen 记录目录更名为
`v125b` 保证版本号全局唯一。

### 归档目录结构

| 路径 | 内容 |
|---|---|
| `solutions/` | 唯一版本源码快照（只读归档，`retained/rejected/timeout` 标注） |
| `evaluator/official_eval.py` | 当前唯一评测入口（proxy-v2；v1 只读历史诊断） |
| `evaluator/archive/legacy-20260901/` | 旧评测器源码（real_model_suite 等） |
| `artifacts/official_eval/` | 当前协议 JSON 与 cache |
| `logs/official_eval/` | 当前协议报告 |
| `artifacts/archive/legacy-*-20260901/` | 旧协议 JSON（real-model-suite / oracle-dashboard / jdrq-diagnostics） |
| `logs/archive/legacy-*-20260901/` | 旧协议报告（evaluations / official-eval / candidates / root-files） |
| `docs/superpowers/archive/plans/` | 已完成/废止计划 |

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
5. 任何排序都以同一 `proxy-v2`、同一 cache、同一设备为前提；不得混用旧
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
