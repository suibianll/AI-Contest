# HiF4 量化竞赛工程

> 数据快照日期：2026-08-31；当前事实以本文件、最新评测日志和 `solution.py` SHA 为准。

华为 2026 算法竞赛 NVFP4 → HiF4 赛道的开发工作区。根目录
`solution.py` 是唯一活跃、可提交的算法文件；历史候选保存在
`solutions/`，不会被运行时引用。

英文版：[README_EN.md](README_EN.md)

## 当前状态

- **官方评测（2026-08-31 第三次修订）**：官方评分**更换了权重，减少了 Linear 样例的
  评分权重**，因此官方总分大幅下降；官方不限制任何 `A@W` 拟合用法，只限制端到端
  运行总时间 **`< 300s`**。当前确认的官方锚点为 **v84：`16517 / 252.563s`（官方通过，
  < 300s）**。新权重总分不能与旧权重分数（v66/v72/v74 的 2 万+）直接比较。见
  [`v84 官方结果`](logs/execution/2026-08-31-v84-official-result.md)。
- 旧权重口径下的官方记录（仅供历史参考，不可与新权重比较）：v074 / C75 `22750 /
  239.387s`、v072 `22662 / 226s`、v066 / C66 `22557 / 217.2s`、v051 / C47b `22451 /
  234s`、v031 / C39-FW 与 v034 / C41b 均 `21864`（`161.3s` / `159.4s`）。
- 官方评测集仍为 **250 个 Linear case + 200 个 Attention case**；官方绝对分数随
  权重方案变化，本地一律不复制 case、不拟合官方绝对分。
- 外部参考：[`youxilee/hif4`](https://github.com/youxilee/hif4) 当前公开代码据
  用户提供的旧权重官方结果为 `24153 / 239s`；未导入本仓库。未修改的 v2.7
  源码在本地 CPU 代理上复测后，最高单模型是 Qwen native `369.527269`，同口径
  Qwen panel 为 `250.327102`（应作为本地最高比较基准）；五模型相加的
  `1085.743597` 仅是诊断量，不能排名或换算官方分数。CUDA 路径还存在外部代码
  的设备混用问题。完整表格见 [`当前主版本算法效果与评测状态`](docs/current-solution-status.md)。
- 历史 v024 得分为 `16043 / 173.8s`，其 Linear 输出监督路径把输出信息
  用于激活侧选择；按当时规则该类 `A@W -> Q(A)` 用法不合规，因此未作为后续
  合规父版本。**官方评测（2026-08-31 修订）已放开全部 `A@W` 拟合限制，只限制
  端到端运行时间**，旧合规判断不再作为当前约束。
- 当前根 `solution.py` 为 **v127**：v106 的 Linear 路径 + PAWV 变长 calibration 修复，
  作为官方 Attention shape 风险的研究候选。活动 `sampled-means-v2`（Qwen、112 Linear +
  96 Attention）复测为 Linear mean `0.522453`、Attention mean `0.842024`、Local API
  `177.039s`、Wall `180.430s`；同一 v2 计划下 v74 为 `0.452721 / 0.657497 /
  165.299s / 168.199s`。旧 `sampled-means-v1`（224/32）的 `0.509408 / 0.828395 /
  151.136s / 161.840s` 仅作历史复现，不再作为当前 A/B 主结果。不要拿本地 300 秒直接
  判断官方时间。
  逐项结果、归档实现
  审计和复现实验配置见 [`当前主版本算法效果与评测状态`](docs/current-solution-status.md)、
  [`算法全景`](docs/algorithm-inventory-and-directions.md)、
  [`归档实现审计`](docs/archive-implementation-audit.md) 与 [`solutions/README.md`](solutions/README.md)。
- **实验史摘要（已归档，不再属于根文件）**：L0–L2（诊断/LRH/CAT）产出 v105/v106/v107；
  L3–L4（Gram gate/GALS）产出 v109/v110；L5–L6（permutation、rank-16、wide rank-4、
  `G_64` hierarchy、structured factor）产出 v111–v118；C1a–C1c（向量化、refresh×2、
  rank-8/`max_blocks=8`）产出 v119–v125，v125 为 legacy precision-only 最高精度
  （panel `295.847849`，API `2653.58s` 不可提交）。**2026-08-31 起 L3–L6/C1 机制已全部
  从根 `solution.py` 裁剪**，根文件回到 v106 Linear 链 + PAWV 变长修复（v127）；这些
  机制只作为 C2/C3 压缩阶段的历史证据保留。下一步执行唯一活跃计划的 C2 低成本跨模型
  guardrail 与 C3 state/runtime 压缩。  逐版本证据见 [`solutions/README.md`](solutions/README.md)
  与 `logs/execution/`。
- **当前目标（2026-08-31 更新，取代旧权重 36000 目标）**：Linear 场景本地
  `linear_mean` 达到 **`0.8`**（当前 v127 `0.522453`，需消除剩余误差约 `58.2%`）；
  Attention 场景尽可能高（活动 v2 根为 `0.842024`；新权重官方锚点 v84 为 `0.739172`）；官方端到端
  `<300s`。**不再设置通用的本地秒数红线**：此前 `sampled API ≤150s` 的门槛建立
  在错误的“本地/官方线性映射”假设上，v100 本地约 150s 仍在官方超时，已证明该
  门槛无效。只有覆盖官方 Attention 变长校准和完整调用结构的 runtime-stress
  profile 才能用于时间筛选；在该 profile 完成前，本地秒数只能作同机 A/B 记录。
  **构成修正：官方 panel 为 250 Linear + 200 Attention（Attention 占比 44.4%），
  而且官方校准长度为 `[10,128,512,1024,1024]`；固定 `seq=128` 的 v2 无法代表
  PAWV/GQRB 等 attention-heavy 候选。**完整推断表、构成错配
  分析与全部官方候选的 v5 复评覆盖矩阵见
  [`当前目标与本地时间推断`](docs/current-solution-status.md)。旧权重口径的
  36000 推导仅作历史方法记录：[`可达性 checkpoint`](logs/execution/2026-08-31-current-results-target-feasibility.md)。
- 当前根源码 SHA256：
  `F15E112C7E832D019EE83D707ACD9D72FEF121A306E4CC3B50DBBC2CBB574924`（规范 LF，
  与 `solutions/20260831_v127_v106-pawv-variable-length-safe_scoreNA_timeNA/` 归档快照一致）。
- 旧版本地评测器（单模型 dev 与 frozen holdout）曾因 calibration/test
  文本重叠不能可靠排序合规候选；`real_data_eval.py`、`holdout_eval.py`、
  `cap_oracle.py` 已于 2026-08-28 **弃用**（文件保留但不再用于任何排序或发布
  判断，`synthetic_attention_eval.py` 仅作性质诊断）；诊断结论见
  [C40 官方结果与评测器诊断](logs/candidates/C40-official-evaluator-diagnosis.md)，
  历史版本可从 git 历史恢复。
- 当前唯一活跃评测器为 `real_model_suite.py`：使用唯一活动 profile
  `sampled-means-v2` 和 Qwen2.5-0.5B，在同一批接近官方 `250:200` 构成的样本上
  同时计算 Linear/Attention 平均 gain 与本地时间。层、role、window、seed 和数据
  revision 都写入 `sample_plan`。旧 `sampled-means-v1` 仅用于历史结果复现，其他
  模型必须显式传入，只作独立 guardrail，不相加。旧
  `official_flow_total`/`panel_score` 仍在 JSON 中兼容保留，但不再是
  报告主指标。完整口径与官方锚点校准见
  [`本地评测统一口径与官方锚点校准`](logs/execution/2026-08-31-local-metric-calibration.md)。
  **注意：v2 的 `seq=128/calib=2` 只适合误差 A/B，不是官方运行时代理；官方
  形状压力 profile（`[10,128,512,1024,1024]`）完成前，禁止用 v2 本地秒数判断
  300s 通过。** v84/v98/v100 的实测反例和修正方案见
  [`统一运行时分析`](logs/execution/2026-08-31-v84-v98-v100-runtime-analysis.md)。
- **官方结果档案（按时间序，细节均见对应日志）**：
  v031/v034 `21864`（旧权重合规锚点）→ v051 `22451` → v066 `22557` → **v072 `22662 /
  226s`、v074 `22750 / 239.387s` 通过（旧权重基线，记录见
  [`v74 官方通过`](logs/execution/2026-08-31-v74-official-pass.md)）** →
  **v84 `16517 / 252.563s` 通过（新权重锚点）**。失败类：v100/v107 官方 Attention
  `wrong answer`（非 timeout；根因为 B2 PAWV 变长 bug，输出差分与边界审计见
  [`v107 输出差分`](logs/execution/2026-08-31-v107-v31-v51-external-attention-output-diff.md)
  与 [`v100 WA 边界审计`](logs/execution/2026-08-31-v100-official-wa-boundary-audit.md)）；
  v98、v121 官方 timeout（v121 记录见
  [`v121 官方 timeout`](logs/execution/2026-08-31-v121-official-timeout.md)）。
  根 `solution.py` 现为 v127（v106 + PAWV 变长修复）研究版本，不是官方候选。
- **官方评测历史修订（2026-08-31）**：端到端超时限制从 420s 收紧为 **`300s`（5 分钟）**，
  且官方不限制任何 `A@W` 拟合用法（`Q(W)`、`Q(A)` 均可自由使用），只限制总运行时间。
  用户确认 **v98 在该限制下官方判为超时**（本地 API `406.24s` > 300s）；v98 不再是
  候选。此前将 v107 误记为 timeout 已纠错：**v107 官方为 Attention `wrong answer`
  （非 timeout）**，本地旧口径 API `481.04s` > 300s 仅作历史风险提示。
  **第三次修订（2026-08-31 晚）**：官方**更换了评分权重，减少 Linear 样例的权重**，
  官方总分据此大幅下降；新权重下确认 **v84 官方通过：`16517 / 252.563s`（< 300s）**。
  新权重总分与旧权重分数（v66/v72/v74 等 2 万+）不可直接比较。协议已升级为 v5，
  详见 [`v98 官方超时`](logs/execution/2026-08-31-v98-official-timeout.md) 与
  [`v84 官方结果`](logs/execution/2026-08-31-v84-official-result.md)。
- v100/v107 Attention WA 的直接根因已由官方自测报错确认：B2 PAWV 用第一个 calibration sample
  的 token 数建立固定 `P^TP` 方阵，再直接累加其他样本；官方接口不保证 calibration
  sample 等长。官方 mini sample 的长度为 `[10, 128, 512, 1024, 1024]`，在第二个
  样本就以 `[10,10] += [128,128]` 抛出 shape mismatch；本地 `seq=32/48` 复现与之
  一致。固定 `seq=128` 的本地 cache 因而漏检。
  v127 已按长度分组 diagonal：校准 V 与在线 V 均按当前行数精确选择统计，未命中
  则回退普通量化；同时删除未使用 low-rank 却仍执行的完整 `P^TP`/`eigh`。官方长度
  模式与公开 API 回归已通过，见 [`v126 修复日志`](logs/execution/2026-08-31-v126-pawv-variable-length-fix.md)。v127 的 v4 采样结果见
  [`v127 sampled`](logs/execution/2026-08-31-v127-sampled-means-qwen.md)。diag-only 的理论增益仍有限，后续若继续提升 PAWV，
  需要保留 `P^TP` 的非对角耦合并另做变长状态设计。
  完整分析见 [`Attention WA 根因`](logs/execution/2026-08-31-v100-v107-attention-wa-root-cause.md)。
- **2026-08-31 归档修复与 v5 复评**：v099–v125 共 28 个归档 `solution.py` 携带同样的
  PAWV 变长 bug，已按 v127 逻辑统一修复（按长度分组的 keyed diagonal），全部通过官方长度
  `[10,128,512,1024,1024]` 形状复现。修复后 v5 `sampled-means-v1` 复评：c39/c66/v72/v74
  等官方通过锚点保持 `0.43–0.44 / 0.667–0.671`；v100-pawv-fixed `0.506715 / 0.828395 /
  150.3s`、v107-pawv-fixed `0.512967 / 0.828395 / 241.5s`、v121-pawv-fixed
  `0.516685 / 0.828395 / 832.9s`（时间不可行）。官方分类不改变。完整见
  [`pawv 归档修复与 v5 复评`](logs/execution/2026-08-31-pawv-archive-fix-and-v5-reeval.md)。
- **活动 v2 官方结果归档复评已完成**：通过/失败/超时源码统一使用 `112L+96A`；
  v98/v100/v107/v121 的本地结果分别为 `0.516969/169.0s`、`0.516969/176.2s`、
  `0.526490/187.1s`、`0.531834/1571.2s`（Linear mean / API），官方裁决仍分别为
  timeout、WA→timeout、Attention WA、timeout。完整表格见
  [`solutions/README.md`](solutions/README.md) 和 [`统一复评摘要`](logs/execution/2026-08-31-official-archive-recheck-v2.md)。

**v84/v98/v100 运行时差异已单独整理**：v84 的旧 sampled-means-v1 CPU 记录
422.615s 不可与 v2 比较；同一 v2 CUDA 配置下 v84 为
0.489389 / 0.739172 / 239.910s（独立复测 234.361s）。v84 对候选 Attention
评估限制 128/256 行，而 v98/v100 对完整 calibration 序列重复执行约 35 次
QK^T/P@V，v100 还增加 PAWV；官方长度 [10,128,512,1024,1024] 因此放大
后两者的 O(T^2d) 成本。详见
[v84/v98/v100 运行时分析](logs/execution/2026-08-31-v84-v98-v100-runtime-analysis.md)。

本地时间和本地分数仅用于同一 profile/device/cache 的候选比较，不冒充官方结果。
官方 300s 是鲲鹏 920B 上官方 450 case 的端到端限制；本地 API 秒数不能直接判定
官方超时。任何官方结果都应与实际提交 SHA、分数和时间一起归档。

## 数据与计划治理（必须遵守）

### 数据及时更新

README 顶部的“当前状态”、[`solutions/README.md`](solutions/README.md)、
[`docs/current-solution-status.md`](docs/current-solution-status.md) 和最新执行日志
共同组成当前事实快照。每次本地评测、官方回传或 active `solution.py` 变更，都必须在
同一提交中更新：

1. 数据日期、模型/数据 revision、完整命令和缓存模式；
2. Linear/Attention 分项、panel、API 时间和 source SHA256；
3. `solutions/README.md` 比较表、当前状态报告和对应 execution log；
4. 若是官方结果，追加官方提交 SHA、分数、时间和日期；若未知，保留 `NA`，不得用本地值代填。

若文档数字冲突，按“根 `solution.py` + 最新可复现评测 JSON/日志 →
`solutions/README.md` → 当前状态报告 → 其他研究文档”的顺序裁决；归档计划和旧日志
只用于历史追溯。每次数据更新还要刷新文档的 `更新日期/数据快照日期`，不能继续沿用旧快照描述。

### 计划写入与执行

仓库同时只能有一份活跃优化计划，位置是
[`docs/superpowers/plans/`](docs/superpowers/plans/)，当前文件见
[`2026-08-31-hif4-active-c1-structured-linear-plan.md`](docs/superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md)。
执行任何优化时**只参考这份 active 计划**、当前根代码、最新评测数据和官方规则；
`docs/superpowers/archive/plans/` 中的文件一律是只读历史，不得作为下一步指令。

计划生命周期规则：

1. 新建计划前先确认 `plans/` 除 README 外没有第二个 `.md`；需要换主线时，在同一提交中把旧计划移入 `archive/plans/`、创建新 active 计划并更新两个 README。
2. 每个步骤必须写清假设、代码入口、数据集/模型、验收指标、预计产物和失败处理；执行后立即填入实际结果、source SHA、日志链接和 `done/rejected/blocked` 状态。
3. 一次实验无论成功、失败、超时或未提交，都先按候选归档流程保存；没有完整源码、SHA 或配置的结果只能标记为不可复现。
4. 计划完成、被替换、明确停止或连续阻塞后，立即归档；不得在旧文件中继续追加新的“下一步”，也不得保留多份“current/active”文字。
5. 归档文件不改写历史结论；若发现实现 bug 或数据错误，新增审计说明或新 active 计划修复，并在索引中标注影响范围。

执行前后至少检查：

```powershell
Get-ChildItem docs\superpowers\plans -File -Filter *.md |
  Where-Object Name -ne README.md
git diff --check
```

第一条命令必须只返回一个 active 计划；如果返回 0 或多个，先整理计划目录，不能开始算法实验。

## 本地评测是否能反映官方方向

以下是旧版 Qwen panel 兼容表，仅用于历史锚点回溯；当前排序请看活动
`sampled-means-v2` 的 Linear/Attention mean 表和对应 sample plan：

| 候选 | 官方分数 | Qwen panel（本地相对分） |
| --- | ---: | ---: |
| C39 | 21864 | 230.096230 |
| C41b | 21864 | 230.096230 |
| C47b | 22451 | 237.541351 |
| C66 | 22557 | 238.282409 |
| v72 / C74 | 22662 | 240.683147 |
| v74 / C75 | 22750 | 242.505358 |

这些 panel 值只证明历史局部排序方向，不是官方绝对分数换算；官方锚点拟合的
诊断结果和适用边界见 [`local metric calibration`](logs/execution/2026-08-31-local-metric-calibration.md)。

## 修订版官方评测锚点（2026-08-29；2026-08-31 更换评分权重）

以下结果按新版 `250 Linear + 200 Attention` 样例统计；**v031–v074 均为旧评分权重
口径**（2026-08-31 晚官方减少 Linear 样例权重后，总分不可与 v84 新权重分数直接比较）。
前三项为用户确认的本地归档提交结果，最后一项是外部仓库参考，不属于本仓库提交：

| 方案 | 分数 | 时间 | 备注 |
| --- | ---: | ---: | --- |
| v031 / C39-FW | 21864 | 161.3s | 合规归档锚点（旧权重） |
| v034 / C41b | 21864 | 159.4s | 合规归档锚点（旧权重） |
| v051 / C47b | 22451 | 234s | 此前本地官方冠军（旧权重） |
| v066 / C66 | 22557 | 217.2s | 前一官方冠军/控制组（旧权重） |
| v072 / C74 | 22662 | 226s | 前一官方冠军；Attention 通过（旧权重） |
| v074 / C75 | 22750 | 239.387s | 旧权重官方基线；Attention 通过 |
| **v84 / C84** | **16517** | **252.563s** | **新评分权重（减少 Linear 权重）下官方通过；< 300s** |
| `youxilee/hif4` | 24153 | 239s | 外部旧权重官方参考；本地最高 Qwen native `369.527269`、panel `250.327102`；五模型 `1085.743597` 仅诊断 |

> **权重变更说明（2026-08-31）**：官方减少 Linear 样例的评分权重，导致总分大幅
> 下降（v84 `16517` < 旧权重 v74 `22750` 即为例证）。两套权重不能互相换算；官方未
> 提供两项权重系数，本地不复制 case 拟合官方绝对分。v84 官方记录见
> [`v84 官方结果`](logs/execution/2026-08-31-v84-official-result.md)。

新版官方时间限制已修订为 **5 分钟（300 秒）**（2026-08-31）；表格中的通过
版本在修订前的 420s 上限下完成评测，其时间仍低于最新 300s 限制。历史 `14613 / 159.2s`、
`14437 / 166.6s` 等数值属于旧评测集，仍保留作历史记录，不与上表直接混算。

## 官方硬约束

1. **官方评测（2026-08-31 修订）不再限制任何 `A@W` 拟合用法**：离线
   `hif4_calibration_and_quantize_weight` 与在线激活量化均可用 `A@W`、输出或残差
   自由优化 `Q(W)` / `Q(A)`，信息源不限。唯一硬约束是端到端运行时间。
2. **官方评分权重（2026-08-31 晚修订）减少了 Linear 样例的权重**，官方总分据此
    大幅下降；同一算法在新权重下的官方总分与旧权重分数不可直接比较。本地排序统一以
    活动 `sampled-means-v2` 的 Linear/Attention mean 为准，不受官方总权重变化影响。
3. 输出必须是合法 HiF4 五字段，API、state、shape、dtype 和设备必须符合要求。
4. 最终官方评测总时间必须严格小于 `300s`（5 分钟）。
5. 不使用 holdout 或官方分数反向调参。

除上述规则外，不设置固定的增益、coverage、beam、单组件非退化或中间时间门槛。
开发阶段允许完整扫描和超过 300 秒的诊断实验；发现精度信号后，再通过算法和实现
优化压入最终时间限制。**不再使用通用的本地秒数红线或固定线性换算**：此前
`sampled API ≤150s` 的经验门槛在 v100（本地约 150s、官方仍 timeout）上失效。
时间筛选必须使用覆盖官方 Attention 变长校准 `[10,128,512,1024,1024]`、完整候选
调用结构和 250:200 case 构成的 runtime-stress profile；在该 profile 完成前，
`timing.api_seconds` 只能用于同一硬件、同一缓存的 A/B 比较，不能判定官方通过。

## 当前算法

当前根为 **v127：v106 Linear 链 + Attention B1/B2 + PAWV 变长修复**（2026-08-31 已把
L3–L6/C1 的实验机制全部裁剪出根文件，它们仅保留在归档中）；v086/C86 仍是不可变历史归档。
评测和优化优先级如下：

| 优先级 | 组件 | 当前机制 | 作用/状态 |
| --- | --- | --- | --- |
| 1 | Linear | BOAT：RMS 对角平衡 + 4/8/16/64 signed-Hadamard | 先压低两侧 operand-local 误差；不构造 Linear 输出 |
| 2 | Linear | Cross-fold Weight-HSDQ：`AᵀA` 二阶增量、15 levels、top-2 block、1 sweep | 只更新离线 `weight_params`；跨 fold 验证后才接纳 |
| 3 | Linear | Gram-hierarchy Activation-HSDQ：静态 `WᵀW`、offset/hierarchy 选择、最多 128 block、2 sweeps | 在线 state 仅含合法静态统计；v4 sampled Linear mean `0.509408` |
| 4 | Linear | Expansive-FFN CAT balance：`rows > channels`、固定 α=0.25 | v106 仅改善 fc_gate；不增加 state 字段 |
| 5 | Attention | reciprocal RMS、K-centering、GQA 对齐、GQRB、PAWV diag-only（v127 已按 `seq_len` 分组修复变长 bug） | 使用真实 non-causal Attention 输出排序；历史 v1 sampled Attention mean `0.828395` |
| 6 | 下一步 | L0–L6、C1a–C1c 已完成并归档（机制已从根文件裁剪）→ **C2 跨模型 guardrail → C3 state/time 压缩**，之后才把裁剪机制以可负担预算回植 | 只参考唯一活跃计划；Attention PAWV 独立延后 |

优化决策只看同一冻结缓存、同一 sample plan 上的两个均值：Qwen
`mean_scores.linear_mean` 和 `mean_scores.attention_mean` 是主指标，其他模型用于
发现结构性回退。不得用官方分数反向调参，也不设置固定的
增益、coverage 或“每个模型必须正向”门槛；只有合规、合法性、非 finite 和
主模型精度方向是当前硬条件。accuracy-first 阶段只记录时间、不因超过 300 秒拒绝；
进入最终提交压缩阶段后，`<300s` 才恢复为硬约束。

### Linear

1. 按 NVFP4 scale 和 E2M1 载荷重建浮点参考。
2. BOAT 用激活/权重 RMS 搜索对角平衡和 4/8/16/64 维 signed-Hadamard；
   候选只使用两侧 operand-local 误差，不构造 Linear 输出。
3. Cross-fold Weight-HSDQ 对宽度满足条件的权重使用 `AᵀA` block Hessian，
   在 15 个 signed levels 上做 top-2 block、1 轮坐标增量；fold 间不泛化的候选
   不进入最终参数。
4. 在线 Activation-HSDQ 使用静态变换后权重 `WᵀW` 的 Gram block 选择
   E6M2 offset/hierarchy，最多处理 128 个 block、每块 2 轮；state 不含输出监督
   或测试样本。

### Attention

当前搜索 reciprocal RMS 平衡、K-centering、GQA 对齐和 16/32/64 维共享
signed-Hadamard。便宜代理扫描后只对前 4 个候选执行完整部署量化和真实
Attention 输出复评；Q/K state 只保存 CPU 静态 Gram、重要性、整数 block/seed
和符号，V 保持独立合法 HiF4 编码。旧 C86 的实验开关、Segment-CVaR 和无收益
V importance 分支不再位于根文件。

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
docs/current-solution-status.md    当前根算法、全层实测和分数归因快照
docs/real-model-evaluator.md        评估器使用说明
docs/research/                      文献调研
docs/superpowers/plans/             唯一活跃实施计划
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

默认命令使用唯一活动的、构成匹配的 `sampled-means-v2`：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --evaluation-profile sampled-means-v2 `
  --sample-layers 8 --sample-test-windows 4 --sample-seed 20260831 `
  --device cpu --algorithm-device cpu --cache-mode read `
  --solution solution.py --candidate-name active `
  --output artifacts\real_model_suite\active-sampled-v2.json `
  --report logs\execution\active-sampled-v2.md
```

当前 `seq=128`、4 个窗口的缓存会选择 4 个分层 Linear layer 和全部 24 个
Attention layer，得到 112 Linear + 96 Attention，Attention 占比 46.2%，接近官方
44.4%；不会复制 case，且会对 profile 所需 layer 执行 calibration。改变任何 seed、
layer/window 数、device、cache 或数据 revision 后，结果必须标记为新的不可直接横比组。
该 profile 的 `sample_plan` 与 `timing.api_seconds` 同时用于均值和时间判断。若要覆盖官方
变长 Attention，还需另行采集包含
`[10,128,512,1024,1024]` 的校准缓存，固定 `seq=128` 的缓存不能代表该成本。

结果字段按下面方式读取：

| 字段 | 用途 |
| --- | --- |
| `results[*].mean_scores.linear_mean` | 当前唯一 Linear 主指标（0–1 gain 平均） |
| `results[*].mean_scores.attention_mean` | 当前唯一 Attention 主指标（0–1 gain 平均） |
| `results[*].sample_plan` | 实际 layer/window/role/seed，比较前必须完全一致 |
| `results[*].timing.local_api_total_seconds` | 当前本地设备六 API 累计耗时 |
| `results[*].timing.wall_seconds` | 本地墙钟时间，含调度/报告开销 |
| `results[*].official_flow_score` / `panel_score` | 旧版兼容字段，不作为新报告主分 |

带 `--solution` 的命令只在本地结果不完整、非法或非 finite 时返回退出码 `2`，
但仍会写出 JSON 和 Markdown；本地 API 超过 300s 不再伪装成官方 timeout。只做锚点
比较时不带 `--solution`。

CPU 全量 Qwen 评测可能很慢，旧 full-layer 数字标为 legacy；日常排序使用上述
sampled profile。官方时间只能以官方平台返回为准，本地 CUDA/CPU 仅能做同机 A/B。

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

当前 clean 根版本的发布回归（格式、合规、reference codec、真实模型套件）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
  tests/test_real_model_suite.py --basetemp=.tmp_pytest\clean-root
```

当前环境结果为 **36 passed**（2026-08-31 加入 sampled-means 测试后实测）。
`test_jdrq.py`、`test_weight_cross64.py`、
`test_weight_full64.py` 和 `test_release_candidate.py` 中仍有针对已删除 C86/JDRQ
私有 helper、实验开关或旧 state schema 的历史断言；它们不再是 clean 根版本的发布
门禁，不能用整库 `pytest -q` 的失败数评价当前算法效果。若重启这些方向，先按
当前 API/state 重新编写测试，再把测试加入发布命令。

如需单独检查真实模型评测器（使用仓库内被忽略的临时目录）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_real_model_suite.py --basetemp=.tmp_pytest\readme-verify
```

### 候选测试顺序与结果保存

每次实验只修改根目录 `solution.py`。先完成语法、合规和单模型真实路径测试，再进行多模型比较；不要直接修改 `solutions/` 中的历史源码。

1. **提交前的快速检查**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
   .\.venv\Scripts\python.exe -m pytest -q `
     tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
     tests/test_real_model_suite.py --basetemp=.tmp_pytest\clean-root
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

    新版默认评测使用 Qwen `sampled-means-v2`，在同一构成匹配样本上比较
    `mean_scores.linear_mean`、`mean_scores.attention_mean` 并记录时间；每份结果必须记录
    sample seed、layer/window index、source case 数、Local API、Wall、设备和
    source SHA256。旧 `panel_score`/`official_flow_total` 只用于读取历史 JSON。

3. **一次性采集多模型真实前向数据**

    `real_model_suite.py` 默认只评估 Qwen2.5-0.5B；需要 GPT-2/OPT/Pythia
    guardrail 时显式通过 `--models` 加入。先采集模型数据，避免每个候选重复执行模型前向：

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

5. **比较本地组件均值**

   默认只比较同一 sample plan 下的两个均值；官方锚点排序和分数拟合只在独立
   校准报告中作事后诊断，不参与候选运行或绝对分数换算。评测器内部仍保留旧
   字段以便读取历史 JSON，但新日志不再把它们写成主分。

   当前主公式为：

   ```text
   score(case) = (MSE_STD - MSE_PLAYER) / MSE_STD
    linear_mean = mean((MSE_STD-MSE_PLAYER)/MSE_STD over sampled Linear cases)
    attention_mean = mean((MSE_STD-MSE_PLAYER)/MSE_STD over sampled Attention cases)
   ```

   标准 NVFP4/HiF4 反量化、HiF4 参数校验和 state 校验全部由评测器独立完成；候选只需实现赛事规定的六个 API。评分器中的 `A@W` 只在候选返回量化结果后用于计算参考误差，不会作为输出传回候选；按官方 2026-08-31 修订，候选可自行自由使用 `A@W` 优化 `Q(W)` 与 `Q(A)`，官方不再限制信息源。

   赛事说明未附官方“标准 HiF4 量化函数”源码；当前独立标准 codec 使用历史已审计实现并在每份报告记录 SHA256。取得官方函数后必须逐位替换并升级评分协议版本。

6. **记录时间但不伪判官方结果**

   `local_api_total_seconds` 是本地设备六 API 累计，`wall_seconds` 是本地墙钟；
   二者只能在相同硬件、cache、shape 和 sample plan 下做 A/B。官方 `300s` 是
   鲲鹏 920B 的官方 450 case 端到端限制，只能由官方平台实际返回确认。

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
    - Sample profile/seed/layer-window plan: ...
    - Local Linear mean / cases: ...
    - Local Attention mean / cases: ...
    - Local API seconds / Wall seconds / device: ...
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
   .\.venv\Scripts\python.exe -m pytest -q `
     tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
     tests/test_real_model_suite.py --basetemp=.tmp_pytest\archive-check
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
  [本地评测统一口径与官方锚点校准](logs/execution/2026-08-31-local-metric-calibration.md)
  与 [`v84 官方结果`](logs/execution/2026-08-31-v84-official-result.md)。
- 候选归档流程见
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/archive/plans/2026-08-26-solution-archive-workflow.md)。
- 归档实现问题与不可复现候选见
  [archive-implementation-audit.md](docs/archive-implementation-audit.md)。
- 多模型真实语料、缓存模式和合规边界见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 官方流程逐 case 求和、独立 codec/校验和排序审计见
  [real-model-evaluator.md](docs/real-model-evaluator.md)。
- 旧优化计划已移入 `docs/superpowers/archive/plans/`，不再作为后续执行依据。
