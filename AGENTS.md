# HiF4 竞赛执行规则

> 最后整理：2026-09-09（清理冗余历史表述，只保留当前有效约束与证据入口）。
> 计划进度、逐版本结果和实验细节由下列文档维护，不在此追加流水账。

## 1. 工作范围与信息入口

- 只完成用户明确提出的任务，答复先给结论并保持简短；不自行扩展评测、实现或文档整理。
- 只改必要代码，只清理自己产生的问题；不做无关重构、格式整理、猜测性防御或提前抽象。
  信任框架保证，仅在用户输入、外部 API 等系统边界做必要校验；不为一次性操作创建辅助抽象。
- 开始优化时读[计划入口](docs/superpowers/plans/README.md)及其中唯一活动计划，再按需读
  [当前状态](docs/current-solution-status.md)、[版本索引](solutions/README.md)、目标父源码和评测器。
  普通问答或文档修改不触发整套评测流程。
- 读取历史证据前，先读全部过期信息清单：[09-02](docs/stale-information-inventory-2026-09-02.md)、
  [09-04](docs/stale-information-inventory-2026-09-04.md)、[09-05](docs/stale-information-inventory-2026-09-05.md)及[09-07](docs/stale-information-inventory-2026-09-07.md)。
  后续修订优先；归档计划和历史日志不提供下一步指令。
- 官方结果优先于活动计划已确认事实，再次是归档 result/log 和本地 JSON/report；推测不得写成事实。
  活动计划的专项规则只适用于该计划，已关闭的 v162 侧向计划不再全局覆盖门禁。
- 当前规则优先级固定为：本文件 → `docs/4b-panel-testing-guide.md` → 唯一活动总计划
  → `workbench/*/state.json` / `queue.md`。`plans/workpackages/` 只存放已完成审计，不提供当前指令；
  workbench 状态只保存历史执行状态，不得重新定义门禁。发现冲突时先同步当前文档再继续执行。

## 2. 当前基线与提交边界

| 用途 | 版本 | 官方分数 / 时间 | 说明 |
|---|---|---|---|
| 当前完整工作父、最优已知可复现方案 | v202 Linear + v195 Attention | 18053 / 281s | 用户官方回传；根与 v202 归档逐位一致，SHA `56DC805D...EFCB2BD` |
| 上一完整官方根（回退对照） | current Linear + v195 Attention | 18053 / 289s | SHA `839ADB1E...761D7F` |
| 用户确认的成功机制锚点 | A@W拟合 + Q/K互逆scale学习 | 21071 / 283s | 距当前工作父 3018 分；源码/配置/SHA待绑定，不替换根父 |
| 用户确认的榜首锚点 | 源码、配置未知 | 21765 / 290s | 距当前工作父 3712 分，不是本地实验结果 |

更早历史父、侧隔离结果和历史时间参考只作证据，见[当前状态](docs/current-solution-status.md)与
[版本索引](solutions/README.md)，不在此保留。

- 当前根 SHA256：`56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
  官方硬限 300s，当前回传余量 19s；本地时间预测和提交时间门已退役。
- 活动计划及阶段只以[计划入口](docs/superpowers/plans/README.md)为准，不在此复制快照。
- 官方提交次数**无限制**；但官方评测稳定，禁止为确定性、时间噪声或批处理研究重复提交相同 SHA
  或逐位等价 A/B。不同算法间 ±1～4 分不证明随机噪声；保留实际裁决。
- 候选单独归档，分别记录官方计分 SHA 与候选归档 SHA；未复测源码不能继承官方结果。
  本地正向不能直接晋级为官方父，须等待官方回传。

## 3. 算法与代码约束

- 正式提交为根 [`solution.py`](solution.py)，单文件、自包含，脱离仓库也能导入六个 API：
  `hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation`、
  `hif4_calibration_attention`、`hif4_dynamic_quantize_q`、
  `hif4_dynamic_quantize_k`、`hif4_dynamic_quantize_v`。
- 不得通过 `importlib`、其他 Python 文件或仓库/归档路径加载实现。编码器、解码器、E6M2、
  scale/lv2/lv3、mantissa/sign 和状态逻辑均留在提交模块内，并通过
  [`evaluator/reference_hif4.py`](evaluator/reference_hif4.py) 的合法状态检查。
- 在线动态 API 只执行校准编译的规则，不带入校准搜索、完整矩阵求逆或未限制的 Python 候选循环。
  Attention 新候选继续遵守 v165 边界：动态 API 无 Gram contraction、无候选循环，复杂计算放在校准。
- Linear 目标是实际输出误差 `XW^T - Q(XR)Q(WR^{-T})^T`；连续变换必须保持乘积不变，
  Hessian/Gram 在最终变换与部署权重坐标系计算。operand MSE、importance 和均值不能替代输出证据。
  A@W 拟合本身没有官方禁令，仍须满足合法 state 与时间约束。
- 每个正式晋级候选必须从当前最高分完整根构建并保留六 API；可只改变一个子系统以保持因果可解释，
  不建立 Linear/Attention 侧父、侧队列或侧分数晋级线，也不得把两个历史侧结果机械组合。活动计划中
  用户明确要求的标准 Linear + Attention 单次诊断提交只判断算法效果，不成为侧父；正向后仍须回装
  当前完整根并由完整官方结果晋级。
- 每个版本一个可解释机制、一个预注册配置，候选数量固定；失败换机制，不扫
  threshold/seed/alpha/offset/fold/coverage/候选数量等邻域，不增加模型/layer/role 专属路由。
- Attention校准、选择、验证分离：只用calibration folds学参数，多折固定聚合；独立 holdout
  记录验证结果，只作诊断，不作否决。
  Linear按用户最新指令直接在全部Qwen3.5-4B校准数据上做A@W低维拟合，不拆fit/select，
  不考虑泛化性；独立窗口Δmean/split/负向损失只记录、不否决探索。合法部署拟合改善、
  control、可达性、单文件和官方300s约束保留，正式晋级仍须官方结果。
- 必须记录 attempted/accepted 或同等计数证明机制可达；死分支 no-op 不能证明收敛或饱和。

## 4. 本地诊断与官方门禁

- 本地 `gain = 1 - MSE_PLAYER / MSE_STD`；STD 为同一 NVFP4 解码输入的标准 HiF4 输出。
  `overall_mean` 是实际 case 等权均值，不是准确率或官方总分，不拟合两侧权重。
- **禁止把本地分数换算官方分数**。本地只做机制否定、误差定位、符号/风险和同机成本诊断。
  本地与官方排序反转后，停止用该 proxy 为同一路线晋级。
- `Δmean`、L1、分位数、Linear `calibration_fit_gain`、Attention 4B paired 和 holdout 均只作诊断，
  不作提交门或晋级门（v188 本地过门官方 −4；L28 fit_gain 0.9486 仅对应官方 +4）。
  机制标签（如 A@W 拟合、校准统计拟合）本身也不阻止官方探索；晋级仍须官方分数、时间和源码 SHA 确认。
- 旧双侧持续计划的专项负向损失、`gain≥0.9`、误差账本、side-score 队列和 OOD 门全部退役；
  历史 `gap` 结论只用于读旧证据，被 OOD 拦截的历史候选记"未获官方验证"，不自动重跑。
  当前按[输出感知舍入边界与 A/W 联合量化计划](docs/superpowers/plans/2026-09-09-output-aware-rounding-and-joint-aw-plan.md)执行。
- 当前测试统一遵循 [4B 面板测试指引](docs/4b-panel-testing-guide.md)：不新增 0.5B、OOD、
  跨模型 GPT-2/opt 或 fresh-default 计时运行。
- 官方时间唯一硬约束为 300s；api_seconds 只作记录与风险提示，
  不得设置任何本地时间公式、预测或提交否决门。

## 5. 评测口径与执行流程

- 官方样例数为用户确认的 50 Linear + 250 Attention（Linear 拟合约 4400/5000）；这是官方分项口径，
  不与侧隔离整包分数比较，case 数不证明隐藏形状/权重，不修改本地协议冒充官方。
  依据见 `logs/execution/2026-09-07-user-21071-mechanism-evidence.md`。
- 日常入口为 [`evaluator/eval.py`](evaluator/eval.py)（eval-v3），复用 proxy-v2 dense cache；
  [`evaluator/official_eval.py`](evaluator/official_eval.py) 是兼容/参考后端，输出不得混排。
  eval-v3 全六 shard 为 4B 的 336 Linear + 72 Attention；历史面板、fold 数与侧隔离口径只用于
  读旧证据，不与当前结果混排。
- 当前本地使用 Qwen3.5-4B 结构代理（24 层，18 DeltaNet + 6 full-attention）；
  面板、窗口与覆盖以 4B manifest 为准，不把本地结构当作官方隐藏输入。
- 只比较相同 evaluator、协议、cache、panel、device 的父子结果。
- 单侧运行严格隔离 API，保持共享 state 调用图，不能按 case 制造 oracle；
  评测只检查接口、合法性和真实形状复杂度，不替候选选算法或参数。

1. 固定当前完整根及 SHA；已有同口径结果不重跑，使用 `--reuse-existing` 零 API 重放。
2. 一个候选只加一个机制和一个固定配置；做六 API、合法 state、有限输出、reachability 和 control smoke。
3. 官方前先跑目标侧 shard0 排除接口错误；活动计划中的算法开发卡随后运行目标侧六 shard，用于判断
   hard-output 优化是否真实发生和修改下一轮算法，但不得把本地数值换算为官方分数。
4. 合法且可达的单一代表候选交官方裁决；不重复相同 SHA 或逐位等价实现。
5. 目标侧六 shard 已在算法开发阶段完成；官方正向后只补必要的双侧 interaction audit，失败后只为
   明确根因运行诊断。
6. 保存 JSON 和 Markdown report，分别写 local proxy、API total、wall time、official 状态。
   接口/环境失败记 `ERROR`，机制否定记 `REJECTED`，官方明确超时记 `TIMEOUT`，
   官方未知记 `unregistered/NA`，不能填本地秒数。无实质算法/复杂度变化不分配版本号。
7. 正式版本归档前做一次脱离仓库单文件导入检查；按活动计划门禁裁决并记录官方回传。

命令模板（使用 CUDA venv，系统 Python 为 CPU-only；输出目录按本次运行命名）：

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution solution.py --linear-only --shards 0 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-name>
```

Attention 改为 `--attention-only`；完整集成审计改为 `--scenario both`。

## 6. 缓存、证据与 Git

- `--calibration-cache-mode`（`off/auto/read/write`，默认 `auto`）缓存**校准产物**
  （weight_states / attention_states），键为 solution 源码 + pack + device 的 identity：
  `read` 缺失即报错；`auto` 读取失败按过期处理并重建；`write` 强制重建。
  **命中时校准 API 的秒数与调用数记为 0**，所以不同缓存状态之间的 `api_seconds` 不可比较；
  误差与分数仍由 `_score` 实算，不受影响。
- `--cache` 指向 4B dense cache（默认 `qwen3.5-4b-proxy-v2.pt`），`--cache-cohort` 声明其口径
  （`old-weight` / `new-weight`，默认 `new-weight`）；cohort 不匹配只报告、不静默合并。
  被 git ignore 的 cache 不作源码证据。
- `--nvfp4-cache-mode` / `--cache-mode` 仅存在于 `evaluator/official_eval.py` 兼容后端，
  eval-v3 入口没有这两个参数。
- **校准缓存必须按 solution SHA 定期清理。** 目录
  `artifacts/official_eval/cache/proxy-v3-calibration/`，文件名
  `<solution SHA 前16位>-<linear|attention|both>-<配置哈希>.pt`。死候选的缓存键永不命中，
  缺失时 `auto` 会自动重建，因此只保留当前根与回退根的缓存即可，其余一律删除。
  **每个新候选的 Linear 六 shard 约 5.6 GB**，不清理会在数天内涨到数百 GB。
  清理入口（默认 dry run，先跑再 `--apply`）：
  `python workbench/cache_cleanup/prune_calibration_cache.py --keep <root前缀> [--keep <回退前缀>] --apply`。
  **多 session 并发时必须加 `--min-age-hours 2`**，否则会删掉其他 session 正在写的活候选缓存
  （2026-09-09 实测：清理后 2 小时内被并发 session 写回 11.21 GB，对应 v204/v205/v211）。
  **`qwen3.5-4b-proxy-v2.pt` 是 dense 输入主缓存，重建需完整 4B 前向，任何清理都不得删除它**
  （脚本已硬编码拒绝）。历史清理记录见 `logs/execution/2026-09-09-cache-cleanup.md`。
- 不覆盖原始 `artifacts/official_eval/*.json`、`logs/official_eval/*.md`、`logs/execution/*.md`；
  修正另写日志并更新状态。比较前检查 `evaluation_scope`。
- 实质代码或状态更新后运行 `git diff --check`、提交、push 并核验 `git status`；
  不提交 ignored 大 cache、`.codegraph/`、临时目录或无关改动。

## 7. 已关闭机制与证据边界

- 用户 21071 成功机制证据优先于历史整族饱和推断。A@W拟合、真正逐列非对称量化、Q/K互逆scale学习
  可按当前唯一总计划注册为"完整根 + 一个增量"的新机制；旧具体实现负结果不撤销，不重复同 SHA/
  逐位等价提交。旧 L21/A21 工作包与本地符号例外全部退役。合法性、control、单文件与官方 300s
  硬限保留。不把一个候选的失败扩写成所有新机制不可行。
- 细节查[当前状态](docs/current-solution-status.md)、[版本索引](solutions/README.md)和
  [计划入口](docs/superpowers/plans/README.md)；09-06 各实验的具体关闭边界以对应计划/日志为准。

### Linear

- 已关闭：旧 full64/单折邻域、Householder 全族、cross-fold minimax 的 fold/Jacobi/coverage/role
  邻域、rank-3/残差系数/fold 扩展（首次 L3 死分支结果无效，修正可达后的负结果有效）。
- LC1（rank-8 A@W 精化）与 LC2（整块 ±1 合法邻域）只关闭各自真实实现，不推广为"根坐标已局部
  饱和"或"A@W 拟合族无效"（`logs/execution/2026-09-08-lc1-lc2-method-audit.md`）。
  L-C3（objective-only）官方 `18031/293s`（−1/+13s）REJECTED
  （`logs/execution/2026-09-08-lc3-official-result.md`）。
- A@W 拟合族增益/additive 形态（v197 官方 `17277/285s` −776、AW1–7、AW9）结构性自闭：
  自适应 scale 吸收增益，重编码整数码必劣于父，与数据无关，不再注册该形态；无结构逐码贪心
  （AW8）过拟合校准窗口。归因见 `logs/execution/2026-09-09-aw-fitting-family-analysis.md`。
- v220 的零码到最小非零有符号码插入已可达但 shard0 负向；该具体插入机制关闭，不通过改成
  8-group/64-block、单个/批量零码或激活比例邻域重试。
- v204（减法定价：关 rank-2 残差段）官方 `18053/286s`（0/+5s）→ rank-2 残差段官方贡献 0 分，
  单次时间差不能单独归因给该段。

### Attention

- 已关闭：per-call 动态 Gram/自适应精化族、+4 scale 窗口、block-smooth refine 覆盖率、
  Jacobian 向 v186 移植及其收缩/clamp/gate 邻域；v187 仅为 clean-room 研究父，不替代完整父。
- 标准 Linear 侧隔离官方分（基线＝标准 Linear + R3 `14405/238s`）：v190/v191 侧分 `14405`（0），
  关闭这两个具体实现及其参数邻域；v192 侧分 `14427`（+22）有精度证据但完整提交超时，
  不再用侧时间差推算降时需求，不缩步数或减轮数重试同一实现。侧隔离只用于测分，
  侧隔离时间对完整根时间没有预测力（v194：侧 −4s / 完整 +5s，方向相反）。
  见 `logs/execution/2026-09-09-standard-linear-attention-side-scores.md`。
- v194（A2/R3 校准等价去重）官方 `18032/285s`，同分慢 5s，`REJECTED_TIME`；等价提速路线不成立。
- v196、v198、v199、v201、v203 官方均 `TIMEOUT(>300s)`，各只关闭该实现，不缩窗/减步/减轮重试。
- v223（A-H1 阈值事件搜索，8 槽×folds≈43 次完整部署路径窗口评估/层）官方 `TIMEOUT(>300s)`，
  本地六 shard mean `+0.003209` 未获官方定价；只关闭该实现，同成本类事件搜索重试前必须先降
  校准成本。
- v222（FIX-A2：A2 mean-gradient + 异常传播修复）官方 `18015/293s`（−38/+12s）REJECTED，
  只关闭该实现；方向定义问题已由 v224 的部署父状态锚定处理。
- v224（A-H1R）官方 `TIMEOUT(>300s)`，本地六 shard `≈+1.2e-6` 未获官方定价；与 v223 同成本类，
  只关闭该实现。v225（A-H3）六 shard `+3.11e-5`，
  收益几乎全部来自 shard5；A-C76.5 残差定向候选六层均未被选择。三者实际均属 Q/K 正交坐标
  变换族，当前关闭 rotation/event/group/seed/block 的直接邻域，转入量化舍入边界自由度。
- v205（减法定价：关 C76.4 旋转搜索）官方 `17969/275s`（−84/−6s）→ C76.4 官方价值 +84 分，
  必须保留，不得为省时间砍掉。砍校准换时间收益极小（本地校准 −30% → 官方仅 −6s），但这两个
  减法候选不能推出通用时间换分速率，也不能据此估算其他实现的可回收时间。
  见 `logs/execution/2026-09-09-v204-v205-subtraction-pricing-official.md`。
- V 侧不注册新候选：per-head 常量不改变块内解，per-channel multiplier 解码不逆缩放会破坏输出，
  五字段不支持 per-token 表；码分配类经 anchor28 探针裁决在当前码语义下 100% 饱和
  （NO_SUPPORTED_MECHANISM），但该饱和不构成最终输出下界
  （`logs/execution/2026-09-08-a28-interpretation-correction.md`）。当前计划只在 Q/K 上研究合法 mantissa
  舍入边界；不据此重开 V 码分配或改变官方解码格式。

### 全局

- 时间余量：根 281s / 硬限 300s，余量 19s。新增正式候选直接从当前最高分完整根构建；
  完整根候选在本地最终回退父状态时不提交。
- 缺口事实（2026-09-09）：根 `18053` 距榜首锚点 `21765` 差 3712 分；v195 完整包 `+21/+9s`
  与侧诊断 `+21/+5s` 不是同一时间口径，不做"剩余秒数换分"或"同量级机制数量"外推。
- cb1/cb2 存在实现错误，其负结果不能证明合法编码空间饱和；合法共享层级必须用五字段复核，
  operand MSE、单元素可表示值并集、非法 oracle 或连续残差不能证明最终输出天花板
  （`logs/execution/2026-09-05-next-plan-evidence-audit.md`）。
- A4/L4/C1、旧 clean-room balance/gamma/refine 和历史负向局部扫描不重开。
- 官方侧贡献比例不等于隐藏权重，也不能据此分摊榜首差距；零收益不能证明隐藏空桶。

## 8. 本文件维护方式

- 每条规则只保留一处当前有效表述；修订直接替换旧表述，不叠加"旧结论 + 更正"段落。
- 官方回传时只更新基线表和 SHA；计划进度维护在计划入口，逐版本分数及证据维护在状态/版本索引。
- 新增实验细节写入计划或执行日志；这里只增加会改变后续行为的约束及证据入口。
