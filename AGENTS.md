# HiF4 竞赛执行规则

> 最后整理：2026-09-09。本文件只保留当前基线、长期约束和工作入口。
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
| 当前完整工作父、最优已知可复现方案 | v202 Linear + v195 Attention | 18053 / 281s | 用户官方回传；根与 v202 归档逐位一致，SHA `56DC805D...EFCB2BD`；相对 v195 同分 / −8s |
| 上一完整官方根 | current Linear + v195 Attention | 18053 / 289s | 可回退历史对照，SHA `839ADB1E...761D7F` |
| 上一完整官方父 | current Linear + R3 Attention | 18032 / 280s | 可回退历史对照，SHA `12352EFD...E24E` |
| 更早完整官方父 | compiled sample-energy | 17636 / 264s | 可回退历史对照，SHA `D66128A6...B0F6` |
| 历史时间参考 | v180 | 17597 / 242s | 只作复杂度证据，不再作为候选父 |
| Linear 侧历史结果 | L28（continuous_linear_l28-proj-vectorized） | 4611 / 286s | +4 vs L4；只作机制证据，不再作为并行父 |
| 历史侧隔离父 | Linear v166 / Attention v168 | 4590 / 226s；14005 / 210s | 只作历史机制证据，不再启动侧隔离计划 |
| Attention 历史正确性参考 | AC0（continuous_attention_ac0-correctness-hardened） | 14395 / 258s | 相对 R3（14405/238s）−10/+20s；只作历史证据 |
| Attention A29 实际机制实现 | a29-final-residual-s | TIMEOUT / >300s | 与 AC0 骨架分开；只关闭该高成本实现，不能把 AC0 的 14395 记为 A29 分数 |
| 用户确认的榜首锚点 | 源码、配置未知 | 21765 / 290s | 距当前工作父 3733 分，不是本地实验结果 |
| 用户确认的成功机制锚点 | A@W拟合 + Q/K互逆scale学习 | 21071 / 283s | 距当前工作父 3039 分；源码/配置/SHA待绑定，不替换根父 |

- 当前根 SHA256：`56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
  当前官方回传时间距硬限 19s；官方硬限为 300s；本地时间预测和提交时间门已退役。
- 活动计划及阶段只以[计划入口](docs/superpowers/plans/README.md)为准，不在此复制快照。
- 官方提交次数**无限制**；历史配额、剩余次数等表述全部失效。
- 用户已确认官方评测稳定；禁止为确定性、时间噪声或批处理研究重复提交相同 SHA 或逐位等价 A/B。
  不同算法间 ±1～4 分不证明随机噪声；保留实际裁决。
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
- Attention校准、选择、验证分离：只用calibration folds学参数，以独立holdout验证；多折固定聚合。
  Linear按用户最新指令直接在全部Qwen3.5-4B校准数据上做A@W低维拟合，不拆fit/select，
  不考虑泛化性；独立窗口Δmean/split/负向损失只记录、不否决探索。合法部署拟合改善、
  control、可达性、单文件和官方300s约束保留，正式晋级仍须官方结果。
- 必须记录 attempted/accepted 或同等计数证明机制可达；死分支 no-op 不能证明收敛或饱和。

## 4. 本地诊断与官方门禁

- 本地 `gain = 1 - MSE_PLAYER / MSE_STD`；STD 为同一 NVFP4 解码输入的标准 HiF4 输出。
  `overall_mean` 是实际 case 等权均值，不是准确率或官方总分，不拟合两侧权重。
- **禁止把本地分数换算官方分数**。本地只做机制否定、误差定位、符号/风险和同机成本诊断。
  本地与官方排序反转后，停止用该 proxy 为同一路线晋级。
- `Δmean`、L1 与分位数只作诊断。v188 已出现本地门通过但官方 −4，故它们不再作为提交或晋级门。
- 旧双侧持续计划的专项负向损失、`gain≥0.9`、误差账本和 side-score 队列已退役；当前按
  [单一完整方案计划](docs/superpowers/plans/2026-09-08-single-solution-optimization-plan.md)执行。
  Linear `calibration_fit_gain`、Attention 4B paired、holdout 和 L1 均只作诊断，不作官方候选排序或提交门。
  机制标签（如 A@W 拟合、校准统计拟合）本身也不阻止官方探索；晋级仍须官方分数、时间和源码 SHA 确认。
- OOD 已退役：eval-v3 的 `--ood` 标记为 retired，4B cache 未抓 OOD 窗口，当前不新增任何 OOD 运行。
  历史 `gap` / `|Δgap| > 0.01` 结论只用于读旧证据，不再是任何形式的门禁；仅被 OOD 拦截的历史候选
  记为“未获官方验证”，不证明机制无效，也不自动重跑。修订依据见
  `logs/execution/2026-09-07-ood-gate-policy-correction.md`。
- 当前测试统一遵循 [4B 面板测试指引](docs/4b-panel-testing-guide.md)：不新增 0.5B、OOD、
  跨模型 GPT-2/opt 或 fresh-default 计时运行。
- 官方时间唯一硬约束为 300s；所有本地时间公式、预测和门禁退役，api_seconds 只作记录与风险提示。
  本地开销可用于标注风险和安排降时优先级，但不得设置 `<280s`、按层外推或其他提交否决门。

## 5. 评测口径与执行流程

- 用户最新确认官方样例数为50 Linear +250 Attention，Linear拟合可约4400/5000；这是官方分项口径，
  不与含标准另一侧的侧隔离整包分数直接比较。case数不证明隐藏形状/权重，不修改本地协议冒充官方。
  依据与研究入口见 `logs/execution/2026-09-07-user-21071-mechanism-evidence.md`。

- 日常入口为 [`evaluator/eval.py`](evaluator/eval.py)（eval-v3），复用 proxy-v2 dense cache；
  [`evaluator/official_eval.py`](evaluator/official_eval.py) 是兼容/参考后端，输出不得混排。
  eval-v3 全六 shard 为 4B 的 336 Linear + 72 Attention；旧兼容 default 的 168 + 120 仅作历史口径。
- 当前本地使用 Qwen3.5-4B 结构代理，24 层含 18 DeltaNet 与 6 full-attention 层；
  面板、窗口与覆盖以 4B manifest 为准，不把本地结构当作官方隐藏输入。
- 只比较相同 evaluator、协议、cache、panel、device 的父子结果；2 折与 reeval5 的 5 折不混排。
  `official-shape-v1`、跨模型、compact/effect/replay/smoke/stress 均不能冒充 default 或官方结果。
- 旧 compact/full-cases/OOD 面板只作历史协议说明，不新增运行；当前使用 4B 目标侧 shard。
- 单侧运行严格隔离 API，保持共享 state 调用图，不能按 case 制造 oracle。评测只检查接口、
  合法性和真实形状复杂度，不替候选选算法或参数。

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
- 不覆盖原始 `artifacts/official_eval/*.json`、`logs/official_eval/*.md`、`logs/execution/*.md`；
  修正另写日志并更新状态。比较前检查 `evaluation_scope`。
- 实质代码或状态更新后运行 `git diff --check`、提交、push 并核验 `git status`；
  不提交 ignored 大 cache、`.codegraph/`、临时目录或无关改动。

## 7. 已关闭机制与证据边界

- 用户21071成功机制证据优先于历史整族饱和推断。A@W拟合、真正逐列非对称量化、Q/K互逆scale学习
  可按当前唯一总计划注册为“完整根 + 一个增量”的新机制；旧具体实现负结果不撤销，不重复同SHA/
  逐位等价提交。旧 L21/A21 工作包与本地符号例外全部退役，不再提供候选或豁免；合法性、control、
  单文件与官方 300s 硬限保留。

以下只保留禁止重试的索引，细节查[当前状态](docs/current-solution-status.md)、
[版本索引](solutions/README.md)和[计划入口](docs/superpowers/plans/README.md)。

- Linear：旧 full64/单折邻域、Householder 全族、cross-fold minimax 的 fold/Jacobi/coverage/role
  邻域、rank-3/残差系数/fold 扩展已关闭。首次 L3 死分支结果无效；修正可达后的负结果有效。
  LC1（rank-8 A@W 精化）与 LC2（整块 ±1 合法邻域）只关闭各自真实实现，不得推广为“根坐标已局部
  饱和”或“A@W 拟合族无效”——LC1 实际引入了 rank-8 求解器，LC2 未执行零值 sign flip；
  见 `logs/execution/2026-09-08-lc1-lc2-method-audit.md`。L-C3（objective-only，
  MSE_STD+numel fold 归一化）官方 `18031/293s`（相对根 −1 分 / +13s）已 REJECTED；LC0 的 `+3`
  在 L28 基线上取得、基线不同，不因 L-C3 撤销，见
  `logs/execution/2026-09-08-lc3-official-result.md`。
- Attention：per-call 动态 Gram/自适应精化族不缩 sweeps 重试；S2 前置条件不满足不启动。
  +4 scale 窗口、block-smooth refine 覆盖率、Jacobian 向 v186 移植及其收缩/clamp/gate 邻域关闭。
  v187 仅为 clean-room 研究父，不替代完整父。
  标准 Linear 侧隔离官方分（2026-09-09，基线＝标准 Linear + R3 `14405/238s`）已给出机制裁决：
  **v190（逐通道闭式互逆）侧分 `14405`，v191（稀疏三角搬运）侧分 `14405`，均与基线逐位相等 →
  机制在官方侧被证伪，关闭机制族**，不做窗口/块对/步长/token/chunk/clamp 邻域重试；
  **v192（32步全矩阵互逆残差）侧分 `14427`（+22）→ 机制有效，只关闭该计算实现的时间形态**，
  需先降时 ≥40s 才可能回收，不缩步数或减轮数重试。
- v194（A2/R3 校准等价去重）官方 `18032/285s`，相对根同分、慢 5s，已 `REJECTED_TIME`；本地
  calibration API −22.5% 不代表官方端到端提速，不再用其他 Linear 载体重复提交该等价实现。
  标准 Linear 侧隔离 `14405/234s`（侧隔离 −4s、完整包 +5s，方向相反）→ **等价提速路线不成立**，
  并由此确认：标准 Linear 侧隔离时间对完整根时间没有预测力，侧隔离只用于测分，
  任何基于它的时间推断都不得写入晋级或否决理由。
- v195（A2 K-center 多窗口梯度聚合修复）官方 `18053/289s`，相对提交时根 `+21/+9s`，已
  `RETAINED` 并切换为当前根；原始完整候选 SHA 为 `839ADB1E...761D7F`。
- v196（A2 原始 4+1 窗口全矩阵互逆残差）官方 `TIMEOUT（>300s）`，根保持 v195；不缩窗、
  不减步数重试。
- v197（Linear 64-block 标量增益 A@W 闭式拟合）官方 `17277/285s`，相对根 `−776/−4s`，
  已 REJECTED；本地 shard0 `−0.2077`（0/56/0）方向一致，疑似部署 block 对齐 bug，
  只关闭该实现，不证明低维 A@W 拟合机制无效。
- v198（GQA 组共享互逆对角：解析初始化+smooth-max+硬门控）官方 `TIMEOUT（>300s）`；
  本地 calibration 无超时信号仍官方超时。只关闭该实现，互逆 scale 机制无官方精度数据点。
- v199（GQA × 64-block hard reciprocal）官方 `TIMEOUT（>300s）`；只关闭该实现。
- v201（hard-logit residual weighted reciprocal）官方 `TIMEOUT（>300s）`；只关闭该实现。
- v202（Linear sample-energy calibration fusion）官方 `18053/281s`，与 v195 同分且快 8s，
  已 `RETAINED` 并切换为当前根；候选 SHA `56DC805D...EFCB2BD`。
- v203（联合 Q/K 合法 hierarchy 邻码选择）官方 `TIMEOUT（>300s）`；只关闭该实现。
- 时间余量事实：当前根 281s / 硬限 300s，余量19s；v190–v192、v196、v198、v199、v201、v203
  的新增 Attention 校准/选择实现均官方 TIMEOUT。后续候选仍须把新增校准成本控制在官方 300s 内，
  新增 Attention 算法先按活动计划与标准 Linear 配对取得官方侧分，再决定是否回装完整根；
  普通完整根候选在本地最终回退父状态时不提交。
- 缺口与速率事实（2026-09-09）：根 `18053`，距榜首锚点 `21765` 差 **3712 分**。按当前最优效率
  4.2 分/秒（v195：+21 分 / +5s）× 余量 19s，时间最多再换约 80 分；按 21 分/机制需约 180 个
  同量级机制。Attention 互逆/残差族八个变体官方结局为 `0/0/+22/TIMEOUT×5`，该族不再出卡。
  见 `logs/execution/2026-09-09-standard-linear-attention-side-scores.md`。
- V 侧不注册新候选：per-head 常量不改变块内解，per-channel multiplier 解码不逆缩放会破坏输出，
  五字段不支持 per-token 表；V-bias 等旧路径已裁决。2026-09-08 用户解禁 V 码分配类后探针裁决
  （anchor28-v-attribution）：当前码语义下 NVFP4→HiF4 重编码的码分配空间 100% 饱和——14 值 offset oracle +
  ratio 1.0 + 无块上限与部署基线逐位相同（六层 0.00% 增益），importance 实测无作用；码分配类同属
  NO_SUPPORTED_MECHANISM。该饱和不构成最终输出下界（0.275 只对应固定 P,V̂，QK-V 负交叉补偿可
  降低 V 项净贡献，见 logs/execution/2026-09-08-a28-interpretation-correction.md）；仅剩码语义变更
  （scale/lv 判据/格式映射，须同步改共享解码端）= A32，属全侧码格式卡，开设与否待用户决策。
- A4/L4/C1、旧 clean-room balance/gamma/refine 和历史负向局部扫描不重开。
  09-06 各块序、source-scale、JDRQ、联合坐标和合法编码实验的具体关闭边界以对应计划/日志为准，
  不把一个候选的失败扩写成所有新机制不可行。
- cb1/cb2 存在实现错误，其负结果不能证明合法编码空间饱和；见
  [证据审计](logs/execution/2026-09-05-next-plan-evidence-audit.md)。合法共享层级必须用五字段复核，
  operand MSE、单元素可表示值并集、非法 oracle 或连续残差不能证明最终输出天花板。
- 官方侧贡献比例不等于隐藏权重，也不能据此分摊榜首差距；零收益不能证明隐藏空桶。

## 8. 本文件维护方式

- 每条规则只保留一处当前有效表述；修订直接替换旧表述，不叠加“旧结论 + 更正”段落。
- 官方回传时只更新基线表和 SHA；计划进度维护在计划入口，逐版本分数及证据维护在状态/版本索引。
- 新增实验细节写入计划或执行日志；这里只增加会改变后续行为的约束及证据入口。
