# HiF4 竞赛执行规则

> 最后整理：2026-09-07。本文件只保留当前基线、长期约束和工作入口。
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

## 2. 当前基线与提交边界

| 用途 | 版本 | 官方分数 / 时间 | 说明 |
|---|---|---|---|
| 完整官方父、低成本候选起点 | v189 | 17616 / 275s | RETAINED，根 `solution.py` |
| 高复杂度新机制的时间预算父 | v180 | 17597 / 242s | 比 v189 快 33s、少 19 分 |
| 历史侧隔离父 | Linear v166 / Attention v168 | 4590 / 226s；14005 / 210s | 仅用于明确的侧隔离计划 |
| 用户确认的榜首锚点 | 源码、配置未知 | 21765 / 290s | 距 v189 4149 分，不是本地实验结果 |
| 用户确认的成功机制锚点 | A@W拟合 + Q/K互逆scale学习 | 21071 / 283s | 外部用户确认，源码/配置/SHA待绑定，不替换根父 |

- v189 官方计分 SHA256：`261202248A0146A2EE45F3DF60BD1979BB8171B7C162921013B0024C848617AF`。
  至榜首 290s 锚点的余量为 15s；官方硬限为 300s；本地时间预测和提交时间门已退役。
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
- Attention、Linear 分开改并检查未修改侧 control。侧隔离实验以 v162 standard 冻结非目标侧；
  两侧分别取得官方结果后才组合检查 interaction。完整父上的单侧增量按活动计划冻结另一侧。
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
- 通用符号/风险门为 `Δmean > 0 且 L1 < 0.02`，L1 是逐 case gain 平均绝对变化。
  v188 已出现通过门禁但官方 −4；门禁不保证官方非负，不再宣称“零误”。
- 两侧专项负向损失政策由 [持续优化计划](docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
  承接：总 L1 只记录，改用 `mean(max(-Δgain,0))<0.02`；独立验证/control不变，时间只以官方300s裁决；历史OOD按下条解释。
  仅适用于该计划两个工作包；通用分析器仍用总L1时，按计划另算专项指标，不能直接套用reject。
- OOD 只作风险诊断，不作提交或方向关闭的一票否决：`gap = gain_in - gain_ood`，Δ 为候选减直接父。
  `|Δgap| > 0.01` 仅提示收益不对称；负值表示 OOD 增益更多，正值不等于 OOD 实际退化。
  同时记录 ID/OOD 的 Δgain、负向 case 与最坏分组；父子须用各自同 SHA 的 in-dist/OOD 配对。
  OOD 实际退化也只记录风险，不单独禁止固定代表候选的官方探索；不得用旧父数值或遗漏 OOD 代替诊断。
- 区分探索提交与正式晋级：满足其他有效门禁的候选，不因 OOD 超阈值或校准拟合机制标签阻止官方验证，
  不需逐次申请 OOD 豁免；本地分析只提供建议。晋级仍须官方分数、时间和源码 SHA 确认。
  仅被 OOD 拦截的历史候选标记为“未获官方验证”，不证明机制无效；不自动重跑历史候选，
  已有官方负结果和其他有效关闭边界不变。修订依据见 `logs/execution/2026-09-07-ood-gate-policy-correction.md`。
- 当前测试统一遵循 [4B 面板测试指引](docs/4b-panel-testing-guide.md)：不新增 0.5B、
  逐候选 OOD、跨模型 GPT-2/opt 或 fresh-default 计时运行。历史 OOD 解释规则仅用于读历史证据。
- 官方时间唯一硬约束为 300s；所有本地时间公式、预测和门禁退役，api_seconds 只作记录与风险提示。

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
- 单侧运行严格隔离 API，保持共享 state 调用图，不能按 case 制造 oracle。官方 mini 只检查接口、
  合法性和真实形状复杂度，不选算法/参数。

1. 固定父版本及 immutable JSON/report；已有同口径结果不重跑，使用 `--reuse-existing` 零 API 重放。
2. 做目标侧最小 smoke，检查六 API、合法 state、有限输出、缓存及机制 reachability。
3. 按活动计划阶段运行目标侧 shard，候选用 `--baseline-solution`；精确匹配
   `(layer, role, test_window, split, length)`、`mse_standard`、`reference_energy`。
4. 记录 mean/median、q25/q75、worst-quartile、正负 case、validation/test 同号率、最坏分组、
   interaction、control 与 API 时间。Linear 看最终 Q(A)Q(W)^T；Attention 看 Q/K/V、QK/QKV、
   logits/probability 及最坏长度/层。旧单 operand 混合坐标诊断不等于同坐标纯量化误差。
5. 前置阶段通过后完成目标侧 4B 六 shard 与冻结侧 control；只有集成调用图检查运行完整双侧。
6. 保存 JSON 和 Markdown report，分别写 local proxy、API total、wall time、official 状态。
   接口/环境失败记 `ERROR`，机制否定记 `REJECTED`，官方明确超时记 `TIMEOUT`，
   官方未知记 `unregistered/NA`，不能填本地秒数。无实质算法/复杂度变化不分配版本号。
7. 正式版本归档前做一次脱离仓库单文件导入检查；按活动计划门禁裁决并记录官方回传。

命令模板（使用 CUDA venv，系统 Python 为 CPU-only；输出目录按本次运行命名）：

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution solution.py --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 7 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\<run-name>
```

Attention 改为 `--attention-only`；完整集成审计改为 `--scenario both`。

## 6. 缓存、证据与 Git

- `--nvfp4-cache-mode auto` 按 scenario/panel/profile 缓存 carrier/scale，只含 evaluator 输入，
  不含候选 state/输出。profile、协议、codec/mode、dense source identity 或数据 hash 不匹配时，
  `read` 拒绝、`auto` 重建；`write` 强制重建，`off` 禁用。
- `--cache-mode auto` 只在 dense cache 缺失时重新前向；`read` 禁止隐式捕获。
  命中仅减少输入准备时间，不改变 API 数量、误差或分数；本机 ignored cache 不作源码证据。
- 不覆盖原始 `artifacts/official_eval/*.json`、`logs/official_eval/*.md`、`logs/execution/*.md`；
  修正另写日志并更新状态。比较前检查 `evaluation_scope`。
- 实质代码或状态更新后运行 `git diff --check`、提交、push 并核验 `git status`；
  不提交 ignored 大 cache、`.codegraph/`、临时目录或无关改动。

## 7. 已关闭机制与证据边界

- 用户21071成功机制证据优先于历史整族饱和推断。A@W拟合、真正逐列非对称量化、Q/K互逆scale学习
  可按当前工作包注册新机制；旧具体实现负结果不撤销，不重复同SHA/逐位等价提交。
  当前工作包对L21-1/A21-1各一个固定代表候选开放本地符号例外的官方探索，负向损失、合法性、
  control、隔离及官方 300s 硬限保留；不得扩展成任意扫参或其他候选的通用豁免。

以下只保留禁止重试的索引，细节查[当前状态](docs/current-solution-status.md)、
[版本索引](solutions/README.md)和[计划入口](docs/superpowers/plans/README.md)。

- Linear：旧 full64/单折邻域、Householder 全族、cross-fold minimax 的 fold/Jacobi/coverage/role
  邻域、rank-3/残差系数/fold 扩展已关闭。首次 L3 死分支结果无效；修正可达后的负结果有效。
- Attention：per-call 动态 Gram/自适应精化族不缩 sweeps 重试；S2 前置条件不满足不启动。
  +4 scale 窗口、block-smooth refine 覆盖率、Jacobian 向 v186 移植及其收缩/clamp/gate 邻域关闭。
  v187 仅为 clean-room 研究父，不替代完整父。
- V 侧不注册新候选：per-head 常量不改变块内解，per-channel multiplier 解码不逆缩放会破坏输出，
  五字段不支持 per-token 表；V-bias 等旧路径已裁决。
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
