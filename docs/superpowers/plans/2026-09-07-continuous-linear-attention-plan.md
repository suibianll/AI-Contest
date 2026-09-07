# Linear / Attention 持续优化总计划

> ACTIVE / READY，2026-09-07。用户要求制定供其他代理持续执行的新计划。
> 本次仅完成计划交接，不启动代理、比赛提交或模型实验。
> 本计划取代 v162 独立恢复计划；两个工作包只有一个共同总契约，不另设活动总计划。

> **当前执行修订**：优先执行[21071成功机制驱动研究](workpackages/21071-evidence-driven-research.md)。
> 用户确认50 Linear/250 Attention、A@W拟合约4400/5000、Q/K互逆变换降低scale与拟合组合21071/283s。
> 旧证据修复中已完成闭环不重跑；当前L走真实逐列输出拟合，A从低成本R3替换为联合scale训练。
> 下面初始父与队列保留设计背景；当前父、优先级及有限官方探索规则以新工作包为准。

## 1. 起点与目标

目标：在六 API、合法 HiF4 编码与官方 300s 内，持续寻找超过现有官方结果的机制，最终挑战用户确认的 21765 锚点。不承诺达到，不将论文准确率或本地 gain 换算官方分数。

| 责任 | 起始官方父 | 官方分数/时间 | 冻结侧 | 工作包 |
|---|---|---|---|---|
| L 代理 | L4 恢复版 | 4607 / 247s | 全部 Attention 为 v162 standard | [Linear](workpackages/continuous-linear.md) |
| A 代理 | R2c 手工梯度栈内旋转 | 14387.8 / 240s | 两个 Linear API 为 v162 standard；V 固定为 R2c 的既有 V 路径 | [Attention](workpackages/continuous-attention.md) |
| 协调者 | 完整 v189 | 17616 / 275s | 单独维护组合与全局索引 | 本文件 §7 |

固定源文件与 SHA256：

- L：`solutions/v162_linear_l4-v189-linear-exact_officialNA_timeNA/solution.py`
  `ACB16F764DB80EDA94EB77FE497A7965C716B527571C241C79B2319529FF5263`。
- A：`solutions/v162_attention_r2c-rotation-in-stack-manual_oodblocked/solution.py`
  `CFDDCED7886A0CABC7251C57387536B86169185E7A6D8BE7A8647FEFAE677CD6`。
  目录名 oodblocked 为历史名称，当前官方状态取 manifest 的 OFFICIAL_PASS。
- 共同零点：`solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py`
  `56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`，1001 / 146s。
- 完整父 v189 SHA：`261202248A0146A2EE45F3DF60BD1979BB8171B7C162921013B0024C848617AF`。

这是从原 v162 两条独立分支持续推进，不是重新复制双标准算法。A 不恢复旧任务书的“V=v162”约束，也不注册新的 V 机制。

## 2. 文件所有权与共享资源

每侧分别使用 `workbench/continuous_linear/`、`workbench/continuous_attention/`：

- `state.json`：official_best、research_parent、pending_official、next_action、最近完成步骤；所有父绑定 SHA。
- `queue.md`：候选机制、证据、优先级、依赖与关闭边界。
- `<run_id>/mechanism.md`、`config.json`、`solution.py`、`manifest.json`、`report.md`。
- 结果：`artifacts/proxy_v3/continuous/<side>/<run_id>/{id,ood,control,timing}`。
- 日志：`logs/execution/continuous-<side>-<run_id>.md`，每轮独立文件，不覆盖历史。
- 官方候选：`solutions/continuous_<side>_<run_id>/`，编号由协调者分配。

代理只写本侧目录、日志和本侧工作包进度，不修改另一侧、根 solution、evaluator、reference、全局状态或 AGENTS。正式提交模块单文件自包含，禁止运行时引用外部候选。

开发和读结果可并行，CUDA/计时串行。继续使用已有公共锁路径 `artifacts/proxy_v3/v162-independent/gpu.lock`，不要另建一把无法互斥的锁。按旧 gpu_lock 的原子创建、PID、side、run_id、finally 释放协议；不要删除另一代理的锁。锁等待期间进行 CPU 代码审查和报告工作。

每轮仅 stage 自己文件，执行 diff-check、提交、push；不得 git add 全仓库。官方回传与组合只由协调者统一登记，避免两个代理写入冲突。

## 3. 持续循环：候选失败不等于研究结束

1. **恢复上下文**：读 AGENTS、过期清单、总计划、本侧 state/queue，再读必要源码和日志；检查已有匹配结果，不重跑父基线。
2. **登记机制**：只选一张机制卡；实现前固定数学假设、直接父、配置、预期改变的误差项、复杂度、验证与停止条件。论文名称不是新颖性证据。
3. **实现和最小验证**：六 API/合法 state、连续不变量、硬前向一致性、机制 reachability、冻结侧 control。异常记 ERROR，修复原因后允许同机制新 SHA，不把无效运行当负结果。
4. **局部诊断**：运行目标侧 shards 0,2；识别真实量化输出变化与误差来源。两 shard 小幅负向不自动关闭一个新机制；若无实现错误且能到达不同合法输出，完成固定六 shard 再裁决。恒等/no-op、结构违规或已确认不可行成本可提前停止。
5. **完整证据**：六 shard、独立 validation/test、成对 OOD、冻结侧 control、跨模型记录、fresh default 六 API 时间；按 §4 分类。
6. **归档和官方探索**：可提交者交协调者，状态 OFFICIAL_PENDING。官方回传后保留分数/时间 Pareto 候选；新官方最佳才替换 official_best。
7. **继续下一轮**：正向父可开展下一个不同机制；负向候选回到本侧官方最佳，换机制，不在失败配置周围扫参数。等待官方时做独立机制的去重、实现与小规模诊断，不把待回传源码悄悄升级成官方父。

每侧最多同时挂起一个新官方包，避免归因混乱；这不是官方提交配额。第二机制可在本地准备。混用多个尚无官方裁决的增量，必须显式记 research_parent，组合不能冒充单步官方收益。

每完成一个机制，将队列补充至最多三个有明确依据的独立后继机制。初始队列耗尽时，先整理剩余误差分组与已测空间，再检索原始论文，提出有实质新自由度/目标/求解规则的一张卡。不得用固定任务列表结束证明“没有剩余算法”。

若确实没有符合约束的新机制，记 NEEDS_NEW_HYPOTHESIS，并给出具体被阻断的选项与必要信息；资源不可用记 RESOURCE_BLOCKED。持续优化不是无限自动运行或无假设空转。本计划不创建定时任务；代理每次交接必须留下下一步可恢复动作。

## 4. 决策规则：探索与晋级分开

### 不可放宽的边界

合法编码、自包含六 API、完整 case 身份、冻结侧逐位 control、校准/holdout 隔离、无测试集选参、部署复杂度与 fresh default 预测 `<280s`。官方硬限 300s。计时缺失不能按零，单侧计时不证明完整组合可提交。

### 本计划局部风险判读

沿用用户已确认的双侧负向损失政策，扩展至本计划两工作包：总 L1 只记录；正常官方探索条件为直接父 Δmean>0、L1_negative=mean(max(-Δgain,0))<0.02，validation/test 两个 split mean 均正。累计 v162、初始强父与直接父分别报告，不因累计正收益大而拒绝。

OOD 必须配对记录 ID/OOD Δgain、Δgap、负向 case 和最坏分组，阈值只提示，不 veto；跨模型也只记录。20% 剩余误差降低、0.9 gain 只作进展标尺，不作继续研究或提交门。

**分析器适配检查**：现有 proxy-v3 analyzer 仍使用通用总 L1 门，不能将它的 `reject` 直接当本计划裁决。代理应在本侧报告从配对 case 显式计算上述专项指标，保留原分析器输出并解释差异，不私改共享 evaluator。

完整六 shard 不满足正常符号/负向损失门时，记 LOCAL_NEGATIVE，切换机制；该结果仅否定本次配置的本地效果，不扩写为全部数学方向无效。若认为这是 proxy 排序反转的新例外，提交包含具体证据和固定候选的探索申请给协调者；本计划不授权自动重开已关闭官方负结果或忽略其他硬边界。

### 官方裁决与成本

- 官方更高且时间合法：登记 SCORE_BEST；官方同分但更快，或分数稍低但明显更快：保留实际 Pareto 点，不能用“噪声带”合并分数。
- 官方负向且无时间优势：OFFICIAL_REJECTED，停止该实现及其参数邻域。
- WA/异常：CONTRACT_ERROR，查部署/状态/数值原因；修复有实质行为变化才新提交，不能重复同 SHA 排查确定性。
- TIMEOUT：关闭当前复杂度方案；仅允许有明确复杂度变化的新实现，不机械减少循环次数重试。
- 本地正向、时间不合格：TIME_HOLD，可研究数学等价的编译/融合；改动影响算法输出则单独登记。
- 本地通过仅代表探索准备完成；正式父晋级仍须官方结果、同源码 SHA、完整配对与时间证据。

## 5. 命令与复现口径

PowerShell 模板，先替换 parent/candidate/run-id，父固定为本轮登记 SHA，不指向可变根文件：

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution <candidate.py> --baseline-solution <parent.py> --linear-only --shards 0,2 --stop-after-nonpositive 7 --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --reuse-existing --output-dir artifacts\proxy_v3\continuous\linear\<run-id>\id
```

完整阶段改 shards 为 `0,1,2,3,4,5`，相同目录使用严格身份检查的 reuse；A 改 `--attention-only` 和目录侧名。`--stop-after-nonpositive 7` 避免通用“两 shard 非正”截断固定六 shard；不是绕过最终专项裁决。

OOD 使用同命令加 `--ood`，目录换为 ood。父/候选均须同 SHA 的 ID/OOD 配对，不能套用旧父 gap。计数必须完整：L 336、A 48；不将32-case screen与48-case full混排。

fresh default（168 Linear +120 Attention），独立 GPU 计时，候选校准必须实际执行：

```powershell
.venv\Scripts\python.exe evaluator/official_eval.py --solution <candidate.py> --name <run-id> --cache-mode read --nvfp4-cache-mode auto --algorithm-device cuda --output artifacts\proxy_v3\continuous\<side>\<run-id>\timing\default.json --report artifacts\proxy_v3\continuous\<side>\<run-id>\timing\default.md
```

这是兼容计时面板，不参与 eval-v3 排名。六 API 实测代入 AGENTS 时间模型；记录预测与后续官方残差。跨模型使用既有 cross_model_eval 入口，运行前读 --help 固定协议，仅描述鲁棒性。侧隔离的 control 单独调用非目标侧六 shard 对应真实输入；完整 both 只用于集成审计。

## 6. 每轮交付账本

至少包含：run_id、side、机制/配置、evaluator/父/候选 SHA、校准划分、cache/panel/device、case count；配对 mean/median/q25/q75/worst-quartile、正负零、L1_total/L1_negative、validation/test同号率、最坏长度/层/role；OOD和control、attempted/accepted/changed codes、state大小、六API时间、预测/官方时间、官方分数与状态、停止原因、next_action。

本地评价是实际最终输出误差，不以 operand MSE、连续变换无误差或单元素可表示集合冒充最终量化效果。父结果身份不符必须重建匹配基线，不跨协议拼接。

## 7. 协调者组合队列

当前立即可排队的集成验证为 L4 + R2c，不必等待两侧下一轮结束。本次制定计划不实际执行组合。

1. 合并单文件，检查 shared helper/常量无污染、两侧输出分别等于被组合侧包；单侧结果各自只归各侧。
2. 完整 both eval-v3、fresh default 六 API 计时、合法性和单文件隔离导入；满足门禁再提交完整包。
3. 记录 `S_additive=S_L+S_A−1001` 与 `interaction=S_combined−S_additive`。当前参考为 **17993.8**，不是官方组合成绩，不假定1分差异为噪声。
4. 官方组合成为可用 Pareto 点后才替换根父；单侧研究依然冻结对侧 standard。
5. 若完整组合超预算，先按六 API 定位；可将 v180 作为独立低成本完整对照，不静默替换 L4 或混用贡献账本。成本机制单独登记，不能把快父换入造成的变化归给新 Attention。

## 8. 当前进度与续跑入口

- L：READY / L21-1，复用真实API闭环，实现真正的逐列非对称量化；旧岭回归去重不证明算法等价。
- A：READY / A21-1，以R3 14405/238s为低成本研究父；A2 14440/274s保留为高分对照，优先联合scale训练替换旧训练。
- C：READY_FOR_INTEGRATION，L4+R2c 完整组合尚未验证。
- 每轮代理更新本侧 state/queue 与工作包进度；协调者收到官方回传后更新本表与全局状态，避免长期停留 DESIGN_ONLY。
