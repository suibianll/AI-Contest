# v162 零点下的 Linear / Attention 独立优化总计划

> 状态：**CLOSED / SUPERSEDED**。原日期：2026-09-06；2026-09-07 由持续优化计划取代。
> 两侧已获官方回传：L4 4607/247s、R2c 14387.8/240s；组合未验证。
> 下文保留原计划设计，不再提供下一步指令。当前入口见 [单一完整方案计划](../../plans/2026-09-08-single-solution-optimization-plan.md)。
> 用户要求：两个子代理分别优化 Linear 和 Attention，均从最初 v162 起步，分别比较效果。
> 本次只制定和交接计划，不启动子代理、不执行算法实验。根 v189 保持不变。
> 本文件是唯一活动总计划；下属两个工作包是可并行执行的独立任务，不是互相依赖的候选。

## 1. 任务分配与固定父版本

| 项目 | Linear 代理 L | Attention 代理 A |
|---|---|---|
| 工作包 | [Linear 任务书](../../plans/workpackages/v162-linear.md) | [Attention 详细计划](../../plans/workpackages/v162-attention.md) |
| 初始源码 | v162 原始单文件 | 同一份 v162 原始单文件 |
| 允许修改 | weight calibration、dynamic activation 及其专用 helper | attention calibration、dynamic Q/K 及其专用 helper |
| 必须冻结 | 全部 Attention API = v162 | 两个 Linear API = v162；本轮 V = v162 |
| 工作目录 | `workbench/v162_linear/` | `workbench/v162_attention/` |
| 结果目录 | `artifacts/proxy_v3/v162-independent/linear/` | `artifacts/proxy_v3/v162-independent/attention/` |
| 执行日志 | `logs/execution/v162-independent-linear.md` | `logs/execution/v162-independent-attention.md` |
| 正式归档前缀 | `solutions/v162_linear_.../` | `solutions/v162_attention_.../` |
| GPU 使用 | 与 A 串行占用 | 与 L 串行占用 |

唯一共同零点文件：
`solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py`。
SHA256：`56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`。
官方结果 **1001 / 146s**。其六 API 都是标准 HiF4，state 为空；兼容 default 两侧 gain 均为 0。

“从 v162 起步”是实际代码与非目标侧语义要求，不是仅在报告中减去 1001。
不能直接复制 v189/v180 的完整优化栈充当新分支。允许审读历史代码，明确机制来源后单独迁入；
已知收益的重现记 RECOVERY，不记新突破。分支内保留 `baseline=v162`、`parent=本侧上一版`
两个指针，后续仍沿用同一 v162 零点，不能静默切换到完整父。

只读强对照也固定身份，不随根文件移动：

- L：`solutions/20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA/solution.py`，
  SHA `9C0EAC6A7CA883A1F8962C11735744271259460F5EBBF23D530A5BBCF12B4646`；
- A：`solutions/20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA/solution.py`，
  SHA `5988AE47EAC2E7DDE7488E06B8F91939F5660A585034280A6D68A8FB6701AC79`；
- 本计划所称“最新完整父强对照”固定为
  `solutions/20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/solution.py`，
  SHA `261202248A0146A2EE45F3DF60BD1979BB8171B7C162921013B0024C848617AF`。
  只调用目标侧读取其性能，不能把其另一侧混入候选或侧隔离官方包。

## 2. 两代理互不干扰的执行约定

- 只写自己目录、自己的任务日志与工作包状态；不改根 solution、AGENTS、全局索引、另一侧文件。
- 不共享可写 helper，不修改 evaluator/reference/cache；v162 的共享 codec 原样保留。
  需要新编码器时新建目标侧专用函数，避免悄悄影响标准 control。
- 提交文件仍必须单文件、自包含六 API，不允许 import 另一候选文件作为实现。
- 两个代理可以同时读代码、开发、分析既有 JSON；CUDA 评测和 fresh 计时必须排队。
  使用 `artifacts/proxy_v3/v162-independent/gpu.lock` 原子独占创建（Python `open(path,'x')`），
  写入 side、PID、run_id、开始时间；释放用 finally。锁存在则等候，不删除对方锁。
  进程崩溃后的锁由协调者核验 PID 后处理；锁属于 ignored 运行产物，不提交 Git。
- 新建 run 目录，不覆盖旧结果；父结果身份完全匹配才复用。baseline 按侧生成一次各自保存，
  不让两代理同时写一份 baseline JSON。
- Git 只 add 本侧文件；全局指针、正式版本号、根替换和最终组合由协调者负责，避免并发编号。
- 用户自行委托两个代理；本计划不是启动后台任务或自动提交官方的通知机制。

## 3. 统一比较账本：累计收益、单步收益、强对照分开

### 3.1 本地

每个 case 的 `g=1-MSE_candidate/MSE_STD`。必须匹配 cache/panel/device/evaluator SHA、
`(layer,role,test_window,split,length)`、mse_standard 和 reference_energy。

每个候选同时报告：

1. **累计效果**：`Δ0 = g_candidate - g_v162`，v162 对齐后应为 0。
2. **本步效果**：`Δstep = g_candidate - g_direct_parent`。
3. **历史强对照差**：同协议本侧对 v166（L）或 v168（A）及最新侧隔离重建对照的差。
   历史强对照不作为代码父，不允许沿用不同 panel 的旧均值直接相减。

主比较只看目标侧；不能用 168+120 的 Overall 或另一侧产生的历史最高 Overall 否决本侧。
两侧各自展示 mean/median/q25/q75、最差四分位、正负零数、分层/role/长度和 validation/test。
L 的 gain 与 A 的 gain 可并列表述，但不能直接当成同一官方分值或排名贡献比较。

平均标准化剩余误差 `r=mean(MSE/MSE_STD)=1-mean(g)`。
相对父的削减 `D=(r_parent-r_candidate)/r_parent`；r_parent=0 时单独记录，不除零。
从 v162 达到 D=20% 只说明优于标准 20%，不说明超过已有 Attention 栈；
“突破现有水平”要求相对强对照也有材料收益，研究目标固定为 **D_strong≥20%**。
20% 是资源投入目标，不是官方分数预测或已有实验事实。

### 3.2 官方

| 版本 | Linear | Attention | 官方总分 | 相对 v162 的贡献 |
|---|---|---|---|---|
| B0=v162 | standard | standard | 1001 | 0 |
| L_i | 优化 L_i | standard | S_Li | C_Li=S_Li−1001 |
| A_j | standard | 优化 A_j | S_Aj | C_Aj=S_Aj−1001 |

各侧单步 `step_gain=S_child-S_same_side_parent`。每项保存实际提交 SHA、官方分数/时间，
未知写 NA。v166 的 C_L=3589、v168 的 C_A=13004 是历史已测锚，不能认作新实验结果。
v162 不为重新确定零点而重复官方提交。

两侧均有独立官方正裁决后，协调者才组合：
`S_additive=S_L+S_A−1001`，`interaction=S_combined−S_additive`。
这只是待检验的可加预测；组合时间不相加，必须完整测量。根 v189 只有在组合官方结果
形成可用的新 Pareto 点后才替换。

## 4. 本次门禁（用户已确认）

**已确认：**合法性、finite、完整 case 身份、未修改侧逐位 control、校准/holdout 隔离、
官方预测 `<280s`、官方硬限 `<300s`、单配置和禁止邻域扫描均继续执行。

**APPROVED（2026-09-06 用户明确选择“本次两侧计划采用负向损失门”）。**
以下专项规则取代本次两侧任务的 `L1_total<0.02`，适用于研究和提交门；不自动修改其他计划。
以 v162 为父，大幅正收益也会违反旧 L1 门，因此本轮将收益与退化分开计算：

- `L1_total=mean(abs(Δ))` 仅记录；
- `L1_negative=mean(max(-Δ,0))<0.02`；
- mean Δ>0，分别报告 median/尾部；独立 validation/test 两个 split mean 均为正；
- 上述分别对 v162 和直接父计算；对已有强对照的进展单独标记，不偷换 baseline；
- OOD 按候选/直接父同 SHA 配对记录 ID/OOD Δgain、Δgap、负向 case 和最坏分组；
  `|Δgap|>0.01` 仅提示收益不对称，不禁止官方探索，不以此关闭机制（2026-09-07 用户授权修订）；
- 官方结论仍以实际分数为准，不宣称上述规则能保证非负。

不需要再次询问这项已确认政策或逐次申请 OOD 豁免。校准/验证隔离、control、合法性和时间门仍须通过；OOD 必须记录但不作否决门。
探索提交允许带着明确的 OOD 风险获取官方证据，正式晋级仍以官方分数、时间和源码 SHA 为准。
负向损失很小也不能保证官方正向；官方未知写 NA，不将本地通过写成已晋级。

## 5. 评测与时间命令契约

固定 cache：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`。
本地环境 `.venv/Scripts/python.exe`，设备 CUDA。manifest 记录输入 hash、源码/配置/评测器 SHA、
GPU/torch、实际命令、父指针、预期/实际 case 数、错误数和 gate 结论。

```powershell
# 从仓库根执行。下面变量在每侧 run manifest 中保存，不能指向根 solution.py。
$zeroPath = 'solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py'
$attentionPath = 'workbench/v162_attention/candidate/solution.py'
$linearPath = 'workbench/v162_linear/candidate/solution.py'

# A 代理：48 Attention cases；第一次从 v162 比较，后续另与直接父做零 API 配对。
.venv/Scripts/python.exe evaluator/eval.py --baseline-solution $zeroPath --solution $attentionPath --name v162-attention --attention-only --shards 0,1,2,3,4,5 --cache artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt --algorithm-device cuda --calibration-cache-mode auto --stop-after-nonpositive 7 --reuse-existing --output-dir artifacts/proxy_v3/v162-independent/attention/id
# L 代理：336 Linear cases；不调用 Attention 作为本侧训练或评测成本。
.venv/Scripts/python.exe evaluator/eval.py --baseline-solution $zeroPath --solution $linearPath --name v162-linear --linear-only --shards 0,1,2,3,4,5 --cache artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt --algorithm-device cuda --calibration-cache-mode auto --stop-after-nonpositive 7 --reuse-existing --output-dir artifacts/proxy_v3/v162-independent/linear/id
```

以上 candidate 路径是待实现产物，不表示文件已存在。后续 run 必须各加版本子目录，不能
沿用该路径覆盖上一版报告。OOD 用同命令加 `--ood`、独立目录，不混入 ID 排名。
control 验证单独调用冻结侧的真实 API 比较五字段/输出/state；不得只检查函数源码一致。
必须覆盖六 shard 实际 case 以及不同调用顺序，确认没有全局 mutable state 污染。

提交前各侧都用其完整六 API 单文件、标准另一侧，单独运行兼容后端 fresh default：
`evaluator/official_eval.py --solution <该侧候选> --cache-mode read --nvfp4-cache-mode auto
--algorithm-device cuda --output <独立json> --report <独立md>`。
不得加 side-only/compact/effect/full-cases/OOD；168+120 default 用于本侧补充验证和完整 API 计时。
时间模型：`170.3+0.115*W_calib+0.694*A_calib+0.734*dyn_act−1.58*dyn_qkv`；
缺失不填零，不用并发 GPU/shard/cache-hit/研究训练耗时直接代入，不将负系数解释为加速。
校准学习必须包含在候选 calibration 的实测内，不把训练移到提交前外部服务规避计时。

## 6. 阶段交接与完成条件

每侧状态只写自己日志，统一字段：`side/run_id/baseline_sha/parent_sha/candidate_sha/
mechanism/config_sha/panel/focus_gain/step_gain/strong_control_gap/L1_total/L1_negative/
OOD_gap/control/coverage/reachability/api_times/time_prediction/official_score/official_time/status`。
研究训练步数、样本数、可训练参数数、全状态字节数必须同时记录。

每侧交付独立 `result.json`、`report.md`、`solution.py`、`config.json`、`manifest.json`，
以及说明“相对标准进步多少、相对已有最佳是否进步、停止在何门”的一段结论。
两侧相互不等待科研结果；仅共享 GPU 排队和最终组合步骤。

此前 v189 残差压力块序计划由本总计划取代为当前研究入口。旧运行由原执行者封存，
不删产物、不把其结果算入 v162 新分支、不根据新 baseline 改写历史裁决。
本计划不承诺达到榜首，失败也不能推导整个合法 HiF4 空间已经耗尽。
