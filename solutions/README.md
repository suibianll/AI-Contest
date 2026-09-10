# HiF4 solutions archive

> 当前完整根：[v237 Linear L-TF2](20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/result.md)，**18518/289s，RETAINED 并晋级**（2026-09-10 用户回传），SHA `ecB1F9E5…` 全值 `ECB1F9E510B5507E1A2DC8B95F8A84E9B51864A420B3828537A328813E2CE554`；相对父 v231 根**同分 / −2s**，余量 9s→**11s**。晋级依据：计划 §2"同分更快且 <300s"+ AGENTS §2 v202 先例（分数不低于当前根）。**本地四行配对效应全部落在同字节 sham null 内、官方仍量出 −2s**——见[官方结果记录](../logs/execution/2026-09-10-v237-linear-official-result.md)。v231 退为历史根（18518/291s）。
> 回退根：[v233 Linear L-TF1](20260910_v233_linear-tf1-gradient-reuse_scoreNA_timeNA/result.md)，**18428/288s**，SHA `0EC89710087D061BF9608196AD4D53A1C6BE98C8A5595596A071ECD05A6821EB`；相对 v230 同分快 4s，严格占优。**L-TF1 尚未并入当前根 v231。**
> 官方已回传：[v235 Linear L-AD1](20260910_v235_linear-ad1-adaround-materialization_scoreNA_timeNA/result.md)，`fe8aec19…`，自 v231 根纯追加；官方 **TIMEOUT（>300s）REJECTED**，见[超时记录](../logs/execution/2026-09-10-v235-linear-official-timeout.md)。**是 v231 的后代（含 K=2），与 v233 不可相加。**
> **未提交**：[v239 Attention A-CT2](20260910_v239_attention-act2-train-tail-reuse_rejected_scoreNA_timeNA/result.md)，`55103e8b…`，训练尾部统计复用（204→198 次调用，同父 A-GR1 六层 state/72 例相同）。**官方 `unregistered/NA`——本卡从未提交**：用户据 **v238 的官方 TIMEOUT**（同族、输出逐位等价、且去除的工作严格更多）判定不花提交名额。**这不是被本地读数否决**——依据是官方结果，不是本地墙钟。归档目录名带 `_rejected` 是用户对"决定不提交"的标注，以归档内 `official-result.json` 的 `unregistered/NA` 为准。
> 官方待回传：[v240 Linear L-MC1](20260910_v240_linear-lmc1-compiled-metric_scoreNA_timeNA/result.md)，`2c344722…`，把固定度量重建从动态路径移入校准（两个追加影子；G 在校准端算出并存入、动态端只加载）。对 v237 根 **336/336 精确零**。**它是搬运而非消除工作**：校准 +50.0 ms/次、动态 −28.3 ms/次，**平衡点 ≈255 次动态调用**；官方调用次数是缺失事实（本地净省约 0.95 s）。**关键发现：stride 也是等价性的一部分**——`cholesky_inverse` 返回转置 stride，`.contiguous()` 会以"前三个字段全对、只有离散码不同"的形态破坏等价。官方 `unregistered/NA`，不写秒数预测。是 v237 的后代（含 L-TF2），与 v233/v235/v238/v239 不可相加。
> 官方已回传：[v238 Attention A-CT1](20260910_v238_attention-act1-gate-reuse_rejected_scoreNA_timeNA/result.md)，`146bb715…`，A-GR1 gate 降本版；官方 **TIMEOUT（>300s）REJECTED**（2026-09-10 用户回传），见[超时记录](../logs/execution/2026-09-10-v238-attention-act1-official-timeout.md)。**它不是 v236 的重复提交**：输出与 v236 逐位等价、gate 实减工作（前向 4→3 / 参考解码 12→8 / V 量化 2→1），测得的约 0.2 s 没把包拉进 300 s——**去重方向由官方结果关闭**。是 v231 的后代（旧的旧），与 v237/v239 不可相加。
> 官方已回传：[v236 Attention A-GR1-on-v231](20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/result.md)，`3319fc35…`，把已官方定价 +29 的 A-GR1 原样重挂到 v231 根（**唯一改动是父替换**）；六shard 复现 v234 每一项（`+0.003845`，21/3/48）；官方 **TIMEOUT（>300s）REJECTED**，见[超时记录](../logs/execution/2026-09-10-v236-agr1-on-v231-official-timeout.md)。**这是 A-GR1 完整包形态的第二次官方否决**：v234 父余量 8s、v236 父余量 9s，父差仅 1s，故本次把 v234 留下的唯一未知量"超限幅度是否 ≤1s"答为**否**——**不存在"换个 Linear 父就能过"的空间**。侧隔离 +29 不受影响，**A-GR1 机制本身不关闭**。**是 v231 的后代（含 K=2），与 v233/v235/v237/v238 不可相加。**
> 以下旧根晋级描述为历史事实，不覆盖当前根。
> L28 `4611/286s` 与其他侧结果降为历史机制证据，不再形成并行父线。
> [L28 回传记录](../logs/execution/2026-09-08-l28-official-result.md)、
> [L23b 超时记录](../logs/execution/2026-09-08-l23b-official-timeout.md)。

> 组合候选 [v202](20260909_v202_linear-sample-energy-fusion_scoreNA_timeNA/result.md)
> 已获用户官方回传 `18053/281s` 并提升为当前根：在 v195 的 Linear 校准中融合 sample-energy
> block order 编译，输出等价且官方时间快 8 秒。

> 当前测试按[4B指引](../docs/4b-panel-testing-guide.md)执行。本文历史0.5B、OOD、跨模型和时间预测结果仅作证据，不构成新测试命令或门禁。

> 上一完整父 v189 为 `17616/275s`。当前执行只以根完整方案为父，官方前仅做六 API smoke 与
> 目标侧 shard0；完整六 shard 只在官方正向后归档或为明确失败诊断运行。

## 2026-09-08 当前单一完整方案候选

以下候选均从当前根独立构建；目录名中的 `scoreNA_timeNA` 是归档时状态，实际官方回传以表格和各自
`result.md` 为准。

> **版本号冲突说明（2026-09-09）：** `v204` 与 `v205` 各有两个并行候选——减法定价系与 AW 拟合系。
> 这是已经发生的历史编号冲突；为保持归档路径稳定，**不重命名目录**。引用时必须写全目录名或注明机制，
> 不得只写 `v204`/`v205`。对应关系：
> `20260909_v204_linear-no-rank2-residual` / `20260909_v205_attn-no-c764-rotation-search` 属**减法定价系**；
> `20260909_v204_linear-aw1-deployed-coordinate_rejected` / `20260909_v205_linear-aw2-hierarchy-gain_rejected` 属 **AW 拟合系**。
> 后续版本继续遵守本文末尾“版本标识全局唯一”，不得再次复用编号。

| 版本 | 机制 | 当前结果 |
|---|---|---|
| [v190](20260908_v190_attn-diag-reciprocal-balance_scoreNA_timeNA/result.md) | diag reciprocal balance | 官方 `TIMEOUT`（`>300s`）；本地 gate 拒绝、回退父 |
| [v191](20260908_v191_attn-block-triangular-transport_scoreNA_timeNA/result.md) | block triangular transport | 官方 `TIMEOUT`（`>300s`）；本地 gate 接受、shard0 delta `-0.0001040638` |
| [v192](20260908_v192_attn-full-reciprocal-residual_scoreNA_timeNA/result.md) | full reciprocal residual | 官方 `TIMEOUT`（`>300s`）；本地 gate 拒绝、回退父、shard0 delta `0` |
| [v193](20260908_v193_attn-joint-qk-product_scoreNA_timeNA/result.md) | joint Q/K product | gate 拒绝，回退父；shard0 delta `0` |
| [v194](20260908_v194_attn-a2-calibration-fused_scoreNA_timeNA/result.md) | A2/R3 校准等价提速 | 官方 `18032/285s`：同分、比当前根慢 5s，`REJECTED_TIME`；本地 calibration API −22.5% 未转化为官方提速 |
| [v195](20260908_v195_attn-a2-center-gradient-aggregate_scoreNA_timeNA/result.md) | K-center 梯度聚合修复 | shard0 delta mean `+0.001935`（6/6/0），非 no-op；官方 `18053/289s`，`RETAINED`，已切换根 |
| [v196](20260908_v196_attn-reciprocal-residual-original-split_scoreNA_timeNA/result.md) | Q/K 互逆残差原始 4+1 配置 | shard0 delta `0`（gate 全拒绝回退父，calibration +37.7%）；官方 `TIMEOUT`（`>300s`），根不变 |
| [v197](20260909_v197_linear-aw1-block-gain_scoreNA_timeNA/result.md) | 64-block 标量增益 A@W 闭式拟合 | 官方 `17277/285s`（相对根 −776/−4s），REJECTED；本地 shard0 −0.2077（0/56/0）方向一致 |
| [v198](20260909_v198_attn-gqa-reciprocal-diag_scoreNA_timeNA/result.md) | GQA 组共享互逆对角（解析初始化+smooth-max+硬门控，冻结 V） | 官方 `TIMEOUT`（`>300s`）；本地 shard0 delta mean `−0.001693`（6/6/0），无精度数据点 |
| [v204（减法定价）](20260909_v204_linear-no-rank2-residual_scoreNA_timeNA/result.md) | 减法定价：关 L-R2 rank-2 残差段 | shard0 delta mean `+0.000134`（30/26/0）；官方 `18053/286s`（相对根 `0/+5s` 噪声）→ rank-2 段官方定价 **0 分**，机制关闭 |
| [v205（减法定价）](20260909_v205_attn-no-c764-rotation-search_scoreNA_timeNA/result.md) | 减法定价：关 C76.4 H16/H32 旋转搜索（Attention 校准 ~30%） | shard0 delta `0`；官方 `17969/275s`（相对根 `−84/−6s`）→ C76.4 官方定价 **+84 分 / ~6s**（14 分/秒，最高效机制），必须保留 |

## 2026-09-09 持续优化候选

| 版本 | 机制 | 当前结果 |
|---|---|---|
| [v199](20260909_v199_attn-gqa-hard-reciprocal_scoreNA_timeNA/result.md) | GQA × 64-block hard reciprocal，真实 hard-output 选择 | 4/6 层产生接受状态，但六 shard 代理 `−0.0000729515`；`REJECTED`，官方 `TIMEOUT` |
| [v201](20260909_v201_attn-hard-logit-residual_scoreNA_timeNA/result.md) | hard-logit residual + softmax Jacobian/V 加权候选排序 | 4/6 层产生接受状态，但六 shard 代理 `−0.0001206117`；`REJECTED`，官方 `TIMEOUT` |
| [v202](20260909_v202_linear-sample-energy-fusion_scoreNA_timeNA/result.md) | Linear sample-energy 编译与首次校准解码融合 | 336 case 逐位等价；官方 `18053/281s`，同分快 8s，`RETAINED` 并切换根 |
| [v203](20260909_v203_attn-legal-hierarchy-selection_scoreNA_timeNA/result.md) | 联合 Q/K 合法 hierarchy 邻码 hard-output 选择 | 仅 shard5 变化且 `−0.0065789294`，总代理 `−0.0010964882`；`REJECTED`，官方 `TIMEOUT` |
| [v204（AW1）](20260909_v204_linear-aw1-deployed-coordinate_rejected_scoreNA_timeNA/result.md) | 部署坐标对齐的 64-block 标量 A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v205（AW2）](20260909_v205_linear-aw2-hierarchy-gain_rejected_scoreNA_timeNA/result.md) | HiF4 8 元素层级组标量 A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v206](20260909_v206_linear-aw3-output-group-gain_rejected_scoreNA_timeNA/result.md) | 输出组 × 64-block 标量 A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v207](20260909_v207_linear-aw4-fine-group-gain_rejected_scoreNA_timeNA/result.md) | HiF4 4 元素细粒度组标量 A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v208](20260909_v208_linear-aw5-output-fine-group-gain_rejected_scoreNA_timeNA/result.md) | 输出行 8-group × 64-block 标量 A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v209](20260909_v209_linear-aw6-broadcast-additive_rejected_scoreNA_timeNA/result.md) | 4 元素组广播 additive A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v210](20260909_v210_linear-aw7-row-local-additive_rejected_scoreNA_timeNA/result.md) | 按输出行独立的 4 元素组 additive A@W 拟合 | 六 shard 336 case 与 v202 逐位相同、无 hard-output 增益；`REJECTED`，官方 `unregistered/NA` |
| [v211](20260909_v211_linear-aw8-output-code-group_rejected_scoreNA_timeNA/result.md) | 冻结 Q(A) 的输出感知 4 码组联合更新 | shard0 `-0.0171391319`（3/53/0），真实翻码但整体回退；`REJECTED`，官方 `unregistered/NA` |
| [v212](20260909_v212_linear-aw9-shared-code-offset_rejected_scoreNA_timeNA/result.md) | 64-block 内单组共享整数 signed-mantissa 偏移 | shard0 `0`（0/0/56），无 hard-output 变化且校准约 5.0×；`REJECTED`，官方 `unregistered/NA` |
| [v213](20260909_v213_linear-aw10-hierarchy-toggle_rejected_scoreNA_timeNA/result.md) | 输出感知 per-group `lv3=1↔2` 合法层级切换 | shard0 `0`（0/0/56），无 hard-output 变化且校准高于父级；`REJECTED`，官方 `unregistered/NA` |
| [v214](20260909_v214_linear-aw11-lv2-toggle_rejected_scoreNA_timeNA/result.md) | 输出感知 per-group `lv2=1↔2` 合法层级切换 | shard0 `0`（0/0/56），无 hard-output 变化且校准高于父级；`REJECTED`，官方 `unregistered/NA` |
| [v215](20260909_v215_linear-aw12-scale-step_rejected_scoreNA_timeNA/result.md) | 输出感知 E6M2 `scale_factor` 相邻码步进 | shard0 `-0.150813`（0/56/0），全 case 回归；`REJECTED`，官方 `unregistered/NA` |
| [v216](20260909_v216_linear-lt2-fixed-order_rejected_scoreNA_timeNA/result.md) | 运行时使用 calibration-compiled activation GPTQ order | shard0 `-0.003884`（13/43/0）；`REJECTED`，官方 `unregistered/NA` |

| [v217](20260909_v217_attention-a5-head-scale_rejected_scoreNA_timeNA/result.md) | A1 输出选择中的单一固定 reciprocal Q/K temperature `1.25` | shard0 `0`（0/0/12），无 hard-output 变化；`REJECTED`，官方 `unregistered/NA` |

| [v218](20260909_v218_attention-c76-1-qonly-range_rejected_scoreNA_timeNA/result.md) | C76.1 固定 Q-only headwise range permutation | shard0 `0`（0/0/12），无 hard-output 变化；`REJECTED`，官方 `unregistered/NA` |
| [v219](20260909_v219_attention-c76-2-fisher_rejected_scoreNA_timeNA/result.md) | C76.2 固定 output-Fisher Q/K importance，blend `0.5` | shard0 `0`（0/0/12），无 hard-output 变化；`REJECTED`，官方 `unregistered/NA` |
| [v220](20260909_v220_linear-aw13-zero-sign_rejected_scoreNA_timeNA/result.md) | L-AW13 输出感知零值到最小有符号码插入 | shard0 `-0.000051551`（14/42/0），实际翻码但整体回归；`REJECTED`，官方 `unregistered/NA` |
| [v221](20260909_v221_linear-aw14-global-residual-basis_rejected_scoreNA_timeNA/result.md) | L-AW14 共享 rank-8 输出残差基 A@W 拟合 | shard0 `-0.032568`（0/56/0），全部回归；`REJECTED`，官方 `unregistered/NA` |
| [v222](20260909_v222_attention-a2-correctness-fix_scoreNA_timeNA/result.md) | FIX-A2：A2 mean-gradient + 异常传播修复（无新算法） | 官方 `18015/293s`（相对根 `−38/+12s`），`REJECTED`，根不变；本地 shard0/shard1 负向方向一致 |
| [v223](20260909_v223_attention-ah1-threshold-events_scoreNA_timeNA/result.md) | A-H1 量化阈值事件搜索（Cayley 路径 8 事件槽 hard-output 选择） | 六 shard 72 case mean `+0.003209`、median 0（+28/−32/0=12），5/6 层接受事件、可达非等价；3/5 接受层 holdout 转劣；事件路径实际从最后一步 Adam 更新前快照出发，不能作为“部署父状态最近阈值”的有效裁决；官方 `TIMEOUT(>300s)`，只关闭该实现，归档源码不修改，修正见 R1/v224 |
| [v224](20260909_v224_attention-ah1r-parent-anchored_scoreNA_timeNA/result.md) | A-H1R 部署父状态锚定阈值事件（切空间 `S=skew(R_parent^T G_R)`，center 固定） | 6/6 层合法接受、`t0_identical=1`、changed-code 小范围可解释（不再是整步回退）；六 shard 等权 `≈+1.2e-6`（单 shard \|delta\|≤8e-6），hard-output 效应在噪声底；移除/旁路四处宽 except 并加动态应用断言；官方 `TIMEOUT(>300s)`，只关闭该实现；与 v223 同成本类，重试前须先降校准成本 |
| [v225](20260909_v225_attention-ah3-gqa-local-hard-event_scoreNA_timeNA/result.md) | A-H3 GQA-group 局部切空间第一事件（正负双向，顺序接受） | 6/6 层有 group 接受（共 18/24）、`t0_identical=1`；六 shard 等权 `+3.11e-5`（基线 `0.533998`→候选 `0.534029`），**增益集中在 shard5 `+1.96e-4`**，其余 ±7e-6 近零/微负；默认早停 2 会在 shard3 漏掉 shard5；官方 `TIMEOUT(>300s)`，只关闭该实现；与 v223/v224 同成本类，重试前须先降校准成本 |
| [v226](20260909_v226_linear-lrb1-residual-rounding_rejected_scoreNA_timeNA/result.md) | L-RB1 静态权重输出残差共享舍入边界（12 个 `(sign, lower-code)` 共享阈值） | 六 shard 全部 `reject`，等权均值 `-4.10e-4`（33/95/208）；168 次 calibration 中 65 层接受、改动 10.38M 个 mantissa 码，但接受层校准 `ΔL` 合计仅 `-2.52e-5`（每层 ≈5e-7），103 次回退全部因 `ΔL ≥ 0`；接受层集中在宽形态，属校准窗口过拟合；本地负向，按 v219/v220/v221 实践归档，未提交官方 |
| [v227](20260910_v227_attention-ag1-joint-affine-gauge_rejected_scoreNA_timeNA/result.md) | A-G1 Q/K 联合仿射 gauge（A2 循环内新增每 KV group 零均值 reciprocal log-scale `s[head_dim]`，与 rotation/K-center 共用同一 Adam） | 六 shard 等权 `-0.005294`（28/32/12），shard2 逐位不变，shard1 `-0.021423` 最差；机制可达（60/72 case 硬输出改变，s=0 逐位恢复父）但净负，误差集中在短序列与 test split；`REJECTED`，按计划 §7 不缩步/缩窗/拆粒度重试，未提交官方，根保持 R0 |
| [v228](20260910_v228_attention-aqb1-q-bias_rejected_scoreNA_timeNA/result.md) | A-QB1 Q 侧加性 logit 偏置（冻结根全部已有 state，每层每 Q head 学 `b_q[head_dim]`，全 5 folds 窗口等权真实部署 MSE 上 Adam 32 步，全 folds 逐层 gate 严格改善才写入） | 六 shard 等权 `-0.053177`（3/69/0），六层全负，shard3（层15）最差 `-0.105430`；机制可达且 gate 真实接受（control 8/8 种子接受、改善 1.9%–2.7%）但全 folds 训练+全 folds gate 仍强负——Q 偏置拟合到的"系统性 logit 偏差"是校准窗口特异而非量化器固有属性；逐通道/逐元素级校准拟合自由度 Attention 侧第三次被否决（继 v227 gauge、A-RB1 舍入边界）；`REJECTED`，按计划 §6 不缩步/调 lr/换 fold/拆 head 粒度重试，未提交官方，根保持 R0 |
| [v229](20260910_v229_attention-amc1-k-mean-recenter_rejected_scoreNA_timeNA/result.md) | A-MC1 K 侧 per-call 均值再定心，冻结完整根 | **官方 TIMEOUT (>300s)，REJECTED**（2026-09-10 用户回传）；精确秒数/分数未知。六 shard 本地 `+0.014923`（26/10/36）未获官方精度定价；SHA `d1c23fa1…c4b247f4d`，根保持 `18053/281s` |
| [v230（Linear L-EM2）](20260910_v230_linear-em2-groupstep-schedule_scoreNA_timeNA/result.md) | 精确度量 Activation 组号主序下降，K=1；完整六 API | **RETAINED 18428/292s**，+375/+11s；原当前根，已由 v233 以同分快 4s 取代为回退根；六 shard 本地 +0.079454（288/0/48）；与 Attention v230 分开登记 |
| [v233（Linear L-TF1）](20260910_v233_linear-tf1-gradient-reuse_scoreNA_timeNA/result.md) | 首遍梯度复用（循环头加 `if _pass:` 保护，pass 0 复用循环前那次梯度）；完整六 API | **RETAINED 18428/288s**，相对父 v230 **同分快 4s**，取代其为**回退根**；低于根 v231 90 分故不成为根。六 shard **336/336 精确零**；每次调用少 2 个矩阵乘（20→18）；本地墙钟低于本机时钟分辨力、官方量出 −4s。**未兑现项：L-TF1 尚未并入根 v231** |
| [v231（Linear L-EM3）](20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/result.md) | groupstep K=2，完整候选；旧v202父、与v230同父构建 | **RETAINED 18518/291s**，相对 v230 +90/−1s；当前根；六shard对v230 +0.027507（286/0/50），SHA `ea79a1c1…5754f1` |
| [v232（Linear L-QF1）](20260910_v232_linear-qf1-quadratic-cost_scoreNA_timeNA/result.md) | 局部二次代价修正（`krba→krbi`，实现 `δᵀGδ`）；v230 根 + 一个 einsum 下标 | **官方 TIMEOUT (>300s)，REJECTED**（2026-09-10 用户回传），精确秒数/分数未知；本地六 shard +0.013739（284/4/48）、零可测时间成本未转化为官方计时；只关闭该实现 |
| [v230（Attention A-FIX1）](20260910_v230_attention-afix1-train-deploy-align_rejected_scoreNA_timeNA/result.md) | A-FIX1 训练/部署前向对齐（`_a2_train_rotation` 训练前向 Q/K 量化从裸 `_dense_to_hif4` 换成完整部署编码 `hif4_dynamic_quantize_q/k`，STE 反向不变；参数化/步数/lr/窗口/gate 全部与根相同） | **官方 TIMEOUT (>300s)，REJECTED**（2026-09-10 用户回传），精确秒数/分数未知；六 shard 等权 `-0.004884`（29/31/12）保留为诊断，未获官方精度定价；对齐前向约 1.4× 校准成本与 v229 同成本类，只关闭该实现，不缩步/缩窗重试；层15 重训 rotation 再次受损 `-0.013820`（继 v227 后第二次）；注意与并行 Linear 线 L-EM2 的 v230 编号冲突，引用须写全目录名 |
| [v235（Linear L-AD1）](20260910_v235_linear-ad1-adaround-materialization_scoreNA_timeNA/result.md) | 去掉 `_adaround_mantissa` 16 模式枚举里的 int64 物化（两条 int64 码表达式逐字不动；float 转换与 `*0.25` 从 16 倍展开张量移到两个小操作数上，`where` 直接在 float 上选；只依赖 device 的掩码提到调用外缓存）；v231 根 + 影子定义 | **官方 TIMEOUT（>300s）REJECTED**（2026-09-10 用户回传，[超时记录](../logs/execution/2026-09-10-v235-linear-official-timeout.md)）。**本轮唯一在本地测出可分辨提速的 Linear 候选（34–52× null），仍超时**——本地提速不构成官方可过的保证。六 shard **336/336 精确零**（0/0/336）；等价性由"逐元素算子与 `where` 交换"对**任意输入**证明，实测对照为 403 单元组 / 4 真实激活组 / 2 真实权重 state 全逐位相同；三臂配对（含同字节 sham null）有 gram 整调用 **+4.26%/+4.63%（null 的 33.9×/52.4×）**，无 gram 的 `o` 路径自带阴性对照按预期为零（0.19×/0.86×）；**动机是 v231 的 9s 余量而非分数**。纯追加 +4903 B，父根字节一个未动。**是 v231 的后代（含 K=2），与 v233 不可相加**；中途 rebase 见日志 §0 |
| [v234（Attention A-GR1）](20260910_v234_attention-agr1-general-reciprocal_scoreNA_timeNA/result.md) | A-GR1 一般非对称互逆矩阵残差（v192 单变量推广：对称零迹 S→一般 M=I+N，Q@M、K@M⁻ᵀ 校准期精确求逆；fit 0-2/gate 3-4、32 步 Adam、逐层全窗口严格改善门、center 同步编译，全部镜像 v192；冻结根 state） | **完整包官方 TIMEOUT（>300s），REJECTED**（2026-09-10 用户回传"v234官方也超时了"），精确秒数/分数未知。父 v230 官方 292s、余量仅 8s，而 A-GR1 的额外时间在**校准侧**（6 层 × 32 步 Adam；部署动态路径无新增算子），是机制自带代价、进不了限；**不缩步/缩窗/减候选重试**，只关闭此实现代表。**侧隔离官方读数 `standard-linear_v234-attn` = `14455/263.7s`（相对 v195 侧基准 `14426/243s` 为 +29/+20.7s，相对 R3 基线 +50）仍然成立**——它测的是**分数**、且是更小的包，本次超时关闭的是**完整包可落地性**，不推翻该分数声明。故本卡结论是"机制有分、代价不可落地"：A-GR1 仍是继 C76.4（+84）、A1（+60）后 Attention 第三大官方正向机制，表达力梯度（对角 0 < 三角 0 < 对称全矩阵 +22 < 一般矩阵 +29）仍获官方确认；与 v229 A-MC1、v230 A-FIX1 同属"侧向正向但完整包计时超限"类，差别是 A-GR1 是该类中**第一个先拿到侧隔离官方正向定价**的成员。本地六 shard `+0.003845`（21/3/48），层15/22 接受，保留为诊断；候选 SHA `4f27fb59…13a9267`；编号 v233 被 Linear L-TF1 占用故取 v234。见[超时记录](../logs/execution/2026-09-10-v234-agr1-official-timeout.md)。**对 v236 的直接影响**：v236 唯一改动是父替换，而校准侧代价基本不随 Linear 父变化、v231 只比 v230 快 1s，故本次超时直接落在 v236 上——其预登记晋级规则的超时分支已被预先回答（秒数未知，不写预测；v236 的行由其执行者串行登记） |
| [v239（Attention A-CT2）](20260910_v239_attention-act2-train-tail-reuse_rejected_scoreNA_timeNA/result.md) | **`_agr1_train` 尾部统计复用**：`final_loss` 循环带上两个按角色标量和，两个 ratio 由它们算出，删掉对同一批 fold 的第二次遍历（两次子串替换 36 行 −74 B，父字节一个未动） | 官方 **`unregistered/NA`**。候选 `55103e8b…`（531018 B）。**审计先于开发**：AST 按 `zip(fold,(m,p))` 位置对应证明同一表达式；仪器化实测 204 次调用中 6 次纯重算，六层 **6/6 入参逐位相同、返回标量逐位相同**，用循环内标量重建 ratio 与上报值**精确相等**；`force_zero` 不可达。**等价性**：六层全量校准 **q/k/v state 逐字节相同**；`_agr1_scale_loss_grad` 204→198、训练段 192 未动；覆盖接受/拒绝/M=I/ineligible/异常回退/确定性。**六 shard 对同父 A-GR1 旧实现 v236 为 72/72 精确零**。**时间：本机测不出**——四行都落在同字节 sham null 内；层22/`_agr1_train` 的 19.83× 伴随 **9/15 轮**（比值大只因 null 中位贴近 0），不当作效应证据。**量级已测定不足**（停滞诊断 §6.3：去重方向合计约 0.2s vs 需要的约 12s），归档不主张改善官方结局；本卡按用户指示照计划跑完。**是 v231 的后代（含 K=2），与 v233/v235/v237/v238 不可相加** |
| [v240（Linear L-MC1）](20260910_v240_linear-lmc1-compiled-metric_scoreNA_timeNA/result.md) | **把固定度量重建移入校准**：`_em1_compile_metric` 按同一表达式/输入/运行设备算出 G 并存入 state，`_em1_metric` 动态端只加载（无 Cholesky、无求逆、无写入） | 官方 **`unregistered/NA`**。候选 `2c344722…`（524220 B）。**对 v237 根 336/336 精确零**（0/0/336）。**这是搬运而非消除**：校准 +50.0 ms/次、动态 −28.3 ms/次（4096 通道，三臂含同字节 null，1/15 与 14/15 轮），**平衡点 ≈255 次动态调用**；本地 144 校准/288 动态 → 净省约 0.95 s，官方若只有 50 次动态则净增约 5.8 s——**官方调用次数是缺失事实**。**关键发现：stride 也是等价性的一部分**——`cholesky_inverse` 返回转置 stride `(1,n)`，`_cpu_state_tensor` 的 `.contiguous()` 会拍平它；数值逐位相同但下游 `.mm()` 走不同 cuBLAS 路径，足以改变离散 mantissa 码（修法：存 G 时不强制 contiguous）。state 体积 +4.75 GB（现 38.05 GB，约 +12%）。**是 v237 的后代，与 v233/v235/v238/v239 不可相加** |
| [v238（Attention A-CT1）](20260910_v238_attention-act1-gate-reuse_rejected_scoreNA_timeNA/result.md) | **A-GR1 gate 里与臂无关的固定计算由每窗两遍降为一遍**：dense 参考 Q/K/V、参考 `target`、父侧 V 五字段各算一次，两臂各自的 Q/K 与各自的 Attention 前向照旧（一次块替换 6 行 −136 B，父字节一个未动） | 官方 **`unregistered/NA`，待用户评测**，不写本地秒数预测。候选 `146bb715…`（528204 B）。**前提实测**（非论证）：两臂差异字段恰为 q/k 的 `learned_rotation` 与 k 的 `learned_center`、**v_state 逐字节相同**；V API 两次调用五字段逐位相同；Q/K/V 都不改写传入 state。**等价性**：真实数据每窗父/候选 loss **精确相等**；六个真实 attention 层全量校准 **q/k/v state 逐字节相同**；覆盖接受(0/22)/拒绝(1/5/8/15)/M=I/ineligible/异常回退。**调用计数**（独立计数器，每窗）：Attention 前向 4→3、参考解码 12→8、hif4 解码 6→5、V 量化 2→1，**Q/K 保持 2/2**（臂相关部分不许减）。**六 shard 对同父 A-GR1 旧实现 v236 为 72/72 精确零**（0/0/72）——处处为零是本卡预期读数，非零才是缺陷；对正式父的变化（等权 `+0.003845`，21/3/48，层 15 `+0.017449`/层 22 `+0.005618`）按 `case_id` 重新核算后继承，**不产出新的机制证据**。**时间**：gate 段 **+18.10%/+13.72%（空对照 15.3×/16.6×，15/15 轮）**，整校准 +0.58%/+0.56%（2.1×/1.5×，边际）；显存峰值三臂相同。`stopped_early` 标注见日志 §5.1。**是 v231 的后代（含 K=2），与 v233/v235/v237 不可相加** |
| [v237（Linear L-TF2）](20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/result.md) | **把 v233 的首遍梯度复用重新在 K=2 根上派生**（不拷贝 v233 提交；本卡唯一改动是循环头加 `if _pass:`，一次子串替换 5 行 +34 B，父字节一个未动） | **官方 18518 / 289s，RETAINED，已晋级为当前完整根**（2026-09-10 用户回传，[记录](../logs/execution/2026-09-10-v237-linear-official-result.md)）：相对父 v231 根**同分 / −2s**，余量 9s→11s；晋级依据为计划 §2"同分更快且 <300s"与 AGENTS §2 的 v202 先例（分数不低于当前根）。候选 `ecb1f9e5…`（516697 B）。**与 v233 的实质差异是 K**：v233 父 K=1、守卫分支永不执行；本卡父 K=2、**pass 1 真正走进该分支**，故算子账是 **38 → 36**（不是 20 → 18），控制 D 实测候选序列 = 父去掉第 2、3 个积、pass 1 的梯度积仍在；控制 E2（pass 1 非有限梯度）在本卡是发布配置。六 shard **336/336 精确零**（0/0/336，`min=max=0.0`，逐 role 全 0）；Attention 四 API 字节码相同、溶解守卫后 AST 与父逐节点相同、五字段逐字节相同（合成/layer0/q 128 行/无 payload/超范围）。**时间：本机分辨不出**——三臂配对（含同字节 sham null）四行 |effect|/|null| 为 0.52/0.52/1.11/1.00×，符号行间翻转、更快轮数 10–17/31；**不主张官方可分辨差异**。**是 v231 的后代（含 K=2），与 v233（v230 后代）不可相加**；`stopped_early` 标注见日志 §4.1 |
| [v241（Attention VK）](20260911_v241_attention-vk-kernel_scoreNA_timeNA/result.md) | 核加权 V 码选择：每 Q head 一个 32 参数相对位置核编译进 `v_state`，代替均匀加权选 V 的 HiF4 码；精确闭式逐元素步长、带状卷积不物化 T×T；校准期逐层 gate + 部署期回退父码 | **本地 Attention 六 shard 均值 `+0.004547`（六片全正）**，逐层 gate 6/6 接受；官方 `unregistered/NA`，未提交。SHA `8D364B3D…86CD` / 530793 B。**风险**：V API Δ +0.0666s/次，250 用例量级 ~17s vs 根余量 11s |
| [v242（Attention VK 共享核）](20260911_v242_attention-vk-kernel-shared_scoreNA_timeNA/result.md) | 与 v241 唯一差别：核从 per-Q-head（16）改为 **per-KV-group（4）**。组内共享是**重参数化**（目标化为 `4·Σ_t(Σ_k w_g[t−k]δ)²`，最小化点不变），每轮带状运算 16→4 次 | **本地六 shard 均值 `+0.004602`**（shard1/5 因 gate 挡下逐位等于父而判 no-op）；**V API Δ 降到 +0.0117s**，量级 ~2.9s，落进余量。官方 `unregistered/NA`。SHA `B11BA4F2…9227` / 530992 B。**是当前可提交形态** |
| [v236（Attention A-GR1-on-v231）](20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/result.md) | **把 v234 的 A-GR1 原样重挂到 v231 根**（本卡唯一改动是父替换；机制/窗口/步数/lr/gate 全不动） | 官方 **TIMEOUT（>300s）REJECTED**（2026-09-10 用户回传），精确秒数/分数未知记 null，见[超时记录](../logs/execution/2026-09-10-v236-agr1-on-v231-official-timeout.md)。**这是 A-GR1 完整包形态的第二次官方否决**：v234 父余量 8s、v236 父余量 9s，**父差仅 1s**，故本次把 v234 留下的唯一未知量"超限幅度是否 ≤1s"答为**否**，两次超时构成双向封闭区间——**不存在"换个 Linear 父就能过"的空间**，A-GR1 校准侧代价与 Linear 父的近旁替换无关（机制属性判断成为实测事实）。侧隔离 +29 不受影响，**A-GR1 机制本身不关闭**；不缩步/缩窗/减候选重试。自 v231 根 `ea79a1c1…` **纯追加** +15193 B（前缀 505762 B `cmp` 逐字节相同）；候选 `3319fc35…`。**Attention 段与 v234 逐字节同构**（`diff(本候选, v234归档候选)` == `diff(v231根, v230根)`，唯一 hunk 是 Linear 的 `_EM1_PASSES 1→2`；该常量只在 Linear 路径被引用），attention-only 不调用任何 Linear API。六 shard **逐项复现 v234**：等权 `+0.003845`（21/3/48），层15 `+0.017449`(10/2/0)、层22 `+0.005618`(11/1/0) 接受，层0/1/8/5 gate parent 逐位不变；72 case 绝对分布也相同——**不产出新的机制证据**。Control PASS（六API独立导入、state 合法、`agr1_attempted=1`、loss 2.0→1.2170、Linear 与 v231 父逐位）。**时间风险**：侧隔离 +20.7s vs 根余量 9s；侧时间对完整包无预测力，故不预测。**是 v231 的后代（含 K=2），与 v233/v235 不可相加** |

### 标准 Linear + Attention 侧隔离官方分（2026-09-09 回传）

口径：**标准 Linear（v162 标准 codec）+ 各 Attention 变体**，基线为标准 Linear + R3
`14405/238s`。这些是 **Attention 侧隔离分**，与完整根 `18053/281s` 不同口径，
不可相加、不可比较、不可从完整分反推。归因见
[侧隔离分登记](../logs/execution/2026-09-09-standard-linear-attention-side-scores.md)。

| 探针目录 | Attention 机制 | 官方分 / 时间 | Δscore vs 14405 | 结论 |
|---|---|---|---|---|
| [standard-linear_v190-attn](20260908_standard-linear_v190-attn_scoreNA_timeNA/) | 逐通道闭式 Q/K 互逆平衡 | `14405 / 246s` | **0** | 官方精确零增益，机制证伪 |
| [standard-linear_v191-attn](20260908_standard-linear_v191-attn_scoreNA_timeNA/) | 块三角输运 | `14405 / 264s` | **0** | 官方精确零增益，机制证伪 |
| [standard-linear_v192-attn](20260908_standard-linear_v192-attn_scoreNA_timeNA/) | 全对称零迹矩阵残差 | `14427 / 272s` | **+22** | 有效但 +34s，完整包 TIMEOUT，不可部署 |
| [standard-linear_v194-attn-speed](20260908_standard-linear_v194-attn-speed_scoreNA_timeNA/) | A2/R3 校准等价提速 | `14405 / 234s` | **0** | 侧隔离 −4s 但完整包 `285s`（+5s），提速路线不成立 |
| [standard-linear_v195-attn](20260908_standard-linear_v195-attn_scoreNA_timeNA/) | K-center 梯度聚合修复 | `14426 / 243s` | **+21** | 与完整包 `18053−18032=+21` 交叉验证，已在根上兑现 |
| [standard-linear_v229-attn](20260910_standard-linear_v229-attn_scoreNA_timeNA/) | A-MC1 K per-call 均值再定心（v195 系根 Attention + recenter） | `14424 / 245s` | **+19**（相对 v195 行 **−2/+2s**） | 本地六 shard `+0.014923` 未迁移官方（方向反转），A-MC1 官方侧价值 −2，K 平移类关闭；侧隔离未超时（245s），完整包 v229 TIMEOUT 不由 per-call 成本单独解释；见[回传记录](../logs/execution/2026-09-10-v229-side-isolation-official.md) |
| [standard-linear_v234-attn](20260910_standard-linear_v234-attn_scoreNA_timeNA/) | A-GR1 一般非对称互逆矩阵残差（v195 系根 Attention + M=I+N） | `14455 / 263.7s` | **+50**（相对 v195 行 **+29/+20.7s**） | **官方正向**：A-GR1 官方侧价值 +29，继 C76.4（+84）、A1（+60）后第三大 Attention 机制；表达力梯度（对角 0 < 三角 0 < 对称全矩阵 +22 < 一般矩阵 +29）获官方确认；与 v234 归档候选 72/72 逐位一致；见[回传记录](../logs/execution/2026-09-10-v234-agr1-side-official.md) |

**Attention Correctness Hardening（2026-09-08）：** [AC0](continuous_attention_ac0-correctness-hardened/result.md)
（`F817E4C2…`，父 R3 `A5C679D7…`）保留为 Attention 正确性参考：原子 Q/K-pair
fallback（删 `except: pass`）、训练 hard forward 走部署五字段路径、统一 transform
reference、GQA/rotation/center shape 校验、校准期 FP64 QK-invariance audit（真实 4B
六层全部 valid，5.2~6.1e-07<1e-6）、fallback_reason 记录；battery 30/30。本地 paired
attention-only vs R3：Δmean +0.003501 / L1 0.011264（72 case，34+/26−/12=）；5/6 层
arm 一致，L15 边际 gate 翻转（R3 identity→AC0 rotation）。**官方回传 14395 / 258s**
（相对 R3 −10/+20s；未触发 <−20 停止门；258s<300s 硬限；A29 骨架与 AC0 同 SHA，
结果绑定 AC0）。[A29 骨架](continuous_attention_a29-boundary-output/result.md) 只是 AC0 逐位复制；
真正 A29 实现 `v163_attention_a29-final-residual-s` 已官方 TIMEOUT；旧 A30/A31 队列已暂停。

最新 Attention 侧历史回传：[Attention A21-1](continuous_attention_anchor21-a1/official-result.json) **14199/244s，OFFICIAL_REJECTED**，相对R3 −206/+6s；源码SHA `870d5848f95887307ad7faa6364b5d7f7480f5b7be6001c812c44ead02bdb48a`。

> **Historical snapshot (2026-09-06; not current instructions):** the superseded umbrella plan was
> [independent Linear / Attention optimization from v162](../docs/superpowers/archive/plans/2026-09-06-v162-independent-linear-attention-plan.md).
> Current execution uses the single full-solution plan linked from `docs/superpowers/plans/README.md`.
> The following entries are historical execution records only.
> [Linear compiled robust calibration-window max block order](20260906_linear-compiled-robust-window-order_rejected/result.md)
> was closed as **CLOSED / R1_REJECTED** after shard0 Linear mean/median deltas of
> `-0.000043038/-0.000117686`; it was archived without R2 or official submission.
> [Linear compiled output-covariance block order](20260906_linear-compiled-output-covariance-order_score-rejected/result.md)
> was closed as **CLOSED / R3_REJECTED_SCORE**: its repaired fresh default Overall was
> `0.687211924573`, below the local high `0.688994940507429`, while the official-time
> predictor passed at `279.215656s`; all evidence was archived without official submission.
> [Compiled calibration sample-energy block order](20260906_linear-compiled-sample-energy_score-tie/result.md)
> was closed as **CLOSED / R3_REJECTED_SCORE_TIE**: its direct-core fresh default Overall
> `0.688994940507` tied the measured local high, while the official-time predictor passed at
> `279.445203s`. The user later reported an official result of `17636/264s` (`+20/-11s` vs v189);
> the original local score-tie decision is retained as history; this archive is now the root working parent.
> [Linear dynamic 32-row block-energy act-order](20260906_linear-dynamic-block-energy32_time-rejected/result.md)
> was closed as **CLOSED / R3_REJECTED_TIME**: its fresh default Overall was `0.688652578052`,
> below the measured local high `0.688994940507`, and the official-time predictor was
> `285.526750s`, so it was archived without official submission and the root remains v189.
> [Linear dynamic carrier-scale act-order](20260906_linear-dynamic-carrier-scale_time-rejected/result.md)
> was closed as **CLOSED / R3_REJECTED_TIME**: its fresh default Overall was `0.687922431205`,
> below the measured local high `0.688994940507`, and the official-time predictor was
> `286.049047s`, so it was archived without official submission and the root remains v189.
> [Linear dynamic block-energy act-order](20260906_linear-dynamic-block-energy_time-rejected/result.md)
> was closed as **CLOSED / R3_REJECTED_TIME**: its fresh default Overall was
> `0.688967415343` (above the local high `0.687776303363`), but the official-time predictor was
> `286.022476s`, so it was archived without official submission and the root remains v189.
> [Integrated Linear carrier-energy act-order](20260906_linear-integrated-carrier-energy_time-rejected/result.md)
> was closed as **CLOSED / R3_REJECTED_TIME**: it preserved the preceding carrier-energy
> local Overall `0.687776303`, but its fresh predictor was `284.291453s` and it did not
> exceed the local high, so it was archived without official submission.
> [Linear dynamic sample-energy act-order](20260906_linear-dynamic-actorder_time-rejected/result.md)
> was closed as **CLOSED / R3_REJECTED_TIME**: its local default Overall was `0.688994940507`
> (above the local high `0.687776303363`), but the official-time predictor was `284.775756s`,
> so it was archived without official submission and the root remains v189.
> [Attention mask-aligned output selector](../docs/superpowers/archive/plans/2026-09-06-attention-noncausal-selector-plan.md)
> was closed as **CLOSED / R2_REJECTED** after the fresh default gate. The previous non-causal
> logit-gain fitting was closed as **CLOSED / R1_REJECTED**. The recovered fixed-order candidate is archived as
> [v189](20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/result.md): official
> `17616/275s`, RETAINED as the previous full parent.
> The fixed-state output-aware JDRQ integration is closed as **CLOSED / J1_REJECTED** after
> 112 paired Linear cases were negative; see the
> [execution record](../logs/execution/2026-09-06-linear-fixed-state-output-aware-jdrq-plan.md)
> and [rejected archive](20260906_linear-fixed-state-output-aware-jdrq_rejected/result.md).
> The conditional-curvature block-order review is closed as **CLOSED / C1_REJECTED**; see the
> [execution record](../logs/execution/2026-09-06-linear-static-gptq-conditional-curvature-plan.md).
> The corrected legal-lattice output oracle is closed as
> **CLOSED / R1_NO_SUPPORTED_MECHANISM**; see the
> [execution record](../logs/execution/2026-09-06-corrected-legal-lattice-output-plan.md).
> The raw Attention source-scale extension is also closed as
> **CLOSED / NOOP_REJECTED**; see the
> [execution record](../logs/execution/2026-09-06-attention-source-scale-proposal-plan.md).
> The aligned Attention source-scale extension is closed as
> **CLOSED / NOOP_REJECTED**; see the
> [execution record](../logs/execution/2026-09-06-attention-aligned-source-scale-plan.md).
> The Attention logit-Fisher pair candidate is closed as **REJECTED / F2_REJECTED** after
> shard4 regressed; see the [execution record](../logs/execution/2026-09-06-attention-logit-fisher-pair-plan.md)
> and [rejected archive](20260906_attention-logit-fisher-pair_rejected/result.md).
> The Linear multi-fold cross-block Hessian extension is closed as
> **CLOSED / B1_REJECTED**; see the
> [execution record](../logs/execution/2026-09-06-linear-crossblock-robust-hessian-plan.md).
> J0 is closed as
> **CLOSED / J0_REJECTED**; see the
> [J0 execution record](../logs/execution/2026-09-06-joint-output-gauge-plan.md). The previous D-A/D-B plan was executed and closed
> without a deployable mechanism. See the
> [execution record](../logs/execution/2026-09-06-hierarchy-partition-activation-plan.md) and stage artifacts.
> The plan was:
> [legal codec verification and output-objective optimization](../docs/superpowers/archive/plans/2026-09-05-legal-codec-and-output-objective-plan-r2-rejected.md)
> (now **CLOSED / R2_REJECTED**). It specified defect tests, a legal hierarchical block solver,
> output-objective diagnostics, and gated candidate validation. The completed codebook
> plan is archived; its broad saturation claims are superseded by this
> [evidence audit](../logs/execution/2026-09-05-next-plan-evidence-audit.md).
> Historical snapshot only: v189 was `17616/275s`; v180 was the `17597/242s` time-budget reference.
> The previous system-identification design is archived as superseded; no new solution version is assigned.

> **Rule correction (2026-09-05):** official submissions are unlimited. Historical
> `x/10`, “remaining slot”, and quota wording is obsolete; see
> [`stale-information-inventory-2026-09-05.md`](../docs/stale-information-inventory-2026-09-05.md).

> **Official update (2026-09-04):** v183 scored **17598 / 279.7s**, tying v182 while taking
> `6.7s` longer, so it is REJECTED under its pre-registered rule and the attention block-smooth
> coverage family is closed. v182 archive SHA `F3E39E99...A438` remains the score parent at
> **17598 / 273s**. The independent side
> parents remain `P_L=v166（4590/226s）` and `P_A=v168（14005/210s）`; v182 is not an isolated
> Linear measurement. The user-reported leaderboard best is **21765 / 290s**, leaving a
> **4167-point** gap. v180 remains the time-budget parent because it gives up only 1 point for 31s.

`solutions/` contains immutable `solution.py` snapshots. The active code is only the repository
root [`solution.py`](../solution.py). Every snapshot below was submitted or retained as an
official-result candidate; official numbers are historical facts and are not replaced by local
proxy scores.

## Canonical re-evaluation

Use [`evaluator/eval.py`](../evaluator/eval.py), never the retired `real_model_suite.py`.
The current evaluator is `eval-v3`: it reuses the fixed `proxy-v2` dense cache, splits the panel
into six shards, caches calibration artifacts, and emits per-case evidence. `official_eval.py`
is intentionally left unchanged as a `proxy-v2` compatibility/reference backend; the older v1
archive is immutable historical evidence only:

```powershell
.venv\Scripts\python.exe -u evaluator\eval.py --official-audit `
  --cohort new-weight --scenario both --shards 0,1,2,3,4,5 `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --calibration-cache-mode auto --algorithm-device cuda `
  --output-dir artifacts\proxy_v3\official-audit
```

The generated audit is diagnostic: official score/time remain independent observations and no
local-to-official score conversion is fitted. Use `--cohort old-weight` explicitly for historical
rows; a cache-cohort mismatch is reported rather than silently mixed.

The protocol fixes Qwen2.5-0.5B, the five Attention calibration lengths
`[10,128,512,1024,1024]`, validation/test holdout windows, independent HiF4 validation, and the
public relative-MSE case score. The default panel is a deterministic stratified real-W/A panel:
168 Linear cases cover every layer/role once and 120 Attention cases cover every layer at each of
the five official lengths. `--full-cases` expands all captured windows for stress; case limits are
still smoke-only. Calibration follows the judge graph: 168 shared layer/role Weight states and 24
shared Attention states, followed by one dynamic call per selected case. The authoritative local
fields are `linear_mean`, `attention_mean`, and the unweighted `overall_mean`; no Linear:Attention
ratio or official-score fit is applied. Each result also contains evaluator-only error-source
controls in `decomposition` and per-case `case_scores`: Linear W/A four-arm output MSE, Attention
Q/K/V/QK controls, and logits/softmax metrics. These controls reuse candidate outputs and do not
change API call counts; use `--no-decomposition` only for a fast smoke run. Local seconds are
same-machine A/B data, not an official-time conversion; `trend_diagnostics` reports known same-
cohort ordering inversions without fitting them.

The eval-v3 audit deliberately uses six balanced shards (336 Linear + 48 Attention cases per
version) so a dense cache is loaded once and calibration artifacts can be reused. Those shard
statistics are diagnostic and must not be mixed with the compatibility backend's 168+120 default
panel.

### Current best and scope rule (2026-09-04)

The highest official score bound to a repository source is **v182: 17598 / 273s**. Its exact
complete parent v180 is **17597 / 242s**; both are Pareto-optimal, so v182 is the score parent and
v180 remains the time-budget parent. v183 `17598/279.7s` is dominated by v182 and rejected. The
independent side parents remain v166 and v168. The
user-reported leaderboard best **21765 / 290s** is 4167 points above v182 and has no synchronized
source or configuration, so it is a target only. v147 is **16579 / 211 s**
(time pass but below v86, rejected); v140 is **15838 / 207 s** (rejected). The pre-A3 parent effect
control is local-only (`Linear=0.588023229`, `Attention=0.757433277`, API `202.317 s`) and is not
an official score.

Do not rank the following together: `default-panel` is the only local proxy-ranking scope;
`effect-panel`/`paired-json-replay` are parent-child mechanism diagnostics, `full-stress` is a
stress check, `smoke-prefix` is interface-only, and GPT-2/hif4/old `official-shape-v1` are
cross-structure or historical probes. Every new JSON records this in `evaluation_scope`; see
[`artifact scope contract`](../artifacts/official_eval/README.md).

The completed 2026-09-01 **historical v1** archive run is in
[`artifacts/official_eval/legacy-v1/archive-official-shape-v1.json`](../artifacts/official_eval/legacy-v1/archive-official-shape-v1.json)
and [`logs/official_eval/archive-official-shape-v1.md`](../logs/official_eval/archive-official-shape-v1.md).
It is immutable historical evidence; do not rank with it.
Among candidates whose functions returned, v121 is the local maximum
(`linear_mean=0.472197763`, `attention_mean=0.833617251`, equal-weight display `28477.289`),
but its API time is `3404.369 s` and its official outcome is timeout. The highest official-pass
candidate under the local API proxy is v084 for Attention (`0.718106989`); v024 is highest for
Linear (`0.450074554`). v002 is recorded as a real local CUDA/CPU device-mix error rather than
silently assigned a score.

## Versions with official outcomes

> 代际声明：v001–v074 为**旧权重**官方分数（与当前评测集不可换算，仅历史证据）；
> v084 起为**新权重**官方分数（当前口径）。v074 另有当前评测集回传 `14561 / 188.9s`
> （2026-09-02），旧权重 `22750` 已失效。

| Version | Source directory | Official score | Official time | Outcome 
|---|---|---:|---:|---
| v001 | `20260826_v001_current-baseline_score10250_time127s` | 10250 | 127 s | pass 
| v002 | `20260826_v002_youxilee-hif4_score15000plus_timeNA` | 15313 | 137 s | pass 
| v013 | `20260827_v013_c10-wide-activation-quadratic_score15799_time144s` | 15799 | 144 s | pass 
| v024 | `20260827_v024_c21-gated-exact-cross-selection_score16043_time174s` | 16043 | 173.8 s | pass 
| v025 | `20260827_v025_c21c-compliance-baseline` | 14437 | 166.6 s | pass 
| v030 | `20260828_v030_c38-beam2-fullcov-official14092_time170.6s` | 14092 | 170.57 s | pass 
| v031 | `20260828_v031_c39-fw-official21864_time161.3s` | 21864 | 161.3 s | pass 
| v032 | `20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s` | 14432 | 216.667 s | pass 
| v034 | `20260829_v034_c41b-mha-k-center_scoreNA_timeNA` | 21864 | 159.4 s | pass 
| v051 | `20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA` | 22451 | 234 s | pass 
| v066 | `20260829_v066_c66-activation-ratio100_scoreNA_timeNA` | 22557 | 217.2 s | pass 
| v072 | `20260829_v072_c74-jdrq-hierarchy_scoreNA_timeNA` | 22662 | 226 s | pass 
| v074 | `20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA` | 22750（旧权重）→ **14561**（当前评测集回传，2026-09-02） | 239.387 s → 188.9 s | pass（**非安全基线**，低于 v84/v86） 
| v084 | `20260830_v084_c84-gram64-sweep5_scoreNA_timeNA` | 16517 | 252.563 s | pass (revised weights) 
| v086 | `20260830_v086_c86-attn-block-final_scoreNA_timeNA` | **16744** | **222.7 s** | **pass (revised weights, new best)** 
| v098 | `20260830_v098_b1-gqrb-margin-active_score293.793700_time406s` | — | >300 s | timeout 
| v100 | `20260830_v100_b2-pawv-diagonly-active_score293.797301_time392s` | — | >300 s | Attention WA / timeout 
| v107 | `20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s` | — | — | Attention WA 
| v121 | `20260831_v121_c1b-structured-refresh2-accepted_score295.811281_time2180s` | — | >300 s | timeout 
| v128 | `20260901_v128_fixed-attn-budget_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v129 | `20260901_v129_fixed-attn-budget-sweep1_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v130 | `20260901_v130_output-weight_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v131 | `20260901_v131_output-weight-qwgram_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v138 | `20260901_v138_attention-static-v86-budget_scoreNA_timeNA` | **15715** | **208 s** | **pass (official, user reported)** 
| v139 | `20260901_v139_linear-output-aware-gain_scoreNA_timeNA` | **15716** | **202 s** | **pass (official, user reported)** 
| v140 | `20260901_v140_linear-roab-pair_rejected` | **15838** | **207 s** | **pass, but rejected: below v86 and 17816** 
| v147 | `20260901_v147_v86-attention-v140-linear_rejected` | **16579** | **211 s** | **pass, but rejected: 165 points below v86; submitted SHA unconfirmed** 
| v155 | `20260902_v155_l5a-permutation-stability_rejected` | **16581** | **208.5 s** | **pass, but rejected: 163 points below v86** 
| v156 | `20260902_v156_l4-weight-decoupled_rejected` | **16580** | **204.3 s** | **pass, but rejected: 164 points below v86** 
| v157 | `20260902_v157_v86-roab-only_rejected` | **16729** | **218.96 s** | **pass, but rejected: 15 points below v86** 
| v158 | `20260902_v158_v86-attention-matrix-smooth_retained` | **16861** | **223 s** | **pass; retained, +117 vs v86** 
| v159 | `20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA` | **17532** | — | **official score reported; runtime unknown** 
| v160 | `20260903_v160_v159-linear-l1batch_v158-attn-a2_scoreNA_timeNA` | **17532** | **232 s** | **pass; score no-op vs v159, source/time-complete experiment parent** 
| v161 | `20260903_v161_v160-attn-s1-qk-gram-refine_scoreNA_timeout` | — | >300 s | **timeout (official, user confirmed); local funnel passed (Qwen default 120 paired +0.0525, 106+/14−; GPT-2 +0.0678 same sign; D1 satisfied locally) but per-call dynamic refinement exceeds the official runtime budget — per-call family closed** 
| v162 | `20260903_v162_standard-baseline-both_scoreNA_timeNA` | **1001** | **146 s** | **pass; calibration anchor measured — official non-zero base score or official STD differs from the local reference codec (local means both exactly 0.0); also establishes the ~146 s official harness time floor** 
| v163 | `20260903_v163_v160-linear_standard-attn_scoreNA_timeNA` | **4587** | **202 s** | **pass; official Linear-side contribution Δ_L = 4587−1001 = 3586, local linear mean 0.633526 (bit-identical to v160, 168 cases), attention mean 0.0; time 202s vs predicted ~186s, within margin** |
| v164 | `20260903_v164_standard-linear_v160-attn_scoreNA_timeNA` | **13945** | **204 s** | **pass; official Attention-side contribution Δ_A = 13945−1001 = 12944; together with v163, endpoint additivity predicts v160 within 1 point**
| v165 | `20260903_v165_standard-linear_v161-attn_scoreNA_timeout` | — | >300 s | **timeout (official, user confirmed); standard Linear is bit-identical to v164, so the result isolates the v161 Cross-Gram64 per-call Attention path as over budget; no score means no Attention accuracy ratio is computed**
| v166 | `20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA` | **4590** | **226 s** | **pass; official +3 over v163 (4587/202s), retained as the new Linear parent side P_L (C_L = 4590−1001 = 3589, G_L = +3, 226s < 300s); rank-1 residual redistribution Linear (fold-median top-2 base-codec residual directions, c=1/4, exact product preservation, single-encode design) + standard Attention (mean 0.0, 0/0/120 vs standard); local linear default 0.636590 vs parent 0.633526 (paired +0.003064, 78+/90−, proj +0.0251), API 282.8s (1.24×)**
| v167 | `20260903_v167_standard-linear_lowrank-gram-attn_rejected` | — | — | **rejected (local, pre-official); side-isolation plan 7.2 low-rank Gram codebook — the designated v165-timeout recovery path. Implementation proven correct (lam=0 ablation is bit-identical to parent, 0.797462). Root cause: the real QK cross-Gram is high-rank (top-2 off-diag ~7% eigen mass), so rank-2 coupling-motivated bumps destroy deep sentinels (L15 0.735→−0.54, L23 0.606→−0.05) under both median and mean fold aggregation, while diagonal-only is a mathematical no-op for nearest-level encodings. No rank neighborhood scan per plan 5; Attention-internal mechanisms exhausted**
| v168 | `20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA` | **14005** | **210 s** | **pass, RETAINED; new Attention parent P_A. Expansion plan A1: per-KV-head multiplicative logit gain folded into the Q/K multiplier path, zero dynamic additions. step_gain +60 over v164 (Attention ratio 0.46 percent), a small positive gain with no local-proxy signal (local Qwen default mean -0.00088, GPT-2 +0.0024); time +6s over v164. Corrected same-day: initially reported 17248/237s in error. Combined-side prediction with v166: 4590 + 14005 - 1001 = 17594 (+62 over v160)** |
| v180 | `20260904_v180_a1-asym-fold-attn_scoreNA_timeNA` | **17597** | **242 s** | **pass, RETAINED as new full official parent; post-official plan D1 A1 Q/K asymmetric fold (alpha=0.3, exponent-sum keeps logits=gamma, only Q/K dynamic-range reallocation; alpha=0 bit-identical to v175). step_gain +3 vs v175 17594; time −3s is recorded but not claimed as a stable speedup because D1 adds no online operator. Compact 4 paired v175 mean +0.000088 (3+/1−); default 120 paired v168 mean +0.000356 (69+/51−, win 0.575), QK interaction +0.01106; GPT-2 −0.008984 model-specific-risk; opt-125m vs v160 −0.000208 (28+/32−) but D1 increment vs v175 +0.00118 (win 0.467, weak positive, not fully agreeing with GPT-2 sign). Gap to 21765 is 4168** |
| v181 | `20260904_v181_a1-qhead-gain-attn_rejected` | — | — | **rejected (local pre-research, clearly negative); post-official plan D2 per-Q-head logits gain control (each Q head own multiplicative gain on top of A1, K per-KV-head shared — breaks GQA group consistency). Clean D2 (D1 fold OFF, A1 symmetric parent): default 120 paired v168 mean −0.002746, median −0.000086, 54+/66−, median MSE ratio 1.000333; D1+D2 stacked was also negative (mean −0.002019, 60/60). Confirms A1 group-consistent structure is load-bearing; D2 family closed and was not submitted** |
| v182 | `20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA` | **17598** | **273 s** | **pass, RETAINED as new full official parent; post-v180 plan L-R2 fused rank-2 residual redistribution (v166 rank-1 → rank-2 U=[u1,u2]/V=[v1,v2], V^T U≈0, Woodbury R^-1=I-UV^T, continuous product exactly preserved; Attention v180 bit-identical). step_gain +1 vs v180 17597; rank-2 family saturated (0<G_L≤20 → close rank-3/coef scan). Hard checks: reachability all 1, vtu_cross_max ~1e-8, continuous-domain rel err 3.96e-7 (float32). Local paired v180: Qwen compact −0.000093, Qwen default +0.000020, GPT-2 +0.001171, OPT +0.025632 — non-negative, no model-specific-risk. Time 273s (+31) within 300s but margin 27s. Gap to 21765 is 4167** |
| v183 | `20260904_v183_attn-bsm-full-refine_rejected` | **17598** | **279.7 s** | **rejected (official 2026-09-04); direction-1 coverage diag product: v182 + attention block-smooth search refine coverage 0.50→1.00 / blocks 131072 (2 constants only, calibration-side, zero online additions; Linear v182 bit-identical). step_gain 0 vs v182 17598 (score tie, no improvement) per pre-registered rule S≤17598 → REJECTED, coverage family closed. Time 279.7s < 300s (+6.7s calibration refine cost, not timeout). Local: Qwen default +0.000511, GPT-2 −0.005, OPT no_effect. Official parent remains v182** |
| v185 | `20260904_v185_cleanroom-robust-operator_rejected` | **8446** | **165 s** | **official REJECTED; clean-room six-API implementation with analytic Linear diagonal transform and low-DOF Attention K-center/QK-balance/logit-gain/+4 gates. Legal and fast, but official −9153 vs v186 confirms severe algorithm underfit rather than timeout. Balance/gamma/refine neighborhood closed** |
| v184 | `20260904_v184_attn-plus4-gate_timeout` | — | **>300 s** | **timeout (official 2026-09-04); dual-window full-calibration +4 gate (each layer calibrated twice: 4-code + 5-code windows, deployed-MSE gate selects). Time-model attribution: dual-window x2 calibration +36.6s official (0.694 x 52.7s local A_calib) → predicted 309.6s from v182 parent 273s, matches actual >300s timeout. Root cause is the dual-window architecture, NOT the 5-code window itself (single-window 5-code calibration runs at 4-code speed: 66.0s vs 66.1s local). Local: Qwen +0.006580 (L11 +0.158 recovered via gate-decision flip), GPT-2 no_effect 12/12 rejected, OPT −0.0002. Timeout per pre-registered rule; single-window +4 variant (the probe, zero calibration cost, Qwen +0.010386) is the time-safe restructure of the same mechanism** |
| v186 | `20260904_v186_attn-plus4-single-window_scoreNA_timeNA` | **17599** | **272 s** | **pass, RETAINED as new full official parent; oracle-decomposition minimal product: v182 + 1-line `_DYNAMIC_OFFSETS (-1,1,2,3)→(-1,1,2,3,4)` (add single +4 E6M2 code to online Q/K/V scale window; hill-climb edge extension cannot reach it across binades). step_gain +1 vs v182 17598; time −1s (time-model predicted 274.0s, actual 272s, within MAE 10.1s — calibration-neutral prediction validated). Local Δmean +0.010344 (largest post-A1 signal, 29x D1) → official +1: reconfirms local mean does not convert to official points but sign gates (Δmean>0, L1=0.0155<0.02) were zero-error. Family officially positive; no code-neighborhood scan (+5/-2 etc.). Gap to 21765 is 4166; time margin 28s** |
| v187 | `20260904_v187_attn-jacobian-sensitivity_research-retained` | **9167** | **169 s** | **official positive / RESEARCH RETAINED; v185 clean-room + analytic final-Attention Jacobian importance for Q/K, KV-group shared and leave-one-fold-out gated. Official +721/+4s vs v185 confirms transfer. 7/24 layers active; local Δmean +0.015187, L1 0.016199. Still −8432 vs v186, so not a full parent; root unchanged** |
| v188 | `20260904_v188_attn-jacobian-port_rejected` | **17595** | **268 s** | **rejected (official 2026-09-04); v186 + v187 Jacobian sensitivity importance ported as final calibration step on the fully-transformed Q/K coordinates (causal/non-causal 0.5, cross-fold median, log shrink 0.25, clamp [0.5,2], LOO deployed-MSE gate; v187 pre-registered constants, no neighborhood scan). step_gain −4 vs v186 17599; time 268s (model predicted 274s, within MAE). Local default 120 vs v186: Δmean +0.000426, L1 0.001114, 6+/4−/110=; gate accepted only 2/24 layers (L12/L22 — the pair-transform-free layers; pair-smooth output-fitted importance wins elsewhere). First sign-gate miss on a near-zero local signal (110/120 cases unchanged): the gate blocks large losses (−165~−1164) but does not guarantee non-negative official deltas for near-zero signals; official ±1~4 is the effective noise band (single-point gains v182/v186 were +1/+1/+3). Jacobian port family closed; root rolled back to v186** |
| v195 | `20260908_v195_attn-a2-center-gradient-aggregate_scoreNA_timeNA` | **18053** | **289 s** | **RETAINED (user-reported official 2026-09-09); K-center multi-window gradient aggregation fix; +21/+9s vs previous root 18032/280s; root switched to this source; platform did not return a separate scored SHA** |
| previous root | `20260908_linear-current-r3-attention_candidate` | **18032** | **280 s** | **RETAINED (user-reported official 2026-09-08); current compiled sample-energy Linear + R3 rotation-center/all-gates Attention; +396/+16s vs previous root 17636/264s; retained as rollback source** |
| v189 | `20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA` | **17616** | **275 s** | **RETAINED (official 2026-09-06); v186 + static deployed-Hessian activation-GPTQ complete 64-block ordering; step_gain +17 vs v186, historical full parent before 17636 root; local default Overall `0.686889608842` remains a proxy-only value** |
| — | `continuous_linear_l28-proj-vectorized` | **4611** | **286 s** | **RETAINED (official 2026-09-08); Linear side parent L4→L28; residual-cross-subspace A@W fit + time-safe rework (batched Cholesky, eigh(64×64) replaces SVD(64×o), vectorized round/clamp projection, math-equivalent); step_gain +4 vs L4 (4607/247s), 286s<300s; fit_gain 0.9453>=0.9 local; supersedes L23b (TIMEOUT)`** |
| — | `continuous_linear_l23b-residual-subspace` | — | >300 s | **TIMEOUT (official 2026-09-08); all-row direct fit + incremental residual; complexity implementation closed, mechanism family reworked as L28** |
| — | `continuous_linear_l23-residual-subspace_timeout` | — | >300 s | **TIMEOUT (official 2026-09-07); evidence only, no re-submit** |
| — | `20260906_linear-dynamic-actorder_time-rejected` | — | — | **REJECTED_TIME (local only); default Overall `0.688994940507` exceeded the local proxy high by `+0.001218637144`, but predicted official time `284.775756s` failed the `<280s` gate; no official submission** |
| v169 | `20260903_v169_standard-linear_v-bias-attn_rejected` | — | — | **rejected (local, clearly negative); expansion plan A2 V output-bias centroid: local Qwen -0.0093 (21+/99-) and GPT-2 0/4 all-negative - final classification per user 'reject clearly-negative optimizations'** |
| v170 | `20260903_v170_standard-linear_fixed-offset-attn_rejected` | — | — | **rejected (local, clearly negative); expansion plan A3 static fixed-offset compile: Qwen -0.0506 (9+/111-) and GPT-2 -0.0551 (1+/3-) - final classification per user** |
| v171 | `20260903_v171_standard-linear_moment-threshold-attn_rejected` | **13657** | **214 s** | **rejected (official 2026-09-04); expansion plan A4 moment-matched mantissa rounding threshold. step_gain −348 vs v168 (14005), Attention ratio −2.69%. Time 214s < 300s; negative from algorithm not timeout. A4 family closed** |
| v172 | `20260903_v172_babai-weight-decode_rejected` | — | — | **rejected (local, clearly negative); expansion plan L2 HiF4 hierarchical Babai decode: compact -0.0483, 0+/48- (zero positive cases) - final classification per user** |
| v173 | `20260903_v173_trellis-vq-weight-decode_rejected` | — | — | **rejected (local, clearly negative); expansion plan L3 fixed-width Trellis/VQ: compact -0.0229, 1+/47- (single positive case) - final classification per user** |
| v174 | `20260903_v174_kronecker-cat_standard-attn_rejected` | **4508** | **190 s** | **rejected (official 2026-09-04); expansion plan L4 Kronecker-compressed analytic CAT. step_gain −82 vs v166 (4590), Linear ratio −2.29%. Time 190s < 300s; L4 family closed** |
| v175 | `20260903_v175_rank1-linear_logit-gain-attn_scoreNA_timeNA` | **17594** | **245 s** | **pass, RETAINED as new full official parent; combination (plan 13) v166 rank-1 Linear + v168 A1 logit-gain Attention. interaction = 17594−4590−14005+1001 = 0 (exact additivity confirmed on official total). S_pred=17594 (+62 over v160, 4171 from leaderboard 21765). Time 245s < 300s** |
| v176 | `20260903_v176_k-outlier-eq-attn_rejected` | **13964** | **205 s** | **rejected (official 2026-09-04); next-stage plan C1 K-side static outlier-channel equalization. step_gain −41 vs v168 (14005), Attention ratio −0.32%. Time 205s < 300s; C1 family closed (consistent with local default −0.004450, GPT-2 −0.002753, opt −0.021851)** |
| v177 | `20260904_v177_c2-group-logit-gain_rejected` | — | — | **rejected (local pre-research, clearly negative); plan C2 A1-fine-grained per-(KV-head, 8-channel-group) logits gain (closed-form 8-param LS per §2b), on P_A=v168. Compact 4 paired v168 mean −0.006858 (1+/3−); default 120 mean −0.006643 (41+/79−, win 0.342), QK interaction −0.0935. B=8 groups negative on v168 branch; not submitted** |
| v178 | `20260904_v178_c2-on-c1-pre-research_rejected` | — | — | **rejected (local pre-research, clearly negative); C2 C1-branch preview (per §2b, C2 on v176 = C1 semantic + group gain). Compact 4 paired v176 mean −0.009194 (1+/3−); default 120 paired v168 mean −0.008610 (47+/73−, win 0.392) — worse than v177. C2 family closed on both branches (v168/C1); not submitted** |
| v179 | `20260904_v179_c3-rand8-ortho-attn_rejected` | — | — | **rejected (local pre-research, clearly negative); plan C3 fixed 8×8 random orthogonal rotation (QuaRot/TurboQuant control, per-KV-head fixed-seed QR, QK^T inner-product-invariant). Compact 4 paired v168 mean −0.229273 (1+/3−), median MSE ratio 1.512 — largest negative among tested mechanisms, matching Longhorn Qwen GQA rotation delocalize evidence. C3 family closed; not submitted** |
| v169 | `20260903_v169_standard-linear_v-bias-attn_rejected` | — | — | **rejected (local, pre-official); expansion plan A2 V output-bias centroid: per-KV-head b = 0.5 * fold-median of mean(O_ref - O_parent), added to V right after the NVFP4 decode. Controls perfect (Q/K states bit-identical). Rejected on three independent grounds: mechanism-level premise absent (output-bias correction only -1.7%..+0.6%, parent output bias is Q/K-dominated), Qwen default -0.0093 (21+/99-, V-only -0.0154), GPT-2 0/4 all-negative (-0.0172) triggering the plan 12-step-9 cross-model structural-reversal block. It was not submitted** |
| v170 | `20260903_v170_standard-linear_fixed-offset-attn_rejected` | — | — | **rejected (local, pre-official); expansion plan A3 static compile of the dynamic scale search: fixed E6M2 offset + exact hierarchy, Q->K->V greedy on real attention output MSE. Winners 11/12 = 0 (standard scale already output-optimal). Cross-model structural reversal, far stronger than v169: Qwen default -0.0506 (9+/111-, all lengths negative) and GPT-2 -0.055 (1+/3-, k_only -0.093). The dynamic refine is a load-bearing part of the official 12944 Attention contribution. A1 multiplier bit-identical (control clean). It was not submitted; A3 closed, next A4**

## 2026-09-01 official-shape-v1 local candidates

These are local reproductions only; no official score/time is inferred from them.  Their
directories follow the same immutable naming rule as the historical archive:
`YYYYMMDD_vNNN_<description>_scoreNA_timeNA`.

| Version | Source directory | Linear mean | Attention mean | API total | Decision 
|---|---|---:|---:|---:|---
| v189 | `20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA` | **0.640258** | **0.752173** | **388.175 s** API / `279.956 s` predicted | **local retained; official `17616/275s`; root switched to v189** |
| — | `20260906_linear-dynamic-actorder_time-rejected` | **0.643867** | **0.752173** | **402.969 s** API / `284.776 s` predicted | **local score high, rejected by official-time gate; root remains v189** |
| v086 (idle rerun) | `20260830_v086_c86-attn-block-final_scoreNA_timeNA` | 0.406668 | 0.719696 | 299.302 s | clean rerun; official 16744/222.7 s pass 
| v128 | `20260901_v128_fixed-attn-budget_timeout` | 0.465655 | 0.837789 | 310.732 s | **official timeout (user confirmed)** 
| v129 | `20260901_v129_fixed-attn-budget-sweep1_timeout` | 0.465655 | 0.836579 | 248.363 s | **official timeout (user confirmed)** 
| v130 | `20260901_v130_output-weight_timeout` | 0.471837 | 0.836579 | 295.437 s | **official timeout (user confirmed); Attention-time risk** 
| v131 | `20260901_v131_output-weight-qwgram_timeout` | 0.473131 | 0.836579 | 294.835 s | **official timeout; high-cost Attention family** 
| v132 | `20260901_v132_output-weight-qwgram-dynsweep2_scoreNA_timeNA` | 0.473131 | 0.834256 | 290.936 s | historical parent; 2 idle runs API<300 
| v133 | `20260901_v133_output-weight-qwgram-gain_scoreNA_timeNA` | 0.483610 | 0.834256 | 287.941 s | historical parent 
| v134 | `20260901_v134_linear-output-activation-cross64_scoreNA_timeNA` | 0.507320 | 0.834256 | 289.042/289.832 s | Linear precision parent; Attention-time risk 
| v135–v137 | three directories explicitly suffixed `_rejected` | 0.500132–0.507163 | 0.834256 | 287.816–296.755 s | **rejected Jacobi/sweep variants** 
| v138 | `20260901_v138_attention-static-v86-budget_scoreNA_timeNA` | **0.507320** | 0.715942 | **192.996/187.935 s** | **official 15715/208 s pass; time parent** 
| v139 | `20260901_v139_linear-output-aware-gain_scoreNA_timeNA` | 0.507278 | 0.715942 | 193.389 s | **official 15716/202 s pass; retained official-result archive** 
| v140 | `20260901_v140_linear-roab-pair_rejected` | 0.507355 | 0.715942 | 205.365 s | **rejected; official 15838/207 s, local-only gain `+0.000035`** 
| v141–v145 (BDLR family) | — (source snapshots deleted; logs/artifacts retained) | 0.281760–0.506256 | 0.715942 | 204.681–211.460 s | **rejected family; selected-column BDLR closed** 
| v147 | `20260901_v147_v86-attention-v140-linear_rejected` | **0.507355 / 0.510050†** | **0.719696** | **222.227 / 300.351 s†** | **official 16579/211s; rejected below v86; submitted SHA unconfirmed** 
| v148 | `20260901_v148_joint-wa-v86-attention-v140-linear_rejected` | **0.509729** | **0.719696** | **369.038 s** | **rejected; A3 precision gain but local time over 300 s** 
| v151 | `20260902_v151_proj-roab-off_rejected` | **0.582528 (targeted)** | **0.942927 (targeted)** | **193.213 s** | **rejected; Qwen role panel no-op, GPT-2 proj-only gain** 
| v152 | `20260902_v152_fc-cat-off_rejected` | **0.583139 / 0.542553 (14/56-case)** | **0.942927** | **199.578/200.432 s** | **rejected; small mixed-sign fc gain** 
| v153 | `20260902_v153_fc-decoupled-activation_rejected` | **0.568754 (targeted)** | **0.942927** | **197.656 s** | **rejected; direct s_q assignment regresses fc** 
| v154 | `20260902_v154_fc-decoupled-scale-fit_rejected` | **0.568754 (targeted)** | **0.942927** | **198.098 s** | **rejected; fitted s_d is a no-op after v153** 
| v155 | `20260902_v155_l5a-permutation-stability_rejected` | **0.570999 (default)** / 0.588162 (effect) | **0.724735 (default)** / 0.757433 (effect) | **248.121 s (default-equivalent)** / 207.196 s (effect) | **rejected; official 16581/208.5s, 163 points below v86** 
| v156 | `20260902_v156_l4-weight-decoupled_rejected` | 0.588131 (effect; default not run) | 0.757433 (effect) | 203.994 s (effect) | **rejected; official 16580/204.3s, 164 points below v86** 
| v157 | `20260902_v157_v86-roab-only_rejected` | NA (legality smoke only) | NA (frozen field-equality check) | NA | **rejected; official 16729/218.96s, 15 points below v86** 
| v158 | `20260902_v158_v86-attention-matrix-smooth_retained` | 0.448180 (default; frozen) | 0.735752 (default) | 295.069 s (default) | **retained; official 16861/223s, +117 vs v86** 
| v159 | `20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA` | **0.705508 CUDA compact / 0.633526 CUDA Linear default** | frozen v158 | **51.055s compact after exact reuse / 269.435s pre-reuse default API** | **official 17532 binds original SHA; current archive not yet resubmitted** 

† The first v147 values come from the original pre-A3 JSON (SHA `9B3EA5...B656`); the second come
from the later direct-merge A3 JSON (SHA `25C245...9C1B`). The archive was modified in place before
the official result was reported, so neither SHA is claimed as the submitted source without further
evidence. The current archived source SHA is `44E377...2672`.

\* The later temporary gain+adyn2 run reported `365.818 s`, but the machine was concurrently busy;
its timing is excluded from runtime ranking. The persisted v133 archive was rerun idle at `291.275 s`;
the active root file was then rerun directly at `287.941 s`.  v134 adds the
output-supervised activation cross term; its two complete runs are recorded in
[`v134 first JSON`](../artifacts/official_eval/v134-linear-output-activation-cross64-official-shape-v1.json)
and [`v134 idle rerun JSON`](../artifacts/official_eval/v134-linear-output-activation-cross64-rerun2-official-shape-v1.json).

The root file currently contains v140 for audit, but the next implementation baseline is the exact
v86 source. v140 keeps the v138 reduced Attention path and adds the ROAB-P2 reciprocal pair
transform to Linear; its local gain is only `0.000035`, and its official result is `15838 / 207 s`
(below v86), so it is rejected. v138 disables
the per-call Attention Gram refinement and shrinks the static candidate set. Two v138 runs are recorded in
[`v138 first JSON`](../artifacts/official_eval/v138-attention-static-v86-budget-official-shape-v1.json)
and [`v138 idle rerun JSON`](../artifacts/official_eval/v138-attention-static-v86-budget-rerun2-official-shape-v1.json).
The v138 official result was reported as **`15715 / 208 s` (pass)**, v139 as **`15716 / 202 s` (pass)**,
and v140 as **`15838 / 207 s` (pass but rejected as inferior)**. The v140 full run is recorded in
[`v140 JSON`](../artifacts/official_eval/v140-linear-roab-pair-official-shape-v1.json);
it gives Linear `0.5073546371`, unchanged Attention `0.7159419612`, and API `205.365 s`.

The BDLR-JAQ trials v141–v145 are recorded as a rejected family summary. Their local Linear means
were `0.281760`, `0.282559`, `0.361154`, `0.506418`, and `0.506256`; all kept Attention at
`0.715942` and stayed below the local time proxy, but none improved v140. Their source snapshots
were deleted to keep the archive compact; the per-run JSON and execution logs remain as evidence.
The selected-column BDLR direction is closed. The next work starts from v86, builds legal structural
oracles, and then tests null-space shaping and subspace-embedded joint vector rounding.

## External hif4 GPT-2 cross-check

The upstream [youxilee/hif4](https://github.com/youxilee/hif4) `real_data_eval.py` was also run
on the v84/v86/v140/v147 snapshots with one fixed 12-layer GPT-2 configuration
(`amax6`, `seq=128`, `calib=2`, `test=2`, `config=current`). Its Linear/Attention means were
`0.586733/0.4477`, `0.586733/0.4727`, `0.599617/0.4661`, and `0.599617/0.4713`, respectively.
This is a diagnostic only: the script repeats a built-in synthetic text when short and obtains
the standard baseline from candidate-private codec functions, so its ordering does not replace
the canonical `proxy-v2` evaluator or official results. See the full
[`external run log`](../logs/execution/2026-09-01-hif4-external-gpt2-v84-v86-v140-v147.md).

### External role attribution

The same hif4 run gives a more actionable result than the aggregate ordering. Relative to v86,
v140 improves static `q/k/v` by `+0.0409/+0.0900/+0.0085`, is nearly neutral on `o` (`-0.0018`),
but regresses `fc` in all 12 GPT-2 layers (`-0.0452`) and has a mixed `proj` regression
(`-0.0153`, including a `-0.1634` layer). A temporary role-gated ablation raises `proj` from
`.5221` to `.5430` when its ROAB is disabled; disabling fc ROAB is a no-op, while disabling fc
BOAT is harmful (`.5107` to `.4599`). The next Linear work therefore freezes q/k/v/o, tests
proj ROAB-off first, and redesigns fc's expansive encoder/scale while retaining BOAT. This is a
role diagnostic, not an official-score claim; the full evidence and protocol caveats are in
[`role attribution log`](../logs/execution/2026-09-01-hif4-external-role-attribution-v140-v86.md).

The first follow-up control, v151, disabled ROAB only for native `rows < channels` (`proj/down`)
matrices while removing the over-budget A3 pass. On the canonical Qwen `proxy-v2` targeted panel,
all seven static Linear role means were identical to the pre-A3 parent (`0.582528` Linear,
`0.942927` Attention); the external four-layer GPT-2 smoke improved only `proj` (`.5029→.5658`).
It is archived as `REJECTED` and remains a cross-model control rather than a new root parent. See
[`v151 execution log`](../logs/execution/2026-09-02-v151-proj-roab-off.md).

The next fc controls were also kept out of root. v152 disabled only expansive CAT (BOAT retained):
Qwen Linear moved `0.582528→0.583139` on the 14-case smoke but only
`0.542366→0.542553` on the paired 56-case panel, with mixed fc layer signs; external GPT-2 moved
fc `.5658→.5709`. v153 then tried the first L1 decoupled encoder, using BF16 `s_q` directly for
fc code assignment without fitting `s_d`, and regressed Qwen Linear to `0.568754` (fc family
`0.334432`). Both are archived as `REJECTED`; the failure points to the planned closed-form stored
scale fit, not another CAT/ROAB switch. See
[`v152/v153 execution log`](../logs/execution/2026-09-02-v152-v153-fc-followups.md).

v154 added the planned fixed-code `s_d` fit in the deployed `Q(W)` Gram metric, but its Qwen role
means were exactly v153 (`fc=.334432`, Linear `0.568754`). It is archived as `REJECTED`; direct
decoupled-scale variants are paused. L3-D0 then found same-fold teacher margin but an exact
layer-3/fold-128 output regression (`fc_gate=-0.094751`, `fc_up=-0.112680`), so the result is
`margin_exists_but_not_compile_safe`. The batched stability probe found no fixed threshold/LUT
student (held-out precision zero), so direct activation encoder compilation is closed. Its only
positive remnant is v155: a fixed four-quartile pressure interleave behind BOAT with a
fold-disagreement gate. It changes only four default Linear cases, gives paired
`+0.000116536` with no regressions, and leaves controls/Attention unchanged; the strict GPT-2
pair is `−0.000153` (fc `−0.000916`), so this is a Qwen-local coordinate control, not a new
official baseline or parent. The first L2 analytic pair-balance probe remains rejected (`fc` paired
mean `-0.314079`, 16/16 regressions). The next experiment is therefore a single-pass
Weight-decoupled or deployment-Gram block-Schur mechanism branched from pre-A3, not another
permutation/scale/CAT sweep.

v157 was the exact-v86 single-variable ROAB experiment. It starts from the reproducible v86
source and adds only the bounded ROAB-P2 reciprocal 2×2 Linear transform, while keeping v86
Attention field-for-field frozen. The direction comes from the clean fixed-Attention official
increment `v138→v140 = +123`, not from a local ranking. No local model panel was run for v157;
legality, continuous-product/covariance invariants, rejected-candidate parent equality, selected
state propagation and isolated import passed. Its SHA is
`984BF752156187B8892894060A99FE52027E2457F37FC23C11657041B29B86E1`. Its official result is
`16729 / 218.96s`: time passes but accuracy is 15 points below v86, so it is rejected. This proves
the earlier `v138→v140 +123` ROAB increment was context-dependent, not portable to exact v86.
v155 (`16581 / 208.5s`) and v156 (`16580 / 204.3s`) remain rejected as well. The next planned
mechanism is a single-pass block-Schur HiF4-GPTQ branch from exact v86.

\* The earlier v086 local `462.239 s` observation was also concurrent-load affected. The clean
idle rerun is `299.302 s` API / `321.996 s` wall; see
[`v086 idle rerun`](../artifacts/official_eval/v086-idle-rerun-20260901-official-shape-v1.json).

## Recording rules

1. Parameter sweeps stay in one unnumbered workbench and one summary log. Allocate a version only
   for a new mathematical algorithm, an official submission, or one representative failure.
2. Final directory names include `retained`, `rejected`, or `timeout`. Unknown official values
   stay `scoreNA_timeNA`; local JSON values never enter Official fields.
3. `result.md` records parent, one algorithm change, exact command/protocol, data/model revisions,
   both means, API/Wall time, source SHA256, an explicit `Status` (`RETAINED`, `REJECTED`,
   `TIMEOUT`, or `ERROR`), official outcome, and next decision.
4. When the official judge changes weights or limits, start a new protocol label and keep old
   outcomes as history; never mix their absolute scores.
5. Small parameter sweeps are grouped under one experiment log and one summary result. Rejected
   micro-variants are not archived one-by-one; their source snapshots may be deleted after JSON and
   execution evidence is retained. Version identifiers are globally unique.
