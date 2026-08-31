# HiF4 唯一活跃优化计划 v5：v125 结构化 Linear 跨模型审计与可压缩路径

> 状态：**ACTIVE**
> 建立日期：2026-08-31
> 适用根：`D:/工作内容/AI竞赛/solution.py`
> 当前精度 parent：v125 C1c structured rank-8 / max-blocks-8 + C1b refresh（sweep2）；
> 当前本地时间 parent 仍为 v106（`<420s`）；v107 官方 Attention WA 后，官方 smoke
> 首选改为 v100（`392.42s`，比 v106 多 `20.23s` API 余量且无完整 `deployment_gram`）
> 根 `solution.py` 规范 LF SHA256：`c9b419717e38bcec69d907d1cab6638409f1fa9a3072892dde9494ef9da3cc8e`
> 主目标：保持 v125（继承 v119）的完整部署 `G_q` exact gate 语义，继续验证 Qwen full-layer
> `linear_mean`；同时把结构化 proposal 的原型实现压缩为可审计的 C1 路径。Attention
> 只作回归检查，不在本计划中扩展 PAWV。

## 1. 唯一执行规则

本文件是唯一可执行计划。执行优化时只读取本文件、根 `solution.py`、最新可复现
JSON/日志和官方规则；`docs/superpowers/archive/plans/` 只作历史证据，不产生顺序。

每个候选严格按以下顺序执行：

1. 先保存 parent SHA 和工作树快照；合成测试必须覆盖合法五字段、二次型增量、
   structured matmul 方向、finite fallback、CPU state 和 state 节点 `<4096`；
2. 先跑 Qwen 五层 `{0,5,11,17,23}` × 七 role screen，固定 cache、至少两折 calibration；
3. screen 只有在 Linear mean/panel 明确高于 parent 且无 role 回退时才进入 full-layer；
   full 只以 Qwen `linear_mean`/panel 晋级，Attention 只防回归；
4. 每次结果（成功、失败、no-op、超时）先归档完整源码、JSON、日志和 README，再
   更新本计划账本；失败候选不能留在根；
5. accuracy-first 阶段时间只记录，不因超过 420s 否决精度；C1 最后才恢复官方
   `<420s` 硬门。任何“加速”候选不得改变 exact deployed-Gram row gate 的数学含义；
6. 本计划的队列完成或连续两个方向无正向后，立即归档本文件，并在同一提交创建
   下一份唯一 active 计划，不能在归档文件追加新的下一步。

### 1.1 评测分层规则（本轮起固定）

为避免每个候选重复消耗数十分钟，精度排序采用 Qwen 主线：

1. 合成/合规/五层七 role screen 是所有候选的第一道门；
2. 只有 screen 明确正向且无 role 回退才跑 Qwen2.5-0.5B 全 24 层；
3. Qwen full 是本地精度排序的唯一硬门。OPT/Pythia 只在 precision parent 变更后或
   每 2–3 个候选做 3–5 层、少量 role 的软 guardrail，不再对每个候选跑五模型全量；
4. “Qwen full 足够”只适用于本地相对排序，不能替代官方提交前的 Attention API smoke、
   state/shape 合规检查和一次跨模型回归。

## 2. 固定基线、目标和约束

固定评测：Qwen2.5-0.5B、24 层、`seq=128`、`calib=2`、`test=4`、`amax6`、CPU、
只读 cache `artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`。

| 指标 | v125（C1c rank-8 / max-blocks-8 + C1b sweep2） |
|---|---:|
| screen Linear mean | `0.53358298` |
| full Linear mean | `0.5097598050` |
| Attention mean | `0.8420394885` |
| Qwen panel | `295.8478489516` |
| native total | `423.3943798775` |
| API time | `2653.580314s`（runtime invalid） |
| LF SHA | `c9b419717e38bcec69d907d1cab6638409f1fa9a3072892dde9494ef9da3cc8e` |

到 `linear_mean=0.9` 仍差 `0.3903987445`，需消除当前剩余归一化误差的
`79.6084%`；这只是本地诊断轴，不能换算官方 36000。所有在线候选必须遵守：

若只在本地诊断轴固定当前 Attention `0.8420394885`，panel 达到 `360` 所需的
Linear mean 实际为

\[
g_L^{360}=\frac{360-200\times0.8420394885}{250}=0.7663684092.
\]

v125 到该值仍需消除 `52.3434%` 的剩余 Linear 误差；`0.9` 是更激进的冗余目标，
不是 36000 的必要条件。官方隐藏分布未知，以上只用于判断所需算法量级。完整推导见
[`当前实验结果与可达性 checkpoint`](../../../logs/execution/2026-08-31-current-results-target-feasibility.md)。

- 输出仍是 HiF4 五字段 `scale_factor/scale_lv2/scale_lv3/sign/mant`；
- state 只含 calibration/offline 生成的 CPU finite 静态统计，不含 token、输出或
  test/holdout 信息；
- proposal 可以使用压缩的 `G_q` 近似，但写回前必须用真实部署
  `G_q=W_q^TW_q` 做逐行 gate；
- 不新增未经压缩的 block-pair Gram，不把 `A@W` 输出监督流入 `Q(A)`。

## 3. 数学基线

对一行 activation 误差 `e=q-x`，部署目标为

\[
J(e)=e^T G_q e,
\qquad G_q=W_q^TW_q.
\]

候选离散坐标改变 `q` 为 `q+\Delta q` 时，精确增量是

\[
\Delta J=2e^TG_q\Delta q+\Delta q^TG_q\Delta q.
\]

v118 的结构化 proposal 将跨 block 残差写成

\[
R_{ij}\approx \sum_{s=1}^{S}c_{j-i,s}K_s,
\qquad K_s\in\mathbb{R}^{64\times64},\ S\le4,
\]

用循环 roll 计算 `R e`，只负责产生离散候选；真正接纳仍比较 parent/candidate
两行的完整 `e^TG_qe`。任何 C1 加速必须证明它与上式在浮点容差内等价，或明确标记
为新的精度候选而不是“实现等价”。

## 4. 执行队列

### C1a：结构化 proposal 的向量化等价实现

**状态：completed（v119）。**

入口：`_structured_gram_matmul`、`_refine_activation_structured`。

把当前逐 row/逐 coordinate 的 15-level Python 循环改成批量 tensor 运算：一次构造
候选 `step`，批量计算局部 `h` 梯度、冻结 structured 梯度和精确 gate 所需的
`G_q` 行积；候选选择顺序、tie break、`_write_codes` 和 exact row gate 必须保持
一致。已用 v118 的宽层样本做 `1e-6` 级字段对照，并完成 screen/full。

验收结果：screen `0.5333753185` 与 v118 完全相同；full Linear `0.5096012555`、
Attention `0.8420394885`、panel `295.8082115559` 的全部分数位与 v118 相同；
API 从 `2249.7464359s` 降至 `2040.5046895s`（`-9.30%`），dynamic 从
`1832.8779521s` 降至 `1633.3390318s`（`-10.88%`），因此 v119 接替为 parent。

### C1b：structured gradient refresh / 多 sweep

**状态：completed（v121 accepted；v120 rejected）。**

当前 proposal 在整行坐标扫描中冻结 `R e`，导致已接受坐标后的跨 block 梯度过期。
C1a 已证明批量化不改变 v118 的精度；本节从 v119 等价 parent 继续，只改变
proposal 梯度刷新策略。
在同一合法 state 上测试两种低自由度变体：每个 block 一次增量 refresh，或最多两轮
block sweep。每次 refresh 只更新 proposal 梯度，候选仍由完整 `G_q` row gate 裁决。
v120 的一次 refresh screen 为 `0.5333730058`，低于 v118 `0.5333753185`，已归档拒绝；
v121 的两轮 refresh screen 为 `0.5333964596`，进入 full 后 Linear `0.5096135327`、
panel `295.8112808759`，相对 v119 分别 `+0.0000122773`、`+0.0030693200`，且
Attention 与除 `proj` 外的 Linear role 不变，故按 accuracy-first 规则接替 parent。
v121 screen/full 产物位于
[`v121 archive`](../../../solutions/20260831_v121_c1b-structured-refresh2-accepted_score295.811281_time2180s/)。
合成单调性、38 项定向测试和 compliance 均通过；full `official_flow_valid=false` 的
唯一原因是 CPU API `2180.45s` 超过 420s，暂不作为精度否决。

### C1c：结构化 rank / block budget 的精度扫描

**状态：completed（v125 precision-only；runtime invalid）。**

在 C1a/b 的最佳实现上逐变量扫描 `S∈{2,4,8}`、`max_blocks∈{2,4,8}`，不同时
改变两个旋钮。state 成本按

\[
\text{bytes}=S\cdot64^2\cdot4+(d/64)\cdot S\cdot4
\]

计算；screen 只保留超过 v118 的组合，full 只跑最高 screen 组合。若两个连续组合
 无增益，停止该族。

已完成第一项 rank 扫描：v122 固定 `max_blocks=4`、`refresh_mode=sweep2`，仅将
`S=4→2`，screen Linear `0.53336284`（较 v121 `−0.00003362`，也低于 v118），
因此拒绝并保留 v121；完整源码/JSON/报告见
[`v122 archive`](../../../solutions/20260831_v122_c1c-rank2-rejected_screen0.533363_time426s/)。

第二项 block budget 扫描：v123 固定 `S=4`、`refresh_mode=sweep2`，仅将
`max_blocks=4→2`，screen Linear `0.53335171`（较 v121 `−0.00004475`），因此拒绝；
完整源码/JSON/报告见
[`v123 archive`](../../../solutions/20260831_v123_c1c-block2-rejected_screen0.533352_time430s/)。

第三项 rank 扫描：v124 固定 `max_blocks=4`、`refresh_mode=sweep2`，仅将 `S=4→8`，
screen Linear `0.53343639`（较 v121 `+0.00003993`），full Linear `0.5096493233`、
panel `295.8202285103`（较 v121 `+0.0089476344`），7 个 role 均不降，故接替当前
precision parent；完整源码/JSON/报告见
[`v124 archive`](../../../solutions/20260831_v124_c1c-rank8-accepted_score295.820229_time2324s/)。
其 API `2323.911178s` 超过 420s，仅作 accuracy-first 记录。随后固定 `S=8` 测试
`max_blocks=8`：screen `0.53358298`，full Linear `0.5097598050`、panel
`295.8478489516`，较 v124 分别 `+0.0001104818`、`+0.0276204413`；Attention
逐位不变，但 API `2653.580314s`，因此只保留为 precision-only 证据，不作为提交版本。
完整源码、screen/full JSON、报告和结论见
[`v125 archive`](../../../solutions/20260831_v125_c1c-block8-precision-only_score295.847849_time2654s/)。
C1c 队列到此停止，不再增加 block budget；下一步按固定分层规则进入 C2/C3。

### C2：跨 fold / 多模型泛化审计

**状态：pending。**

对 C1 最高 parent 在固定 Qwen screen 后增加低成本 OPT-125m、Pythia-160m
（若 cache 可用）五层复筛；统计 `proj` 与其他 role 的 delta、proposal recall 和
`J_64` 下降。多模型只作软 guardrail，不以单个弱模型覆盖 Qwen 主门禁；但若出现
明显结构性回退，禁止扩大 state 或 block budget。

### C3：state/时间压缩 checkpoint

**状态：pending。**

汇总 C1/C2 的 precision、recall、`J_64`、state bytes 和 API 时间。按单变量顺序执行：

1. 用 `e^TG_qe=||eW_q^T||^2` 的部署权重因子替代宽投影的 dense `G_q`，要求 keep
   mask/五字段与 reference 一致；
2. 对 selected-block sparse `delta` 使用
   `2e^TG_q delta + delta^TG_q delta` 做增量 exact gate，避免 parent/candidate 两次
   dense quadratic form；
3. structured sweep2 接受一个 block 后增量更新 circular kernel gradient，不再对完整
   row/block tensor 重跑 `_structured_gram_matmul`；
4. 仍不足时才验证 block 轴 FFT circular convolution；
5. 输出 v100/v106/v125/压缩候选的精度、state 峰值和 API Pareto，最终恢复 `<420s`
   硬门并做 Attention contract smoke。

未经 exact-equivalence 对照的近似不能成为提交 parent。C3 完成后归档本计划；下一份
唯一 active 计划才进入共享正交 butterfly/Givens frame 和冻结 activation state 后的
完整离散 JDRQ-weight，不在本计划继续增加 rank/block/offset。

## 5. 候选账本

只登记固定 cache 的 full-layer accepted parent；screen/rejected/no-op 候选在对应
`solutions/` 与 execution log 归档。

| 版本 | Linear | Attention | panel | API time | 状态 |
|---|---:|---:|---:|---:|---|
| v116 | 0.5093045894 | 0.8420394885 | 295.734045 | 739.42s | 前一 parent |
| v117 | 0.5095117268 | 0.8420394885 | 295.785829 | 2019.48s | 前一 parent |
| v118 | 0.5096012555 | 0.8420394885 | 295.808212 | 2249.75s | L6d precision parent |
| v119 | 0.5096012555 | 0.8420394885 | 295.808212 | 2040.50s | C1a 等价时间 parent |
| v120 | NA（screen `0.5333730058`） | NA | NA | 419.63s screen | C1b block refresh rejected |
| **v121** | **0.5096135327** | **0.8420394885** | **295.811281** | **2180.45s** | **当前 parent；C1b completed** |
| v122 | NA（screen `0.53336284`） | NA | NA | 425.70s screen | C1c rank-2 rejected |
| v123 | NA（screen `0.53335171`） | NA | NA | 429.95s screen | C1c max_blocks-2 rejected |
| **v124** | **0.5096493233** | **0.8420394885** | **295.820229** | **2323.91s** | 前一 precision parent；C1c rank-8 |
| **v125** | **0.5097598050** | **0.8420394885** | **295.847849** | **2653.58s** | **precision-only accepted；runtime invalid** |

## 6. 完成条件

C1a–C3 完成，或连续两个方向有充分 `not actionable` 证据后：把本文件标记
`COMPLETED`，保存每个候选的源码/配置/结果/结论，移入
`docs/superpowers/archive/plans/`；同一提交更新两个计划 README、根 README、当前
状态、算法清单、实现审计和 `solutions/README.md`，并创建下一份唯一 active 计划。
