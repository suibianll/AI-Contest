# HiF4 唯一活跃优化计划 v5：v118 结构化 Linear 精度与可压缩路径

> 状态：**ACTIVE**
> 建立日期：2026-08-31
> 适用根：`D:/工作内容/AI竞赛/solution.py`
> 当前精度 parent：v119 C1a structured proposal vectorization（与 v118 精度等价）
> 根 `solution.py` 规范 LF SHA256：`c9c45a7911594b4b378d0c5e2769187d76dc587d79b6da9fa5f5a487e4b7cb11`
> 主目标：保持 v119（继承 v118）的完整部署 `G_q` exact gate 语义，继续提升 Qwen full-layer
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

## 2. 固定基线、目标和约束

固定评测：Qwen2.5-0.5B、24 层、`seq=128`、`calib=2`、`test=4`、`amax6`、CPU、
只读 cache `artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`。

| 指标 | v119（v118 等价） |
|---|---:|
| screen Linear mean | `0.53337532` |
| full Linear mean | `0.5096012555` |
| Attention mean | `0.8420394885` |
| Qwen panel | `295.8082115559` |
| native total | `423.2878345580` |
| API time | `2040.504690s` |
| LF SHA | `c9c45a7911594b4b378d0c5e2769187d76dc587d79b6da9fa5f5a487e4b7cb11` |

到 `linear_mean=0.9` 仍差 `0.3903987445`，需消除当前剩余归一化误差的
`79.6084%`；这只是本地诊断轴，不能换算官方 36000。所有在线候选必须遵守：

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

**状态：pending。**

当前 proposal 在整行坐标扫描中冻结 `R e`，导致已接受坐标后的跨 block 梯度过期。
C1a 已证明批量化不改变 v118 的精度；本节从 v119 等价 parent 继续，只改变
proposal 梯度刷新策略。
在同一合法 state 上测试两种低自由度变体：每个 block 一次增量 refresh，或最多两轮
block sweep。每次 refresh 只更新 proposal 梯度，候选仍由完整 `G_q` row gate 裁决。

验收：合成目标必须单调不增；Qwen screen 需高于 v118，且 `proj` 之外不能出现
结构性回退；否则归档为 `rejected`，不扩大 sweep。

### C1c：结构化 rank / block budget 的精度扫描

**状态：pending。**

在 C1a/b 的最佳实现上逐变量扫描 `S∈{2,4,8}`、`max_blocks∈{2,4,8}`，不同时
改变两个旋钮。state 成本按

\[
\text{bytes}=S\cdot64^2\cdot4+(d/64)\cdot S\cdot4
\]

计算；screen 只保留超过 v118 的组合，full 只跑最高 screen 组合。若两个连续组合
无增益，停止该族。

### C2：跨 fold / 多模型泛化审计

**状态：pending。**

对 C1 最高 parent 在固定 Qwen screen 后增加低成本 OPT-125m、Pythia-160m
（若 cache 可用）五层复筛；统计 `proj` 与其他 role 的 delta、proposal recall 和
`J_64` 下降。多模型只作软 guardrail，不以单个弱模型覆盖 Qwen 主门禁；但若出现
明显结构性回退，禁止扩大 state 或 block budget。

### C3：state/时间压缩 checkpoint

**状态：pending。**

汇总 C1/C2 的 precision、recall、`J_64`、state bytes 和 API 时间，评估两条保持
exact gate 的路线：批量使用现有 dense `G_q`，或离线保存可证明上界的 block-row
压缩统计。未经 exact gate 证明的近似不能成为提交 parent。只有 checkpoint 后才把
`<420s` 恢复为最终官方硬门。

## 5. 候选账本

只登记固定 cache 的 full-layer accepted parent；screen/rejected/no-op 候选在对应
`solutions/` 与 execution log 归档。

| 版本 | Linear | Attention | panel | API time | 状态 |
|---|---:|---:|---:|---:|---|
| v116 | 0.5093045894 | 0.8420394885 | 295.734045 | 739.42s | 前一 parent |
| v117 | 0.5095117268 | 0.8420394885 | 295.785829 | 2019.48s | 前一 parent |
| v118 | 0.5096012555 | 0.8420394885 | 295.808212 | 2249.75s | L6d precision parent |
| **v119** | **0.5096012555** | **0.8420394885** | **295.808212** | **2040.50s** | **当前 parent；C1a completed** |

## 6. 完成条件

C1a–C3 完成，或连续两个方向有充分 `not actionable` 证据后：把本文件标记
`COMPLETED`，保存每个候选的源码/配置/结果/结论，移入
`docs/superpowers/archive/plans/`；同一提交更新两个计划 README、根 README、当前
状态、算法清单、实现审计和 `solutions/README.md`，并创建下一份唯一 active 计划。
