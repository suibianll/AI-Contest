# 当前主版本：算法与评测状态

更新：2026-09-01。根目录 [`solution.py`](../solution.py) 是当前已验证且满足本地 API `<300s` 代理的活动版本；归档源码只读。
本页只描述当前状态，历史算法的逐项证据见 [`algorithm-inventory-and-directions.md`](algorithm-inventory-and-directions.md)
和 [`archive-implementation-audit.md`](archive-implementation-audit.md)。

## 评测口径

唯一活动评测器是 [`evaluator/official_eval.py`](../evaluator/official_eval.py)，协议
`official-shape-v1`。它固定 Qwen2.5-0.5B 的 24 层、250 Linear + 200 Attention case，
Attention calibration 长度 `[10,128,512,1024,1024]`，并在每个 API 边界独立校验参数和 state。
`Linear calibration=2` 是公开本地数据包的明确假设，因为赛事说明书没有给出 Linear 折数；
该假设不能冒充官方隐藏数据。

每个 case 的分数为

\[
s=(MSE_{STD}-MSE_{PLAYER})/MSE_{STD},
\]

报告主指标是 250 个 Linear case 的算术平均 `linear_mean` 和 200 个 Attention case 的
算术平均 `attention_mean`。`total_sum` 是 450 个 case 分数的和；`equal_weight_45000_scale`
只是乘 100 的等权显示。官方最近减少 Linear 权重且未公布系数，因此不能从本地值拟合官方
绝对分。

时间分为六 API 的 `api_total_seconds` 和含本地评分开销的 `wall_seconds`；二者仅能做同机、
同 cache、同协议 A/B，不能直接推断鲲鹏 920B 的 `<300s` 结论。

## 活动根版本

根代码目前是 v140（v138 静态 Attention + ROAB-P2 Linear reciprocal pair transform），
SHA256 为 `52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`。实现保留：

1. BOAT 对角平衡与 signed-Hadamard 等价变换，满足 `X'W'^T=XW^T`；
2. Cross-fold Weight-HSDQ，在 64 通道块上使用校准激活统计做合法 HiF4 离散搜索；
3. Gram-hierarchy Activation-HSDQ，使用最终部署 Q(W) 的静态 Gram 做有限预算激活候选筛选；
4. Expansive-FFN CAT balance，仅在 `rows>channels` 的形状启用并失败回退；
5. `Q(A)`/`A@W` 输出监督的分块权重精修、`Q(W)^T W` 分块 cross 项、output gain 和
   在线输出目标激活精修；
6. Attention 的低复杂度 reciprocal balance、K-centering、少量 block-Hadamard/GQRB 静态
   shortlist；候选只在 128-token view 上评分，关闭动态 Q/K Gram sweep，dense PAWV 已移除。

已从根代码裁剪并仅保留在归档中的方向：Global Activation-LRH、final deployed-Gram
row gate、GALS、block-local permutation、L6 rank/factor 系列和 C1 refresh/rank 系列。
它们的 full-layer 结果普遍超过 300 秒或没有稳定的跨折收益，不是当前发布门禁。

## 新协议复测结果

批量结果写入 [`artifacts/official_eval/archive-official-shape-v1.json`](../artifacts/official_eval/archive-official-shape-v1.json)，
报告写入 [`logs/official_eval/archive-official-shape-v1.md`](../logs/official_eval/archive-official-shape-v1.md)。
表中本地数值必须来自同一 cache 和同一设备；`error` 是该归档在本机 CUDA 上的真实运行错误，
不能用零分替代。

| 版本 | Linear mean | Attention mean | API total(s) | Wall(s) | 官方裁决 |
|---|---:|---:|---:|---:|---|
| v001 | 0.315499 | 0.449961 | 36.868 | 59.582 | pass |
| v002 | — | — | — | — | 本机 device-mix error |
| v013 | 0.437667 | 0.639206 | 71.433 | 95.076 | pass |
| v024 | 0.450075 | 0.639206 | 76.148 | 99.192 | pass |
| v025 | 0.387146 | 0.639206 | 77.815 | 101.034 | pass |
| v030 | 0.265321 | 0.639206 | 93.005 | 116.100 | pass |
| v031 | 0.374651 | 0.639206 | 87.008 | 109.642 | pass |
| v032 | 0.356338 | 0.639206 | 143.127 | 165.348 | pass |
| v034 | 0.374651 | 0.639206 | 87.730 | 109.926 | pass |
| v051 | 0.355097 | 0.639206 | 163.484 | 185.502 | pass |
| v066 | 0.357919 | 0.639242 | 165.801 | 187.881 | pass |
| v072 | 0.365587 | 0.639242 | 165.616 | 187.545 | pass |
| v074 | 0.372739 | 0.639242 | 180.526 | 202.576 | pass |
| v084 | 0.406668 | 0.718107 | 279.191 | 300.848 | pass（新权重） |
| v086（空闲重测） | 0.406668 | 0.719696 | 299.302 | 321.996 | 官方 16744/222.7s pass；本地 API<300 |
| v098 | 0.465655 | 0.833573 | 406.681 | 429.285 | timeout |
| v100 | 0.465655 | 0.833617 | 417.747 | 439.896 | Attention WA/timeout |
| v107 | 0.469211 | 0.833617 | 436.719 | 459.727 | Attention WA |
| v121 | 0.472198 | 0.833617 | 3404.369 | 3429.645 | timeout |
| v127 | 0.465655 | 0.833617 | 416.465 | 439.617 | 新协议 root 基线 |
| root-v127-no-pawv | 0.465655 | 0.833573 | 405.851 | 428.122 | PAWV 移除 |
| v128 | 0.465655 | 0.837789 | 310.732 | 332.557 | **官方 timeout（用户确认）** |
| v129 | 0.465655 | 0.836579 | 248.363 | 270.606 | **官方 timeout（用户确认）** |
| v130 | 0.471837 | 0.836579 | 295.437 | 317.607 | **官方 timeout（用户确认）** |
| v131 | 0.473131 | 0.836579 | 294.835 | 317.708 | **官方 timeout（用户确认）** |
| v132 | 0.473131 | 0.834256 | 290.936 | 314.251 | 两次空闲重测，历史父版本 |
| v133 | 0.483610 | 0.834256 | 287.941 | 310.621 | 历史父版本 |
| v134 | 0.507320 | 0.834256 | 289.042/289.832 | 312.315/313.181 | 历史 Linear 精度父版本；Attention 时间风险 |
| v138 | **0.507320** | 0.715942 | **192.996/187.935** | 216.324/210.855 | **当前根；v86 级静态 Attention 时间父版本** |
| v139 | 0.507278 | 0.715942 | 193.389 | 217.196 | 输出感知连续 gain 回退，拒绝 |
| v140 | **0.507355** | 0.715942 | **205.365** | 229.337 | **当前根；ROAB-P2 正向 Linear 候选** |
| v141 | 0.281760 | 0.715942 | 204.681 | 228.127 | BDLR rank-4 列式修正回退，拒绝 |
| v142 | 0.282559 | 0.715942 | 211.460 | 234.842 | BDLR 锚点冻结仍回退，拒绝 |
| v143 | 0.361154 | 0.715942 | 207.445 | 230.788 | BDLR 仅动态激活仍回退，拒绝 |
| v144 | 0.506418 | 0.715942 | 208.414 | 232.178 | BDLR 阻尼 0.02 回退，拒绝 |
| v145 | 0.506256 | 0.715942 | 208.513 | 232.206 | BDLR 阻尼 0.005 回退，拒绝 |

本轮统一复测中，旧归档**最高本地等权显示**为 v121：`linear_mean=0.472197763`、
`attention_mean=0.833617251`、`equal_weight_45000_scale=28477.289`，但 API/Wall 均远超
300 s，且官方历史裁决为 timeout；它不是可提交版本。v132 两次空闲机器重测 API 为
`290.936s` 和 `289.318s`，仅作历史父版本记录。当前根 v133 直接空闲重测为 Linear
`0.483610`、Attention `0.834256`、API `287.941s`、wall `310.621s`，API 满足 `<300s`；
同代码归档等价副本复测为 `291.275s`，同样满足限制。完整字段以[v133 根重测 JSON](../artifacts/official_eval/v133-active-root-rerun-20260901-official-shape-v1.json)、
[v133 归档等价重测 JSON](../artifacts/official_eval/v133-gain-adyn2-equivalent-idle-rerun-20260901-official-shape-v1.json)为准；
wall 字段仅作诊断，不作为官方计时。

v134 在相同 cache 上完成两次完整重测，Linear `0.5073195`、Attention `0.8342565`
逐位一致，API 分别为 `289.042s` 与 `289.832s`（均低于本地 300 秒代理），wall
分别为 `312.315s` 与 `313.181s`；详见 [v134 首次 JSON](../artifacts/official_eval/v134-linear-output-activation-cross64-official-shape-v1.json)、
[v134 第二次 JSON](../artifacts/official_eval/v134-linear-output-activation-cross64-rerun2-official-shape-v1.json)。
官方分数/时间尚未登记，不能由本地代理推断官方通过。

时间质量更正：最后一次 `root-v127-output-weight-qwgram-gain-adyn2` 运行时有其他程序同时
占用机器，报告的 `365.818s` API / `397.341s` wall 是受干扰观测，不作为超时判定；原始
JSON 保留，详见 [`时间质量更正`](../logs/execution/2026-09-01-timing-quality-correction.md)。
随后已在无其他 Python/评测进程的机器状态下连续重测 v132 两次，并分别重测 v133 根文件与
归档等价候选，API 均低于 `300s`。
v86 官方结果为 **`16744 / 222.7s`，新权重下通过**。

v86 的本次空闲 `official-shape-v1` 重测为 Linear `0.406668`、Attention `0.719696`、
API `299.302s`、wall `321.996s`；此前 `462.239/501.257s` 是受并发干扰的原始观测，
保留但不再代表当前本地时间。v128 fixed-attn-budget 的官方评测已由用户确认超时
（`>300s`，官方分数未返回）；其本地 `310.732s` 仅作代理记录，不能改写为官方时间。
v129 fixed-attn-budget-sweep1 虽然本地 API `248.363s`、wall `270.606s` 均低于 300 秒，
但官方评测同样已确认 timeout（`>300s`，分数未返回）；这进一步证明本地秒数不能保证
官方通过。

v130 的本地 API `295.437s`、wall `317.607s`，官方也已确认 timeout（`>300s`，分数未返回）。
该版本的本地时间分解为 Attention calibration `115.461s`、动态 Q/K/V `34.459s`；对比
官方通过的 v86（本地对应 `55.347s` 与 `5.761s`），Attention 路径是当前最明确的官方
超时风险。v129 本地总时长更低却同样 timeout，说明不能用单一比例映射本地与官方时间。
后续计划将 Attention 先收敛到 v86 级别的静态低复杂度，再在此时间父版本上恢复必要的
Linear 精度组件。

v131 的本地 API `294.835s`、wall `317.708s`，官方同样已确认 timeout（`>300s`，分数未
返回）。它与 v129/v130 共享 Attention calibration `115s+`、动态 Q/K/V `35s` 左右的高成本
路径，因此本轮只把它登记为官方超时，不把超时错误归因到 v131 新增的 Linear Q(W)-Gram。

v138 已完成该时间重构：在保持 v134 Linear `0.5073195` 的同时，把本地 Attention
calibration/动态 Q/K/V 降到 `36.25/3.71s` 量级，总 API `187.935–192.996s`；两次
结果逐位一致。它只作为官方时间的更保守候选，最终仍需平台实测。

v140 在 v138 上加入 ROAB-P2：校准阶段学习 reciprocal 2×2 pair transform，在线同时变换
激活与权重并以合法输出重构误差选择坐标系。统一复测 Linear `0.5073546371`，相对 v138
提升 `+0.0000351323`；Attention 保持 `0.7159419612`，API `205.365s`、wall `229.337s`。
该正向候选已归档，官方分数/时间仍未登记。

v141–v145 依次测试 rank-4 选列跨块修正、冻结锚点、仅动态激活以及阻尼 `0.02/0.005`。
Linear 分别为 `0.281760/0.282559/0.361154/0.506418/0.506256`，Attention 均为
`0.715942`，API `204.681–211.460s`。这些候选都没有超过 v140，故列式 BDLR 路线已关闭，
活动根恢复为 v140，下一步转入对称联合层级码字更新。

官方历史锚点（独立于本地代理）：v74 `22750 / 239.387s`（旧权重，通过）；v84
`16517 / 252.563s`（新权重，通过）；**v86 `16744 / 222.7s`（新权重，通过，当前
官方最佳：分数比 v84 高 `+227`、时间比 v84 快 `29.863s`）**；v98/v121 timeout；
v100/v107 Attention WA。

v86 尤其重要：它是**唯一在官方 300s 限制内通过、且改动了 Attention 路径**
（C86 Q/K 共享 block-Hadamard）的候选，证明 Attention 侧改动本身不必然导致
超时，与 v098/v100/v121 的 B1 GQRB / B2 PAWV 路径形成对照。方向含义：官方时间
风险来自"per-seq_len 分组 + Python 循环"型 Attention 机制（PAWV/GQRB），而非
Attention 改动本身。

## 下一步

当前 active 计划是 [`2026-09-01-hif4-linear-0.8-under-300s-plan.md`](superpowers/plans/2026-09-01-hif4-linear-0.8-under-300s-plan.md)。它只比较两个最终目标：精度和官方时间；
本地统一使用 `official-shape-v1` 记录 Linear/Attention 均值和六 API 时间。当前根 v140 已完成
完整 450-case 复测；v141–v145 回退并归档，活动根恢复为 v140。任何新版本必须保留源码 SHA、统一命名归档和
可复现 JSON。

## 归档规则

- `solutions/` 只存版本源码和 `result.md`，不覆盖、不重写历史；本地结果不得填入 Official 字段。
- 活动计划目录只保留一份计划，完成或废止的文件移动到 `docs/superpowers/archive/plans/`。
- 活动评测输出只写 `artifacts/official_eval/`、`logs/official_eval/`；旧
  `artifacts/real_model_suite/` 和旧 evaluator 不再读取。
