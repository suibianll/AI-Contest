# Attention Q 侧加性 logit 偏置补偿计划（A-QB1）

> 状态：CLOSED / REJECTED，2026-09-10。
> 从属于[Linear 完整输出交叉残差纠码与双线协调计划](../2026-09-10-linear-cross-residual-correction-plan.md)。
> 当前完整根为 v202 Linear + v195 Attention，官方 `18053/281s`。本文件只负责 Attention A-QB1，
> 统一父版本、版本号、组合和根切换由总协调计划处理。
> 前一张 Attention 卡 A-G1（v227）已本地 REJECTED 归档，归因见
> `workbench/full_solution/attention-ag1-joint-affine-gauge/diag/diag_report.md`。

## 1. 目标与新自由度

v227 归因的三条事实决定本卡设计：

1. A-G1 的 s 是精确算术 gauge，收益只能来自量化舍入边界的窗口特异移动，单窗口 gate 对其系统性
   反定价（4 个接受层 gate 改善与 eval delta Spearman = −1）。
2. 联合训练在固定 32 步内拖垮了 rotation 本身：层 15 在根里接受 rotation（gate +2.43%），
   在 v227 联合训练后翻车为 identity，直接丢根收益。
3. 根用同一单窗口 gate 给 rotation 定价拿到官方 +21：窗口稳定收益（旋转改变量化误差的整体形状）
   存在，窗口特异收益（逐通道 gauge 抖动）不存在。

因此本卡冻结根的全部已有状态（包括根自己 gate 接受的 rotation/center），不重训任何已有参数，
只增加一个从未存在过的自由度：

**A-QB1：根最终旋转坐标下的 per-Q-head 加性偏置 `b_q[head_dim]`。**

根的 K 侧已有加性 center（`K' = KR + c`），且 c 已在 A2 循环里按输出损失训练——K 侧加性自由度
已被覆盖（再加 b_k 与 c 严格冗余）。Q 侧从未有加性项：Q-center 会改变 softmax 输入本身，
不是 gauge，因此历史 rotation/center/scale 族都没有触到它。

量化对 Q 的逐元素舍入误差在通道维有系统分量（偏向可表示网格），经 `QK^T` 形成逐 logit 的系统偏差
`E_j[δℓ] ≈ K̂' E[δq]`。学习一个固定 `b_q` 使 `Q' = QR + b_q` 后的部署输出逼近 dense teacher，
即让偏置主动抵消这一系统分量。它是有意改变真实函数的补偿项，优化目标就是最终输出本身，
不存在"被 softmax 吸收"或"被自适应 scale 还原"的恒等路径。

## 2. 数学与合法性

- 部署变换：`Q'' = (Q R) + b_q`，`b_q` 为每层每 Q head 一个长度 `head_dim` 的 float32 向量，
  存于 `q_state`（CPU）；K/V 与根逐位一致。
- 在 `_nvfp4_to_hif4` 的 `learned_rotation`（及 center）之后、`_dense_to_hif4` 之前逐元素加；
  动态 API 只做一次加法，无候选循环、无矩阵求逆、无校准搜索。
- 输出五字段仍然是合法 HiF4；合法性检查不变。

## 3. 固定训练算法

1. 从根 R0 复制完整单文件候选；冻结根的全部 calibration state 与 A2 结果，不进入 `_a2_train_rotation`，
   不改变 rotation/center/scale 的任何数值（层 15 教训：新机制叠加在根已接受的臂之上）。
2. 对 6 个 full-attention 层，初始化 `b_q = 0`，用现有 `_m_attention_backward` 经 STE 穿过量化器，
   以**全部 calibration folds**（窗口等权，不用单窗口）的真实部署归一化 MSE 为损失，Adam 固定 32 步
   （lr、β、clip 与根 A2 相同配置），只更新 `b_q`。
3. gate：训练结束后用全部 calibration folds 的完整部署路径（动态 Q/K/V + causal attention）计算
   case 等权真实 MSE，与父逐层比较；该层严格改善才把 `b_q` 写入 `q_state`，否则该层保持父
   （`b_q` 缺省等价全零）。gate 只选父/唯一候选，不循环、不扫阈值。
4. 一个固定配置：不扫步数、lr、fold 数、head 分组、正则或截断范围。

## 4. 与已关闭族的边界

- 不是 rotation/event/group/seed/block 邻域：不改变任何正交/对角变换， encoder 决策规则不变。
- 不是 reciprocal scale / affine gauge（v190/v191/v198/v199/v217/v227）：加性偏置改变真实 logit，
  不是乘积不变变换；也不在 v227 的参数邻域内（不同参数、不同损失结构、不同 gate 口径）。
- 不是 K-center：c 作用在 K 上且已在根中训练；本卡不动 K。
- 不是 A-RB1 舍入边界、不是 V 码分配、不是 per-call 动态族、不是 Jacobian importance 移植。

若本卡失败，关闭"Q 侧加性偏置"这一实现，不以步数/lr/fold/head 粒度重试。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-aqb1-q-bias/`
（若该目录名与既有冲突，用 `attention-aqb1-q-logit-bias/`）。
日志：`logs/execution/2026-09-10-attention-aqb1-q-bias.md`。
输出：`artifacts/proxy_v3/attention-aqb1-<run-id>/`。

1. 从 R0 复制完整单文件候选，不修改根或归档源码；
2. control：`b_q=0` 时 Q/K/V 五字段与输出和根逐位一致；合成非零 `b_q` 时 Q 五字段与输出改变、
   K/V/Linear 不变；六 API 脱离仓库独立导入；`validate_state` 通过；
3. 记录每层 `||b_q||`、非零通道、gate 父/候选 loss、accepted/reverted、Q changed count；
4. GPU 串行约束：运行评测前确认无其他进程占用（`nvidia-smi`，显存 <2GiB 才启动），
   先跑 Attention shard0 排除接口错误，再用 `--shards 0,1,2,3,4,5 --stop-after-nonpositive 6`
   跑完整六 shard（禁用默认早停）；
5. 本地正负只记录；合法且可达的非等价候选归档一个版本（官方状态 `unregistered/NA`，
   官方评测由用户统一进行）；六层全部回退或输出逐位相同记 `NO_EFFECT`。

## 6. 完成条件

- 无可达非等价输出：`NO_EFFECT`；本地净负且形成非等价候选：归档 `REJECTED`；
- 本地非负候选：归档并标记等待用户统一官方评测；
- 完成后本文件归档，不追加偏置粒度/步数/正则邻域。

## 7. 结果（2026-09-10，关闭）

A-QB1 已实现为 v228，本地六 shard 强负：等权 `-0.053177`（3/69/0，72 case），六层全负，
shard3（层15）最差 `-0.105430`。机制可达且 gate 真实接受（control 8/8 种子接受、改善
1.9%–2.7%），但全 folds 训练 + 全 folds gate 仍失败——Q 偏置拟合到的系统性 logit 偏差是
校准窗口特异而非量化器固有属性。结合 v227（gauge）与 A-RB1（舍入边界），逐通道/逐元素级
校准拟合自由度在 Attention 侧第三次被否决。按 §6 关闭本实现，不追加步数/lr/fold/head 粒度
邻域；未提交官方，官方状态 `unregistered/NA`。归档：
`solutions/20260910_v228_attention-aqb1-q-bias_rejected_scoreNA_timeNA/`，执行记录
`logs/execution/2026-09-10-attention-aqb1-q-bias.md`。根保持 R0。
