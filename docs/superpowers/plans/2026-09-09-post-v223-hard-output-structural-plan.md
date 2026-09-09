# v223 后完整根离散输出与结构优化计划

> ACTIVE，2026-09-09。所有正式候选从当前最高分完整根 `solution.py` 构建：
> v202 Linear + v195 Attention，官方 `18053/281s`。v222、v223 已归档，不修改其源码；
> v222 官方 `18015/293s`，已 REJECTED，根不变；v223 官方结果到达时只补结果记录，
> 不阻塞本计划继续执行。

## 1. 这轮要解决的问题

当前停滞不是缺少连续优化器，而是连续方向没有被正确转成部署后的离散收益：

1. v223 从最后一步 Adam 更新前的 `theta_pre/center_pre` 生成事件，事件路径的 `t=0` 不是实际
   gate 选中的部署父状态。数百万 Q/K 翻码主要来自回退最后一步 Adam，而不是跨过一个量化阈值。
2. 全局 rotation 方向同时改变所有 GQA group，容易让少数有益翻码被大量无关翻码抵消。
3. 过去大量 Attention 候选只是在互逆 scale、步数、窗口或槽位上换参数；这类邻域已经关闭。
4. Linear 的标量 gain/additive 会被自适应 scale 吸收，逐码贪心又容易拟合校准窗口；后续只能做
   有共享结构、实际改变合法 hard code 的更新。

本轮不再优化 smooth loss 本身。每张卡都直接回答：部署字段是否变化、哪些码变化、最终
Attention output 或 Linear `A@W` 输出误差是否变化。

## 2. 固定执行方式

- 每次只执行当前队首一张卡；每张卡都从当时最高分完整根复制到新的
  `workbench/full_solution/<name>/`，不从 v222/v223 或其他归档源码继续改。
- 开发阶段只做与本卡相关的代码检查：六 API 可导入、合法 state、有限输出、父状态 control、
  attempted/changed/accepted 计数。Attention 变换相关的校准包装、A2 训练、动态 rotation 应用和
  动态 center 应用共四条路径均不得用宽泛 `except Exception` 静默回退。
- 先运行目标侧 shard0。确认实现可达、父 control 正确后即可归档并提交一个代表候选；目标侧六 shard
  在提交后用于补全机制记录，不作为提交前门禁，也不阻塞下一张卡。本地结果只用于找实现错误和
  解释翻码，不换算官方分数，也不设置本地时间门。
- 一个新数学机制只形成一个固定配置和一个正式候选。正式候选保留完整六 API；官方分数更高，
  或同分且更快，并且时间 `<300s`，才替换根。
- 官方候选提交后立即开始下一张卡，不等待回传；回传后按候选自己的源码、版本和类型补记结果。

## 3. R1：A-H1R 部署父状态锚定的阈值事件

实现目录：`workbench/full_solution/attention-ah1-parent-anchored/`。

### 实现

1. 先运行当前根原有 A2 trainer 和 gate，得到最终实际部署父状态：
   - gate 选择 rotation：父状态是最终 Adam 更新后的 `rotation/center`；
   - gate 选择 identity：新增的 `learned_rotation` 为 identity、`learned_center` 为 zero；当前根原有
     C76.4 rotation、permutation、multiplier 和 hierarchy state 全部保留，不把整个 Q/K state 清空。
2. 在该父状态重新计算 calibration windows 的 mean output gradient。center 固定为实际部署
   `center_parent`，不参与本卡更新；rotation 使用父正交矩阵上的
   切空间方向 `S = skew(R_parent^T G_R)`，路径固定为
   `R(t) = R_parent @ cayley(tS)`。
   不再保存或使用 `theta_pre/center_pre`。
3. 冻结父 permutation、offset、`scale_factor/lv2/lv3` 只用于解析量化边界；求出严格正的 Q/K
   首批阈值事件，float32 相同事件合并，并用 `nextafter` 跨过边界。
4. 固定评估最早 8 个不同事件和父状态。每个事件都重新走真实 Q/K encode/decode、当前 V 路径和
   最终 Attention output MSE；严格优于父状态才保存。

### 在 shard0 前必须证明

- `t=0` 的 rotation、固定 center、Q/K 五字段及最终输出与父状态逐位一致。
- 每个可用事件确实跨过预测边界；相邻事件槽的 Q/K hard-code 状态不能完全相同。
- changed-code 由事件集合逐项解释；若再次出现由父状态错位造成的整步回退，判实现错误并修复，
  不进入 4B。
- R1 候选必须移除或旁路根中的四处宽泛异常：v189 校准调用回退、A2 训练失败回退、动态
  `learned_rotation` 应用回退、动态 `learned_center` 应用回退。shard0 同时断言
  `a2_arm != "fallback"`，并在 state 含 learned rotation/center 时断言动态路径实际应用对应变换；
  否则 `t=0` 一致性可能在未变换状态上平凡通过而掩盖错误。

### 记录与结束

记录每层父 arm、rotation 方向范数、事件数、8 个 `t`、Q/K changed-code、父子最终 loss、接受事件和
holdout。shard0 若形成合法、可达、非等价候选，归档为下一个正式版本并提交官方；随后补六 shard。
若六 shard 均无接受或整体回到父状态，关闭“全局方向最近阈值”机制；不改事件数、步长、窗口或
seed 重试，已提交候选仍等待官方独立裁决。

## 4. R2：A-H3 GQA-group 局部 hard-event 坐标更新

实现目录：`workbench/full_solution/attention-gqa-local-hard-event/`。

这不是缩小 R1 步长，而是解决全局方向把不同 KV head 的收益互相抵消。

### 实现

1. 从完整根的实际部署 Q/K 状态开始，冻结 V 和所有未处理 GQA group。
2. KV group 按索引固定顺序处理一轮。每个 group 只在自己的正交切空间计算 final-output mean
   gradient，分别沿 `-gradient` 和 `+gradient` 求第一个真实 Q/K hard-code 阈值。
3. 每个 group 只比较三个状态：当前状态、正方向第一个事件、负方向第一个事件。全部走完整
   Attention hard-output；必须翻码且严格降低 calibration folds 聚合 loss 才接受。
4. 接受后将该状态作为下一个 group 的父状态；不重复第二轮，不改变 group 顺序，不扩大到第二、
   第八或更多事件。

### 记录与结束

逐 group 记录两个阈值、Q/K changed-code、完整输出 delta 和接受方向；最后记录整体 holdout。
若有 group 可达但全部拒绝，关闭“head 局部正交事件”机制。shard0 若 accepted>0，形成一个正式
候选并提交官方，再补六 shard记录；不按接受 group 数拆成多个版本。

## 5. R3：A-C76.5 残差定向的 C76.4 正交候选

实现目录：`workbench/full_solution/attention-c765-residual-directed-rotation/`。

依据：减法定价候选 `v205_attn-no-c764-rotation-search` 表明，删除 C76.4 后官方少 84 分，因此保留
C76.4；本卡不改变其动态路径，只增加一个与现有固定 seed 不同、由最终输出残差解析生成的校准候选。

### 实现

1. 保留当前 C76.4 的 H16/H32、原候选和部署逻辑不变；新候选使用 C76.4 搜索开始前的同一份
   `q_state/k_state`、calibration Q/K/V 和 reference output。
2. 对每个 calibration window 先由该父状态得到 `Q_hat/K_hat/V_hat` 和最终输出残差，再通过现有
   Attention backward 得到 `dQ_hat/dK_hat`。对 KV group `g` 累计右变换梯度
   `G_g = Σ(Q_g^T dQ_g + K_g^T dK_g)`，其中同组所有 Q heads 先求和，所有窗口按 case 等权平均。
3. 对 block size `B∈{16,32}`，将 `G_g` 切成连续 `B×B` 对角块，并定义
   `C = (G_block + G_block^T)/2`。用 `torch.linalg.eigh(C)`，选择 `|eigenvalue|` 最大的特征向量；
   绝对值并列取索引较小者，向量首个非零元素固定为正。该块符号固定为
   `s_j = +1 (u_j>=0), -1 (u_j<0)`，拼成 `[kv_heads, head_dim]` signs。
4. 候选变换严格使用当前动态实现已有的 `x -> (x * signs) @ H_B`，不引入 permutation。H16 和
   H32 各生成一个候选；不搜索 seed、阈值、特征向量组合或候选数量。
5. 两个新候选与当前 C76.4 已有候选一起走现有完整 deployed-MSE 选择。动态 API 仍只应用最终选中的
   正交变换，不增加在线候选循环。

### 预检与结束

先用一个实际 4B Attention 层确认新候选不是现有 seed 的逐位重复，并且保持浮点 QK 内积不变。
若 shard0 候选重复或不可达，关闭“残差定向 C76.4”机制。若被选择且产生不同 hard output，形成
一个正式候选并提交官方，再补其余五个 shard；六层全部重复或全部不被选择时关闭该机制，不继续换
eigensolver、seed、block size 或符号规则。

## 7. 明确不执行的工作

- 不修改 v222、v223 或任何 `solutions/` 已归档源码。
- 不启动 A-H2；v223 的 center 梯度只有约 `2e-8` 到 `4e-8`，没有可解释方向。
- 不重开 Q/K 互逆 scale/残差的步数、窗口、block、slot、clamp 邻域。
- 不重开 Linear 标量 gain/additive、AW8 任意逐码贪心、L-H1 普通逐列正负码重选。
- 不重开 v220 零码到最小非零有符号码插入；改变 64-block/8-group 粒度仍属于同一机制邻域。
- 不运行 0.5B、OOD、GPT-2/opt、fresh timing，不用本地分数预测官方分数或本地秒数否决提交。
- v222 已官方 REJECTED；不为等待 v223 官方结果暂停开发，也不把 v223 未确认结果并入当前根。

## 8. 版本、归档和结果更新

1. 开发文件只放 `workbench/full_solution/<name>/`，包含构建脚本、配置和一份结果摘要。
2. 通过本卡实现检查并形成一个非等价完整候选后，才分配下一个未使用版本；正式目录一次性写入
   `solution.py`、`config.json`、`verification.json`、`result.md`，之后不修改归档源码。
3. 本地失败写清 `ERROR`、`NOOP` 或 `REJECTED_LOCAL`；官方未知写 `unregistered/NA`；官方回传
   使用 `RETAINED`、`REJECTED` 或 `TIMEOUT`，不拿本地秒数填官方时间。
4. 官方正向后更新根、当前状态、版本索引、计划队列和执行日志；官方非正向只更新候选自己的结果
   与公共索引，根不变。
5. 每轮结束清理死候选的 calibration cache，只保留当前根与回退根；并发时使用
   `--min-age-hours 2`，不得删除 dense 主缓存 `qwen3.5-4b-proxy-v2.pt`。

## 9. 当前执行队列

| 顺序 | 工作 | 开始条件 | 完成条件 | 下一动作 |
|---:|---|---|---|---|
| 0 | v222 结果与 v223 元数据收口 | DONE | v222 公共记录已同步；v223 元数据已归档 | 进入 R1 |
| 1 | R1 A-H1R 父状态锚定 | 立即 | 正确性证明、shard0、归档/提交；六 shard 后补 | 进入 R2 |
| 2 | R2 A-H3 group 局部事件 | R1 已提交或关闭 | 单轮四组 shard0、归档/提交；六 shard 后补 | 进入 R3 |
| 3 | R3 A-C76.5 残差定向 C76.4 | R2 已提交或关闭 | 新候选去重、shard0、归档/提交；其余 shard 后补 | 汇总官方结果并制定新计划 |

三张卡全部结束后，本计划立即归档。不得在表尾继续追加相似参数变体；下一计划只依据这三张卡的
hard-output、官方结果和实际代码机制重新制定。
