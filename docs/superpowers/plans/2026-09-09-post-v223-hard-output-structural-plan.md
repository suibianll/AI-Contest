# v223 后完整根离散输出与结构优化计划

> ACTIVE，2026-09-09。所有正式候选从当前最高分完整根 `solution.py` 构建：
> v202 Linear + v195 Attention，官方 `18053/281s`。v222、v223 已归档，不修改其源码；
> 官方结果到达时只补结果记录，不阻塞本计划继续执行。

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
  attempted/changed/accepted 计数。异常直接抛出，不用宽泛 `except Exception` 把错误伪装成回退。
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
   - gate 选择 identity：父状态是 identity rotation 和 zero center。
2. 在该父状态重新计算 calibration windows 的 mean output gradient。rotation 使用父正交矩阵上的
   切空间方向 `S = skew(R_parent^T G_R)`，路径固定为
   `R(t) = R_parent @ cayley(tS)`；center 路径从实际 `center_parent` 出发。
   不再保存或使用 `theta_pre/center_pre`。
3. 冻结父 permutation、offset、`scale_factor/lv2/lv3` 只用于解析量化边界；求出严格正的 Q/K
   首批阈值事件，float32 相同事件合并，并用 `nextafter` 跨过边界。
4. 固定评估最早 8 个不同事件和父状态。每个事件都重新走真实 Q/K encode/decode、当前 V 路径和
   最终 Attention output MSE；严格优于父状态才保存。

### 在 shard0 前必须证明

- `t=0` 的 rotation、center、Q/K 五字段及最终输出与父状态逐位一致。
- 每个可用事件确实跨过预测边界；相邻事件槽的 Q/K hard-code 状态不能完全相同。
- changed-code 由事件集合逐项解释；若再次出现由父状态错位造成的整步回退，判实现错误并修复，
  不进入 4B。
- R1 候选必须移除或旁路从根继承的 `hif4_calibration_attention` 宽泛 `except Exception`（两处：
  v189 校准调用回退与 A2 训练失败回退），并在 shard0 记录中断言 `a2_arm != "fallback"`；否则
  `t=0` 一致性证明可能在 identity/identity 上平凡通过而掩盖真实错误。

### 记录与结束

记录每层父 arm、方向范数、事件数、8 个 `t`、Q/K changed-code、父子最终 loss、接受事件和
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

## 5. R3：A-C77 残差定向的 C76.4 正交候选

实现目录：`workbench/full_solution/attention-c77-residual-directed-rotation/`。

依据：减法定价候选 `v205_attn-no-c764-rotation-search` 表明，删除 C76.4 后官方少 84 分，因此保留
C76.4；本卡不改变其动态路径，只增加一个与现有固定 seed 不同、由最终输出残差解析生成的校准候选。

### 实现

1. 保留当前 C76.4 的 H16/H32、原候选和部署逻辑不变。
2. 在每个 KV group 内，用父 Q/K hard-output 对最终 Attention 输出的残差反传，构造对称的
   Q/K 误差耦合矩阵；取其固定最大绝对特征方向，按首个非零元素为正统一符号。
3. 将该方向确定性地投影为一个 signed-Hadamard 排列/符号候选；每个 block size 只生成一个，
   不搜索 seed、阈值或候选数量。
4. 新候选与当前 C76.4 已有候选一起走现有完整 deployed-MSE 选择。动态 API 仍只应用最终选中的
   正交变换，不增加在线候选循环。

### 预检与结束

先用一个实际 4B Attention 层确认新候选不是现有 seed 的逐位重复，并且保持浮点 QK 内积不变。
若 shard0 候选重复或不可达，关闭“残差定向 C76.4”机制。若被选择且产生不同 hard output，形成
一个正式候选并提交官方，再补其余五个 shard；六层全部重复或全部不被选择时关闭该机制，不继续换
eigensolver、seed、block size 或符号规则。

## 6. R4：L-Z1 共享层级内的零码激活

实现目录：`workbench/full_solution/linear-zero-code-activation/`。

依据：L-H1 已证明普通逐列正负码重选没有独立残余；LC2 又没有执行零值 sign flip。本卡只处理
父权重中量化为零、但完整 `A@W` 输出残差支持非零贡献的元素，并以自然层级组为单位更新，避免
退回 AW8 的无结构逐码贪心。

### 实现

1. 从当前根最终部署权重和冻结后的 activation state 计算
   `E = XW^T - Q(XR)Q(WR^{-T})^T`。
2. 对每个自然 8 元素 HiF4 层级组，只查看父码为零的位置；用 `E` 与部署 activation 的相关项
   确定最小非零 magnitude 的符号。
3. 每组只构造一个联合候选：把该组内所有预测为负增量的零码同时激活为最小非零合法码，然后在
   固定共享 `scale_factor/lv2/lv3` 下重算完整组的 A@W 输出误差。父状态始终参与比较。
4. 按 64-column block 固定顺序单轮处理；必须使完整 block 输出误差严格降低才接受。动态 activation
   路径不变，部署仅保存合法权重五字段。

### 预检与结束

先在一个真实 weight block 上证明：存在零码、候选合法、changed>0，并且候选与 AW8 的“任意单码
逐个贪心”实现不同。预检无可达状态则关闭，不运行 4B。shard0 若形成合法非等价候选，提交官方并
再补六 shard；六 shard 若 accepted=0 或实际输出不改善，关闭零码激活机制，不扫描激活比例、阈值、
block size 或多轮次数，已提交候选仍等待官方独立裁决。

## 7. 明确不执行的工作

- 不修改 v222、v223 或任何 `solutions/` 已归档源码。
- 不启动 A-H2；v223 的 center 梯度只有约 `2e-8` 到 `4e-8`，没有可解释方向。
- 不重开 Q/K 互逆 scale/残差的步数、窗口、block、slot、clamp 邻域。
- 不重开 Linear 标量 gain/additive、AW8 任意逐码贪心、L-H1 普通逐列正负码重选。
- 不运行 0.5B、OOD、GPT-2/opt、fresh timing，不用本地分数预测官方分数或本地秒数否决提交。
- 不为等待 v222/v223 官方结果暂停开发，也不把它们未确认的结果并入当前根。

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
| 0 | v222/v223 元数据收口 | 已完成实现 | 新增元数据提交，公共状态同步 | 不等待官方，进入 R1 |
| 1 | R1 A-H1R 父状态锚定 | 立即 | 正确性证明、shard0、归档/提交；六 shard 后补 | 进入 R2 |
| 2 | R2 A-H3 group 局部事件 | R1 已提交或关闭 | 单轮四组 shard0、归档/提交；六 shard 后补 | 进入 R3 |
| 3 | R3 A-C77 残差定向 C76.4 | R2 已提交或关闭 | 新候选去重、shard0、归档/提交；其余 shard 后补 | 进入 R4 |
| 4 | R4 L-Z1 零码激活 | R3 已提交或关闭 | 预检、Linear shard0、归档/提交；六 shard 后补 | 汇总官方结果并制定新计划 |

四张卡全部结束后，本计划立即归档。不得在表尾继续追加相似参数变体；下一计划只依据这四张卡的
hard-output、官方结果和实际代码机制重新制定。
