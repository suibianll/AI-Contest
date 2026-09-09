# 当前最高分根持续优化计划

> ACTIVE，2026-09-09。当前官方根保持为 v202 Linear + v195 Attention，`18053/281s`，
> SHA256 `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
> 所有新实现从该根生成到 `workbench/full_solution/`；在官方正向前不覆盖根，也不修改
> `solutions/` 中任何已归档源码。

## 1. 当前问题

持续实验没有推进，不是候选数量不足，而是执行链存在三个明确问题：

1. A2 训练记录的是多窗口 mean loss，但更新使用未除以窗口数的梯度总和，目标与更新不一致。
2. Attention 校准用两个宽泛 `except Exception` 静默回退，代码错误会被伪装成 identity/fallback。
3. A2 前向使用真实 hard encode/decode，反向仍是 STE。它可以产生方向，但不能证明离散输出会改善；
   后续算法必须直接比较部署路径的 hard-output loss，不能再把训练 loss 下降当成优化结果。

## 2. 立即执行：FIX-A2 正确性修复

实现目录：`workbench/full_solution/attention-a2-correctness-fix/`。

只做以下三项修复，不同时加入新算法：

1. `grad_theta` 和 `grad_center` 在加入正则、裁剪及 Adam 更新前同时除以训练窗口数。
2. 修正 `_a2_train_rotation` 的三返回值类型标注。
3. 移除 `hif4_calibration_attention` 的宽泛异常吞噬；输入或算法错误直接抛出，正常的
   gate identity 选择仍保留。

执行顺序：

1. 运行构建脚本，确认 parent SHA 正是当前根。
2. 运行静态与小张量 smoke：六 API、合法 state、有限输出、异常不被吞掉。
3. 运行 Attention shard0；记录 rotation/identity 数、训练窗口数和 hard-output 父子变化。
4. shard0 确认实现可达后运行 Attention 六 shard。这里的数值只用来判断修复是否改变部署输出，
   不换算官方分数。
5. 如果六 shard 有实际变化，分配一个新的正式版本并以“当前完整根 + FIX-A2”提交官方；官方
   分数更高，或同分且更快，并且 `<300s` 时才替换根。若逐位等价、官方降分或超时，修复记录
   归档，官方根仍为 v202。

## 3. 后续算法队列

每次只执行一项。前一项结束并归档后再开始下一项，不批量制造候选。

### A-H1：量化阈值事件搜索

- 目标：解决 STE 方向与 hard code 不一致。
- 连续方向：先运行 FIX-A2 的 mean-gradient trainer，得到最后一步更新前的 `theta0/center0` 及
  mean gradient `g_theta/g_center`。若 FIX-A2 官方未并入根，A-H1/A-H2 均从当前完整根构建，
  方向由根原 trainer 计算、梯度仍按窗口数归一化（归一化只影响方向定义，不改变根的部署状态）。
  固定
  `d_theta = -g_theta / max(||g_theta||, 1e-12)`、
  `d_center = -g_center / max(||g_center||, 1e-12)`，路径为
  `theta(t)=theta0+t*d_theta`、`center(t)=center0+t*d_center`，只搜索 `t>0`。
- 事件生成：仅在生成阈值时冻结父状态的 permutation、offset、`scale_factor/lv2/lv3`。对每个
  calibration Q/K 元素，在 `t=0` 解析计算 Cayley 路径下的变换值 `x0` 和导数 `dx/dt`；对每个
  相邻 HiF4 magnitude 中点 `b`，计算线性化边界
  `t=(b-|x0|)/(sign(x0)*dx/dt)`，只保留有限且严格为正的 `t`。`x0=0` 时直接使用
  `d|x|/dt=|dx/dt|`。这一步只生成事件位置，最终收益仍由真实非线性路径重算。
  `t` 升序排列；float32 表示相同的 `t` 合并成一个事件，使用 `torch.nextafter(t,+inf)` 保证
  跨过预测边界。生成阶段不以 smooth loss 排序。
- 固定候选：固定 8 个事件槽，填入最早的 8 个**不同**事件；事件不足时剩余槽标记 unavailable，
  不复制状态，零事件则记不可达并结束。槽位数不根据本地耗时、结果符号或设备调整，也不扫描邻域。
- 真实评估：每个 `t` 都重新走完整部署 Q/K encode/decode、当前 V 路径和最终 Attention output
  MSE；动态 Q/K API 只读取校准选出的 rotation/center，不包含事件循环。
- 选择与验证：候选参数只按 calibration folds 的 case 等权 normalized output MSE 聚合选择，
  `t=0` 父状态始终在候选集中；严格更优才接受。选定后只在独立 holdout 记录父子 loss，holdout
  不改变候选、不否决提交，也不用于修改任何参数。
- 时间处理：记录 shard0 calibration API 时间和事件评估次数，只标注风险；不据本地时间减少事件、
  预测官方时间或阻止提交，官方 `<300s` 是唯一时间裁决。
- 产物：方向范数、原始/去重事件数、8 个 `t`、每个事件 changed-code 数（按 Q 侧 / K 侧拆分
  记录，供 A-H2 前提归因：center 只影响 K 码，rotation 同时影响 Q/K；被接受事件只翻 K 码
  才支持"收益只来自 center"）、fold 聚合最终 loss、holdout 记录、被接受的事件序号和最终 state。
- 失败处理：若事件可达但最终输出均不改善，关闭该阈值事件机制；不改步长、事件数或 seed 重试。

### A-H2：K-center 离散坐标更新

- 前提：A-H1 证明 rotation 与 center 混合事件无法定位收益，或收益只来自 center。
- 做法：冻结当前部署 rotation。对 KV head `h`，用 FIX-A2 在 calibration folds 聚合得到的 center
  mean gradient 定义单位方向 `u_h=g_h/max(||g_h||,1e-12)`；从当前 `center_h` 分别沿 `-u_h`
  和 `+u_h` 求冻结父 `scale_factor/lv2/lv3` 下第一个严格为正的 K 编码阈值，并在阈值处使用
  `torch.nextafter(t,+inf)`。因此每个 head 恰好两个候选；方向无阈值时该方向记不可达。
- 顺序：KV head 按索引升序处理且只处理一轮。每个 head 都以之前已接受 head 的 center 为当前状态，
  重新生成该 head 的两个最近事件；相同阈值同时翻码，不按单元素拆候选。
- 选择：两个候选和当前状态都走完整 hard encode/decode 与最终 Attention output MSE，只按
  calibration folds 的 case 等权聚合选择；必须实际翻码且严格降低聚合 loss 才更新该 head。
  全部 head 完成后在独立 holdout 只记录一次父子结果，不参与选择或提交决定。
- 产物：每个 head 的方向范数、正负阈值、changed-code 数、fold loss、接受方向及最终 holdout 记录。
- 失败处理：关闭 center 离散更新，不扫描 head 顺序、轮数或幅度。

### L-H1：真正的逐列非对称权重量化

- 目标：回到用户给出的 `A@W 拟合 + 逐列非对称量化` 成功机制，而不是继续增益/additive 邻域。
- **第 0 步表达性预检**：先把“逐列非对称”固定定义为：同一输入列内，正权重和负权重允许选择
  不同 magnitude code，但最终仍只输出合法五字段，不引入正负双 scale、zero-point 或自定义解码。
  用一个含正负权重的 64-column 合成块验证：候选五字段通过 `evaluator/reference_hif4.py`，且相对
  根编码产生不同的合法解码值。若目标必须依赖五字段无法保存的正负双 scale，则直接判
  `NOT_EXPRESSIBLE`，不进入实现。
- **第 0 步残余空间预检**：在同一合成块上冻结最终部署 activation state，枚举一列的合法
  sign/mant 邻码并计算完整输出误差；必须存在至少一个“根 Weight GPTQ 不选、但 A@W 完整输出目标
  严格更优”的合法状态，才证明该方向不是 AW1-AW14 的标量 gain/additive 或同目标重选码。
  未找到反例则记 `NO_DISTINCT_RESIDUAL` 并关闭，不制作 4B 候选。归因对照见
  `logs/execution/2026-09-09-aw-fitting-family-analysis.md`。
- 做法：从当前根的最终部署权重坐标出发，用全部 Linear 校准行构造输出残差；对自然 64-column
  block 内每列分别累计正值和负值的输出加权误差，再联合选择合法 sign/mant 与共享层级 scale。
  这张卡改变正负码分配，不再拟合一个会被自适应 scale 吸收的标量 gain。
- 选择：校准目标直接使用完整 `XW^T - Q(XR)Q(WR^{-T})^T`；记录 attempted/changed/accepted。
- 失败处理：若能翻码但输出目标负向，关闭该非对称编码实现；不退回 AW1-AW14 的组大小邻域。

## 4. 归档方式

每项结束后只做一次归档：

- 开发中：源码、构建脚本、固定配置和验证结果留在对应 `workbench/full_solution/<name>/`。
- 形成正式候选后：新建一个未使用的版本目录，复制完整单文件并记录 parent SHA、candidate SHA、
  本地结果和官方状态；不改任何既有版本目录。
- rejected/timeout：在新候选自己的 `result.md` 写清原因，根不变。
- retained：官方结果确认后才把候选复制为根，并同步状态、版本索引和计划。

## 5. 当前队列

| 顺序 | 工作 | 状态 | 下一动作 |
|---:|---|---|---|
| 1 | FIX-A2 目标/异常修复 | 已实现，待 Attention shard0 | 验证部署输出是否变化 |
| 2 | A-H1 量化阈值事件搜索 | 未开始 | FIX-A2 结束后执行 |
| 3 | A-H2 K-center 离散坐标 | 未开始 | 仅按 A-H1 归因决定是否执行 |
| 4 | L-H1 逐列非对称权重量化 | 未开始 | Attention 两项结束后执行 |
