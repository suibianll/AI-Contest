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
   分数更高且 `<300s` 才替换根。若逐位等价或官方不增分，修复记录归档，官方根仍为 v202。

## 3. 后续算法队列

每次只执行一项。前一项结束并归档后再开始下一项，不批量制造候选。

### A-H1：量化阈值事件搜索

- 目标：解决 STE 方向与 hard code 不一致。
- 做法：保留当前 A2 的低维 rotation/center 方向，但不按连续 learning rate 更新；计算沿该方向
  第一次会改变 Q/K 合法编码的阈值事件，按方向顺序评估事件状态。**8 是上限而非定额**：
  先实测单事件校准成本，再按根 19s 官方余量决定实际评估的事件数，超预算就截断事件数
  （这是预注册的时间截断规则，不属于扫事件数邻域）；单事件成本本身就超预算则整卡不提交。
  前例 v196/v198/v199/v201/v203 的新增 Attention 校准全部官方 TIMEOUT，实现前必须先给出
  单事件实测成本。所有事件评估放在校准内完成，动态 Q/K API 不含候选循环（v165 边界不变）。
- 选择：每个状态都走真实动态 Q/K encode/decode 和最终 Attention output MSE；只用 calibration
  folds 学习与聚合，接受条件是 folds 聚合严格优于父状态**且独立 holdout 验证不为负**
  （与 AGENTS.md §3 校准/选择/验证分离一致）。
- 产物：changed-code 数、每个事件的最终 loss、被接受的事件序号和最终 state。
- 失败处理：若事件可达但最终输出均不改善，关闭该阈值事件机制；不改步长、事件数或 seed 重试。

### A-H2：K-center 离散坐标更新

- 前提：A-H1 证明 rotation 与 center 混合事件无法定位收益，或收益只来自 center。
- 做法：冻结当前 rotation，只对每个 KV head 的 center 使用编码边界生成一个正向和一个负向候选；
  逐 head 走真实 hard-output loss，固定一轮顺序更新。
- 选择：候选必须实际翻码且整层最终 output loss 下降；否则保持父 center。
- 失败处理：关闭 center 离散更新，不扫描 head 顺序、轮数或幅度。

### L-H1：真正的逐列非对称权重量化

- 目标：回到用户给出的 `A@W 拟合 + 逐列非对称量化` 成功机制，而不是继续增益/additive 邻域。
- **第 0 步表达性预检**：先证明目标逐列非对称编码能用五字段合法 state 表达并通过
  `evaluator/reference_hif4.py` 检查；再论证它相对根已有 Weight GPTQ（Hessian 度量逐列最优 +
  误差反馈）存在真实残余空间，而不是退化为又一次输出加权重选码（AW 族九连败归因见
  `logs/execution/2026-09-09-aw-fitting-family-analysis.md`）。预检任一不通过则不进入实现，
  直接记录关闭依据。
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
