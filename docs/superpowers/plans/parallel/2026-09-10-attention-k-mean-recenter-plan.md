# Attention K 侧 per-call 均值再定心计划（A-MC1）

> 状态：本地正向已归档 v229，待官方裁决，2026-09-10。
> 从属于当前活动总计划（Linear 线）。当前完整根为 v202 Linear + v195 Attention，官方 `18053/281s`。
> 本文件只负责 Attention A-MC1；版本登记、组合与根切换由总协调线处理。
> 前两张 Attention 卡：A-G1（v227 REJECTED）、A-QB1（v228 REJECTED），归因见各自执行日志。

## 1. 目标与定位

规则级自由度盘点（2026-09-10，`agent-7` 只读分析）结论：Q·K 不变量只允许四类结构变换
（对角 scale、置换、正交变换、K 平移），前三种均已关闭或已在根中；scale/hierarchy/mantissa 搜索
距全 255-code 穷举仅 0.015–0.08%；逐通道/逐元素校准拟合参数三连否（gauge/偏置/边界表）。
K 平移类中唯一未部署的形态是 **per-call 均值再定心**，本卡给它一次性裁决，结束后 K 平移类关闭。

机制定义：在根最终部署坐标（rotation + 根 center 之后、`_dense_to_hif4` 之前）对当前调用的 K
做 token 维均值再定心：

`K'' = K' - mean_tokens(K')`

量化前严格 softmax 不变（logits 只加逐 query 常数 `Q·mean(K')`，softmax 后消失）；所有收益或
损失只来自量化输入分布的移动。它**不含任何校准拟合参数**——均值来自当前调用本身，因此不受
v227/v228 的窗口过拟合失败模式影响；相反它直接针对该失败模式（测试窗口与校准窗口的统计漂移）：
根 center 为静态的层，per-call 均值把当前窗口重新拉回零中心；根 center 为 mode-2 per-call
midrange 的层，残差均值 = mean−midrange 缺口（离群点敏感 vs 稳健中心的差异）。

## 2. 合法性与成本

- 动态 API 只做一次 token 维 reduce + 减法：无候选循环、无 Gram contraction、无求逆，
  符合 v165 边界；与已部署的 mode-2 per-call midrange 同属轻量 per-call reduce 模式。
- state 只新增每层一个 CPU 标量 arm 标志（`k_state`）；五字段格式不变。
- 不是 C37：C37 是 per-call 样本统计改写 refine 的 importance 权重（Linear 侧机制，−2.7pp 否定）；
  本卡不改 importance/选择规则，只在编码前平移输入，且 K 平移有精确 softmax 不变性保护。
- 不是 mode 2 邻域：不替换根 center，在其之上叠加残差均值；不以 midrange/mean/中位数互换重扫。

## 3. 固定算法

1. 从 R0 复制完整单文件候选；冻结根全部 state，不重训任何已有参数。
2. 校准：对 6 个 full-attention 层，分别用全部 calibration folds（窗口等权）的完整部署路径
   真实 MSE 比较「父」与「父 + 残差均值再定心」两臂，每层严格改善才把该层 arm 标志写入
   `k_state`，否则保持父。无训练、无步数、无超参。
3. 部署：`hif4_dynamic_quantize_k` 路径在根 center 之后对 K 做 `K -= K.mean(dim=token)`（按 head
   分组聚合均值，与 center 的广播形状一致），仅在 arm 标志开启时执行。

## 4. Control（全部必须通过并记录）

1. arm 关闭：Q/K/V 五字段与输出与根逐位一致；
2. arm 开启 + 合成非零均值偏移输入：K 五字段改变、dense softmax 输出数值不变（<1e-6）、
   Q/V/Linear 逐位不变；
3. 六 API 脱离仓库独立导入；`validate_state` 通过；
4. 记录每层 arm、gate 父/候选 loss、K changed count。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-amc1-k-mean-recenter/`。
日志：`logs/execution/2026-09-10-attention-amc1-k-mean-recenter.md`。
输出：`artifacts/proxy_v3/attention-amc1-<run-id>/`。

GPU 串行（`nvidia-smi` 显存 <2GiB 才启动）；先 shard0 排除接口错误，再
`--shards 0,1,2,3,4,5 --stop-after-nonpositive 6` 跑满六 shard；本地正负只记录；
合法且可达的非等价候选归档一个版本（`unregistered/NA`，官方评测用户统一进行）；
六层全回退或逐位相同记 `NO_EFFECT` 并关闭 K 平移类。

## 6. 完成条件

- `NO_EFFECT` 或本地净负：归档 `REJECTED`，K 平移类关闭，不以 mean/midrange/中位数/trimmed-mean
  等中心变体重试；
- 本地非负：归档并等待用户统一官方评测；
- 完成后本文件归档。若本卡关闭，Attention 侧规则级与平移类自由度全部耗尽，后续只剩绑定
  21071 锚点源码或官方裁决驱动的新证据。

## 7. 执行结果（2026-09-10，v229）

- 归档：`solutions/20260910_v229_attention-amc1-k-mean-recenter_officialNA_timeNA/`，候选
  SHA256 `d1c23fa11198e56f15ac8f64e033c00333dcd2d5660cec773598624c4b247f4d`；执行日志
  `logs/execution/2026-09-10-attention-amc1-k-mean-recenter.md`。
- Control 全部 PASS：arm 关闭逐位恢复父；arm 开启+偏移输入 K 五字段变化且 dense softmax
  max|Δ|=1.86e-9（精确不变）、Q/V/Linear 逐位不变；六 API 独立导入；`validate_state` 通过；
  gate 接受/回退双路验证；`root_rotation_frozen=True`。
- 六 shard（72 case，`--stop-after-nonpositive 6` 跑满）等权均值 `+0.014923`（26/10/36）：
  层0 `+0.000000`（0/0/12）、层1 `+0.020038`（10/2/0）、层8 `+0.000000`（0/0/12）、
  层15 `+0.079126`（12/0/0）、层22 `+0.000000`（0/0/12）、层5 `-0.009628`（4/8/0）。
  candidate overall `+0.548920` vs baseline `+0.533998`；API total（诊断，1 次校准缓存命中）
  29.725s。3/6 层 gate 接受（层1/5/15），层0/8/22 回退逐位不变。
- 解读：本计划线首个本地正向 Attention 候选；机制无校准拟合参数，不受 v227/v228 窗口
  过拟合模式影响；层15 收益集中且均匀（12/12 case 约 +0.079），与"纠正窗口统计漂移"假设
  一致；层5 gate 接受但 eval 净负（gate/evals 口径差异，如实记录）。
- 裁决：**本地非负分支**——按 §6 归档 v229，官方状态 `unregistered/NA`，等待用户统一官方
  评测。根保持 v202 Linear + v195 Attention（`18053/281s`）不变。
