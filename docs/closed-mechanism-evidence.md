# 已关闭机制与证据边界（2026-09-10 迁移）

> 从 AGENTS.md 原第 7 节迁移，保留具体实现关闭约束与证据。按目标机制查阅，不作为当前计划入口。
> 下文仓库路径均相对仓库根；历史 REJECTED 不等于官方负向，未提交者仍为官方未知。
> 版本与分数快照以后续官方记录为准；不得由有限实验推断整族饱和。

- 用户 21071 成功机制证据优先于历史整族饱和推断。A@W拟合、真正逐列非对称量化、Q/K互逆scale学习
  可按当前唯一总计划注册为"完整根 + 一个增量"的新机制；旧具体实现负结果不撤销，不重复同 SHA/
  逐位等价提交。旧 L21/A21 工作包与本地符号例外全部退役。合法性、control、单文件与官方 300s
  硬限保留。不把一个候选的失败扩写成所有新机制不可行。
- 细节查[当前状态](current-solution-status.md)、[版本索引](../solutions/README.md)和
  [计划入口](superpowers/plans/README.md)；09-06 各实验的具体关闭边界以对应计划/日志为准。

### Linear

- 已关闭：旧 full64/单折邻域、Householder 全族、cross-fold minimax 的 fold/Jacobi/coverage/role
  邻域、rank-3/残差系数/fold 扩展（首次 L3 死分支结果无效，修正可达后的负结果有效）。
- LC1（rank-8 A@W 精化）与 LC2（整块 ±1 合法邻域）只关闭各自真实实现，不推广为"根坐标已局部
  饱和"或"A@W 拟合族无效"（`logs/execution/2026-09-08-lc1-lc2-method-audit.md`）。
  L-C3（objective-only）官方 `18031/293s`（−1/+13s）REJECTED
  （`logs/execution/2026-09-08-lc3-official-result.md`）。
- A@W 拟合族增益/additive 形态（v197 官方 `17277/285s` −776、AW1–7、AW9）结构性自闭：
  自适应 scale 吸收增益，重编码整数码必劣于父，与数据无关，不再注册该形态；无结构逐码贪心
  （AW8）过拟合校准窗口。归因见 `logs/execution/2026-09-09-aw-fitting-family-analysis.md`。
- v220 的零码到最小非零有符号码插入已可达但 shard0 负向；该具体插入机制关闭，不通过改成
  8-group/64-block、单个/批量零码或激活比例邻域重试。
- v204（减法定价：关 rank-2 残差段）官方 `18053/286s`（0/+5s）→ rank-2 残差段官方贡献 0 分，
  单次时间差不能单独归因给该段。

### Attention

- 已关闭：per-call 动态 Gram/自适应精化族、+4 scale 窗口、block-smooth refine 覆盖率、
  Jacobian 向 v186 移植及其收缩/clamp/gate 邻域；v187 仅为 clean-room 研究父，不替代完整父。
- 标准 Linear 侧隔离官方分（基线＝标准 Linear + R3 `14405/238s`）：v190/v191 侧分 `14405`（0），
  关闭这两个具体实现及其参数邻域；v192 侧分 `14427`（+22）有精度证据但完整提交超时，
  不再用侧时间差推算降时需求，不缩步数或减轮数重试同一实现。侧隔离只用于测分，
  侧隔离时间对完整根时间没有预测力（v194：侧 −4s / 完整 +5s，方向相反）。
  见 `logs/execution/2026-09-09-standard-linear-attention-side-scores.md`。
- v194（A2/R3 校准等价去重）官方 `18032/285s`，同分慢 5s，`REJECTED_TIME`；等价提速路线不成立。
- v196、v198、v199、v201、v203 官方均 `TIMEOUT(>300s)`，各只关闭该实现，不缩窗/减步/减轮重试。
- v223（A-H1 阈值事件搜索，8 槽×folds≈43 次完整部署路径窗口评估/层）官方 `TIMEOUT(>300s)`，
  本地六 shard mean `+0.003209` 未获官方定价；只关闭该实现，同成本类事件搜索重试前必须先降
  校准成本。
- v222（FIX-A2：A2 mean-gradient + 异常传播修复）官方 `18015/293s`（−38/+12s）REJECTED，
  只关闭该实现；方向定义问题已由 v224 的部署父状态锚定处理。
- v224（A-H1R）官方 `TIMEOUT(>300s)`，本地六 shard `≈+1.2e-6` 未获官方定价；与 v223 同成本类，
  只关闭该实现。v225（A-H3）官方 `TIMEOUT(>300s)`，本地六 shard `+3.11e-5` 未获官方定价；
  收益几乎全部来自 shard5；A-C76.5 残差定向候选六层均未被选择。三者实际均属 Q/K 正交坐标
  变换族，当前关闭 rotation/event/group/seed/block 的直接邻域，转入量化舍入边界自由度。
- v205（减法定价：关 C76.4 旋转搜索）官方 `17969/275s`（−84/−6s）→ C76.4 官方价值 +84 分，
  必须保留，不得为省时间砍掉。砍校准换时间收益极小（本地校准 −30% → 官方仅 −6s），但这两个
  减法候选不能推出通用时间换分速率，也不能据此估算其他实现的可回收时间。
  见 `logs/execution/2026-09-09-v204-v205-subtraction-pricing-official.md`。
- V 侧不注册新候选：per-head 常量不改变块内解，per-channel multiplier 解码不逆缩放会破坏输出，
  五字段不支持 per-token 表；码分配类经 anchor28 探针裁决在当前码语义下 100% 饱和
  （NO_SUPPORTED_MECHANISM），但该饱和不构成最终输出下界
  （`logs/execution/2026-09-08-a28-interpretation-correction.md`）。当前计划只在 Q/K 上研究合法 mantissa
  舍入边界；不据此重开 V 码分配或改变官方解码格式。
- 2026-09-10 规则级空间全部裁决（`logs/execution/2026-09-10-attention-*.md`）：v227（A-G1 联合
  仿射 gauge）本地六 shard `−0.005294` REJECTED（gauge 收益窗口特异、gate 反定价）；v228（A-QB1
  Q 侧加性偏置）六层全负 `−0.053177` REJECTED；A-QC1（Q 侧 per-call 数据中心化）`NO_EFFECT`
  （6/6 层 gate 全拒、逐位相同，不占版本号）；v229（A-MC1 K 侧 per-call 均值再定心）本地正向
  `+0.014923`，完整包官方 TIMEOUT（>300s），标准 Linear 侧隔离官方 `14424/245s`
  （相对 v195 侧基准 `14426/243s` 为 **−2/+2s**）——本地正向未迁移官方，A-MC1 官方侧价值
  为 −2，K 平移类正式关闭（`logs/execution/2026-09-10-v229-side-isolation-official.md`）。
  Q·K 不变量四类结构变换（对角 scale/置换/正交/K 平移）与两侧 per-call
  定心规则均有定论，且 K 平移已有官方负向定价；scale/hierarchy/mantissa 搜索距穷举 oracle
  仅 0.015–0.08%。不注册 gauge/偏置/中心变体（mean/midrange/中位数/trimmed）重试；
  Attention 后续只由官方回传或 21071 锚点源码绑定驱动。

### 全局

- 时间余量：根 281s / 硬限 300s，余量 19s。新增正式候选直接从当前最高分完整根构建；
  完整根候选在本地最终回退父状态时不提交。
- 缺口事实（2026-09-09）：根 `18053` 距榜首锚点 `21765` 差 3712 分；v195 完整包 `+21/+9s`
  与侧诊断 `+21/+5s` 不是同一时间口径，不做"剩余秒数换分"或"同量级机制数量"外推。
- cb1/cb2 存在实现错误，其负结果不能证明合法编码空间饱和；合法共享层级必须用五字段复核，
  operand MSE、单元素可表示值并集、非法 oracle 或连续残差不能证明最终输出天花板
  （`logs/execution/2026-09-05-next-plan-evidence-audit.md`）。
- A4/L4/C1、旧 clean-room balance/gamma/refine 和历史负向局部扫描不重开。
- 官方侧贡献比例不等于隐藏权重，也不能据此分摊榜首差距；零收益不能证明隐藏空桶。
