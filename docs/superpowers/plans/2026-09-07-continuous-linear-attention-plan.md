# Linear / Attention 持续优化总计划

> 最新Linear回传：L23 **TIMEOUT**，未回传精确耗时/分数；旧PENDING记录已失效。关闭本复杂度实现，后继修正全数据求解并消除重复全矩阵计算，另存新SHA；父L4及根v189不变。[回传记录](../../../logs/execution/2026-09-07-l23-official-timeout.md)。

> ACTIVE，2026-09-07。当前阶段：READY_FOR_DESIGN；本轮只更新研究计划。

## 当前执行入口

[21071机制驱动的下一轮计划](workpackages/21071-evidence-driven-research.md)是当前双侧机制、
配置、验证、组合和后继分支的唯一明细；旧L21/A22执行卡保留作历史设计。
测试按[4B指引](../../4b-panel-testing-guide.md)，证据纠偏见[记录](../../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)。

目标是复现可验证的A@W拟合与Q/K互逆学习收益，再挑战21071、21765；不承诺分数。
官方50/250为样例数量，不推断权重或隐藏数据关系；外部21071/283s不继承为仓库成绩。

## 当前父与队列

| 侧 | 官方父/对照 | 下一步 |
|---|---|---|
| Linear | L4 4607/247s | R0证据纠偏及去重 → L23 A@W低维拟合（残差监督子空间，先去重） |
| Attention | 研究父A22-2 14424/271s；高分A2 14440/274s；低成本R3 14405/238s | A23父坐标下Q/K量化块scale乘积目标 |
| 完整包 | v189 17616/275s | 检查已有组合，缺失则登记C23-0：L4+A22-2 |

侧父完整路径/SHA、official_history、pending_official由
[Linear state](../../../workbench/continuous_linear/state.json)与[Attention state](../../../workbench/continuous_attention/state.json)维护。
Linear冻结v162 standard Attention；Attention冻结v162 standard Linear和父V。
本地负向不扩写整族关闭，官方明确负结果不重试；未回传不晋级。

## 执行、隔离与交付

1. 每个候选一个机制和固定配置，去重后才实现；Linear直接用全部4B校准数据拟合与选择，不考虑泛化；Attention保持训练/选择/holdout分离。
2. 先六API接口、合法state、连续不变量、真实硬前向及reachability；再4B目标侧shard0和固定六shard。
3. Linear以4B校准数据上的合法部署A@W拟合改善作为本地收益条件，独立窗口均值/split/负向损失不作门。
   Attention正常门仍为Δmean>0、validation/test各正、平均负向损失<0.02；总L1仅记录。
   通用分析器reject不能替代专项指标；Linear新用户指令优先；Attention旧卡符号豁免不自动延伸新卡。
4. 时间仅记录API秒数；官方300s是唯一时间硬限，不跑0.5B、OOD、跨模型或fresh-default计时。
5. 合法单文件、训练机制contract fuzz、control与源码SHA确认后才准备官方包；官方不限次数，
   禁止重复同SHA/逐位等价提交。组合单独验证，不能相加侧分/时间冒充官方结果。
6. 每侧最多一个待官方包；失败回官方父换机制，等待时做独立后继设计，不无限空转。

继续使用workbench/continuous_linear、continuous_attention各自目录；每轮config/source/manifest/report
和独立日志落盘，旧结果不覆盖。CUDA共用原有artifacts/proxy_v3/v162-independent/gpu.lock，
遵守PID/原子创建/finally释放规则，不删除其他任务的锁。本计划不启动代理或自动化。
只stage本轮文件，diff-check、commit、push并核对status。

## 当前进度

R0纠偏文档已写；L23/A23为DESIGN_READY，机制去重及编译可行性尚待执行；
C23-0为INTEGRATION_PLANNED，尚未验证是否已有同SHA组合结果。
没有新增候选源码、测试结果或官方提交，根solution.py不变。

### L23 直接拟合版（2026-09-07 用户指令：不拆fit/select、官方裁决）

- L23 去重（dedup-r0.md）与数学检查（math_check.py）完成；候选改为全校准行
  严格改善接受（无 holdout 守门）。脱离仓库导入检查 + CPU contract fuzz 全 PASS。
- 4B 六 shard paired（父 L4，336 例）：candidate gain +0.3426 vs 父 +0.5243，
  Δmean −0.18、0/336/0 —— **record-only**（直接拟合政策下不阻止官方探索）。
  拟合质量：各层 L_all 相对父降 50–62%，accepted 88–110/144（宽）/24–29/40（窄），机制可达。
- 已归档 `solutions/continuous_linear_l23-residual-subspace/`（SHA
  `33D1DA51…E35D`），注册官方探索（PENDING_OFFICIAL）。官方比分与 300s 为唯一裁决；
  正向则登记新 Linear 侧父并考虑 L23+A22-2 组合，负向只关闭该实现、不扫邻域。
