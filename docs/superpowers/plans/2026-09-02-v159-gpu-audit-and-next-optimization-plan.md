# v159 GPU 审计与下一步优化计划

> 状态：**ACTIVE**
>
> 更新：2026-09-02
>
> 父版本：v159，源码 SHA256
> `0508045A0DDD0F17679DCA827C265CFC7588E76081D3AECEFF555D257DD4242`

## 1. 已确认事实

- v159 官方分数为 **17532**，官方时间未提供；相对 v158 `16861/223s` 提升 `671` 分，说明
  合并后的 Linear GPTQ 主框架有效。
- 用户确认的 17816 仍高于 v159 `284` 分，但其完整源码、Attention 配置和官方时间没有同步；
  这 `284` 分不能归因给某个本地模块。
- CPU Linear compact 为 `0.705515`，相对 v158 配对 `+0.149191`，56/56 case 改善；它只证明
  当前公开 Qwen holdout 上机制稳定，不解释官方绝对分。
- 当前 GPU default 不是低分，而是 `ERROR`：校准阶段把必须返回的 CPU state 提前用于 CUDA
  张量计算，首个错误在 `solution.py:8135`；同类使用还位于 8187、8272 附近。
- CPU compact 的 API `167.570s` 中 Weight calibration 为 `131.693s`（78.6%），Activation
  dynamic 为 `35.877s`（21.4%）。复杂度第一目标是 Linear calibration，而不是 Attention。
- v159 的 W-only/A-only 控制臂均大幅退化、W+A 却显著改善，属于强耦合坐标方案；后续不能
  独立删改 Weight 或 Activation 路径来推断收益。

## 2. 执行顺序

### P0. 修复 GPU 可执行性，不改数学

只修改校准内部临时张量：计算时使用 `best_d/best_perm` 的当前 device 版本，返回 state 时仍复制
到 CPU，保持公开状态合法。同步检查三处 transformed-sample 路径，不增加兼容层或辅助抽象。

验收：

1. 单个 Linear state 的 CUDA smoke 通过；
2. CPU/CUDA 输出字段、shape 和合法性一致；
3. 同一小样例的分数只允许正常数值舍入差异；
4. 这一步不分配新版本、不提交官方评测。

### P1. 建立唯一有效的 GPU 基线

1. `.venv` + CUDA，先跑 v158 与 v159 的 Linear-only compact；父版本只运行一次并保存 JSON。
2. v159 使用同 cache、同 device 与 v158 精确配对，检查 mean、median、q25、worst quartile、
   negative cases、validation/test 同号率和逐 role。
3. compact 通过后只跑 **Linear-only default 168 cases**；Attention 已冻结，不重复 120 个
   Attention case，也不跑 `--full-cases`。

门槛：GPU compact 不应出现系统性负 case；若 CPU/GPU 排序或逐 case 符号明显变化，先停止并
审计数值路径，不能进入算法优化。

### P2. 先降复杂度，再提高精度

先做不改变候选集合和输出的等价优化，每项单独验证：

1. 变换后的 calibration samples 只构造一次，复用于 ratio、adaptive regularization 和 offset
   选择；避免三次重复 smooth/permutation/Hadamard。
2. `H_act_base`、Weight Gram、importance 和可复用分解只计算一次；禁止为同一 state 重建相同
   矩阵。
3. 对候选 metric 做批量计算或共享中间量，不改变候选、接受条件和最终 state。

等价优化要求逐 case 输出不变；目标是先降低 Weight calibration 时间，不能用删 case 或缩短
输入伪造加速。完成等价优化后，再用未编号 workbench 分别消融 adaptive-reg、adaptive-offset 和
joint smooth 搜索；一次只关闭一项，compact 配对和 default 尾部均通过后才保留。

### P3. 解释 17532 与 17816 的差距

第一优先是补齐 17816 的完整提交源码、Attention 配置和官方时间，并与 v159 做源码级差分。
在这些信息缺失时，不围绕本地 proxy 调参去“拟合 284 分”。若需要官方 A/B，只允许一个变量：

- Linear 完全固定，只替换可确认的 Attention 配置；或
- Attention 完全固定，只验证一个 Linear 复杂度消融。

本地已知 v159 compact role 为 q/k/v `0.822/0.830/0.821`、o `0.688`、proj `0.651`、
fc_gate/fc_up `0.595/0.532`。这些只把 fc/proj 标为诊断重点，不作为官方权重推断。

### P4. 下一精度机制

在 P0–P3 完成前不增加新算法。之后从 v159 分支，冻结 v158 Attention，只做一个 Linear 数学
变化：在现有 A/W 联合坐标中实现单次、共享分解的 block residual/GPTQ 更新，优先覆盖
fc/proj 形状，但使用统一规则，不写 layer/role 特调表。

验收固定为：

- 两个 calibration fold 与 validation/test holdout 同方向；
- median、q25、worst quartile 不退化，不能只看 mean；
- W-only/A-only/W+A interaction 可解释；
- GPU Linear default 时间相对 v159 不增加，官方时间仍只认回传；
- 一次只提交一个数学机制，失败不继续扩大 block、offset、seed、rank 或 damping 网格。

## 3. 最小测试集

- 接口/设备修复：1 个代表性 state smoke，仅判正确性。
- 日常机制：固定 56-case Linear compact，读取已有 NVFP4 cache。
- 晋级审计：仅一次 168-case Linear default。
- Attention：只有 Attention 代码变化时才跑；当前完全跳过。
- `--full-cases`：无明确 stress 问题时不运行。

## 4. 当前决策

当前立即执行顺序是 `P0 GPU 修复 → P1 GPU compact/default → P2 等价降复杂度`。在取得有效 GPU
profile 和 17816 完整配置前，不继续增加 GPTQ 轮次、搜索候选或 Attention 机制。
