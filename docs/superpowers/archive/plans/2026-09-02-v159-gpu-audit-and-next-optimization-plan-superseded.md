# v159 GPU 审计与下一步优化计划

> 状态：**ACTIVE**
>
> 更新：2026-09-02
>
> 当前 v159 归档 SHA256：
> `13C9CF0BFCF2277F0828D8CC1A18A8F7414DB183F3E27DD898D52597ACC5EC79`

## 1. 已确认事实

- v159 原始官方提交 SHA `0508045A...4242` 的分数为 **17532**，官方时间未提供；相对 v158
  `16861/223s` 提升 `671` 分。当前归档只增加数学等价的 GPU device 修复，尚未官方复测。
- 17816 完整源码无法提供；它只保留为不可复现的官方锚点，不再作为等待项或代码父版本。
- CPU Linear compact 为 `0.705515`，相对 v158 配对 `+0.149191`，56/56 case 改善；它只证明
  当前公开 Qwen holdout 上机制稳定，不解释官方绝对分。
- GPU 修复后的 compact 为 `0.705508`，median/q25/worst-quartile
  `0.685426/0.591841/0.540094`，56/0/0；相对同设备 v158 为 `+0.149185`，56/56 改善。
- GPU Linear default 为 `0.633526`，median/q25/worst-quartile
  `0.626581/0.536043/0.434968`，167/1/0；唯一负例是 layer 22 `o`、length 10。
- GPU default API `269.435s`、wall `291.145s`；Weight calibration `208.971s`（77.6%），
  Activation dynamic `60.464s`（22.4%）。复杂度第一目标仍是 Linear calibration。
- v159 的 W-only/A-only 控制臂均大幅退化、W+A 却显著改善，属于强耦合坐标方案；后续不能
  独立删改 Weight 或 Activation 路径来推断收益。

## 2. 执行顺序

### P0. 修复 GPU 可执行性，不改数学（DONE）

只修改校准内部临时张量：计算时使用 `best_d/best_perm` 的当前 device 版本，返回 state 时仍复制
到 CPU，保持公开状态合法。同步检查三处 transformed-sample 路径，不增加兼容层或辅助抽象。

验收：

1. 单个 Linear state 的 CUDA smoke 通过；
2. CPU/CUDA 输出字段、shape 和合法性一致；
3. 同一小样例的分数只允许正常数值舍入差异；
4. 这一步不分配新版本、不提交官方评测。

结果：三处临时 transformed-sample 计算改用当前 device 的 `best_d/best_perm`，返回 state 仍为
CPU；单 state、动态 Activation、compact 和 default 均通过。修复已直接同步进 v159 归档，
未创建新版本。

### P1. 建立唯一有效的 GPU 基线（DONE）

1. `.venv` + CUDA，先跑 v158 与 v159 的 Linear-only compact；父版本只运行一次并保存 JSON。
2. v159 使用同 cache、同 device 与 v158 精确配对，检查 mean、median、q25、worst quartile、
   negative cases、validation/test 同号率和逐 role。
3. compact 通过后只跑 **Linear-only default 168 cases**；Attention 已冻结，不重复 120 个
   Attention case，也不跑 `--full-cases`。

门槛：GPU compact 不应出现系统性负 case；若 CPU/GPU 排序或逐 case 符号明显变化，先停止并
审计数值路径，不能进入算法优化。

结果：CPU/GPU compact mean 仅差约 `7e-6`；v159 对 v158 同设备 56/56 改善。Linear default
已保存为后续唯一父基线，不再重复运行。

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

当前进度：前两项等价复用已完成。transformed calibration samples 只生成一次，
`weight_hat.T @ weight_hat` 只计算一次并由 Activation Gram/GPTQ Hessian 共享。相对 P0 CUDA
compact，56/56 case 输出完全相同；API `52.321→51.055s`。计时存在约 0.5 秒波动，因此只记为
小幅去重，不外推 default 或官方时间。下一步先做热点分解，再决定是否消融 adaptive-reg、
adaptive-offset 或 joint smooth；不继续进行无证据重构。

### P3. 17816 边界（CLOSED）

17816 完整源码无法提供，因此不再等待或尝试源码级复原，也不围绕本地 proxy 拟合 284 分差。
后续全部优化从当前 v159 归档继续，Attention 固定为 v158。

本地已知 v159 compact role 为 q/k/v `0.822/0.830/0.821`、o `0.688`、proj `0.651`、
fc_gate/fc_up `0.595/0.532`。这些只把 fc/proj 标为诊断重点，不作为官方权重推断。

### P4. 下一精度机制

在 P2 完成前不增加新算法。之后直接修改 v159 现有归档，冻结 v158 Attention，只做一个 Linear 数学
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
profile 后进入 P2；P0/P1 已完成。17816 不再作为阻塞项，不增加 GPTQ 轮次、搜索候选或
Attention 机制。P2 的两项等价复用已同步到 v159 归档；后续同算法优化继续更新该归档，不创建
新版本。
