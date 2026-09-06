# Linear 编译校准输出协方差块序计划

状态：`CLOSED / R3_REJECTED_SCORE`

## 假设与边界

父版本为根 v189（官方 `17616/275s`）。上一计划证明了“校准样本能量 ×
权重重要性”的固定块序有本地正向，但最终只与本地最高分持平，已归档；本计划
只测试一个不同的、输出目标对齐的块序统计，不继承上一候选的样本能量块序。

在最终 Linear 变换坐标中，校准协方差为 `C_A`，部署量化权重的输出 Gram 为
`H_W = Q(W)^T Q(W)`。对每个合法 64-channel block，固定按

```text
score(block) = trace(C_A[block, block] · H_W[block, block])
```

排序，再复用 v189 的静态 activation-GPTQ 路径。该统计是对角能量乘积的完整
块内协方差扩展，直接近似该 block 的输出能量，而不是重新量化或改变 codec。
`C_A` 与 `H_W` 都已在父校准流程中生成；只保存一个 block permutation，不增加
在线矩阵运算或 state 维度。对没有完整校准协方差的超宽输入，保留父块序，避免
引入额外估计或路由。

禁止扫描块大小、混合权重、平方/绝对值、fold、layer/role 路由及任何阈值邻域；
本计划只允许这一条固定的块内输出协方差规则。

## 执行顺序

### R0：单文件与合法 state

从根 v189 复制单文件候选，运行 `py_compile`、六 API 导入、合成合法调用和状态
检查。确认排序有限、是完整合法 64-block permutation，连续变换不变，Attention
与父逐位一致，在线路径只读取已编译 order。

### R1：Linear 前两 shard

使用 `evaluator/eval.py`、固定 proxy-v2 cache、CUDA、`--linear-only
--shards 0,1`，baseline 为根 v189。记录 paired mean/median、L1、正负 case、
尾部、validation/test、order reachability 和未修改 Attention control。出现
明显回归、no-op 或 state/control 问题即关闭，不扫统计邻域。

### R2：六 shard、OOD、fresh default

R1 通过后运行 Linear 六 shard 与 OOD，再运行一次 fresh default。只将
`L1 < 0.02`、`|delta gap| <= 0.01` 作为风险诊断；本地分数不换算官方分数。
时间使用六 API 分解模型，预测必须 `<280s`。

### R3：裁决

只有 fresh Overall 严格高于当前本地最高 `0.688994940507429` 且时间通过，才按
用户规则归档候选并提交官方；否则完整归档为 `REJECTED`，根保持 v189。官方网页
未通过工具确认前，官方字段写 `unregistered/NA`。失败后不在本计划内扫描公式邻域。

## 执行结果

R1 初始运行因超宽输入缺少父版本 hdiag 回退而判为无效，已保留并修复。修复后 R1、
R2 六 shard 与 OOD 均通过接口/风险检查；fresh default 为
`0.640810865681/0.752173407020/0.687211924573`，官方时间模型预测
`279.215656s`。Overall 低于本地最高 `0.688994940507429`，故关闭为
`CLOSED / R3_REJECTED_SCORE`，未提交官方，根保持 v189。

候选与证据归档于
`solutions/20260906_linear-compiled-output-covariance-order_score-rejected/`；
执行记录见 `logs/execution/2026-09-06-linear-compiled-output-covariance-order-plan.md`。
