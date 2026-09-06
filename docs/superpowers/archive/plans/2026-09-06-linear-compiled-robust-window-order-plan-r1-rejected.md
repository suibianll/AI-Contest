# Linear 编译校准稳健窗口极值块序计划

状态：`CLOSED / R1_REJECTED`

## 假设与边界

父版本为根 v189（官方 `17616/275s`）。已关闭的输出协方差块序 fresh Overall
`0.687211924573` 低于本地最高，根不变。本计划注册一个独立的低成本校准统计：
对每个 64-channel block，先在最终部署坐标中计算每个校准窗口的

```text
window_score(block) = sum((A_window[block]^2) * activation_importance[block])
```

再固定使用该 block 在所有校准窗口上的最大值排序。它选择对最坏校准窗口最有
输出压力的 block，仍只编译一次 block permutation；codec、Linear/Attention
状态、连续变换和在线算子全部保持 v189。该规则从根 v189 独立构造，不复用上一
候选的块序 state。

不得扫描 mean/max/median、权重混合、block 大小、fold、layer/role 路由、阈值或
候选邻域；本计划只允许上述一个固定的窗口极值聚合。没有校准样本或非 64 整除
宽度时保留父静态路径。

## 执行顺序

### R0：单文件与合法 state

从根 v189 复制单文件候选，运行 `py_compile`、六 API 导入、合成合法调用和状态
检查。确认每个 order 是有限且完整的合法 64-block permutation，Attention
control 与父逐位一致，在线只读取已编译 order。

### R1：Linear 前两 shard

使用 `evaluator/eval.py`、固定 proxy-v2 cache、CUDA、`--linear-only
--shards 0,1`，baseline 为根 v189。记录 paired mean/median、L1、正负 case、
尾部、validation/test、order reachability 和未修改 Attention control。接口、
合法性或明显回归失败即关闭，不扫描聚合邻域。

### R2：六 shard、OOD、fresh default

R1 通过后运行 Linear 六 shard 与 OOD，再运行一次 fresh default。只把
`L1 < 0.02`、`|delta gap| <= 0.01` 作为风险诊断；本地分数不换算官方分数。
按六 API 分解模型预测官方时间，必须 `<280s` 才具备提交资格。

### R3：裁决

只有 fresh Overall 严格高于当前本地最高 `0.688994940507429` 且时间通过，才按
用户规则归档候选并提交官方；否则完整归档为 `REJECTED`，根保持 v189。官方结果
未确认前写 `unregistered/NA`。

## 执行结果

R0 通过。R1 shard0 的 Linear mean/median delta 为
`-0.000043038/-0.000117686`，worst-20% 为 `-0.001098366`；shard1 mean 为
`+0.000191295`。shard0 触发方向门禁，按固定规则关闭为
`CLOSED / R1_REJECTED`，不运行 R2、不扫描聚合邻域、不提交官方。

候选与 R1 证据归档于
`solutions/20260906_linear-compiled-robust-window-order_rejected/`；执行记录见
`logs/execution/2026-09-06-linear-compiled-robust-window-order-plan.md`。
