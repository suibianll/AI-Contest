# Attention softmax-logit Fisher 2×2 Q/K 配对计划

> 创建：2026-09-06  
> 状态：**CLOSED / F2_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 当前本地最高完整 default：Linear `0.641778372`、Attention `0.752173407`、
> Overall `0.687776303`（carrier-energy 候选，因时间预测 `281.401s` 未提交）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

当前 Attention 的 A1/A1 asymmetric fold 与普通 2×2 pair balance 已固定；本计划只
加入一个不同的统计目标：用 softmax 对 logit 的局部 Fisher 权重

```text
F_ts = p_ts · (1 - p_ts)
```

在 calibration 中对 Q/K 的 pair covariance 加权，再解一次 GQA-local SPD 2×2 平衡
变换。Q 使用 `M`，K 使用 `M^{-T}`，因此连续 `QK^T` 不变；部署 state 只增加/覆盖
已有合法 `pair_transform`，不保存 token、probability、V 或 Jacobian 张量。

这与已关闭的 output-Jacobian pair 不同：它不使用 `V`、输出残差或
`p·(V-O)`，只刻画 softmax logit 本身的 Fisher 几何；也不启用已关闭的
Q/K Fisher importance、head scale、rotation、source-scale 或参数邻域。配置固定为
causal/non-causal Fisher 平均、相邻二维坐标、一个 SPD 解和一次 validation gate，
禁止扫描 Fisher 混合、温度、pair stride、ridge、seed、head/layer 路由或阈值。

## 2. 固定执行顺序

### F0：单文件与不变量 smoke

从 v189 研究源码构造单文件候选，检查 `py_compile`、六 API、有限 Fisher/covariance、
合法 state；用合成 GQA 输入验证 `QK` 连续点积误差在浮点容差内。

### F1：Attention eval-v3 双 shard

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1`，baseline 为 v189，
逐 case 检查 mean/median、L1、正负 case、Q/K/V 与 logits/probability 来源、
reachability 和未修改 Linear control。若两 shard 没有一致正向信号，立即关闭。

### F2：六 shard 与 OOD

只有 F1 通过才运行 Attention 六 shard；要求 state/输出全有限、Q/K 连续不变量通过、
`L1 < 0.02`，并记录 QK/QKV interaction。OOD 只检查
`|Δ(gain_in-gain_ood)| <= 0.01`，不把本地分数换算为官方分数。

### F3：default、时间与提交条件

运行一次 fresh default。只有完整 Overall 严格高于当前本地最高
`0.687776303` 且六 API 分解时间预测 `<280s` 时，才归档源码、SHA、result、manifest
和仅含 `solution.py` 的 zip，并执行用户要求的提交/推送流程；官方平台回传前登记
`unregistered/NA`。若分数不超过本地最高或时间超门，归档为 `REJECTED`/`REJECTED_TIME`，
不提交、不扫描邻域，根 `solution.py` 保持 v186。

## 3. 证据边界

- 这是单一 Attention 机制，不与 Linear 候选合并；根文件不在官方回传前切换。
- 使用 `evaluator/eval.py` eval-v3 和同一 cache/panel/device；分片计时不能代替 fresh
  default 时间预测。
- 官方提交次数无限制，但每次提交仍须通过合法性、单一机制和 `<280s` 时间门；本地
  Overall 不等价于官方绝对分。

## 4. 执行裁决（2026-09-06）

- F0 通过：六 API 可导入；合成 GQA 连续 `QK` 最大绝对误差 `9.54e-7`。
- F1 双 shard 均正均值：shard0 `+0.013007`（L1 `0.015736`），shard1 `+0.005376`
  （L1 `0.010917`）；但分别含 `2` 个负 case，作为继续信号而非晋级结论。
- F2 在 shard4 自动停止：shard2 仅 `+0.000355`，shard3 `0/8` 变化，shard4
  `-0.001774`、正/负/零 `1/1/6`，最坏 case `-0.026800`（layer 4、validation、
  length 128）。因此机制证据否定，F3/OOD/default 未执行，不提交官方，不扫描邻域。
- 根 `solution.py` 仍为 v186；候选源码 SHA256
  `6C3150FAA44020A71C69205E6F8D0B43E6DC0431F6B60B7D10C135E743B5523C`，归档目录和
  详细 eval-v3 证据见对应 `solutions/` 与 `artifacts/proxy_v3/`。
