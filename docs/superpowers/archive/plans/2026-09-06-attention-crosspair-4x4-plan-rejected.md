# Attention 相邻 pair 的 4×4 GQA 配对变换计划

> 创建：2026-09-06
> 状态：**CLOSED / C2_OOD_REJECTED**
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）
> 当前本地最高完整 default：Overall `0.687776303`（carrier-energy 候选，时间预测
> `281.401s`，未提交）
> 根正式版本：v186（官方 `17599/272s`）

## 1. 目的与唯一机制

v189 的基础 Q/K 状态已经有独立 2×2 pair 变换；本计划只测试一个不同的坐标粒度：
把相邻两个二维 pair 组成固定的四维 super-pair，在其上拟合一次普通 covariance
平衡的 SPD 4×4 变换。Q 使用 `M`，K 使用 `M^{-T}`，所以连续 `QK^T` 保持不变。

新矩阵只捕捉相邻 pair 之间的交叉协方差；不使用 softmax Fisher、V、输出残差、
output Jacobian、head scale、rotation、source-scale 或动态搜索。固定相邻分组、单个
4×4 SPD 解、一次独立 validation gate；禁止扫描 super-pair stride、ridge、收缩、
seed、矩阵维度或其它邻域。部署 state 只保存合法 `pair_transform`。

## 2. 固定执行顺序

### C0：单文件与连续不变量 smoke

从 v189 研究源码构造候选，检查 `py_compile`、六 API、4×4 state 的有限性和合法形状；
用合成 GQA 输入验证 Q/K 连续点积最大误差在浮点容差内，并确认旧 2×2 state 仍可回退。

### C1：Attention eval-v3 双 shard

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1`，baseline 为 v189。
记录 mean/median、L1、正负 case、Q/K/V 与 logits/probability 来源、reachability 和
未修改 Linear control。若两个 shard 没有一致正向信号，立即关闭。

### C2：六 shard 与 OOD

只有 C1 通过才运行六 shard；要求输出有限、连续 QK 不变量通过、`L1 < 0.02`，并记录
QK/QKV interaction、最坏 layer/length。OOD 只检查相对父版本的
`|Δ(gain_in - gain_ood)| <= 0.01`，不把本地分数换算官方分数。

### C3：default、时间与提交条件

运行一次 fresh default。只有 Overall 严格高于 `0.687776303` 且六 API 分解时间预测
`<280s` 时，才归档源码、SHA、result、manifest 和仅含 `solution.py` 的 zip，并执行
提交/推送流程；官方回传前登记 `unregistered/NA`。否则归档为 `REJECTED` 或
`REJECTED_TIME`，不提交、不扫邻域，根 `solution.py` 保持 v186。

## 3. 证据边界

- 这是单一 Attention 交叉 pair 机制，不与 Linear 候选合并；根文件在官方正向前不切换。
- 使用 `evaluator/eval.py` eval-v3 和同一 cache/panel/device；分片时间不能替代 fresh
  default 时间预测。
- 官方提交次数无限制；官方分数和本地 proxy 分数完全分开记录。

## 4. 执行裁决（2026-09-06）

- C0 通过：六 API 可导入；4×4 连续 GQA `QK` 最大绝对误差 `9.54e-7`，旧 2×2 identity
  回退通过。
- C1 双 shard 正向：shard0 mean delta `+0.001844`（L1 `0.001844`，正/负/零
  `2/0/6`），shard1 `+0.009313`（L1 `0.010591`，正/负/零 `7/1/0`）。
- C2 ID 六 shard mean `0.758912691`，父 `0.752772355`，delta `+0.006140336`；
  但 OOD shard0/1 分别为 `-0.001240` 和 `-0.015858`，shard1 L1 `0.021338`、
  最坏 delta `-0.072815`（layer 19、validation、length 10），OOD 门禁失败并停止。
- C3 fresh default 未执行，官方未提交；候选源码 SHA256
  `3BA683A046083A5E9E043CC135532D01912A2CC437A5290ABB2D5D6E695E75F3`，根
  `solution.py` 仍为 v186。
