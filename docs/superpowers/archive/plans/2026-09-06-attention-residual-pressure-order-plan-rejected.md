# Attention residual-pressure tied permutation 计划

> 创建：2026-09-06  
> 状态：**CLOSED / R1_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 当前本地最高完整 default：Overall `0.687776303`（carrier-energy 候选，时间预测
> `281.401s`，未提交）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

前两轮 Attention permutation/temperature 候选在 v189 上 no-op，因为它们分别使用
幅值排序或 reciprocal allocation，未改变最终 state。本计划只测一个不同统计目标：
在父版本已选的 `d/center` 连续坐标中，对每个 Q/K 通道实际运行一次基础合法 HiF4
量化，累计逐通道归一化重构残差压力，再将同一 KV-head 内的高压通道按固定顺序聚合。

Q 与其对应 K 使用同一个 tied KV-head permutation，因此连续 Q·K logits 不变；候选
只改变已有 `permutation` state，仍由真实 attention-output scorer 和父 gate 选择。
不使用 Fisher/Jacobian、temperature、独立 permutation、4×4/cross-pair、rotation、
source-scale、block-size 或参数邻域，也不增加在线运算。

## 2. 固定执行顺序

### R0：单文件、合法性与连续不变量 smoke

从 v189 研究源码生成单文件候选，运行 `py_compile`、六 API 独立导入、合成 GQA
calibration/dynamic 调用和 `validate_state`/`validate_hif4_params`。验证 residual
pressure/order 有限、同 KV-head 的 Q/K permutation 一致、连续 QK 点积误差在浮点容差内，
输出无 NaN/Inf。

### R1：Attention eval-v3 前两片

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1`，baseline 为 v189。
记录 mean/median signed delta、L1、正负 case、Q/K/V、logits/probability、最坏
layer/length、非 identity permutation reachability 和 calibration API 时间。任一片无
稳定正向、出现明显尾部回归或状态错误即关闭；不扫描统计窗口或排序规则。

### R2：六片与 OOD

只有 R1 通过才运行 Attention 六 shard，检查 `L1 < 0.02`、有限输出、未修改 Linear
control、QK/QKV interaction；随后运行同 SHA OOD 六 shard，仅用
`|Δ(gain_in-gain_ood)| <= 0.01` 作拟合门禁。

### R3：default、时间、归档与提交

运行一次 fresh default。只有 Overall 严格高于 `0.687776303` 且分解时间模型预测
`<280s`，才归档源码/SHA/result/manifest 和仅含 `solution.py` 的 zip，并按用户要求
尝试提交与推送；官方平台上传未通过工具确认前登记 `unregistered/NA`。否则归档为
`REJECTED`/`REJECTED_TIME`，根 `solution.py` 保持 v186。

## 3. 证据边界

- 这是单一 tied residual-pressure permutation 机制，不与 Linear 候选合并，不修改根文件。
- 使用 eval-v3、同一 cache/panel/device 和 v189 parent；分片分数不等价于 default 或官方。
- 候选统计只使用 calibration，固定全量已捕获窗口、一个压力定义和一次排序；不扫描
  fold、threshold、seed、候选数、层/头路由或其它邻域。

## 4. 执行裁决（2026-09-06）

- R0 通过：`py_compile`、六 API 独立导入、合成 GQA 调用、合法 state、有限输出和
  tied Q/K 连续 QK 不变量检查通过。
- R1 shard0：8 个 Attention case，delta mean `-0.005305`、L1 `0.020938`、
  正/负/零 `2/2/4`；最坏 layer12/length128 为 `-0.061995`。
- R1 shard1：8 个 Attention case，delta mean `0`、L1 `0`、正/负/零 `0/0/8`。

R1 已出现负向且超过 L1 门限，R2/R3 未执行；不运行 OOD/default，不分配版本号，
不提交官方，不扫描压力定义或 permutation 阈值。候选源码 SHA256 为
`87375D16F2A71827243E633C4C464121E853393E63E91A2939EFC112B5285CAE`，根 `solution.py`
仍为 v186。
