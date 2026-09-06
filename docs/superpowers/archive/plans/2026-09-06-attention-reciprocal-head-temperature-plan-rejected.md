# Attention reciprocal per-KV-head temperature 计划

> 创建：2026-09-06  
> 状态：**CLOSED / T1_NOOP_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 当前本地最高完整 default：Overall `0.687776303`（carrier-energy 候选，时间预测
> `281.401s`，未提交）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

在 v189 的 A1 per-KV-head logits gain、D1 非对称 fold 和已有 Q/K state 之后，测一条
尚未执行的 reciprocal allocation：对每个 KV head 采用同一个预注册的 reciprocal
temperature factor，使 Q 乘以该 factor、K 乘以其倒数。连续 Q·K logits 保持不变，
变化只来自 Q/K 两个独立 HiF4 lattice 如何分配动态范围。

候选仅打开已有
`_ATTN_OUTPUT_HEAD_SCALE=True`，固定使用代码内的
`_ATTN_OUTPUT_HEAD_SCALE_FACTORS=(0.50, 0.75, 1.25, 1.50, 2.00)` 候选池；不修改
factor 列表、A1/D1、center/permutation/pair-smooth、Linear/V 或在线路径。真实
attention-output scorer 和已有安全 gate 负责选择；不使用 Fisher、Jacobian、独立
headwise permutation、4×4/cross-pair、rotation、source-scale 或其它参数邻域。

## 2. 固定执行顺序

### T0：单文件与不变量 smoke

从 v189 研究源码生成单文件候选，运行 `py_compile`、六 API 独立导入、合成 GQA
calibration/dynamic 调用和 `validate_state`/`validate_hif4_params`。验证各 KV-head
factor 有限、Q/K 连续点积误差在浮点容差内、输出无 NaN/Inf；Linear/V control 不变。

### T1：Attention eval-v3 前两片

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1`，baseline 为 v189。
记录 mean/median signed delta、L1、正负 case、Q/K/V、logits/probability、最坏
layer/length、factor reachability 和 calibration API 时间。任一片无正向或出现明显
尾部回归即关闭，不修改 factor 候选池。

### T2：六片与 OOD

只有 T1 通过才运行 Attention 六 shard，并检查 `L1 < 0.02`、有限输出、未修改
Linear control 和 QK/QKV interaction；随后运行同 SHA OOD 六 shard，仅用
`|Δ(gain_in-gain_ood)| <= 0.01` 作拟合门禁。

### T3：default、时间、归档与提交

运行一次 fresh default。只有 Overall 严格高于 `0.687776303` 且分解时间模型预测
`<280s`，才归档源码/SHA/result/manifest 和仅含 `solution.py` 的 zip，并按用户要求
尝试提交与推送；官方平台上传未通过工具确认前登记 `unregistered/NA`。否则归档为
`REJECTED`/`REJECTED_TIME`，根 `solution.py` 保持 v186。

## 3. 证据边界

- 这是单一 reciprocal per-KV-head allocation 机制，不与 Linear 候选合并，不修改根文件。
- 使用 eval-v3、同一 cache/panel/device 和 v189 parent；分片分数不等价于 default 或官方。
- 不扫描 factor、fold、seed、head/layer 路由、threshold 或其它邻域；官方提交次数无限制，
  但本地分数严格超过当前最高且预测时间 `<280s` 才进入提交流程。

## 4. 执行裁决（2026-09-06）

- T0 通过：`py_compile`、六 API 独立导入、合成 GQA calibration/dynamic 调用、
  `validate_state`/`validate_hif4_params` 和有限输出检查通过；Q/K state multiplier
  有限且形状合法。
- T1 shard0：8 个 Attention case，candidate mean `0.777742183683`，delta mean `0`、
  L1 `0`、正/负/零 `0/0/8`。
- T1 shard1：8 个 Attention case，candidate mean `0.805886376364`，delta mean `0`、
  L1 `0`、正/负/零 `0/0/8`。

两片共 16 个 case 与 v189 逐位一致；factor 候选未产生部署变化。T2/T3 未执行，
不提交官方、不分配版本号、不扫描 factor 或邻域。候选源码 SHA256 为
`162CFCF2C0403D38176B16328FCE1A4D5DED010E196924286B9588D7757FC013`，根 `solution.py`
仍为 v186。
