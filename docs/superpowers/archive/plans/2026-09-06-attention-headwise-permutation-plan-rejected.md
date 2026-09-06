# Attention Q/K 独立 headwise permutation 计划

> 创建：2026-09-06  
> 状态：**CLOSED / P1_NOOP_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 当前本地最高完整 default：Overall `0.687776303`（carrier-energy 候选，时间预测
> `281.401s`，未提交）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

v189 的 Attention 校准已经包含 tied Q/K headwise ordering：同一 KV head 的 Q/K
使用同一个通道排列。当前只测一个尚未执行的独立机制：在相同真实 attention-output
校准目标下，分别允许 Q-only、K-only 和 Q+K independent headwise permutation
候选，让两个独立 HiF4 lattice 各自使用其输出误差更敏感的通道布局。

候选只打开研究代码已有的
`_ATTN_OUTPUT_HEADWISE_PERMUTATION=True`，保留
`_ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES=4`、所有父版本 scale/center/D1/pair-smooth、
Linear/V 路径和部署 refine 不变。它不使用 Fisher、Jacobian、4×4/cross-pair、
rotation、source-scale、head gain 或任何参数邻域；在线只读取已有合法 permutation
state，不增加动态搜索。由于 Q/K permutation 是独立的，本机制不宣称连续 QK 不变量，
而是直接用实际 causal attention output 误差验证离散布局是否有收益。

## 2. 固定执行顺序

### P0：单文件与合法 state smoke

从 v189 研究源码生成单文件候选，运行 `py_compile`、六 API 独立导入和合成 GQA
调用；检查 Q/K state 的 permutation 为有限整型、长度与 head/channel 形状匹配，
输出无 NaN/Inf。Linear/V control 必须与父路径保持不变。

### P1：Attention eval-v3 前两片

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1`，baseline 为 v189。
记录 mean/median signed delta、L1、正负 case、Q/K/V、logits/probability、最坏
layer/length 和 permutation reachability。若两片没有稳定正向、出现明显尾部回归或
state/接口错误，立即关闭；不运行其它排列候选。

### P2：六片与 OOD

只有 P1 通过才运行 Attention 六 shard，并检查 `L1 < 0.02`、未修改 Linear control、
QK/QKV interaction 和实际输出有限性。随后运行同 SHA OOD 六 shard；仅用
`|Δ(gain_in-gain_ood)| <= 0.01` 作拟合门禁，不把本地 proxy 换算成官方分数。

### P3：default、时间、归档与提交

运行一次 fresh default。只有 Overall 严格高于 `0.687776303` 且分解时间模型预测
`<280s`，才归档源码/SHA/result/manifest 和仅含 `solution.py` 的 zip，并按用户要求
尝试提交与推送；官方平台上传未通过可用工具确认前登记 `unregistered/NA`。不满足
任一条件则归档为 `REJECTED`/`REJECTED_TIME`，根 `solution.py` 保持 v186。

## 3. 证据边界

- 这是单一 Attention headwise permutation 机制，不与 Linear 候选合并，不修改根文件。
- 使用 eval-v3、同一 cache/panel/device 和 v189 parent；分片分数不等价于 default 或官方。
- 不扫描候选数、fold、threshold、seed、排列基、层/头路由或其它邻域；官方提交次数无限制，
  但本地门和时间门仍是提交前置条件。

## 4. 执行裁决（2026-09-06）

- P0 通过：`py_compile`、六 API 独立导入、合成 GQA calibration/dynamic 调用和
  `validate_state`/`validate_hif4_params` 均通过，输出有限。
- P1 shard0 与 shard1 均为逐 case no-op：16 个配对 case 的 candidate/base 输出
  完全一致，Attention `delta_mean=0`、`L1=0`、正/负/零 `0/0/8`（每片）。校准路径
  实际运行了 C76.1 分支，但独立 permutation 没有形成被部署的 state；这不是正向证据。
- P2/P3 未执行；不提交官方，不分配版本号，不扫描排列候选或其邻域。候选源码 SHA256
  为 `5F94099DAFD5F44201430A943DDEC75EBCE3AB8F735C90EED5774D92ECA69683`，归档于
  `solutions/20260906_attention-headwise-permutation_rejected/`；根 `solution.py`
  仍为 v186。
