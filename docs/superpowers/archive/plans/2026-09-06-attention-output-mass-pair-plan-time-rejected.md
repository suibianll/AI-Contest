# Attention 输出质量加权 2×2 Q/K 变换计划

> 创建：2026-09-06  
> 状态：**CLOSED / M2_TIME_REJECTED**  
> 父版本：v189（local Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 参考：输出 Jacobian pair 变体（local positive，但时间门失败）

## 1. 目的与唯一机制

上一轮证明完整输出 Jacobian 加权 pair transform 有小幅正向信号，但新增完整敏感度
计算会把时间推到 `280s` 门外。本计划只利用 base Attention calibration 已经执行的
causal attention probability 质量统计：捕获每个 query-head/token 与 key-token 的
`p²` mass，用它对同一合法 2×2 Q/K covariance 做一次固定加权拟合。

固定规则：

1. 不新增校准 softmax/matmul；仅复用已有 `_attention_head_square_mass` 的 causal
   `p²` 统计，按 GQA group 聚合 K 侧；
2. 使用 v189 完整 Q/K/V state，奇数位 calibration windows 拟合、偶数位真实部署
   attention gate 验证；
3. 只生成一个 SPD reciprocal 2×2 pair-transform candidate，拒绝则保留父 state；
4. 不改变 ridge、token/head 参数、候选数量、V state 或 Linear state，不扫描此前
   output-Jacobian 的权重、blend、ridge 或 token 邻域。

该机制与完整 Jacobian 版本不同：它只检验已有 causal attention mass 是否足以提供
输出路由权重，且校准期零新增 Q/K attention 计算。

## 2. 固定执行顺序

### M0：单文件与状态 smoke

从 v189 复制候选，检查 `py_compile`、六 API、有限输出、合法 pair-transform shape、
连续 Q·K 不变量和 mass capture 可达；root `solution.py` 保持不变。

### M1：Attention eval-v3 六 shard 配对

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1,2,3,4,5`，baseline
为 v189；记录 48 个 case 的 mean/median/L1、Q/K/V 与 interaction、长度/层尾部、
mass reachability 和 API 时间。前两个 shard 明确负向或全程 no-op 即关闭，不调参数。

### M2：default/OOD 与时间

只有 M1 有实质正向变化才运行 OOD 和 proxy-v2 default-panel fresh audit；要求
Attention 严格高于 v189、Linear control 逐位不变、OOD `|Δ(gain_in-gain_ood)|<=0.01`，
并用六 API 分解模型预测 `<280s`。

### M3：归档与官方

全部门禁通过才分配 v190、保存源码/SHA/manifest/result/zip，并准备官方提交包；官方
网页上传未通过工具确认前记 `unregistered/NA`，root 不自动切换。失败或官方负向后
关闭该固定 mass 机制，不扫描参数邻域。

## 3. 实际执行结果

### M0

单文件编译、六 API 导入、有限输出、合法 pair-transform shape、连续 Q·K 不变量和
mass capture 均通过。capture 输出为有限的 query/key `p²` mass，候选源码 SHA-256 为
`10154C7EF5FAC891433687BA34B88BCECF4E60C6C8BCE0B12A7753BB32D44EBF`；root v186
未修改。

### M1：eval-v3 Attention 六 shard

48 个配对 case 的候选 mean 为 `0.752881086113999`，v189 baseline 为
`0.752772354840816`，delta `+0.000108731273184`。shard delta 依次为
`0`、`+0.001415867840441`、`-0.000763480201340`、`0`、`0`、`0`；收益不具备
材料幅度。M2 OOD 候选为 `0.751526904172546`，父为 `0.751857367978909`，相对
父的 gap 变化约 `+0.0004392`，未触发 OOD 门禁。

### M2：fresh default 与时间

固定 proxy-v2 fresh default 的结果为：Linear `0.640258324429894`，与 v189
`0.640258324429894` 逐位一致；Attention `0.752399782610857`，相对 v189
`0.752173407020070` 为 `+0.000226375590787`；Overall
`0.686983932005295`，相对 v189 `0.686889608842467` 为 `+0.000094323162828`。
API 分解为 `W_calib=271.393797799712s`、`A_calib=60.488939199597s`、
`dyn_act=59.605983000249s`、`dyn_qkv=3.135593600005s`，校准时间模型预测
`282.286164185503s`，不满足 `<280s`。

## 4. 决策

虽然默认 Attention 有极小正向变化且 OOD 通过，但固定 mass pair 机制未通过时间门，
不分配 v190、不提交官方、不扫描参数邻域。候选源码、manifest、fresh-default JSON
和报告归档于 `solutions/20260906_attention-output-mass-pair_time-rejected/`；root
`solution.py` 仍为 v186。
