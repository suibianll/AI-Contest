# Linear 校准输出能量加权静态 GPTQ 块序计划

> 创建：2026-09-06  
> 状态：**CLOSED / E2_TIME_REJECTED**  
> 父版本：v189（local Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，official `unregistered/NA`）  
> 根版本：v186，`solution.py` 保持不变

## 1. 目的与唯一机制

v189 已验证依据部署量化权重列能量生成静态 64-block GPTQ 顺序；本计划只测试一个
未测过的顺序统计：用 calibration NVFP4 激活经过与在线完全相同的连续变换后，计算
每个 block 的实际校准激活能量，并与当前部署坐标中的权重列能量相乘，按该输出能量
降序排列 block。它不改变 HiF4 编码、权重参数、激活 state 数量、Hessian、refine
预算、Attention 或动态 API 算子。

固定规则：

1. 先调用 v189 父校准，保持其所有 state 与权重编码逐位不变；
2. 从 calibration 输入取固定的确定性行样本，应用 `_static_actorder_dense_from_state`
   的同一 `smooth_inv/permutation/block_smooth/residual` 坐标链；
3. 计算 `score_b = Σ_{j∈b} E[X_j²]·importance_j`，只写入一个完整
   `gptq_block_order` 排列和标记字段；没有额外候选或阈值搜索；
4. 行样本不可用时退回 v189 hdiag 顺序，保证 state 合法且父路径可复现。

该统计与 v189 的纯权重 hdiag 块序不同，也不改变已有 Linear 量化目标；失败后不扫描
混合权重/激活比例、排序方向、采样数、block 大小或 role/layer 路由。

## 2. 固定执行顺序

### E0：单文件与 state smoke

从 v189 单文件复制，运行 `py_compile`、脱离目录导入六 API；检查 order 是
`0..blocks-1` 的 CPU `int16` 完整排列，父权重与非 Linear state 不被改写，有限输出和
连续 Linear 变换不变量通过。

### E1：eval-v3 Linear 配对

固定 proxy-v2 dense cache、CUDA、v189 baseline，先跑 Linear shard `0,1`。若前两个
shard 的配对均值非正或 order 不可达，立即关闭；若通过，再完成六 shard 的 336 cases。
记录 mean/median/L1、q25/q75、worst quartile、validation/test 同号率、role/layer
尾部、未修改 Attention control 和实际 API 时间。

### E2：OOD、default 与时间

只有 E1 有实质正向变化才运行 OOD 六 shard及 fresh default。要求：Linear 严格高于
v189、Attention 与 v189 逐位一致、`|Δ(gain_in−gain_ood)|<=0.01`，并以六 API
分解模型预测 `<280s`；跨模型仅记录，不作门禁。

### E3：归档与官方

所有门禁通过且 local default Overall 严格高于当前最高有效父版本时，保存源码、SHA、
manifest、完整 JSON/Markdown、OOD 证据和 `solution.zip`，登记下一个版本候选并准备
官方提交；官方网页上传未由工具确认前写 `unregistered/NA`，root 不自动切换。
官方回传正向才升级父版本；负向/超时则关闭该固定顺序机制，不扫描邻域。

## 3. 停止条件

接口、state、有限值、case identity 或时间硬门失败均停止本配置。局部 proxy 只用于
筛选和归因，不能换算官方分数；Linear/Attention `0.9` 是长期目标，不以本候选的局部
变化宣称达到。

## 4. 实际结果

E0 通过，候选 SHA-256 为
`2C37E66A01550A2B83818AE76A45DACC AFBD625D90D1652804BBBB1FBE5A03B`（去除空格即为
`2c37e66a01550a2b83818ae76a45daccafbd625d90d1652804bbbb1fbbe5a03b`）；root v186 未修改。
E1 六 shard 的 Linear mean 为 `0.637795463532374`，父 v189 为 `0.636799488553279`，
delta `+0.000995974979095`；六个 shard 均为正，L1 均小于 `0.02`，order reachability
均为 `1`。

E2 OOD Linear 为 `0.649708907181630`，父为 `0.648734547234220`，delta
`+0.000974359947410`；相对 in-dist 的 gap 变化约 `+0.0000216`，通过 OOD 门。
fresh default 为 Linear `0.641470702272374`、Attention `0.752173407020070`、
Overall `0.687596829250581`，相对 v189 Overall `0.686889608842467` 为
`+0.000707220408114`。时间分解模型预测 `281.514913071177s`，未通过 `<280s`。

## 5. 决策

该固定统计带来正向本地 Linear 变化，但 fresh default 时间超门，不分配新正式版本、
不提交官方、不扫描排序/采样/混合邻域。候选及 fresh default 证据归档于
`solutions/20260906_linear-calibration-energy-actorder_time-rejected/`。
