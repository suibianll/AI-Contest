# A-MC1 K 侧 per-call 均值再定心执行记录

> 计划：[Attention K 侧 per-call 均值再定心计划（A-MC1）](../docs/superpowers/plans/parallel/2026-09-10-attention-k-mean-recenter-plan.md)。
> 根 R0：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256 `56DC805D...EFCB2BD`。
> 前卡 A-G1（v227）、A-QB1（v228）均 REJECTED，归因见各自执行日志；本卡按 A-MC1 计划设计
> （无校准拟合参数，per-call 均值来自当前调用，针对 v227/v228 的窗口统计漂移失败模式）。

## 1. 实现（2026-09-10）

- 候选：`workbench/full_solution/attention-amc1-k-mean-recenter/candidate/solution.py`（根逐位副本 + 纯新增段）。
- 机制：冻结根全部 state（含根 gate 接受的 rotation/center）；在根 rotation+center 之后、
  `_dense_to_hif4` 之前，对当前调用的 K 按 head 分组做 token 维均值再定心
  `K -= mean_tokens(K)`。量化前 softmax 精确不变（logits 只加逐 query 常数 `Q·mean(K)`，
  softmax 后消失），全部收益/损失只来自量化输入分布的移动。
- 无训练、无步数、无超参、无校准拟合参数；动态 API 只多一次 token 维 reduce + 减法
  （符合 v165 边界，与根 mode-2 per-call midrange 同属轻量 per-call reduce 模式）。
- 校准：对 6 个 full-attention 层，用全部 5 folds（窗口等权）完整部署路径真实 MSE 比较
  「父」与「父+残差均值再定心」两臂，严格改善才写 `k_state["k_mean_recenter"]=1`，否则保持父。

## 2. Control 结果（全部 PASS，GPU 空闲时运行）

脚本：`workbench/full_solution/attention-amc1-k-mean-recenter/control.py`，原始结果
`control_results.txt`。

1. **arm 关闭逐位恢复父**：Q/K/V 五字段与最终输出逐位一致。
2. **arm 开启 + 合成偏移输入**：K 五字段显著变化（scale_factor 172、lv2 237、lv3 933、
   sign 8198、mant 10711 处），dense softmax max|Δ|=1.86e-9（<1e-6，精确不变），Q/V 与
   Linear 逐位不变。
3. **六 API 独立导入**：脱离仓库 importlib 加载通过。
4. **合法 state**：`validate_state` 通过；`root_rotation_frozen=True`（根 rotation 逐位不动）。
5. **gate 双路验证**：接受与回退两条路径均验证。

## 3. 评测

- shard0：`artifacts/proxy_v3/attention-amc1-shard0-20260910/`，接口正常，
  `reasonableness_issues: 0`。
- 六 shard（72 case，`--stop-after-nonpositive 6` 跑满）：
  `artifacts/proxy_v3/attention-amc1-sixshard-full-20260910/candidate/`。
- shard↔层对应：shard0→层0、s1→层1、s2→层8、s3→层15、s4→层22、s5→层5。

| shard（层） | delta_mean | +/-/0 |
|---:|---:|---|
| 0（层0） | +0.000000 | 0/0/12 |
| 1（层1） | +0.020038 | 10/2/0 |
| 2（层8） | +0.000000 | 0/0/12 |
| 3（层15） | +0.079126 | 12/0/0 |
| 4（层22） | +0.000000 | 0/0/12 |
| 5（层5） | -0.009628 | 4/8/0 |

等权均值 `+0.014923`，合计 26/10/36；manifest candidate overall `+0.548920` vs baseline
`+0.533998`；API total（诊断，1 次校准缓存命中）29.725s。

## 4. 结果与裁决

- 裁决：**本地正向，归档 v229，待用户统一官方评测**，官方状态 `unregistered/NA`。
- 归档：`solutions/20260910_v229_attention-amc1-k-mean-recenter_officialNA_timeNA/`
  （本地正向候选，目录名不带 `rejected`），候选 SHA256
  `d1c23fa11198e56f15ac8f64e033c00333dcd2d5660cec773598624c4b247f4d`。
- 解读：本计划线首个本地正向的 Attention 候选。3/6 层 gate 接受（层1/5/15），
  3 层回退（层0/8/22，对应 shard 逐位不变）。层15 全部 12 case 均匀改善约 +0.079，
  与"per-call 均值纠正测试窗口相对校准窗口统计漂移"的机制假设一致；机制无校准拟合参数，
  不受 v227/v228 的窗口过拟合失败模式影响。**层5 如实记录：gate 接受（全 folds 真实 MSE
  严格改善）但 eval 窗口净负 `-0.009628`（4/8/0），属 gate/evals 口径差异。**
- 根不变：v202 Linear + v195 Attention，官方 `18053/281s`。

## 附：标准 Linear 侧隔离组合（2026-09-10，用户指示）

- 组合：`solutions/20260910_standard-linear_v229-attn_scoreNA_timeNA/solution.py`
  （v229 attention 源码 + v162 标准 Linear 尾块，构建器
  `workbench/standard_linear_attention_probes/build.py`，SHA256
  `ffe736a3face76defe06fc4b66c63b09364359ad524c27bd7a7cede4e7eea931`）。
- 静态核验：verify.py PASS（组成 + 六 API）。
- 运行时核验：`artifacts/proxy_v3/standard-linear-v229-attn-equiv-20260910/`，
  组合 vs v229 归档做 attention-only 六 shard 配对，72 case 全部逐位一致
  （delta 全零，overall `+0.548920` 与 v229 精确一致）——组合未改变 attention 行为。
- 用途：完整包 v229 官方 TIMEOUT；侧隔离形态剥离 Linear 校准（侧基线 238s 口径），
  官方侧分对照 `standard-linear_v195-attn` 的 `14426/243s`，正向差值即 A-MC1 官方侧价值，
  同时判读 TIMEOUT 是否由完整包 Linear 校准占用导致。官方评测待用户统一进行。
