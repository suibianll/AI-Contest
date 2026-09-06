# 64-block 层级分区与激活误差解剖计划

> 日期：2026-09-06。状态：**CLOSED / D_B_REJECTED**。
> 这是上一份合法编码/输出目标计划关闭后的独立研究计划；根 `solution.py` 保持 v186，
> 不继承上一计划的 R2-L 失败候选，也不重开已关闭的 scale、mantissa、rank、动态 per-call
> 或 Jacobian importance 邻域。

## 1. 目标与证据边界

上一轮 R1 exact 合法块解在 v186 固定 refine 配置下没有 material operand gap，R2-L 的
固定 8 码联合输出规则也未通过 LOO/holdout。当前仍有两个在既有证据中未直接测量的开放项：

1. 激活侧 X 的非网格误差是否主要由 clip/长尾造成；
2. 在合法 `lv2/lv3 ∈ {1,2}` 不变时，只改变 64-block 内元素到 8×8×4 子组的归属，
   是否能降低实际输出误差。

本计划先做不改正式代码的 D-A/D-B 离线诊断。oracle 结果不是候选、不是官方分数预测，
也不能直接证明可部署规则。

## 2. 固定输入与抽样

- 父源码：根 `solution.py` v186，SHA 由每次 manifest 记录；输入固定为
  `artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`。
- 层固定 `[0,8,15,23]`；Linear 记录全部七 role，重点汇总 `fc_gate/fc_up/proj`；
  Attention 记录 Q/K，使用每个 calibration window、head 0 及最后 KV group。
- 每个 Linear 矩阵均匀取 32 个输出行和 4 个 64-channel block；激活取每个 calibration
  window 的均匀 32 token。所有索引按 `floor(k*(n-1)/(m-1))` 去重排序。
- 变换、父动态输出、合法字段和解码全部复用已核验 v186 calibration artifact 与
  `evaluator/reference_hif4.py`；不重新捕获模型，不读取 holdout 选择参数。

## 3. D-A：激活侧误差解剖

对每个固定 layer/role/calibration window，记录父动态量化后的最终坐标：

- effective scale、`abs(x)/effective_scale` 的 p50/p90/p99；
- ratio > 7 的 clip element/block fraction；
- 最近合法码重建误差中 clip 部分与非 clip 网格部分的 SSE 占比；
- 父动态解码误差、有限值、实际调用和输入身份。

判定：D-A 只用于区分 clip 主导与 round/grid 主导。若发现可由现有 API 可见的、通用且
不引入动态 per-call 搜索的压缩机制，另立机制卡；否则仅关闭激活侧该诊断，不改源码。

## 4. D-B：层级子组分区 oracle

在最终连续坐标中，对每个选定 64-block 比较：

1. 当前固定连续分区；
2. 一个固定的 shared-pressure block-local permutation：用 calibration 激活 RMS 与
   weight RMS 的归一化最大压力排序，使高压力元素集中到独立 4 元素子组。

两臂均用 `reference_hif4.py` 合法 exact block solver 重新编码；candidate 只改变元素归属，
不放宽五字段。Linear 同时报告 operand SSE 与选定输出行的
`X_hat W_hat^T` 对 `X W^T` MSE；Attention 记录 Q-only/K-only block replacement 的
完整 GQA output、logits 和 probability MSE。Permutation 只作共享坐标 oracle，不能把单侧
重排误写成可部署变换。

固定门：至少 3/4 层的重点 role 在 output oracle median gain > 1%，且至少一个深层和一个
非深层均有正向；否则标记 `NO_MATERIAL_GROUPING_ORACLE`，不实现候选。若通过，下一计划
才允许设计一个预注册的、非 output-aware 的 shared-pressure 规则，并重新做完整 eval-v3。

## 5. 产物与停止条件

实现 `workbench/hierarchy_partition_activation_probe.py` 及最小测试；产物根为
`artifacts/proxy_v3/hierarchy-partition-20260906/{d-a,d-b}/`，每次 run 保存
`result.json`、`report.md`、`run_manifest.json`。D-A/D-B 失败或数值异常记 `ERROR`，
oracle 无余量记 `REJECTED`；不分配正式版本，不修改根源码。

只有 D-B 通过固定门才另立 R3 计划。无论结果如何，保留原始产物并更新当前状态、计划索引
和执行日志；运行 `pytest`、`py_compile`、`git diff --check`。官方提交仍只在一个完整本地
候选超过当前最高且满足时间门后进行。

## 6. 执行裁决（2026-09-06）

- D-A 完成 140 条记录（4 层 × 7 role × 5 window）。clip 元素占比 `0.045359`，但最近
  合法网格 SSE 中占比 `0.088882`；clip 是次要误差源，未形成独立可部署机制。
- D-B 完成 28 个 Linear state 与 20 个 Attention window。重点层 output oracle 中位 gain
  依次为 L0 `-0.017062`、L8 `-0.075558`、L15 `-0.014854`、L23 `-0.042745`，
  固定门通过层数 `0/4`；Attention Q/K/joint 中位 gain 为
  `+0.000005/-0.001716/-0.001534`。裁决 **NO_MATERIAL_GROUPING_ORACLE**，不进入 R3，
  不修改根源码。
- 产物：[`D-A result`](../../../artifacts/proxy_v3/hierarchy-partition-20260906/d-a/result.json)、
  [`D-B result`](../../../artifacts/proxy_v3/hierarchy-partition-20260906/d-b/result.json)、
  [`执行记录`](../../../logs/execution/2026-09-06-hierarchy-partition-activation-plan.md)。

本计划已完成并移入 `docs/superpowers/archive/plans/`；下一方向必须新建独立活动计划。
