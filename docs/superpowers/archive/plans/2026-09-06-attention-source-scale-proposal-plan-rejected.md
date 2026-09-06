# Attention source-scale proposal 优化计划

> 创建：2026-09-06  
> 状态：**CLOSED / NOOP_REJECTED**  
> 父版本：v186；研究比较副本：v189

## 1. 机制

计划把现有 Linear source-scale proposal 的固定 `median/q75/max` 源统计扩展到
Attention Q/K dynamic API，使用 canonical `_dense_to_hif4` 和原有合法层级 solver。
V、Linear、transform、state 和 online 候选数量不变；不使用自定义 decoder、非法字段、
`A@W` 反推或不受限搜索。

## 2. 结果

S0 通过。S1 使用固定 `proxy-v2` cache 和 CUDA eval-v3，在 shard 0、1 的 16 个
Attention case 上候选与 v186 baseline 逐位一致：`delta_mean=0`、`L1=0`、
positive/negative/zero=`0/0/16`。直接调用也确认 Q/K source proposal codes 能生成，
但最终五字段参数完全未改变，说明 raw source scale 在 Q/K 已变换坐标下没有进入更优
候选。按计划早停，未运行 OOD、default 或官方提交。

## 3. 产物

- 执行记录：[`2026-09-06 执行记录`](../../../logs/execution/2026-09-06-attention-source-scale-proposal-plan.md)
- 候选：`workbench/attention_source_scale_proposal_solution.py`
- S1 manifest：`artifacts/proxy_v3/attention-source-scale-20260906/full-s1/candidate/manifest.json`

根 `solution.py` 未修改，未分配 v190，官方状态不变。后续使用新的坐标对齐机制卡，
不重开本计划的 raw proposal 配置或参数邻域。
