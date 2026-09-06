# Attention 变换坐标对齐 source-scale 执行记录

日期：2026-09-06  
计划：[`2026-09-06-attention-aligned-source-scale-plan.md`](../../docs/superpowers/archive/plans/2026-09-06-attention-aligned-source-scale-plan-rejected.md)  
状态：`CLOSED / NOOP_REJECTED`

## A0

候选源码：`workbench/attention_aligned_source_scale_solution.py`  
SHA256：`fba3f7dee40a5cb2ddd7d8c6a5ef13ef585fe9bbd16f570f9bad61db439167a4`

- `py_compile` 与六 API 导入通过。
- source scale 的最终坐标 gather/RMS 形状检查通过。
- `multiplier=1` 时回到 raw proposal；V、Linear 和 fallback 路径未改。

## A1

命令使用 `evaluator/eval.py`、固定 `proxy-v2` cache、CUDA、Attention 六 shard、
baseline `solution.py`，输出目录为
`artifacts/proxy_v3/attention-aligned-source-scale-20260906/full-a1/`。

评测在 shard 0/1 后按 no-op 停止：16 个配对 case 的 candidate/base 输出逐位一致。
配对统计为 delta mean `0`、median `0`、L1 `0`、正/负/零 `0/0/16`；shard 0/1
candidate local attention mean `0.791814`（8 cases），仅是该顺序前缀，不是 default
panel。API 诊断总计约 `21.984s`，无官方时间等价性。

独立直接校准检查确认 proposal tensor 可生成，但所有 Q/K 的五字段参数差异均为零。
因此没有继续运行 default/OOD，也没有创建官方候选包。

## 裁决

坐标对齐没有使 source proposal 进入实际 winner。raw 与 aligned source-scale proposal
族关闭，不调整统计量、阈值、offset、RMS/mean/max 或 role/layer 邻域；根 `solution.py`
仍为 v186，v189 继续作为本地最高控制。

