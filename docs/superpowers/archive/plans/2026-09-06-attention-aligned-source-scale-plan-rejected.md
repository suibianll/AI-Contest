# Attention 变换坐标对齐 source-scale 优化计划

> 创建：2026-09-06  
> 状态：**CLOSED / NOOP_REJECTED**  
> 父版本：根 `solution.py` v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）  
> 研究父：v189（本地 Overall `0.686889608842`，官方 `unregistered/NA`）

## 1. 机制

在 raw source-scale proposal 已确认 no-op 后，本计划只把每个 NVFP4 source 16-value
group 的 scale 按 Q/K 最终 permutation 搬运，并乘以固定 multiplier 的 RMS，使 proposal
在变换后坐标中对齐；其余 source-scale 统计、E6M2 编码和 canonical hierarchy solver
全部保持不变。它不改变合法值域、在线候选数量、V 或 Linear。

## 2. 执行结果

- A0：单文件导入、`py_compile`、形状映射和 `multiplier=1` 回归通过；V/Linear 未改。
- A1：`eval-v3` Attention 六 shard 按计划运行，因 shard 0/1 的 16 个配对 case 全部
  逐位一致而提前停止。候选 SHA256 为
  `fba3f7dee40a5cb2ddd7d8c6a5ef13ef585fe9bbd16f570f9bad61db439167a4`。
  配对 delta mean `0`、median `0`、L1 `0`、正/负/零 `0/0/16`；直接校准检查也显示
  proposal 可生成但没有任何 Q/K 五字段变化。
- API 诊断总计约 `21.984s`（仅两个 shard，非官方时间）；未进入 default/OOD/官方。

## 3. 裁决

坐标对齐修复仍未让 proposal 进入实际 winner，故整个 source-scale proposal 族关闭，
不继续扫描 RMS/mean/max、统计量、阈值、offset 或 role/layer 邻域。根 `solution.py`
保持 v186；v189 不变并继续作为本地比较控制。

## 4. 证据

- 候选：`workbench/attention_aligned_source_scale_solution.py`
- A1：`artifacts/proxy_v3/attention-aligned-source-scale-20260906/full-a1/`
- 执行记录：`logs/execution/2026-09-06-attention-aligned-source-scale-plan.md`

