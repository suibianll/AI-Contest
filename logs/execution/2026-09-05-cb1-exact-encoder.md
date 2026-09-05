# 2026-09-05 cb1 混合 W pipeline 单元证明执行日志

> 计划：[`2026-09-05-nvfp4-codebook-exact-conversion-plan`](../docs/superpowers/plans/2026-09-05-nvfp4-codebook-exact-conversion-plan.md)
> 阶段：P1 单元证明 + G1 裁决
> 产物：脚本 `workbench/cb1_exact_encoder.py` + 报告
> `artifacts/proxy_v3/cb1-exact-encoder-20260905/run-001/cb1_report.json`
> 时长：14 秒 quick（2 层 × 2 role × 76 块 + 14 块）
> 输入：cached proxy-v2 pack（shard 0）

## 1. 实施步骤

1. 写 P1 脚本 `workbench/cb1_exact_encoder.py`：
   - Fraction 精确表（同 P0），K=20 候选（j ∈ [-2,2] × 4 m6 mantissas）
   - 对每个 (row, 64-block)：从标准 E6M2 sf 出发，对每个候选 sf 评估
     (exact_count desc, mse asc) lex 选 sf
   - 输出 mant/lv2/lv3/sign 的 HiF4 参数；MSE 与 `sol._dense_to_hif4` (refine=0)
     在同一 dense 输入下比较
2. 调试过程修复 7 处维度/broadcast bug：
   - `pack.layers` 是 int 而非 list
   - scale reshape (R, B, 4) 而非 (R, B, 4, 4)
   - `cnt_cpu[:, None, :, :].expand(R, K, 4, 7)` broadcast 对齐
   - `lv2/lv3` 维度错误（scalar (R,) → 应扩展到 (R, 4)）
   - mant 计算 newline 合并导致 NameError
   - `mse_rel` 公式中 `float()` 无法转换张量
   - `mse_hif4_decode` 中 sf 维度需 (R, 1, 1) 而非 (R, 1)
   - v186 输出字典键名（mant/sign/scale_*）与 hybrid（mantissa/sign/sf/lv2/lv3）不同
3. **关键设计 bug 修复**：最初用 `quant_block.abs()`（NVFP4 code 整数）作为
   HiF4 mant 输入，导致 mant 全为 1.75、误差 100× 于 v186。**改用 dense_block 作为
   HiF4 mant 输入**（mant = RTN(dense / denom)）。

## 2. 结果（quick 模式 L00 fc_gate/proj）

```
[w] L00 fc_gate  n= 14 hybrid=2.300e-04 v186=3.794e-06 ratio=60.619 ex=0.000 in 0.4s
[w] L00 proj     n= 76 hybrid=1.434e-04 v186=2.484e-06 ratio=57.744 ex=0.000 in 0.5s

G1 verdict: CLOSE_W->P4  (mean_ratio=59.182, better_or_equal=0/2)
```

## 3. G1 裁决：**CLOSE_W → P4**

**理由**：

1. **MSE 远差于 v186**：hybrid MSE 是 v186 baseline 的 58–60 倍（fc_gate 60.6、proj 57.7）。
   G1 阈值是 ≤ 0.90×，实测 59×，**远高于阈值**。
2. **精确占比 0**：hybrid 在 dense 输入下没有触发"精确落位"路径——精确 cell 数 = 0。
   这与 P0 的 0.789 完全相反。

## 4. 根因分析（机制证伪）

P0 PASS 的精确占比建立在"**NVFP4 code 已是严格 E2M1 × E4M3 scale 乘积**"前提上——
P0 测的是"给定 NVFP4 (quant, scale) 对，能否找到 sf 使 c × (m4/m6) × 2^j ∈ S"。
P1 hybrid 改用 dense 输入（NVFP4 反量化结果，含 BF16 snap 残差），mant = RTN(dense /
denom) 不再保证对齐 HiF4 格点——dense 不是严格的 E2M1 × E4M3 scale 结构，无法触发
精确路径。

**关键机制结论**：
- P0 的"精确占比 0.789"是 **NVFP4→HiF4 严格码点对齐**的可达上界，但**只在 NVFP4
  数据被忠实保留时成立**——NVFP4 反量化产生 BF16 snap 后，这个上界不再可达。
- v186 baseline（refine=0）已通过 `_dense_to_hif4` 联合搜索 sf 和 mant，能在 dense
  输入下找到 MSE 极小解；hybrid 只搜 sf 不搜 mant，永远输给联合搜索。
- **机制类别错位**：本计划核心机制是"per-(row, 64-block) sf 选择"，但 v186 已经把
  sf 选择 + mant RTN 一起做——hybrid 退化 = v186 refine=0（已实测 MSE 3.8e-6 vs
  hybrid 2.3e-4）。**新机制没有提供超越现有 `_dense_to_hif4` 的解空间**。

## 5. P0 数据 vs P1 实施的语义错位

- **P0 测的是**：给定真实 NVFP4 输入，最优 sf 选择能保留多少精确值；
- **P1 想做的是**：在 dense 输入上做精确编码，让 HiF4 落到 NVFP4 码点上。

两者不是同一个问题：dense 输入已经丢失了 NVFP4 码本结构（BF16 snap 后，dense 值
不再是 E2M1 × E4M3 的严格乘积）。

## 6. 风险登记更新

- **R5 符号门禁失手带**：未触发。MSE 远差于 v186，直接归因机制失败。
- **新增 R6 机制证伪**：P0 数据上界可达 ≠ 端到端机制有效；dense 输入下 NVFP4 码本
  对齐结构已丢失。

## 7. 关闭 W 侧 → 评估 P4

按计划 §3 护栏：**若 G1 失败，P4 评估仍然开启**（因为 P0-F2 attention 侧精确占比 ≥ 20%）。

P4 设计：Q/K 动态编码器按 64-块分支——对齐块走码本精确路径，其余块走现有
rotation/pair_transform 路径。但 P1 的语义错位问题同样适用：dense 输入不含 NVFP4
码点结构，attention X 侧 dense 输入也不会保留精确路径。

**真正的 P4 重启点**应是：**Q/K 是否本身有"输入端已经是严格 NVFP4"的情况**？X 路径
NVFP4 量化由 `_quantize_nvfp4_dynamic_per_call` 调用，理论上 quant 是 E2M1、scale
是 E4M3——但 X 输入本身不是 NVFP4（X 是 BF16 activation），所以 X 端走 NVFP4 量化
后输入到 attention API 时是 NVFP4 (quant, scale) 对，**此时 P0 同样的"精确可达上界"
成立**。

P4 启动条件应重新定义为：**Q/K NVFP4 (quant, scale) 对上，P0 精确占比 ≥ 20%**——这是
P0 quick 模式未抽样 attention X 的原因。P4 实施第一步：**复用 cb0 脚本扩抽样
attention X 的 NVFP4 数据**，复测 P0-F2 精确占比。

## 8. 下一步

- **P4 重启评估**：扩 cb0 脚本抽样 attention X 的 NVFP4 (quant, scale) 对；
  若精确占比 ≥ 20%，启动 P4 attention 运行时分支编码器（与 v168 类似但只对
  对齐块生效）
- **W 侧正式关闭**：本计划 W 侧机制（per-(row, 64-block) sf 选择）通过 P1
  实测证伪，**不作为可提交候选**。
- 计划状态更新：标记 P1 CLOSED-W → P4 待评估；不修改根 solution.py。
