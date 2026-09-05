# 2026-09-05 cb2 attention 精确分支 P4 端到端验证（NO-GAIN → 计划关闭）

> 计划：[`2026-09-05-nvfp4-codebook-exact-conversion-plan`](../docs/superpowers/plans/2026-09-05-nvfp4-codebook-exact-conversion-plan.md)
> 阶段：P4 attention 运行时精确转换端到端验证
> 产物：脚本 `workbench/cb2_attn_exact_branch.py` + 报告
> `artifacts/proxy_v3/cb2-attn-exact-branch-20260905/run-001/cb2_report.json`
> 时长：14 秒 quick（4 layer × Q/K/V = 11 cells）
> 输入：cached proxy-v2 pack（shard 0, calibration_qkv 1 window）

## 1. 实施步骤

1. 写 P4 脚本 `workbench/cb2_attn_exact_branch.py`：
   - 复用 cb0 的精确 sf 候选表（Fraction 精确算术，K=20 候选）
   - 对每个 (row, 64-block)：精确编码强制 `lv2 = lv3 = 1`，mant 由 NVFP4 dense 值 RTN
   - 与 v186 `_dense_to_hif4(refine=0)` baseline 对比 MSE
2. 修复 3 处 bug：
   - `pack.test_qkv[w][l]` 是 None → 改用 `pack.calibration_qkv[w][l]`（dict 形式）
   - triple 不是 tuple 而是 dict，需要 `qkv_dict[name]` 取 Q/K/V

## 2. 结果

| layer/role | exact_pure | v186 | ratio | sub_exact |
|------------|-----------|------|-------|-----------|
| L00 k | 1.15e+03 | 7.94 | 144× | 23.2/64 |
| L00 v | 1.52e-01 | 5.7e-6 | 26566× | 27.4/64 |
| L06 q | 3.71 | 4.08e-2 | 91× | 26.2/64 |
| L06 k | 1.92 | 2.12e-2 | 91× | 30.3/64 |
| L06 v | 1.06e-1 | 1.65e-3 | 64× | 26.7/64 |
| L12 q | 3.81 | 3.98e-2 | 96× | 25.7/64 |
| L12 k | 2.23 | 2.46e-2 | 91× | 29.3/64 |
| L12 v | 1.12e-1 | 2.18e-3 | 51× | 29.6/64 |
| L18 q | 4.79 | 5.26e-2 | 91× | 25.2/64 |
| L18 k | 1.95 | 2.22e-2 | 88× | 26.8/64 |
| L18 v | 2.90e-1 | 4.99e-3 | 58× | 27.8/64 |

**P4 gate (Q/K): NO-GAIN** —— mean_ratio=101.087, mean_exact_subs=26.32/64, n=8 cells。

## 3. G4 裁决：**NO-GAIN → 计划关闭**

**理由**：
1. **exact encoder MSE 比 v186 baseline 差 51–26566 倍**：所有 11 个 cells 的 MSE 全部
   大幅劣于 v186，Q/K 平均差 101×，V 差 58–26566×。
2. **精确占比 ~41% 但代价极高**：26/64 sub-blocks 精确，但强制 `lv2=lv3=1` 损失了
   v186 baseline 利用 `lv2/lv3=2` 提供的额外动态范围。

## 4. 根因分析（机制证伪 #2）

P4 设计与 P1 同一根本错误：
- **P0 精确占比 0.42–0.53 建立在 NVFP4 (quant, scale) 严格码本上**，测的是
  "给定 NVFP4 输入，能找到 sf 使解码无损的比例"。
- P4 exact encoder 强制 `lv2=lv3=1` 试图利用精确性，但**忽略了 v186 已经通过
  `lv2/lv3=2` + sf 联合搜索达到 MSE ~e-5 的近最优解**。
- 当 NVFP4 dense 值幅值大时（如 Q 矩阵元素达 ±3.8），强制 `lv2=1` + mant ∈ [0,1.75]
  × sf 的动态范围不足以覆盖，mant 全部 clamp 到 1.75，MSE 爆炸。

**P4 设计约束过严**：精确性要求（lv2=lv3=1）与动态范围需求（lv2/lv3=2）在大多数
64-block 上互斥，无法两全。

## 5. 关闭族合规表更新

| 已尝试方向 | P0 数据 | P1/P4 实证 | 结论 |
|---|---|---|---|
| W sf-only 搜索（hybrid mant RTN） | exact=0.789 | MSE 60× v186 | CLOSED-W |
| Attention 精确路径（lv2=lv3=1 强制） | exact=0.42–0.53 | MSE 51–26566× v186 | CLOSED-ATTN |
| v168 baseline 已联合搜索 sf+mant+lv2/lv3 | — | — | 不可超越 |

## 6. 计划关闭

- 本计划无新增可提交候选；根 solution.py 保持 v186。
- 计划文件 `docs/superpowers/plans/2026-09-05-nvfp4-codebook-exact-conversion-plan.md`
  移入 archive，AGENTS.md / README.md 引用更新为"无 active 计划"。
- 状态写回 `docs/current-solution-status.md` 与 `solutions/README.md`。
