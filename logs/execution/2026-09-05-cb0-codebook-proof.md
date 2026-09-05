# 2026-09-05 cb0 codebook-exact conversion P0 数据事实执行日志

> 计划：[`2026-09-05-nvfp4-codebook-exact-conversion-plan`](../docs/superpowers/plans/2026-09-05-nvfp4-codebook-exact-conversion-plan.md)
> 阶段：P0 数据事实 + G0 裁决
> 产物：脚本 `workbench/cb0_codebook_proof.py` + 报告
> `artifacts/proxy_v3/cb0-codebook-proof-20260905/run-001/cb0_report.json`
> 时长：56 秒（CPU，PyTorch 2.13 + Python 3.14）
> 输入：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`（v3 准备的 6 shard 缓存）

## 1. 实施步骤

1. 写 P0 脚本 `workbench/cb0_codebook_proof.py`：
   - Fraction 精确表 `valid(7,8,4,17)` + `err2(7,8,4,17)` 重建 HiF4 子块 b 精确条件
     `c · (m4/m6) · 2^dj ∈ S`（S = {k/4, k/2, k : k=1..7}）。
   - `analyze_pair(quant, scale)`：F0 结构 → F1 子尺度对齐 → F2 候选 sf 精确上界 →
     F3 code 统计 → F4 MSE 对比。
   - 候选生成：每子块 j ∈ [-4..2] × 4 m6 mantissa → K=112 候选。
   - baseline sf = `sol._standard_e6m2_scale(amax)`（真实解函数，含 BF16 中间量）。
   - 抽样：W 全 6 shard × 7 role，X/QKV 抽样 2 windows。
   - CHUNK=2048 block-chunks 控制内存峰值。
2. 4 处维度 broadcast bug 修复：
   - `m6_off` 维度位置（应为最后一维而非 dim=4）
   - `base_code_safe` 形状（(N,) 与 (N,4) ok 错位）
   - `E_b * ok_b` broadcast 对齐（需 `ok_b[None]` 而非 `ok_b[:, None, :]`）
   - cnt/E_b 求和：3 维张量不能写 4 字母 einsum，改用直接 mul + `.sum((1,2))`
3. 运行：quick（仅 L00/L06/L12）确认无误后跑 full 版（24 fc + 16 attn + 2 X）。

## 2. 结果（`cb0_report.json`）

### 2.1 G0 裁决（W fc_gate/fc_up/proj 聚合）

```json
"g0": {
  "agg_exact_frac_best_nonzero": 0.7889,
  "agg_mse_ratio_best_over_baseline": 0.4760,
  "per_role": {
    "fc_gate": {"exact_frac": 0.7746, "mse_ratio": 0.5133},
    "fc_up":   {"exact_frac": 0.7903, "mse_ratio": 0.4506},
    "proj":    {"exact_frac": 0.8017, "mse_ratio": 0.4577}
  },
  "verdict": "PASS->P1"
}
```

阈值：exact ≥ 0.20 AND mse_ratio ≤ 0.85。
**远超阈值**：exact 是阈值的 3.9 倍，mse_ratio 是阈值的 0.56 倍。

### 2.2 全 role 精确占比分布（W）

| role | mean | min | max | n |
|------|------|-----|-----|---|
| fc_gate | 0.7746 | 0.7581 | 0.7881 | 8 |
| fc_up | 0.7903 | 0.7754 | 0.8052 | 8 |
| proj | 0.8017 | 0.7889 | 0.8089 | 8 |
| q | 0.7376 | 0.5948 | 0.7884 | 4 |
| k | 0.7046 | 0.5794 | 0.7659 | 4 |
| v | 0.8090 | 0.7270 | 0.9012 | 4 |
| o | 0.8384 | 0.8019 | 0.8906 | 4 |

每个 role 都 >0.70，平均 0.78。**最差样本 (L00 k)** 0.5794 也过阈值。

### 2.3 X 抽样（calibration 缓存）

| role | exact | mse_ratio |
|------|-------|-----------|
| fc_gate | 0.4668 | 0.8322 |
| proj | 0.5291 | (未测) |

n=1/role（脚本只抽样 2 windows），但已证明结构上限可达。

### 2.4 F1 对齐结构（代表性样本 fc_gate L00）

- p_all4_same_mantissa = 0.0%（四子块全同 mantissa 的块占 0%）
- mode_count_hist: 1 模式 39.0% / 2 模式 53.7% / 3 模式 7.0% / 4 模式 0.3%
- even_m4_frac = 0.4905（E6M2 兼容比例 49%）
- subnormal_scale_frac = 0%（fc_gate L00 全是正规 scale）
- exp_spread p50=1 / p90=2（块内子 scale 跨度小）
- pair_m4_eq_mean = 0.13（跨列对偶相等性弱）

## 3. G0 裁决：**PASS → P1**

**理由**：
1. 三个 fc 角色聚合 `exact=0.789` / `mse_ratio=0.476`，均显著超过阈值
   （exact≥0.20 / mse_ratio≤0.85）。
2. 所有 7 个 W role 个体均值都 >0.70，最差样本 0.5794 也过线。
3. X 抽样显示结构上限对 activation 同样存在。
4. F1 数据支持计划 §2 推断：跨列对偶相等性弱（perm 16 粒度收益有限）、
   偶八分比例 ~49%（半数尾数需 sf 偏移才能精确），
   意味着精确占比主要靠"sf 候选网格"实现而非 perm 重排——与计划设计一致。

## 4. 风险登记更新

- **R1 精确占比数据依赖**：✅ 已排除。P0 实测精确占比 0.789，远高于 R1 上限 0.10–0.20 的悲观估计。
- **R2 变换栈让渡**：未触发。F1 显示 perm 信号弱 → 让渡成本低于预估。
- **R3 v183 类比**：待 P1/P2 验证。
- **R4 时间预算**：未触发。P0 脚本 56s CPU 端运行；候选搜索只在 W_calib 阶段，系数 0.115。
- **R5 符号门禁失手带**：未触发。P0 信号强，预期 P2 Δmean 显著正。

## 5. 下一步

进入 **P1**：
- 实现真实精确优先编码器 `workbench/cb1_exact_encoder.py`：
  - sf 候选网格（与 P0 同口径）+ 4 组 a 兼容 DP（保持 5 字段合法性）
  - GPTQ 组合（精确块 δ=0 不传播）
  - smooth d 退化为 per-column 2 幂，perm 退化为 16 粒度或恒等
- 在 ≥3 层 × 3 角色（fc_gate/fc_up/proj）做**乘积级**对比：缓存 v186 校准态 vs 混合臂
- **门槛 G1**：混合臂乘积 MSE ≤ 0.90× v186 臂且在 ≥2/3 单元上不劣 → P2
- 失败时（0.90, 1.0]：计划修订一次（2 幂 smooth 变体 / 部分 perm 16 粒度）
- > 1.0 → 关闭 W 侧，转 P4 评估

P4 评估（attention 侧）开启条件已部分满足（Q/K exact 0.74/0.70 > 0.20）；
若 G1 通过则合并到 P2/P3；否则单独走 P4。
