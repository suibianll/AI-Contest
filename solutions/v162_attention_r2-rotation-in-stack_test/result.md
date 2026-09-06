# v162_attention R2：旋转部署进 v189 Attention 栈（OOD 门拒绝）

> 日期：2026-09-06。分支：v162 独立 Attention（A 代理）。
> 状态：**REJECTED / OOD_GATE**。按"无论成败都归档"纪律存档；不晋级、不替换 R1。

## 源码与身份

- **solution.py SHA256：`0B56CCA1F557E44CC6409822C64D29919D6955ABC164C59105138594022A9C7C`**
  （typing 导入修正后、全部评测前的定稿版）
- 直接父：R1（`v162_attention_r1-v189-attnstack-recovery_officialNA_timeNA`，
  SHA `3619BFEB...BFC8`）。
- 机制：在 R1 栈之上，按 A2 冻结配置训练每 KV group 旋转（标准编码器 surrogate），
  注入点在 `_nvfp4_to_hif4` 全部栈变换之后、HiF4 编码之前（连续 QK 与 R1 严格不变）；
  gate 在真实部署路径上逐层决策（identity 严格更优则回退），失败安全回退 R1 行为。
- 构建：`workbench/v162_attention/build_r2_candidate.py`（确定性补丁）。

## 评测结果（r2-screen / r2-id / r2-ood / r2-default）

- ID 48-case：mean **+0.773821** / median +0.754861，48 正/0 负，L1_neg 0；
  split test +0.772330 / validation +0.775313 均正；24/24 层为正（+0.59~+0.96）。
- fresh default：attention_mean **0.767021**（vs R1 0.752173，+0.0148）；linear 0.0。
- 时间：A_calib 89.873s（含 24 层训练+gate 评估）→ **228.936s < 280s** ✅。
- **OOD 门未过**：候选 gap = 0.773821 − 0.757615 = **+0.016207**；
  相对直接父 R1（同 SHA in-dist/OOD 配对，R1 gap +0.000915）的 **Δgap = +0.015292 > 0.01 → BLOCKED**。
  收益主要留在 WikiText 分布内，与 09-04 OOD 标定 gap 家族（Attention +0.015~0.022）特征一致。

## 裁决与教训

- **REJECTED / OOD_GATE**：不晋级分支最佳；不通过重训、缩容量、换折、扩校准集等邻域手段修复
  （失败换机制纪律）。
- 教训：标准编码器 surrogate 训练的旋转与 v189 栈组合后分布鲁棒性下降；后续新机制需在校准
  目标中直接体现部署路径，或使用分布更稳的参数化。
- 分支最佳保持 R1（default 0.752173 / shard48 +0.752772，全门通过）。


## 官方结果（2026-09-06 补记）

- **official：wrong answer（Attention 场景）**，同 SHA 以 `_test` 重提交仍 WA（确定性非法，非偶发）。
- 与 A2 共同点为旋转代码路径；按 v107 判例优先排查未设防异常出口。
