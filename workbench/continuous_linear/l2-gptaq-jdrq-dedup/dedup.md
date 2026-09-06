# L2 去重记录：GPTAQ 非对称输出残差补偿 vs JDRQ（DUPLICATE_CLOSED）

> 日期：2026-09-07。侧：Linear。run_id：`l2-gptaq-jdrq-dedup`。
> 结论：**DUPLICATE_CLOSED** —— GPTAQ 卡与已关闭 JDRQ（J1_REJECTED）同目标、
> 同更新变量、同离散规则、同 Xh/state 依赖；按工作包 L2 准入条件跳过，不重跑。

## 逐项对比

| 维度 | JDRQ（v189 上 J1_REJECTED） | L2 GPTAQ 卡（工作包描述） | 是否同一 |
|---|---|---|---|
| 目标 | `min ||Y − Xh·Wh^T||²`，teacher = 真实部署变换输出 `X_t W_t^T`（`_jdrq_calibration_products`）；残差 `residual = y − z·w^T` | `min||Xh Wh^T−Y||²`，含 H=Xh^T Xh、B=Xh^T Y | **同** |
| 更新变量 | Weight 离散码（mantissa ±0.25/0，`_jdrq_refine_mantissa_coordinates`；row-wise hierarchy `_jdrq_refine_rowwise_hierarchy`；offset `_jdrq_refine_hierarchy_offsets`） | 固定的逐列残差补偿规则 | **同（码级残差补偿）** |
| 离散更新规则 | exact one-coordinate signed-mantissa descent on product residual：block 按 `Z^T residual` 梯度平方 top-k 选，per-row 三分（−0.25/+0.25/0）能量 argmin，多 pass | 论文残差项而非仅 Hessian 扰动 | **同（残差引导、非仅 Hessian）** |
| Xh/state 依赖 | 冻结一次真实 dynamic activation codec（`_nvfp4_to_hif4`），保持 state 不重建 | 冻结一次真实 dynamic activation state，量化后保持 | **同** |
| 证据 | 112 配对 case mean Δ −0.000226、worst −0.00448；主要负向 proj/o/fc_gate | — | JDRQ 已关闭 |

## 结论

L2 描述与 `_jdrq_select_weight_candidate`（C72+C73，输出感知 A@W 产品选择）
在目标、变量、残差引导离散更新和冻结激活依赖上一致；"论文残差项 vs 仅
Hessian 扰动"正是 JDRQ 已实现的 residual-guided 形式。按工作包 L2 准入：
"若只是同目标同更新的换名/阻尼/系数变化，标 DUPLICATE_CLOSED，跳过"。

**不创建候选，不分配版本，不回填 v190；父 L4 不变。**

## 证据

- JDRQ 实现：L4 源码 `_jdrq_*` 段（`_jdrq_calibration_products` →
  `_jdrq_ridge_projection` → `_jdrq_make_target` → `_jdrq_refine_*` →
  `_jdrq_select_weight_candidate`）。
- JDRQ 官方/本地关闭：
  `logs/execution/2026-09-06-linear-fixed-state-output-aware-jdrq-plan.md`。
- GPTAQ 依据：arXiv 2504.02692（工作包 L2 引用，纸面去重未授权重跑）。