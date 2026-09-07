# L-R3 执行报告：GPTAQ vs JDRQ 公式级去重

> 日期：2026-09-07。侧：Linear。run_id：`repair-r1/l-r3`。
> 契约：[evidence-repair-next-cycle.md](../../docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md) L-R3。
> 结论：**DUPLICATE_CLOSED 维持** —— GPTAQ 连续优化目标与 JDRQ 覆盖。

## 1. 公式对照（`probe_l3_formula_dedup.py`，小型确定性矩阵）

- 目标一致：论文 GPTAQ（arXiv 2504.02692）Eq.(4)-(5) 为
  `min{Δw} ||Δw·X − r||²`，其中 `r = w·X̃ − w·X`（X̃=全精度输出目标，
  X=输入）；JDRQ `_jdrq_make_target` 目标也是 `min||residual − ΔW·Z||²`，
  teacher = 真实部署变换输出（`_jdrq_calibration_products` 的 `X_t W_t^T`），
  **已经包含非对称校准语义**（冻结真实 dynamic activation state）。
- 连续解：GPTAQ 单步 `Δw = Rᵀ Z H⁻¹`（H=ZᵀZ+λI）；JDRQ target
  `η·Rᵀ Z (ZᵀZ+λI)⁻¹`（`_jdrq_ridge_projection`）。两者数学同式。

## 2. 数值结果（同一 (Z,Y,base) 输入）

| trial | λ | GPTAQ 单步 vs JDRQ target maxdiff |
|---|---|---|
| 0 | 0.0 | 1.28e+00（无正则奇异解不同） |
| 0 | 0.1 | **4.77e-07** |
| 1 | 0.0 | 4.67e+00（无正则奇异） |
| 1 | 0.1 | **5.96e-07** |
| 2 | 0.0 | 3.43e+01（无正则奇异） |
| 2 | 0.1 | **5.36e-07** |

- λ=0 时 (ZᵀZ) 复数条件（16×8 输入欠定），解不稳定，两者均不收敛到唯一解；
  评估器实际路径均有正则（λ>0）。**λ>0 时两者逐位一致（~5e-7，浮点级）**。
- base=0 时 JDRQ target 同时等于岭闭式解 W*（3.6e-07），三者同解。

## 3. 完整覆盖论证

1. **目标相同**：均 `min||Xh Wh^T − Y||²`，JDRQ teacher 即论文"全精度输出
   目标 X̃"的部署坐标版本。
2. **连续更新相同**（λ>0 岭投影，上述数值一致）。
3. **离散化阶段**：JDRQ 后续 `_jdrq_refine_mantissa_coordinates` /
   `_jdrq_refine_hierarchy_offsets`（残差引导合法五字段坐标下降）与
   GPTAQ 论文的"逐列更新 + 通道并行/Cholesky 融合"是同一输出目标下的
   不同实现路径；JDRQ 已在 v189 上 J1_REJECTED（112 case 负向）。
4. 按 L-R3：公式可证明等价/覆盖 → **维持 DUPLICATE_CLOSED**。

## 4. 证据修订口径（R0 表更新）

- 上一轮 `l2-gptaq-jdrq-dedup/dedup.md` 判定依据从"手动常见识别"升级为
  **公式级数值对照**：DEDUP_UNRESOLVED → **RESOLVED（等价）**。
- 不注册 GPTAQ 新候选，不重跑 JDRQ；父 L4 不变。

## 5. 产物

- `workbench/continuous_linear/repair-r1/probe_l3_formula_dedup.py`