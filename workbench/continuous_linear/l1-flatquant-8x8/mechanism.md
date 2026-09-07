# L1 机制终裁：HiF4 层级结构化可逆变换（FlatQuant 8×8 T1⊗T2）

> run_id：`l1-flatquant-8x8`（侧：Linear）
> 日期：2026-09-07（证据修复后更新）
> 契约：[证据修复工作包](docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md)
> + [L 工作包](docs/superpowers/plans/workpackages/continuous-linear.md) L1。
> 直接父：L4 `ACB16F76...F5263`（官方 4607/247s）。

## 1. 最终裁决：COST_HOLD（方向未判定）

**状态：COST_HOLD，不注册候选，不关闭可逆变换数学方向。**

| 维度 | 证据 | 结论 |
|---|---|---|
| 正确性（矩阵指数梯度/STE/inference_mode/T=I 恒等） | `probe_l1_correctness_cost.py`：grad 1.2e-2~1.8e-2（double）、STE \|g−1\|=0、T=I 恒等 | **PASS** |
| 完整硬前向成本 | `probe_l1_correctness_cost.py`：0.62（o）/1.73（fc_up）/3.24（proj）s/step | **COST_HOLD**：32 步×168 state ≈ 3,300–17,000s 本机不可承受 |
| T=I 部署一致基线 | `probe_t_identity_parity.py`：简化前向与父 MSE ratio 0.45–0.80（缺 rank 配对 gram/actorder/importance） | **重建不可用** → 方向未判定 |
| 缓存/复用后继 | `report-l-r2-cost-followup.md`：唯一可预计算 X^T X（省 <5%）；weight/act GPTQ 依赖 T 无免费复用 | 无足够增益方案 |

## 2. 累计证据（可信口径）

- 旧 fast-path 34/35 退化：**无效**（round(x/scale) 自制输入 + 简化 codec）。
- L-R2 简化 STE 3/3 退化：**无效**（T=I 前向与父不一致，基线错位）。
- 真实输入 12/12 逐位一致（`probe_real_api_closure.py`）：基线路径本身可靠，
  但 T 方向判定仍缺"插 T 完整校准变体"。

## 3. 为何不改规则绕过 COST_HOLD

工作包 L1 明确：完整硬前向每步重建不可承受时记 COST/DESIGN_HOLD；禁止
以成本高改用简单 codec 后仍称同一候选；禁现场减步数或扫 rank。已评估
缓存方案不足以改变结论 → 保持 COST_HOLD，FlatQuant 作为"需要解析/非训练
求解的新机制"另立卡（未授权自动重跑）。

## 4. 下一步（按官方 P3 桶证据）

官方 P3（2026-09-05）确认：**Linear 官方增益 100% 落在 fc+proj 大形状桶**
（W2=fc_gate/fc_up 1818、W3=proj 1767）；q/k/v/o 官方零收益。本地 fc/proj
gain 0.50–0.56 也最低。→ 新机制只应在 fc/proj expansive/wide 形状上找增量。

## 5. 产物

- `workbench/continuous_linear/repair-r1/probe_l1_correctness_cost.py`
- `workbench/continuous_linear/repair-r1/probe_t_identity_parity.py`
- `workbench/continuous_linear/repair-r1/probe_l1_direction_full.py`
- `workbench/continuous_linear/repair-r1/report-l-r2*.md`
- `workbench/continuous_linear/repair-r1/report-t-identity-correction.md`
- `artifacts/proxy_v3/continuous/linear/repair-r1/l1-cost.json`