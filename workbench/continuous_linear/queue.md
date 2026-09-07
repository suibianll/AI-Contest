# L 侧机制队列（continuous-linear）

> 更新：2026-09-07（evidence-repair 后）。契约：[持续优化总计划](../docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
> + [证据修复工作包](workpackages/evidence-repair-next-cycle.md)。

## 已裁决

| 状态 | 卡 | 依据 | 结果 |
|---|---|---|---|
| CLOSED / COST_HOLD | L1 FlatQuant 8×8 T1⊗T2 训练式 | 正确性 PASS；完整硬前向 0.62-3.24s/step × 32 步不可承受；T=I 重建基线不可用（方向未判定）；缓存无足够增益 | [l1 终裁](l1-flatquant-8x8/mechanism.md) |
| CLOSED / DUPLICATE_CLOSED | L2 GPTAQ | JDRQ 同目标同更新（公式级 5e-7） | [l2 去重](l2-gptaq-jdrq-dedup/dedup.md) |
| CLOSED / DUPLICATE_CLOSED | L3 合法层级 | 修正版合法网格同目标全负 | [l3 去重](l3-legal-hierarchy-dedup/dedup.md) |
| CLOSED / REJECTED | 旧 L1 34/35 退化 | PROBE_INVALID_FOR_DEPLOYMENT（round 自制输入+简化 codec） | 撤回 |

## 关键证据（2026-09-07 修订）

- **真实 API 闭环 12/12 逐位一致**（`repair-r1/probe_real_api_closure.py`）；
  smooth/perm/hadamard 可逆（1e-13）；rank 非纯可逆（连续参照偏差 1e-4 级）。
- **四臂归因 UNIDENTIFIABLE**：真实部署值下 W-only/A-only 是混合坐标（E10/E01
  比 E00 大 3–300×），interaction 巨大；仅完整 E11 有效。旧"A-only 61-79%"作废。
- **官方 P3 桶证据**：Linear 官方增益 100% 落 fc+proj 大形状桶（fc 1818 +
  proj 1767）；q/k/v/o 零收益。本地 fc 0.528 / proj 0.564 / qkv 0.768 ——
  官方敏感桶即本地最差桶，最新候选应只瞄准 fc/proj expansive/wide 形状。

## 队列（新假设，NEEDS_NEW_HYPOTHESIS 已记录）

见 [`needs-new-hypothesis.md`](needs-new-hypothesis.md)。被阻断的具体选项：

1. **训练式可逆变换（FlatQuant）**：COST_HOLD（每步完整硬前向不可承受），
   除非有解析/非逐 state 训练的求解（需另立卡并预注册）。
2. **fc/proj 增量的机制**：块序族、JDRQ/GPTAQ、合法编码搜索、Householder、
   rank-3/系数/fold、CAT/BOAT/ROAB 已关闭；fc/proj 的官方增益在 v160/v189
   已通过 smooth+perm+CAT64+GPTQ+rank2+hdiag-actorder 栈实现。
3. **跨层/输出侧变换**：eval-v3 每层激活为真实前向捕获，中间误差不传播；
   输出行重排不改变逐行独立编码误差（no-op）。
4. **组合路线（不经新机制）**：L4+R2c 组合（协调者 C-R1）加性预测 17993.8，
   无需新 L 机制即可推进，但那是协调者职责。

## next_action

向协调者提交 L1 终裁（COST_HOLD）+ 证据修复完成（L-R1/R2/R3）；L 侧等待
新假设或重开边界；不空转、不重跑已关闭族。若用户重开任一关闭边界（如
fc/proj 块序邻域、FlatQuant 解析求解），从 L4 `ACB16F76...F5263` 构造候选。