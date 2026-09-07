# L 侧机制队列（continuous-linear）

> 更新：2026-09-07。契约：[持续优化总计划](../docs/superpowers/plans/2026-09-07-continuous-linear-attention-plan.md)
> + [证据修复工作包](workpackages/evidence-repair-next-cycle.md)。

## 已裁决

| 状态 | 卡 | 依据 | 结果 |
|---|---|---|---|
| CLOSED / REJECTED | fc/proj 解析旋转（正交特征基 T） | 5/5 真实 fc/proj state 全部退化（+3%~+17%） | [fcproj-analytic-t](fcproj-analytic-t/report.md) |
| CLOSED / FLAT-DEGRADE | fc/proj 解析旋转（Kronecker T1⊗T2） | 4/4：2 退化、2 持平，无材料改善 | 同上 |
| CLOSED / COST_HOLD | L1 FlatQuant 训练式 | 正确性 PASS；完整硬前向不可承受；方向未判定 | [l1 终裁](l1-flatquant-8x8/mechanism.md) |
| CLOSED / DUPLICATE_CLOSED | L2 GPTAQ | JDRQ 同目标同更新（公式级 5e-7） | [l2 去重](l2-gptaq-jdrq-dedup/dedup.md) |
| CLOSED / DUPLICATE_CLOSED | L3 合法层级 | 修正版合法网格同目标全负 | [l3 去重](l3-legal-hierarchy-dedup/dedup.md) |

## fc/proj 解析求解已测空间（2026-09-07）

1. **正交特征基 T**（eigh 块内 Gram 平均）：5/5 退化。
2. **Kronecker 解析 T**（min||T1⊗T2−G||_F² SVD）：2 退化 + 2 持平。
3. P1（2026-09-05）：**v186 连续变换族已饱和**（X_tW_t==XW^T 全 336 case
   机器精度内）；官方 4166 分差只能靠减少量化扰动，不是找连续等价变换。
4. P2：mantissa 0.25 网格是唯一有系统余量的字段约束（Linear W −77%，
   100% 同号），但无合法实现路径 → NO_SUPPORTED_MECHANISM。
5. E2E_REFINE（E6M2 scale offset 精化，默认 False）属 AGENTS §7
   "alpha/offset/sweep/block 数…局部扫描"关闭边界，不启动。
6. `_WEIGHT_PRODUCT_SELECTOR` / `_jdrq_select_weight_candidate` 在 L4 主路径
   **未被调用**（v189 有意未接入的死代码）——不是遗漏候选。

## 结论与 next_action

fc/proj 桶是官方唯一增益桶（P3），但该桶上可识别的解析机制均已测负或属
关闭边界：旋转解析（正交/Kronecker）9 个 state 无改善；P2 字段解析无合法
路径；训练式 COST_HOLD。**无新候选超过本地最高 0.6368，不触发提交**。

下一步选项：(a) 协调者推进 L4+R2c 组合（C-R1，加性 17993.8）；(b) 用户
提供新机制方向或重开某关闭边界（如 P2 mantissa 的合法实现、E2E refine 的
预算内形式）；(c) 背景论文检索新自由度（本侧不空转等待时做下去重）。