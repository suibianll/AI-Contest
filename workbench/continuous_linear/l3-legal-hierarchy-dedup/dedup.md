# L3 去重记录：合法共享层级输出选择 vs 修正版合法离散网格（DUPLICATE_CLOSED）

> 日期：2026-09-07。侧：Linear。run_id：`l3-legal-hierarchy-dedup`。
> 结论：**DUPLICATE_CLOSED** —— 工作包 L3 目标（依据输出损失选择合法共享
> 层级）已被修正版合法离散网格计划（R1_NO_SUPPORTED_MECHANISM）覆盖。

## 对比

| 维度 | 修正版合法离散网格（2026-09-06 已关闭） | L3 卡（工作包描述） |
|---|---|---|
| 目标 | 合法 scale×lv2×lv3×mantissa 联合输出 oracle（真实 A@W 输出误差） | 合法 scale/lv2/lv3/mantissa/sign 联合决策，真实 A@W 误差 |
| 对象 | 28 state × 112 独立 holdout case（64-block 内） | 一个完整 64 块先证明非 no-op 且 A@W 误差可下降 |
| 求解 | 固定模式数 14,622,720，参考 HiF4 五字段语义 | 合法联合决策，固定候选总数 |
| 结果 | LOO mean -0.00314；holdout mean -0.00032、median -0.000054、正 case 10/112 | — |

同一合法候选集合、同一输出目标；旧计划已用真实 NVFP4 输入得出 holdout
全负（无材料余量）。按工作包 L3："若现有求解器已覆盖同一合法候选集合与
同一目标，直接去重，不重开阈值/窗口/候选数量扫描。"

**不创建候选，不重跑；父 L4 不变。**

## 证据

- 计划结论：`docs/superpowers/archive/plans/2026-09-06-corrected-legal-lattice-output-plan-rejected.md`
- R1 数据：`artifacts/proxy_v3/corrected-legal-lattice-20260906/r1-fixed-panel/result.json`
- 执行记录：`logs/execution/2026-09-06-corrected-legal-lattice-output-plan.md`