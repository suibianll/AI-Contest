# S1 结构解剖报告（fc/proj 权重 + Attention Q/K）

> 计划 [`2026-09-05-targeted-fcproj-qk-mechanism-research-plan`](../../docs/superpowers/plans/2026-09-05-targeted-fcproj-qk-mechanism-research-plan.md)
> S1 阶段。产物：`artifacts/proxy_v3/targeted-autopsy-20260905/run-001/autopsy.{json,md}`。
> 纯离线只读（复用 v186 校准缓存 + 已验坐标镜像），无版本号、无官方提交、无计时。

## 对象与量

对 P3 官方证明收益所在对象：fc_gate/fc_up/proj（72 个 state，24 层）与 Attention Q/K
（96 个 dynamic case，24 层 × 2 windows），重建编码前连续张量（W_t / Q_t / K_t），与已量化
params 展开逐元素 `r=|x|/(scale_factor·lv2·lv3)`（mantissa 网格坐标，0.25 一格，
合法上限 1.75；层级 lv2/lv3∈{1,2}）。

## 结果（mean/median/max）

| 统计 | fc/proj 权重 | Q/K |
|---|---|---|
| clip_frac（r≥1.75，饱和） | 4.6% / 4.7% / 4.8% | 4.3% / 4.6% / 7.4% |
| high_frac（r≥0.75） | 48.8% / 49.1% / 50.0% | 46.7% / 48.7% / 52.6% |
| low_frac（0<r<0.5） | 36.2% / 35.9% / 44.9% | 36.6% / 36.2% / 57.6% |
| zero_frac | 0.0% | 2.1%（个别 case 到 46%，为全零注意力输入窗口） |
| r_p50 | 0.73 | 0.69 |
| r_p90 | 1.57 | 1.54 |
| r_p99 | 1.91 | 1.90 |
| r_max | 2.4–4.4 | 2.2–3.0 |

最差 clip 层：权重 L9/L12/L18/L20 fc_gate（4.75–4.78%）；Q/K L1 k（7.4%）、L7 k/q、L15 k。

## 判定

1. **round 受限主导，clip 受限很小**：饱和元素仅约 4.6%（这些是层级求解器判定"不值得
   为该 8/4 通道组加倍"的组内大值；即使坐标无 clip，也要先解决 round 主导部分）。
   R1 实验已证明去掉 round 网格可回收 Q/K 单侧误差的 ~82% —— 与该分布一致：大部分误差
   来自 3-bit 网格，而不是饱和。
2. **r 分布已相当健康**：p50≈0.7、p90≈1.6、p99≈1.9 —— 层级与坐标已把大部分元素推进
   网格上半区；无"大段低分辨率浪费"迹象。zero≈0（权重）/2%（Q/K 偶发全零窗口）说明
   scale 空间利用充分。
3. 结论：在这些官方收益对象上，剩余误差是 **3-bit mantissa 网格的固有量化噪声（主导）+
   约 4.6% 组内离群饱和（次要）**。前者合法不可调（格式固定）；后者只能由"降低块内离群
   幅度"的坐标手段缓解——该类手段（rotation/block smooth/pair/permutation）已在 v186 连续
   域饱和（P1：连续偏差≈0，即已把可用的保语义坐标手段用尽）。
4. **未发现任何新的合法编译目标** → 按计划 §2 结束分支，S2/S3 不触发。fc/proj 与 Q/K
   作为官方收益对象，其剩余空间被格式能力硬限；对榜首 4166 分差距的合理解释维持为
   "机制代际差距（榜首可能采用更高表达力的非等效合法表示）"，且无法在当前 v186 合法字段
   内仅凭参数/规则关闭。

## 记录与去向

- 计划文件：`docs/superpowers/plans/2026-09-05-targeted-fcproj-qk-mechanism-research-plan.md`
  → 标记 COMPLETED-CLOSED 并移入 archive。
- 根 `solution.py` 保持 v186；无新的版本号、无父变更、无官方提交。
