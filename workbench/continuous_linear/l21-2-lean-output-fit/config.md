# L21-2 设计登记：低成本输出拟合主干（条件卡）

> run_id：`l21-2-lean-output-fit`。侧：Linear。日期：2026-09-07。
> 契约：[linear-output-followthrough.md](../../docs/superpowers/archive/plans/linear-output-followthrough.md) §5 L21-2。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。触发条件：L21-1 逐列条件求解
> 本地 3/3 holdout 退化（校准 fold 过拟合），且逐列 OBQ 循环成本高。

## 1. 目标

在 L4 真实 activation_state 与 GPTQ 权重编码基础上，构建"精简主干 + 一次输出
求解"，固定两个逻辑对照并**分开报告**（不混 mean）：
- **精简父**：相对 L4 删除组件后的主干（测量删除损失）。
- **精简父 + 拟合**：在精简父上做一次输出求解（测量新增拟合收益）。

禁止：不用简化 codec 冒充 L4；不训练激活变换（保留父 smooth/perm/hadamard/
rank/importance/顺序全部语义）；不恢复 32 次硬前向训练结构。

## 2. 组件清单（登记删除/保留）

| 组件 | L4 状态 | L21-2 精简父 | 说明 |
|---|---|---|---|
| smooth scale（best_d） | 保留 | **保留** | 父部署坐标；删除会大幅降低 |
| permutation | 保留 | **保留** | 父部署坐标 |
| block-hadamard（CAT64） | 保留 | **保留** | 父部署坐标 |
| rank-1/rank-2 残差（residual_u/v） | 保留 | **删除** | 逐 state 折叠拟合，是 L21-1 过拟合高风险组件；删除其部署坐标修正，观察删除损失 |
| weight _gptq_quantize_weight | 保留 | **保留** | 父权重编码（一次编译） |
| static actorder hdiag | 保留 | **保留** | 父权重编码链 |
| activation 动态编译（Xh） | 保留 | **保留一次** | 只编一次 |
| 逐列 OBQ 条件求解循环 | L21-1 新增 | **删除** | 过拟合且成本高；改为"块级一次求解" |
| 块级一次输出求解（LS + 格点化） | 无 | **新增** | 无逐列 F 更新循环；对每 64 块解 LS、格点化、块两臂接受 |

## 3. 求解设计（块级一次求解，区别于 L21-1）

对每个 64 列块 B（部署列顺序，一遍）：
1. `H_B = Σω_f Xh_B,f^T Xh_B,f`；`D_B = Σω_f Xh_B,f^T (R_f + Xh_B,f W_B)`；
   ridge 标量与父同（记录 0.2，diag 加）。
2. `Z = solve(Hλ, Dλ)`（连续 LS，含 parent 锚 `λ·W0`）——**不进入逐列
   F 更新循环**。
3. 一次性格点化：对 Z 每个元素取合法最近格点 `scale·code`（译回五字段
   sign/mant，scale/lv2/lv3 固定父），clamp 合法范围。
4. 块两臂：真实校准目标 L(W) 严格改善才替换；更新残差 R。
5. 奇异/非有限块保留父（计数报告）。

## 4. 对照与验收

| 对照 | 期望读数 |
|---|---|
| 精简父 vs L4 | 删除损失（应为负或小幅正；负向可接受，仅诊断） |
| 精简父+拟合 vs 精简父 | 新增拟合收益（必须正向才继续） |
| 精简父+拟合 vs L4 | 综合；若删除 loss 淹没拟合 gain → 与 A21-1 教训一致，不可混 mean |

representative states：L0-o / L11-proj / L0-fc_up（真实 API 闭环）。局部诊断，
不提交官方。若精简父+拟合相对 L4 达正且 no-op 排除 → 进入扩展验证；
否则记录并换机制。

## 5. 成本说明

- 每 state 一次校准（含父权重 GPTQ 一次编译）+ 每块一次 64 阶 LS；
  与 L21-1 逐列循环相比显著更轻；只报告本机耗时，不换算官方。