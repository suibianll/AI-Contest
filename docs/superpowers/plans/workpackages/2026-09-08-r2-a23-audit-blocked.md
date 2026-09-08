# R2：A23 Q/K 互逆 scale 机制零 API 审计（BLOCKED）

依据 [单一完整方案优化计划 §7 R2](2026-09-08-single-solution-optimization-plan.md)。
零模型 API，纯代码审计。

## 审计对象与结论

**结论：A23 机制 NOT_TRANSPLANTABLE / NOT_APPLICABLE 于当前根（12352EFD）。**

## A23 机制（官方 14437/276s，父 A22-2 14424/271s）

A23 的机制（[A23 mechanism](../archive/solutions%2Fcontinuous_attention_anchor23-a1%2Fmechanism.md) 语义）：
**只改变 A22-2 的训练目标**——从"Q 与 K 各自相对父 amax 的平方和相加"
改为"联合乘积 `a_Q(f,g,b)·a_K(f,g,b)/a_Q0·a_K0`"，正则 `mean(S²)`。参数化是
A22-2 的 `exp(±S)` 互逆残余（K-center 同步 `c·exp(−S)`），gate 是完整父 C' 的
真实 readout MSE 严格下降。

## 根（12352EFD）Attention 实际结构（A2 系，非 A22/A23 系）

| 维度 | 根 | A22-2/A23 | 结论 |
|---|---|---|---|
| 训练目标 | `_a2_train_rotation`（L11373）最终输出 MSE `residual.square().mean()/mse_std`（L11444） | amax² 可加（A22-2）/ 乘积（A23） | **目标族不同** |
| 参数化 | A2 Cayley 旋转 `theta`（`_m_cayley_pair`）+ K-center | `exp(±S)` 互逆残余 | **参数化不同** |
| amax 目标入口 | 根无（`amax` 仅在 Linear 权重编码/helper；`block_amax` 在 L7227/7432 = Linear block encode） | A23 是 amax 目标族修改 | **无同构目标入口** |
| `torch.exp` | 仅在 Linear SmoothQuant（L4387/4400） | Attention 无 exp(±S) | **无互逆残余机制** |
| 训练入口 | `_a2_train_rotation`（唯一，调用于 L11584 `windows[:-1]`） | A22/A23 残余训练 | A23 无对应入口 |

## 判定依据（计划 R2 原文）

> "只有差异可分离时，才在当前完整根上移植同一机制；不得同时换成最终残差 CG、码级代理或新的训练主干。"

- A23 的"scale 乘积目标"是 **amax 目标族**内的修改（A22-2 可加 → A23 乘积）；
  当前根的 Attention 目标是 **最终输出 MSE**（A2 系），两者目标族完全不同。
- 要在根上"移植 A23 机制"必须把根的目标从最终输出 MSE 换成 amax 乘积——
  这正是计划明令禁止的"换成新的训练主干"。
- 根的 Attention（A2：`_a2_train_rotation` final-output + Cayley + center）是
  与 A22-2/A23 并行的另一条官方验证线（A2=14440/274s 为最高 Attention 分）。

## 处置

- **R2 BLOCKED / NOT_APPLICABLE**：A23 机制在当前完整根上无可分离的同构入口。
- 不注册 A23-root-port；不把 A2 最终输出目标改成 amax 目标（计划禁止换训练主干）。
- A23 官方 +13 保持为 A22-2 系内的证据，不移植到根。
- 下一轮只有 R1（L-C3 objective-only，已 READY_FOR_OFFICIAL）这一已知方向可推进；
  其余不注册邻域或替代机制（计划 §7 末尾）。

## 证据

- 根（12352EFD）Attention 训练：`_a2_train_rotation`（L11373-11496，目标 L11444）。
- A23：`solutions/continuous_attention_anchor23-a1/`（父 A22-2，官方 14437/276s）。
- 零 API：均为读代码 / 函数位置定位，无模型前向。