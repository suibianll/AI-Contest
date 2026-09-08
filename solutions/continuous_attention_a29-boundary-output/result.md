# continuous_attention A29：Boundary-Output（骨架占位）

> 日期：2026-09-08。父：`continuous_attention_ac0-correctness-hardened`
> （SHA256 `F817E4C24CAAA8D1325A5A0045F8A0F057EE0DC5EB4C0B7167298FE67BB4F5A2`）。
> 本目录 = AC0 `solution.py` 的逐位复制（当前 SHA 与 AC0 相同）。**机制尚未实现**，
> 以下为已冻结的机制约束（主指令 §18-20），实现前须按仓库纪律补机制卡并验证。

## 唯一新增机制（预注册）

final-output residual driven Q/K code-boundary compensation：

```
S* → α_boundary → Q_H(Qe^{αS*}) → K_H(Ke^{-αS*}) → Attn
```

- 方向来源：final Attention output residual → softmax Jacobian → 解析 Q/K transform
  方向（禁止连续 STE 梯度直接训练 reciprocal transform——连续域梯度为零）。
- 每层 **1 个非 identity proposal**；identity parent 永远作为 fallback。
- 禁止：多步 signGD、大量 step sweep、连续 loss 迭代、surrogate mantissa-only loss、
  简化 lv2/lv3 codec、多 rank/seed sweep。
- 硬门：`L_candidate = MSE(Attn(ĥQ_c, ĥK_c, ĥV), Attn(Q,K,V)) < L_parent`，
  Q/K/V 全部来自真实动态 API（五字段部署路径），才部署。

## 禁止事项（沿用主指令 §22）

- 修改历史 `solutions/*/solution.py` 或根 `solution.py`。
- coupled Q/K transform 使用 `except Exception: pass`；单边 Q/K fallback。
- trainer 使用简化 encoder / 非部署 V；连续 reciprocal transform 的 STE loss 判收益。
- 在无 correctness parity 的情况下叠新算法（AC0 已通过 battery 30/30）。

## Status

- **本地**：骨架占位，未实现机制，未评测。
- **官方**：`unregistered / NA`。
- 下一步：按机制卡（softmax-Jacobian 方向 + normal-equation 解析解 + 1/64
  code-boundary 单 proposal + 真实 hard gate）实现并逐项验证；任何 AC0 已确认的
  correctness 不变量不得破坏。