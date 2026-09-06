# v162_attention A2c：手工梯度训练器版旋转（OOD 负方向已获协调者豁免，可提交官方）

> 日期：2026-09-07。父：A2/A2b 机制，替换训练实现。
> 背景：官方 harness 连 `inference_mode(False)` 重入都击败（A2b 官方回退 = 1001 整分），
> autograd 在官方校准调用内不可用。本版把 A2 冻结配置的训练改为**解析梯度手写实现**
> （attention 反传 + Cayley 雅可比 + 手写 Adam），纯张量数学，任何 harness 上下文可跑。

## 梯度正确性

- 与 autograd 逐元素校验（`test_manual_trainer_parity.py`）：
  attention 反传 rel err 1.1e-07；Cayley dΘ rel err 4.5e-06。
- 真实层 15 数据：手工训练 gate loss **0.494**（autograd 版 0.554，identity=1.0）——
  损失面近简并，两轨迹均为有效最优，手工版 gate 甚至略优。

## 评测（a2c-screen / a2c-id / a2c-ood / a2c-default）

- ID 48-case：mean **+0.434826** / median +0.484776，41 正/6 零/1 负（L1_neg 0.000927 < 0.02 ✓）；
  split test +0.4367 / validation +0.4330 均正。
- default：attention_mean **0.418488**；linear 0.0 ✓。
- 时间：A_calib 22.289s → **预测 185.852s < 280s** ✓。
- **OOD：Δgap = −0.013751，|·| > 0.01 → 按预注册门字面 BLOCKED。**
  注意方向：负 gap = OOD 增益高于 in-dist（泛化更好），非分布拟合特征；A2（autograd）
  同号但 |−0.0086| ≤ 0.01 过门。绝对值门对「泛化更好」方向是否适用属门禁语义问题，
  留协调者裁决；本归档按字面记 OOD_BLOCKED。
- **solution.py SHA256 见 manifest**；契约模糊测试 CLEAN（inference_mode 内实测训练+部署成功）。

## 官方

- unregistered / NA。若协调者裁决负 gap 方向不构成禁止，本版可直接重提交（WA 免疫 + 训练可部署）。


## 协调者裁决（2026-09-07）

- 用户豁免 OOD 门在**负 gap 方向**的适用（OOD 增益高于 in-dist
  属泛化更好，非该门要拦截的分布拟合方向）。本版状态改为**可提交官方**。


## 官方结果（2026-09-07 回传）

- **official score：9538 / 175s**。`C_A = S_A − 1001 = 8537`。
- 分数 ≠ 父（1001）⇒ 手工训练器在官方 harness 实际部署成功（WA 免疫 + 训练可用的双重证明）。
- 时间预测 185.852s vs 官方实测 175s（差 10.9s，在模型 MAE 10.1s 量级）。
- 机制价值：每 KV group 可学习正交旋转单独贡献 C_A=8537，为 v189 栈锚（13008）的 65.6%；
  本地比例（0.4185/0.7522=0.556）→ 官方比例（0.656）首次给出该机制类的官方/本地映射点（仅记录，不拟合）。
- 分支矩阵更新：R1（栈，C_A=13008）> A2c（旋转，C_A=8537）；A2c 为分支第二官方锚。
