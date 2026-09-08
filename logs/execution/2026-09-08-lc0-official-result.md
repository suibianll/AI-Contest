# LC0（correctness-hardened）官方回传：4614 / 294s

用户回传 LC0 官方成绩 **4614 / 294s**。相对父 L28（4611/286s）：**+3 分 / +8s**，294s < 300s 通过。

## 候选身份

- 候选：`solutions/continuous_linear_lc0-correctness-hardened/solution.py`
- 源码 SHA：`E702D7C1F087097906E752DEFA5CBFF05C2E41585883B0EC8D5E4F937AA4AF3D`（已核对一致）
- 父：L28（`44D7E964…`）
- 改动：L28 的 correctness hardening（MSE_STD 归一化 A@W 校准目标、严格激活形状检查、
  无静默 GPTQ 回退、规范零 sign 写回、显式合法格点 tie 语义）

## 判定

1. **官方正向 +3**（4611→4614），核心机制：**MSE_STD 归一化 A@W 校准目标**
   （`ω_f = 1/(F·numel·MSE_STD_f)`）——与纠偏指令建议的 `ω_f = 1/max(MSE_STD,f)` 一致，
   证实该归一化方向正确。
2. **本地 shard0 微负（-0.0027）但官方 +3**——再次确认 4B proxy 不排序官方
   （用户已确认本地/官方多次反转）。
3. **时间 294s 距 300s 仅 6s 余量**，无法与 Attention 组合；L28（286s）仍为组合时间父。
4. LC0 登记为**已确认官方正向的候选**，但**不升级为侧父**（+3 分 / +8s 的 Pareto 不如 L28）。

## 机制继承

LC0 的 MSE_STD 归一化目标可直接应用到 L-A0/A1 Full-64 Direct A@W 路线
（其 ω 权重当前用 `1/MSE_STD`，已一致）。这是纠偏指令与 LC0 的共识点。