# L-A0 Full-64 Direct A@W solver：能力验证与关键诊断（2026-09-08）

依据用户纠偏指令（复现 Full-64 Direct A@W HiF4 Fitting，目标官方 Linear ≈0.88）。

## 实现

从 L4（ACB16F76）构建 `_la0_full64_direct_fit`：
- 每 64 输入块：`H_B = Σω_f X̂ᵀX̂ + λI`（64×64）、`G_B = Σω_f X̂ᵀR_f`（64×o）
- `ΔW_B = H_B⁻¹G_B`（**完整 64 维，不 rank-8 压缩**）
- 合法投影（round/clamp，固定 scale/lv2/lv3，只改 mant/sign）
- 接受判定用**真实输出 loss**（最终五字段解码 + 真实动态激活），非二次型近似
- 权重 ω_f = 1/MSE_STD,f

## 能力验证（4B shard0，CPU，零模型前向）

| 指标 | 结果 |
|---|---|
| **校准 S_fit**（`1-ΣMSE_player/ΣMSE_std`） | **1.0000**（全 role，远超 0.85/0.88 目标） |
| 连续解 unweighted err | 6e-4 → 5.8e-6（-99%） |
| 合法投影后 err（逐步长） | 6e-4 → 1.8e-5（-97%，关键：写回 sign/mant reshape 修复后） |
| W_cand vs 父 | max diff 0.018，finite，scale/lv2/lv3 不变，合法 |

## 4B shard0 paired（GPU，父 L4）

| 指标 | 结果 |
|---|---|
| 独立窗口 gain | **0.0298**（vs L4 0.509，delta -0.479）——**严重过拟合** |
| api_total | **342.4s（>300s 超时）** |

## 关键诊断

1. **过拟合根因**：Full-64 自由参数 = 每块 64×o（fc_up 589,824/块 × 40 = 23.6M）远超校准行 N=138。
   S_fit=1.0 是记忆而非泛化；独立窗口 gain 0.03 证实。
   **已知 0.88 官方方法必有 L-A0 未捕捉的正则/结构约束**（这是核心机制差异）。
2. **时间超时**：342s。接受判定逐块重算全输出（`xh_folds @ W_cand_block.t()`）太慢；
   且 Full-64 每块 [64,o] 大矩阵。需 L-A1 时间重构（充分统计量、loss delta 二次型、output chunk）。

## 结论与方向

- Full-64 Direct A@W **校准拟合能力确认**（远超 L28 rank-8），但**无约束解过拟合**。
- 按用户指令：holdout 下降不否决官方探索，但**时间 342s>300s 是硬约束**；
  过拟合（gain 0.03）暗示与已知 0.88 方法存在**核心机制差异**。
- 下一步：L-A1 时间重构（压入 <300s）+ 分析已知 0.88 方法的结构约束
  （正则/低秩隐式约束/不同目标），而非继续调 L-A0 邻域。