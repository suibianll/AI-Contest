# continuous_attention A2：全长 K/V 训练（被 A1 支配，关闭）

> 日期：2026-09-07。父：A1。唯一变化：训练 K/V 支持集从 ≤128 采样改为全长（Q 仍 ≤32）。
> 假设：采样改变 softmax 分母/竞争关系，全长应改善目标。**假设未兑现**。

## 评测（a2full-screen / a2full-id / a2full-ood / a2full-default）

- ID 48-case：mean +0.779157（A1 +0.782914，**−0.0038**）；48 正/0 负。
- default：attention_mean **0.771298**（A1 0.773281，−0.0020）——未超本地最高。
- OOD：Δgap +0.010339（A1 +0.0083）——风险旗数值亦更差。
- 时间：A_calib 138.4s（A1 107.1s，+31s）→ 预测 259.7s（A1 239.0s，+20.7s）。
- 成本探测记录：全长单层训练 3.2s/层（快于采样的每层 ~4.5s），但 default 全流水
  实测 A_calib 138.4s——probe 外推（76.7s）与实测偏差源于 default 面板校准的实际
  窗口/精化组合；以实测为准。

## 裁决

- **A1 在精度/OOD/时间三轴均占优 A2 → A2 卡关闭（DOMINATED）**。分支最佳保持 A1
  （default 0.7733，pending_official）。全长目标损失更高（0.2785 vs A1 0.2589）——
  采样本身对校准窗有正则效应，全长并非更优目标。
- solution SHA256 见 git（CFDDCED7 家族 + `_A2_MAX_KV_TOKENS=10**9` 单行差异）。
