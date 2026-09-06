# v162_attention R2c：栈内旋转手工训练器版（OOD 正向 gap，字面阻止）

> 日期：2026-09-07。父：R2/R2b。手工梯度训练器移植进 v189 栈（同 A2c，parity 已验）。
> 解决了 R2b 的"训练无法在官方部署"问题（inference_mode 内实测训练+gate+部署全链路正常），
> 但机制的 in-dist/OOD 不对称性依旧。

## 评测（r2c-screen / r2c-id / r2c-ood / r2c-default）

- ID 48-case：mean **+0.776564** / median +0.752453，48 正/0 零/0 负，L1_neg 0；
  split test +0.780033 / validation +0.773095 均正。
- default：attention_mean **0.767788**（R1=0.752173，+0.0156）；linear 0.0 ✓。
- 时间：A_calib 相近 → **预测 225.560s < 280s** ✓。
- **OOD：Δgap = +0.012242 > 0.01 → 字面 BLOCKED**（正向 = in-dist 虚高的分布拟合方向，
  协调者 2026-09-07 豁免仅覆盖负方向，本版不适用豁免）。

## 裁决

- **OOD_BLOCKED 维持**。按"失败换机制"纪律不扫邻域修复；除非协调者对正向 gap 另行裁决。
- 分支最佳/可提交版仍为：R1（14009/211s 官方锚）与 A2c（负 gap 已豁免，可测官方价值）。
