# continuous_attention A1：部署对齐旋转训练（本地新高，待官方）

> 日期：2026-09-07。父：R2c/R3 机制（官方 14387.8 / 14405）。隶属[已归档持续优化总计划](../../docs/superpowers/archive/plans/2026-09-07-continuous-linear-attention-plan-superseded.md)。
> 唯一机制变化（A1 卡）：训练硬前向改用**旋转插入点的真实坐标**（栈变换复刻 U，逐位对照
> 通过）+ **父实际精化编码**（importance/offsets/refinement 全量）+ **部署量化的 V**——
> 修复 R2c 的训练/部署错位（R2c 训练用标准编码器代理 + 标准 V）。配置完全继承 R2c
> （32步/lr0.01/clip1.0/reg1e-3/等间隔采样/末窗 gate），手工梯度保留。

## A0 部署目标验证（verify_a0.py，全部通过）

- 复刻逐位对照：U 复刻 → rotate → 部署编码 → decode == 六 API 部署输出（层 0/15/23、
  Q/K 双侧、随机正交 R）**逐位一致**；层 23 的 block-smooth 分支缺口已补齐。
- 契约 battery：5 种几何（含 MHA/non-pow2 head_dim）、Lq<Lkv、inference_mode/no_grad、
  state 合法、有限输出 PASS。
- 训练前向修复记录：设备迁移、`_dequantize_hif4` 命名、块平滑分支（原 R2c 复刻缺失）。

## 评测（a1-screen / a1-id / a1-ood / a1-default）

| 门 | 结果 |
|---|---|
| ID 48-case | mean **+0.782914** / median +0.772866，48 正/0 零/0 负，L1_neg 0 |
| split | test +0.7804 / validation +0.7854 均正 |
| **OOD** | Δgap **+0.0083 ≤ 0.01 ✅**（R2c +0.0122 → 部署对齐训练修复一半） |
| default | attention_mean **0.773281**（**本地新高**，R2c 0.767788 +0.0055；R3 0.765023） |
| 时间 | A_calib 107.1s（部署对齐训练较贵）→ **预测 239.0s < 280s** ✅ |
| 契约 fuzz | CLEAN |

## 官方

- **unregistered / NA（pending_official）**。官方矩阵：R3 14405 > R2c 14387.8 > R1 14009；
  本地映射（仅记录）：R2c 0.7678→14387.8，A1 0.7733→待回传。
- **solution.py SHA256 见 manifest**。


## 官方结果（2026-09-07 回传）

- **official score：14389 / 256.3s**。vs R2c（14387.8）+1.2 = 官方中性；vs R3（14405）−16。
- 时间 256.3s vs 预测 239.0s（+17.3s，官方精化训练更贵），仍 <280s 但余量收窄。
- **OOD gap 排序第 3 数据点**：R3(0.0065)→14405 > A1(0.0083)→14389 > R2c(0.0122)→14387.8，
  单调一致。结论：部署对齐训练官方中性；R3 保持最佳；本卡关闭，队列转 A2 成本探测。
