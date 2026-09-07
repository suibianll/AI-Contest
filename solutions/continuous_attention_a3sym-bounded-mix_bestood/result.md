# continuous_attention A3sym：旋转+K-center+有界对称混合（官方 REJECTED——OOD 排序信号跨机制证伪）

> 日期：2026-09-07。父：A1/A2 机制族。新自由度：每 KV 组对称非正交混合
> `T = exp(S)`（S 对称零迹初始 0，特征值钳位 [−log2/2, +log2/2] ⇒ cond(T) ≤ 2），
> Q 乘 T、K 乘 T^{-1}（连续 QK 严格不变），与旋转联合 32 步解析梯度训练
> （Daleckii–Krein 矩阵指数导数，parity 8.1e-06）。
> 训练前向为 A0 逐位验证的部署路径复刻（含 block-smooth 分支修复）。

## 实现事故记录（已修复）

1. 第一版部署的 `_a2_apply_sym_mix` einsum 要求 num_heads==groups——Q（14 heads）
   触发维度错误被 except 吞掉 ⇒ **Q 侧混合静默丢失**（只有 K 侧生效，QK 不变性破坏）
   ⇒ 首轮评测灾难性负分（ID −4.99）。修复为 GQA per-group 映射 + 硬断言。
2. Daleckii–Krein 初版缺对称化（非对称 W 下原始伴随非对称）→ parity 0.82；对称化后 8.1e-06。
3. 两个 bug 均为部署/实现层，非机制本身；修复后重评（a3sym2-* 全新目录，无缓存复用）。

## 评测（a3sym2-screen / a3sym2-id / a3sym2-ood / a3sym2-default）

| 门 | 结果 |
|---|---|
| ID 48-case | mean +0.768727 / median +0.751162，48 正/0 零/0 负，L1_neg 0 |
| split | test +0.7746 / validation +0.7629 均正 |
| **OOD** | **Δgap −0.000459 —— 全候选最佳 OOD 档案**（R3 +0.0065 / A1 +0.0083 / R2c +0.0122） |
| default | attention_mean 0.765190（A1 0.773281 −0.0081；R2c 0.767788 −0.0026） |
| 时间 | A_calib 126.2s → **预测 253.936s < 280s** ✅ |
| 契约 fuzz | CLEAN |

## OOD 排序信号预测（供协调者决策）

族内三点单调关系：R3(+0.0065)→14405 > A1(+0.0083)→14389 > R2c(+0.0122)→14387.8。
A3sym 的 −0.0005 严格低于 R3 的 +0.0065 ⇒ **按排序信号预测官方 ≥ 14405**。
但本地 in-dist 低于 A1/R2c（−0.008/−0.003）——按「超过本地最高才提交」规则不触发；
是否花费一次官方提交测「OOD 排序信号在负方向的外推」由协调者决定。

## 官方

- **unregistered / NA**。solution.py SHA256 见 git（CFDDCED7 家族 + sym-mix 扩展）。


## 官方结果（2026-09-07 回传）

- **official score：13572 / 288s**。`C_A = 12571`——兄弟候选最差
  （A2 −868、R3 −833、A1 −817），288s > 280s 提交门。
- **OOD 排序信号跨机制证伪**：最佳 OOD 档案 ⇒ 最差官方分；该信号仅在 R2c 同源
  训练器族内单调，跨机制无效。裁决 **REJECTED**，卡关闭，不扫邻域。
