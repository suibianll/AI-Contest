# L25：块级 lv2/lv3 联合码选择（REJECTED）

研究循环卡 L25，靶点 E2（合法投影损失 0.0320）。2026-09-08 实现并本地验证，**关闭**。

## 机制

L23b 固定父 scale/lv2/lv3，只改 sign/mant。E2 的投影损失可能来自父格点过粗
（shard0 诊断 frac_above_grid=0，无 clip）。L25 对每个块额外评估**细格点固定候选**
（lv2=1、lv3=1，scale 仅 sf），若该合法写回（同步 scale_lv2/scale_lv3=1）使 L_all
严格下降则接受，否则保留父格点。一次固定组合，不搜索。五字段合法复核通过
（`ref.validate_hif4_params` PASS）。

## 证据（4B shard0 paired，父 L4）

| 指标 | L23b | L25 |
|---|---|---|
| shard0 fit_gain（校准折叠） | 0.9478 | **0.9453** |
| 宽层 accepted | 144/144 | 144/144 |
| shard0 delta_mean（vs L4 面板） | -0.2182 | **-0.2182（逐位相同）** |

**关闭原因**：细格点候选在真实 4B 数据上从未改变部署输出（面板 delta 与 L23b
逐位一致），说明接受判定下细格点不占优；校准拟合精度反而略降（0.9453<0.9478）。
按卡证伪判据（"部署误差未降 → 关闭该组合"）**关闭**。

## 关闭粒度

只关闭"lv2/lv3=1 细格点联合选择这一实现"。E2 格仍 OPEN。E4（0.6143）仍是最大
OPEN 格（L24-C 分桶、L24-A carry-over 已关闭）。L23b（13639FB2）保持
READY_FOR_OFFICIAL，fit_gain 0.9486 ≥ 0.9 目标已达成。

## 产物

- 源码：`workbench/continuous_linear/l25-lv23-joint/candidate/solution.py`
  （SHA `8231071FCCFB407C…`）
- 归档：`solutions/continuous_linear_l25-lv23-joint_rejected/`
- 评测：`artifacts/proxy_v3/continuous/linear/l25-lv23-joint/shard0/`