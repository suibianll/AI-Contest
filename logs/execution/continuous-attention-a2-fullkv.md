# continuous-attention A2：全长 K/V 训练（2026-09-07，DOMINATED 关闭）

- 成本探测：单层全长训练 3.2s（快于采样）；default 实测 A_calib 138.4s（+31s vs A1），
  probe 外推失准（76.7s）——以 fresh default 实测为准。
- 结果：ID +0.7792 / default 0.7713 / OOD +0.0103 / 时间 259.7s——三轴均劣于 A1
  （+0.7829 / 0.7733 / +0.0083 / 239.0s）。全长目标损失 0.2785 vs 采样 0.2589。
- 裁决：DOMINATED，卡关闭。A1 保持分支最佳（pending_official）。全长采样正则效应记录。
