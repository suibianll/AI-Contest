# L24-C：动态 activation scale 分桶加权拟合（REJECTED）

研究循环卡 L24-C，靶点 E4（部署失配 0.6143）。2026-09-08 实现并本地验证，**关闭**。

## 机制

L23b 拟合使用 fold 级输出能量 omega `ω_f = 1/(F·max(‖Y_f‖²,1e-12))`，实测给 10 行短窗
fold0 每行权重是 128 行 fold1 的 **17.76 倍**（proj），10 行短窗贡献总权重 0.0138 vs
128 行窗 0.00997 —— 校准拟合被短窗主导，而部署全是 128/512 行长窗（E4 失配来源）。

L24-C：按 `floor(log2(行 activation max))` 分桶，每桶等权。一个固定配置（`_L24_C_SCALE_BUCKET=True`、
`_L24_C_BUCKET_GRID=16`），不扫参。

## 证据（4B shard0 paired，父 L4）

| 指标 | L23b | L24-C |
|---|---|---|
| shard0 fit_gain（校准折叠） | 0.9478 | **0.9249** |
| shard0 delta_mean（vs L4 面板） | -0.2182 | **-0.2416** |
| pos/neg/zero | 0/56/0 | 0/56/0 |

**双向负向**：分桶加权既降低校准拟合精度，又未改善部署失配。按卡证伪判据
（E4 未降 → 关闭该分桶方式）**关闭**。

## 关闭粒度

只关闭"activation-scale 分桶加权这一实现"。E4 格仍 OPEN，可尝试其他机制（非分桶邻域）。
L23b（13639FB2）保持 READY_FOR_OFFICIAL。

## 产物

- 源码：`workbench/continuous_linear/l24-c-scale-bucket/candidate/solution.py`
  （SHA `585387E960C4BD80…`）
- 归档：`solutions/continuous_linear_l24c-scale-bucket_rejected/`
- 评测：`artifacts/proxy_v3/continuous/linear/l24-c-scale-bucket/shard0/`
- 账本：`artifacts/continuous/linear/error_ledger_linear_2026-09-08.json`