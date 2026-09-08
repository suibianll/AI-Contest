# L-C3 官方回传：18031 / 293s（REJECTED）

用户回传 L-C3 官方成绩 **18031 / 293s**。相对根（12352EFD，18032/280s）：
**−1 分 / +13s**。按单一完整方案计划 §2 唯一晋级指标（官方总分更高且 <300s；
同分取更快）——**不晋级，根保持不变**。

## 候选身份

- 候选：`solutions/continuous_linear_lc3-objective-only/solution.py`
- 计分 SHA：`C231328D1F8EE5B2DB02869EF071838F862EFE893F0DBAE9CF59A6293D440CC8`（核对一致）
- 父：根 solution.py（`12352EFD…`，官方 18032/280s）
- 机制：L-C3 objective-only——`_linear_output_candidate_metrics` per-case 分母
  `‖Y_f‖²` → `numel_f × MSE_STD_f`（LC0 在 L28 上官方 +3 的归一化，移植到根）

## 判定

1. **官方 −1 / +13s**（18032→18031；280→293）。本地 shard0 Δ−0.0164（微负）与
   官方 −1 方向一致，但都不决定官方；官方裁决：**机制移植未在根上转正**。
2. **LC0 的 +3 不被否定**：那是 L28 侧父基座上的验证；本移植在根
   （compiled-linear + R3 attention）上未转移，属不同基座/上下文。
3. 按计划 §3.6 关闭 L-C3 该具体移植实现；根 12352EFD 保持工作父。
4. 不扫该 fold 归一化的邻域（计划 §7 末尾：不注册邻域/替代机制）。

## 证据

- 归档：`solutions/continuous_linear_lc3-objective-only/`（manifest 已更新 REJECTED）
- 合成/合约/shard0 验证见 manifest（机制实现正确、diff 忠实、本地微负）
- 官方回传：2026-09-08 用户报告