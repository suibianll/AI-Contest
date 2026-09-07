# continuous-attention A1：部署对齐旋转训练（2026-09-07）

> 契约：continuous-attention.md A1 + evidence-repair §6 A-R1。父：R2c/R3（官方 14387.8/14405）。

## A0 部署目标验证（全部通过后才训练）

- 复刻逐位对照：`_a1_stack_transform`（decode→K-center(mode)→multiplier→signs·H64→
  block-smooth→pair-transform）+ `_a1_deployed_encode`（sample-importance/精化/decode）
  与六 API 部署输出**逐位一致**（层 0/15/23 × Q/K）。修复记录：块平滑分支缺失（层 23
  启用 block_smooth_size=16）、设备迁移、`_dequantize_hif4` 命名。
- 契约 battery：5 几何/Lq<Lkv/inference/finite/state 全 PASS。

## A1 训练与评测

- 训练前向：U 坐标（栈变换后）旋转 → 部署精化编码 → decode；V 用部署量化 V
  （R2c 用标准编码 V——目标错位修复点）。梯度：STE 恒等 + 手工 attention/Cayley 反传。
- ID +0.7829（48/0/0）、default **0.7733 本地新高**（R2c +0.0055）、OOD +0.0083 过门、
  时间 239.0s、fuzz CLEAN。
- 归档 `solutions/continuous_attention_a1-deployed-aligned_officialNA_timeNA/`
  （SHA 65EB37F2...4964A，pending_official 已入 state.json）。

## 官方回传（2026-09-07 上午）

- **A1 官方 14389 / 256.3s**：vs R2c +1.2（官方中性），vs R3 −16。时间 +17.3s 超预测
  （官方精化训练更贵），<280s 但余量收窄。
- **OOD gap 排序第 3 数据点**：R3(0.0065)/14405 > A1(0.0083)/14389 > R2c(0.0122)/14387.8
  ——单调一致。该族内 OOD gap 是官方分的有效排序信号。
- 裁决：A1 卡关闭（官方中性，R3 保持最佳）。队列转 A2 成本探测（预期 TIME_HOLD）→ A3 去重。
