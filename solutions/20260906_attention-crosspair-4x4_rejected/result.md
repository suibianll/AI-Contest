# Attention cross-pair 4×4（REJECTED）

日期：2026-09-06

父版本：v189（源码 SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`）

候选源码 SHA256：`3BA683A046083A5E9E043CC135532D01912A2CC437A5290ABB2D5D6E695E75F3`

状态：**REJECTED / C2_OOD_REJECTED**。根 `solution.py` 未改，官方状态保持
`unregistered/NA`。

## 机制

把相邻两个二维 pair 固定组成四维 super-pair，在 v189 已冻结 Q/K 状态后拟合一次
普通 covariance 平衡的 SPD 4×4 变换；Q 使用 `M`，K 使用 `M^-T`。不使用 Fisher、
V、输出残差或 Jacobian，在线只应用已编译的 `pair_transform`。

## 证据

- C0：六 API 可导入；连续 `QK` 最大绝对误差 `9.54e-7`；2×2 回退通过。
- C1 shard0：mean delta `+0.001844`，L1 `0.001844`，正/负/零 `2/0/6`。
- C1 shard1：mean delta `+0.009313`，L1 `0.010591`，正/负/零 `7/1/0`。
- C2 ID 六 shard：candidate `0.758912691`，parent `0.752772355`，delta `+0.006140336`。
- C2 OOD shard0/1：mean delta `-0.001240/-0.015858`；shard1 L1 `0.021338`，最坏
  delta `-0.072815`（layer 19、validation、length 10）。OOD 门禁失败，未运行 C3。

完整 eval-v3 证据：

- `artifacts/proxy_v3/attention-crosspair-4x4-20260906/c1/`
- `artifacts/proxy_v3/attention-crosspair-4x4-20260906/c2/`
- `artifacts/proxy_v3/attention-crosspair-4x4-20260906/c2-ood/`
