# 归档 PAWV 变长 bug 批量修复与 v5 复评

> 日期：2026-08-31
> 官方规则（2026-08-31 修订）：端到端时间限制 300s，不再限制任何 `A@W` 拟合用法。
> v98 官方 timeout；v100/v107 官方 Attention WA（非 timeout），根因为 B2 PAWV 变长
> calibration shape mismatch；v121 官方 timeout。见
> [`v98 官方 timeout`](2026-08-31-v98-official-timeout.md) 与
> [`Attention WA 根因`](2026-08-31-v100-v107-attention-wa-root-cause.md)。

## 修复对象与方式

v099–v125 共 **28 个归档 `solution.py`** 携带 B2 PAWV 变长 bug：`_build_pawv_metric`
用 `q_samples[0].shape[0]` 建立固定 token 方阵，却对每个校准样本直接
`metric += einsum(P,P)`，官方长度 `[10,128,512,1024,1024]` 在第二样本即触发
`[10,10] += [128,128]` RuntimeError。v126/v127 已修复；本次将其余 28 个归档统一
打补丁，与 v127 修复逻辑逐字段一致：

1. `_build_pawv_metric` 改为按 `seq_len` 分组返回 `dict[str, torch.Tensor]`（keyed diagonal）；
2. 校准调用改为 `row_diagonals.get(str(int(v.shape[0])))`；
3. `v_state` 改为 `{"row_diagonals": {key: CPU tensor ...}, "row_lowrank": None}`；
4. `hif4_dynamic_quantize_v` 按 `dense.shape[0]` 精确长度查表，未命中回退 `state["row_diagonal"]`。

修复脚本与验证脚本：`.pytest_tmp/fix_pawv.py`、`.pytest_tmp/verify_pawv_fix.py`（临时，
已清理）。**28 个文件批改后全部 `ast.parse` 语法通过；v100/v107/v121 三条代表性候选在
官方长度 `[10,128,512,1024,1024]` 与 `[10,32]` 变长校准下 `PASS`**
（keyed keys、形状、finite、精确长度查表、未命中回退均通过）。

受影响的 28 个目录：

- `20260830_v099_...` 至 `20260830_v107_...`（v099–v107 全部 9 个）
- `20260831_v108_...` 至 `20260831_v125_...`（v108–v125 全部 18 个）
- `20260831_v125b_c1c-block8-screen-positive`（2026-09-01 归档整理时重命名，版本号唯一化）

## v5 sampled-means 复评（Qwen2.5-0.5B，224 Linear / 32 Attention，seed 20260831）

| 候选 | 官方原结果 | Linear mean | Attention mean | Local API (s) | 报告 |
|---|---|---:|---:|---:|---|
| c39 (v031) | 21864 / 161.3s | 0.439775 | 0.667092 | 80.500 | [`official-anchors`](2026-08-31-official-anchors-sampled.md) |
| c41b (v034) | 21864 / 159.4s | 0.439775 | 0.667092 | 79.094 | 同上 |
| c47b (v051) | 22451 / 234s | 0.433744 | 0.667092 | 116.557 | 同上 |
| c66 (v066) | 22557 / 217.2s | 0.432060 | 0.671106 | 187.353 | 同上 |
| v72 | 22662 / 226s | 0.432117 | 0.671106 | 228.777 | [`v072`](2026-08-31-v072-sampled.md) |
| v74 | 22750 / 239.4s | 0.440305 | 0.671106 | 218.619 | 已有记录 |
| v98 | timeout | 0.506715 | 0.828323 | 219.040 | [`v098`](2026-08-31-v098-sampled.md) |
| v100-pawv-fixed | WA | 0.506715 | 0.828395 | 150.251 | [`v100 修复`](2026-08-31-v100-pawv-fixed-sampled.md) |
| v107-pawv-fixed | WA（非 timeout） | 0.512967 | 0.828395 | 241.506 | [`v107 修复`](2026-08-31-v107-pawv-fixed-sampled.md) |
| v121-pawv-fixed | timeout | 0.516685 | 0.828395 | 832.920 | [`v121 修复`](2026-08-31-v121-pawv-fixed-sampled.md) |

## 结论与边界

- **修复有效**：v100/v107/v121 的 `solution.py` 现能通过官方变长形状复现；本地
  `sampled-means-v1` 全部 `local_result_valid=true`。
- **不改变官方分类**：本地通过不改变官方 300s 与格式判定。v100 官方 WA、v107 官方 WA
  （非 timeout）仍是官方记录；v121 本地 API 832.92s 仍远超 300s，不可提交。
- **精度链**：修复后 v107 Linear `0.512967` 高于 v127 `0.509408`（sampled 口径），
  v121 `0.516685` 更高但时间不可行；这为下一步在 300s 内压缩 C1b/C1c 结构性计算
  提供了精度上限参考，不构成提交推荐。
- 所有新报告均为 v5 协议、同一 seed/cache/shape，跨候选可直接比较；官方锚点分数仅作
  历史记录，不参与本地换算。