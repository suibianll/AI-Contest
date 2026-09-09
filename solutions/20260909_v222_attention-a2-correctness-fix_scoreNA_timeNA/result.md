# v222：FIX-A2 正确性修复（mean gradient + 异常传播）

- 机制：活动计划 FIX-A2——A2 trainer 的 `grad_theta`/`grad_center` 按训练窗口数取均值
  （修复 mean loss 记录与求和梯度更新的目标不一致），修正 `_a2_train_rotation` 三返回值标注，
  移除 `hif4_calibration_attention` 的宽泛异常吞噬。不含新算法。
- parent：根 `solution.py`（v202 Linear + v195 Attention），SHA `56DC805D...EFCB2BD`。
- candidate SHA256：`8d3474bb15784e7916c212fe37e94f5d343481de3353b559fae35e47c0123699`。
- 构建/验证：`workbench/full_solution/attention-a2-correctness-fix/`（build.py、verify.py 六项
  smoke 全过：六 API、合法 state、mean 梯度、异常传播）。
- 脱离仓库单文件导入检查通过（六 API 可导入，SHA 与候选一致）。

## 本地结果（eval-v3，Qwen3.5-4B proxy-v2，CUDA，attention-only）

- shard0：paired delta mean `−0.000108`，+/-/0 = `5/7/0`，min `−0.0157`，max `+0.0187`；
  `reasonableness_issues=0`。部署输出真实变化，机制可达、非逐位等价。
- 六 shard 运行（`attention-a2-correctness-fix-all6`）：**评测器在 2 个 shard 后提前停止**
  （连续非正 paired delta）。shard0 `−0.000108`、shard1 `−0.015073`（5/7/0，min −0.0742），
  24 case 总 delta ≈ `−0.0076`。修复改变了部署输出但本地方向为负。
- 时序（只记录）：shard0 candidate calibration API `5.385s`（父 8.82s 口径来自 v205 归档，
  不同运行不可比）；本地时间不预测官方时间。

## 官方结果

- `unregistered/NA`。按计划 §2 步骤 5：六 shard 有实际变化 → 分配正式版本提交官方裁决；
  官方分数更高或同分更快且 `<300s` 才替换根，降分/超时则归档、根仍为 v202。

## 证据位置

- 归档：`solutions/20260909_v222_attention-a2-correctness-fix_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attention-a2-correctness-fix-shard0/` 与
  `attention-a2-correctness-fix-all6/`（stopped_early=true）
