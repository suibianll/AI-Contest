# A-GR2（一般互逆残差的真实目标训练）执行日志 — 2026-09-10

计划卡：`docs/superpowers/plans/parallel/2026-09-10-attention-agr2-trueobjective-plan.md`。
工作目录：`workbench/full_solution/attention-agr2-trueobjective/`（control.py、
control_results.txt、agr2_block.py、build.py）。
父：v231 Linear (L-EM3 K=2) + v195 Attention 归档
`solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py`，
SHA `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1`（现场核验；
工作区根 `solution.py` 被并行 Linear 会话改脏，未用作父或 baseline）。

## 结果：NO_EFFECT

六 shard 72/72 case 与父逐位相同（delta 全 0），六层 gate 全 parent →
按计划 §6 不占版本号、不归档、不开探针条目，只留本日志。

候选 SHA256 `0c84601d8a8d7b69db915d5c61b21d92833befd2827b12b276dbb9658c2b350b`
（v231 归档纯追加，前缀 `cmp` 逐字节验证；仅存 workbench，未归档）。

## 机制

A-GR1（v234，侧隔离官方 +29）的单变量改动：训练目标从逐 64 块 amax 尺度比代理
换成 gate 同款完整部署路径真实输出 MSE（STE 反向，训练对象只有 M=I+N；
rotation/center 冻结）。其余全部与 A-GR1 相同。

## 实现过程修正（两处，均有 control 证据）

1. **Adam eps 吞梯度**：裸 MSE 下梯度量级 ~1e-15（d_output=2·residual/numel），
   Adam 固定 eps=1e-8 使其归一化失效，且 M=I+N 的微小更新低于 fp32 在 1.0 附近的
   分辨率（6e-8），SVD 投影后 N 恒为 0（control 实测 n_norm=0.0、inv_err=0.0）。
   修正：训练损失按根 A2/A-FIX1 惯例除以每窗标准 HiF4 MSE（逐窗正常数，不改变
   逐窗最优解；gate 仍裸 MSE）。修正后 n_norm 5.6–5.9（合成）、14.5–17.8（真实）。
2. **final_loss 报告 bug**：初版训练后未重算（恒 0.0）→ 修正为训练后完整前向
   重算归一化损失。

## Control（control_results.txt，全 PASS，CPU——GPU 当时被 Linear 会话占用）

1. 0 训练步：Q/K/V 五字段与最终输出与父逐位一致。
2. 互逆性：编译对 fp64 最大偏差 2.95e-7；量化前 dense logits 最大偏差 1.18e-8。
3. 六 API 脱离仓库独立导入通过；validate_state / validate_hif4_params 通过。
4. V/Linear 与父逐位不变；可达性：合成 8/8 attempted=1，4/8 训练损失下降，
   2/8 gate 接受（双臂均自然出现）。
5. 解析梯度 `grad_N = G_m − P·G_p^T·P` 公式级 float64 有限差分 worst relerr
   **2.89e-7 < 1e-6**（eps=1e-5；fp32 einsum helper 的自身舍入 ~1e-5 级另测，
   与 fp32 训练精度一致）。

## 本地评测（GPU 串行，后台 30s 轮询 <2000 MiB 后启动；baseline=v231 归档）

- shard0（接口检查）：status ok，reasonableness_issues 0，SHA 前缀匹配。
- 六 shard（`--stop-after-nonpositive 6`）：records 6 无异常；**72/72 case
  delta 精确 0**（0/0/72）。api total delta 合计约 +35s（缓存口径混合，仅诊断）。

真实 4B 逐层 gate 审计（校准缓存）：

| 层 | 根 a2 arm | agr2 arm | train(归一化) | gate 父 | gate 候选 | n_norm |
|---|---|---|---|---|---|---|
| 0 | rotation | parent | 0.4555→0.4577 ↑ | 4.0923e-4 | 4.2122e-4 | 17.18 |
| 1 | rotation | parent | 0.4998→0.5224 ↑ | 1.6597e-3 | 1.6817e-3 | 15.16 |
| 5 | rotation | parent | 0.3011→0.2748 ↓ | 1.2467e-3 | 1.2847e-3 | 15.38 |
| 8 | identity | parent | 0.4494→0.3877 ↓ | 3.6691e-3 | 3.8543e-3 | 14.50 |
| 15 | rotation | parent | 0.5798→0.5833 ↑ | 2.2210e-3 | 2.4655e-3 | 16.93 |
| 22 | rotation | parent | 0.6926→0.6917 ↓ | 1.4342e-2 | 1.4413e-2 | 17.79 |

attempted 6/6、accepted 0/6。N 真实移动（奇异值钳位 [0.7071, 1.4142] 生效），
但 STE 穿完整部署编码的梯度在 32 步内连 fit 窗真实 MSE 都未稳定下降（三升三降），
gate 层层劣于父。

## 结论

在 M=I+N 同一自由度上，"训练目标对齐真实部署 MSE"单变量改动给出否定答案：
瓶颈审计 §1 的"代理目标错位"假设不成立——A-GR1 的 amax 代理反而训出了 2 层
（15、22）可过 gate 的 M。本实现关闭，不以缩步/缩窗/调 lr/改归一化重试。
A-GR1（v234）的代理目标形态维持为该自由度唯一正向实现。
