# v162_attention R1：v189 Attention 栈迁入 + 标准 Linear（分支最佳候选）

> 日期：2026-09-06。分支：v162 独立 Attention（A 代理）。
> 契约：[总计划](../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> + [A 任务书](../docs/superpowers/archive/plans/v162-attention.md)。
> 分支前缀 `v162_attention_`，全局版本号/根替换/组合由协调者负责。

## 源码与身份

- **solution.py SHA256：`3619BFEB0E017555BD8FE31410888F80950A0128E3A2B94F643BD3F20B30BFC8`**
- 零点父：v162 `56101559...C000A`（官方 1001/146s）；分支父链：A2 旋转 gate 臂 → 本包。
- 机制（RECOVERY，按总计划 §1"明确机制来源后单独迁入；已知收益记 RECOVERY"）：
  v189（v186 Attention 栈：Smooth-QK center / multiplier / signs·H64 / pair-transform /
  logit-gain / K center_mode=4 / V importance refinement）整体保留；
  文件尾追加 v162 标准 Linear 覆盖定义（shadow def + 独立 `_branch_*` codec helper），
  零删除、零注意力侧改动。构建：`workbench/v162_attention/candidate_v2/solution.py`。

## 逐位验证

- Linear 侧：56 项比较（4 层 × 7 role × {weight calib, dyn act}）== v162 standard 逐位一致。
- Attention 侧：12 项 API 探针（4 层 × Q/K/V，同校准 state）== v189 逐位一致。
- 脱离仓库单文件导入：6/6 API 组返回 ✓。
- 工具：`workbench/v162_attention/verify_r1_port.py`。

## 本地门禁（全部通过）

| 门 | 结果 |
|---|---|
| eval-v3 六 shard ID（baseline=v162） | mean **+0.752772** / median +0.724958，n=48，48 正/0 零/0 负，L1_neg **0.000000** |
| split | test +0.753569 / validation +0.751976 均正 |
| OOD（候选/直接父成对，同 SHA） | 候选 gap +0.000915（in +0.752772 / ood +0.751857），\|Δgap\| ≤ 0.01 ✅ |
| fresh default（compat，六 API，168+120） | attention_mean **0.752173**（与 v189 历史值 0.752173407020 完全一致）；linear 0.0；overall 0.313406 |
| 时间模型 | W_calib 0.668s、A_calib 65.710s、dyn_act 1.466s、dyn_qkv 3.306s → **211.832s < 280s** ✅ |
| 强对照 | 同协议 v168 +0.7494 / v189 +0.7571（32-case 面板）：本包为 v189 Attention 侧的逐位复现 |

## 官方结果

- **official score：14009 / 211s**（2026-09-06 用户回传）。`C_A = S_A − 1001 = 13008`，
  与 v189 加性模型的 Attention 侧锚（≈13008）一致；时间预测 211.832s vs 官方实测 211s。
  本包成为 Attention 分支官方锚：`S_A = 14009`。
- 预注册判读：`C_A = S_A − 1001`；机制来源为 v189 Attention 栈（其完整组合官方锚 17616/275s，
  本包以标准 Linear 替换 static-actorder Linear，W_calib 项更小，时间余量更大）。
- 分支本地最高 Attention：default **0.752173** / shard48 **+0.752772**。
