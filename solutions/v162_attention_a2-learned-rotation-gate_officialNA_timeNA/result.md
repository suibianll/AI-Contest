# v162_attention A2：每 KV group 可学习正交旋转（gate 臂，任务书 A2/A3 全流程）

> 日期：2026-09-06。分支：v162 独立 Attention（A 代理）。
> 契约：[总计划](../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> + [A 任务书](../docs/superpowers/plans/workpackages/v162-attention.md)。
> 冻结：两个 Linear API、dynamic V、standard codec = v162。

## 源码与身份

- **solution.py SHA256：`19159AB9904FB3F10CD89FC20D188CF3428546AE0B8B0E96E4AFBAC28F841940`**
- 零点父：v162 `56101559...C000A`（官方 1001/146s）；直接父 = 零点（分支首机制）。
- 机制（新机制，非重复：08-26 learned butterfly 仅计划未执行；novelty 表见执行日志）：
  每 KV group 一个可学习正交 `R_g∈R^{64×64}`（Cayley(Θ)+规范化 Sylvester Hadamard 初始化），
  同组 Q head 与 K head 共享 R_g，动态路径 NVFP4 解码 → group matmul → v162 标准 HiF4 编码；
  校准内按 A2 冻结配置训练（Adam lr=0.01 × 32 步、STE、reg 1e-3·mean((C−I)²)、
  等间隔确定性采样 128 KV/32 Q），第 5 校准窗为 gate，gate 上先 learned vs H、
  胜者再 vs identity，严格更优才部署（平局归 identity）。config：`workbench/v162_attention/config.json`。

## 必要测试（A1，全部通过）

T1 R=I 逐位对齐 / T2 GQA 连续 QK 误差 2.67e-05 / T3 调用次序与 token 数无关 /
T4 五字段合法 + decoded==研究前向 / T5 STE 梯度有限非零 + 训练冒烟 / T6 Linear+V 逐位不变。
工具：`workbench/v162_attention/tests_a1.py`。

## 本地门禁（全部通过）

| 门 | 结果 |
|---|---|
| eval-v3 六 shard ID（baseline=v162） | mean **+0.429820** / median +0.465527，n=48，42 正/6 零/0 负，L1_neg **0.000000**（L1_total 0.4298 仅记录） |
| split | test +0.434912 / validation +0.424728 均正 |
| 逐层部署 | 21/24 层部署旋转（attempted 24 / accepted 21），层 0/2/8 gate 回退 identity |
| OOD | Δgap −0.008639，\|·\| ≤ 0.01 ✅（in +0.4298 / ood +0.4385，无分布拟合特征） |
| fresh default（compat 168+120） | attention_mean **0.422443**；linear 0.0；overall 0.176018 |
| 时间模型 | W_calib 0.680s、A_calib 23.706s（含训练）、dyn_act 1.697s、dyn_qkv 0.843s → **186.744s < 280s** ✅ |
| 真实 control | 60 项比较（4 层 × 7 role + dyn V）与 v162 逐位一致 ✅ |
| 自包含 | 脱离仓库单文件导入 6/6 API ✓ |
| 四臂/强对照 | gate 优于 H 臂（+0.4213 vs +0.3768）；未超 v168 +0.7494 / v189 +0.7571 → RECOVERY_ONLY 标签 |

## 状态与官方

- 状态：**CLEAN_ROOM_PROGRESS / RECOVERY_ONLY**，全门通过，分支第一可登记机制；
  被 R1（v189 栈迁入）超越后不再是分支最佳，但保持独立登记资格。
- **official score：unregistered / NA**；官方贡献按 `C_A = S_A − 1001` 登记（若单独提交）。
