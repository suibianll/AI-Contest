# A1a Matrix-Smooth 4×4 组内扩展 — Qwen compact 哨兵 REJECTED

日期：2026-09-03
状态：**REJECTED（阶段 B，Qwen attention compact 哨兵即停）**

## 1. 候选定义

- 父版本：v160 归档源码，SHA `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`；
- 候选源码：`workbench/a1_matrix_smooth4.py`，SHA256
  `7e7df03f4229fb390be45b25dde63447762bdef2512edc38580a71780f916a5b`；
- 唯一机制变化：Attention Matrix-Smooth 的组内平衡 block 从 2×2 扩大到 4×4
  （`_ATTN_PAIR_SMOOTH_SIZE = 4`；`_apply_attention_pair_transform` 从 matrices shape 推断
  pair size；fit 内 `eye` 参数化；head_dim % 4 != 0 时回退 2×2）。ridge、fit/validation
  划分、A1 gate、importance 推导全部沿用 v158，无任何参数搜索；
- 连续域不变性保持：`k_matrix = inv(q_matrix)^T`，单元检查 `q@k^T=I` 误差 `1.2e-7`，
  部署 logits 相对误差 `2.0e-7`（GQA q=14/kv=2/head_dim=64 随机数据）；
  head_dim=6 时正确回退 2×2。

## 2. 阶段 A（接口与不变性）

- 六 API 脱离仓库导入 ✓；常量 `_ATTN_PAIR_SMOOTH_SIZE=4` ✓；
- 数值单元检查（`workbench/a1_unit_check.py`）全部通过：4×4 state shape
  `(14,16,4,4)/(2,16,4,4)`、`q@k^T=I`、部署 logits 不变、fallback。

## 3. 阶段 B（Qwen attention compact，NVFP4 cache 命中）

- Parent：v160 归档源码同面板运行
  （`artifacts/official_eval/a1-parent-v160-attn-compact.json`，4 layer states，4 cases）；
- 候选：`artifacts/official_eval/a1-matrixsmooth4-attn-compact.json`，
  report `logs/official_eval/a1-matrixsmooth4-attn-compact.md`；
- 结果：Attention mean `0.718819`，paired mean Δgain **`-0.078643`**、median Δ
  `-0.002349`，改善/回归/不变 **`0/2/2`**，结论 `consistent_regression`；
- 误差源：layer 23 K-only `-164.36`、layer 15 Q/K `-21.8/-18.7`；两个不变 case 是
  A1 gate 在那些层拒绝了 4×4 变换（保持 v160 行为），两个回归 case 是 gate 接受但
  真 holdout 变差；
- API `10.826s`（无时间风险）；未运行 GPT-2 与 default（失败即停）。

## 4. 失败机制归因

- 2×2→4×4 使每 block 平衡自由度从 3（SPD 2×2 对称参数）升到 10；Attention 校准
  长度 `[10,128,512,1024,1024]` 折内样本有限，10 参数拟合方差大；
- A1 gate 的 odd-window 验证折与 compact 真 holdout 分布不同：gate 在 2/4 层接受了
  4×4，但接受层在 holdout 上整体回归——过拟合验证折而非泛化；
- v158 的 2×2 恰在偏差-方差甜点：更小 block 无足够跨通道补偿，更大 block 拟合方差
  占优。该结论同时间接否定 A1b（跨 head 联合均衡自由度更高，方差问题只会更严重），
  A1（组内扩展）整体关闭。

## 5. 决定

- A1a/A1 `REJECTED`，不调参（不试 3×3、不改 ridge、不缩 block、不按层路由）；
- 证据不删除：parent/candidate JSON 与 report 保留；
- 按活动计划顺序进入 **A2 第 0 步**（深层 K 结构诊断，零 API）：读
  `v160-a2-attn-default-candidate.json` 分解与 layer 22/23 K 投影结构特征，仅当存在
  跨层一致且跨模型可验证的解析病因时才构造 A2 候选；
- 单元检查脚本 `workbench/a1_unit_check.py` 保留（可复现不变性验证）。
