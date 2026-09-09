# v227 / A-G1（Q/K 联合仿射 gauge）六 shard 净负事后诊断

日期：2026-09-10。纯 CPU 事后分析，未运行任何 GPU 任务，未修改源码。
输入证据：
- 校准缓存 `artifacts/official_eval/cache/proxy-v3-calibration/165e1a6bc50abd41-attention-*.pt`（v227，6 文件）与
  `56dc805d6e5a3aef-attention-*.pt`（根，6 文件）。
- 六 shard per-case JSON `artifacts/proxy_v3/attention-ag1-sixshard-full-20260910/candidate/{candidate,baseline}-attention-shard*.json`
  （baseline 已核验 = 根 SHA `56dc805d…EFCB2BD`）。
- 脚本：`diag_calibration_audit.py`（Q1/Q3/Q4）、`diag_paired_cases.py`（Q2）；
  结果：`diag_calibration_audit.json`、`diag_paired_cases.json`。

## shard↔层↔case 对应关系（先决事实）

- `evaluator/proxy_v3_eval.py:38` `shard_layers(total=24, shard) = range(shard, 24, 6)`，每 shard 4 层；
  面板 FA 池为 panel 层 {0,1,5,8,15,22}（模型层 {3,7,11,19,23,31}），每 shard 恰好 1 个 FA 层：
  **shard0→层0, shard1→层1, shard2→层8, shard3→层15, shard4→层22, shard5→层5**。
- 每个 shard 的 12 个 attention case = 同一层的 12 个 test 窗口（`test_qkv_windows` 全部 12 个，
  split 混合 validation/test，长度 {10,128,512,1024} 各 3）。**case 不过全部 6 层；层与 shard 一一对应**。
- 校准窗口 5 个（train split，长度 10/128/512/1024/1024）：前 4 个训练 A2，最后 1 个（len 1024）做单窗口 gate。

## Q1 接受率与 gate 方向

v227 audit（每层的 a2_arm 等字段从缓存 q_state 读出，无 fallback）：

| panel层 | shard | arm | gate_rel_impr | train_loss | max|s| | 根对照 arm / gate_rel_impr / train_loss |
|---|---|---|---|---|---|---|---|
| 0 | 0 | rotation+scale | **+4.99%** | 0.6818 | 0.2385 | rotation / +4.54% / 0.7562 |
| 1 | 1 | rotation+scale | **+8.81%** | 0.7752 | 0.2158 | rotation / +11.76% / 0.7011 |
| 5 | 5 | rotation+scale | **+3.41%** | 0.4816 | 0.2095 | rotation / +7.06% / 0.4635 |
| 8 | 2 | identity | −13.61% | 0.5574 | 0.2337 | identity / −2.86% / 0.5933 |
| 15 | 3 | identity | −1.12% | 0.7481 | 0.2364 | **rotation** / **+2.43%** / 0.7073 |
| 22 | 4 | rotation+scale | +0.035% | 0.9603 | 0.2145 | rotation / +0.43% / 0.9646 |

- **6 层中 4 层接受**（0、1、5、22），2 层 gate 拒绝（8、15），无 fallback。
- 接受层 gate 改善幅度 0.035% ~ 8.81%。
- **shard2 全部 12 case 逐位不变**的直接原因：层 8 在候选与根下都被 gate 拒绝（identity），
  两侧部署状态逐位相同 → 输出逐位相同。不是"该 shard 覆盖的层全回退"的巧合，而是该层在根里也是 identity。
- 关键差异：**层 15 在根里接受 rotation（gate +2.43%），在 v227 里联合训练后 gate 变 −1.12% 被拒**——
  v227 在层 15 上退化成"无 rotation"，直接丢掉了根已有的收益。
- 联合训练在 4/6 层上连 gate 窗口本身都比根的纯 rotation 差（层 1：8.81% vs 11.76%；层 5：3.41% vs 7.06%；
  层 15：−1.12% vs +2.43%；层 22：0.035% vs 0.43%），仅层 0 略好。train_loss 同样在层 1/5/15/22 高于根。
  即加入 s 自由度后，固定 32 步 Adam 下 rotation 本身的优化质量被拖低。

## Q2 接受层与 eval 恶化的对应

per-case paired delta（candidate gain − baseline gain），72 case 总体 mean **−0.005294**，pos/neg/zero = 28/32/12：

| 层(shard) | arm | mean delta | pos/neg/zero | min / max |
|---|---|---|---|---|
| 0 (s0) | rotation+scale | −0.002287 | 6/6/0 | −0.0412 / +0.0117 |
| 1 (s1) | rotation+scale | **−0.021423** | 6/6/0 | −0.0880 / +0.0697 |
| 5 (s5) | rotation+scale | −0.004880 | 5/7/0 | −0.0252 / +0.0149 |
| 8 (s2) | identity(两侧同) | +0.000000 | 0/0/12 | 0 / 0 |
| 15 (s3) | identity(根=rotation) | −0.003990 | 7/5/0 | −0.0853 / +0.0286 |
| 22 (s4) | rotation+scale | +0.000814 | 4/8/0 | −0.0039 / +0.0104 |

- 负 delta 集中在**接受了联合候选的层**（1、5、0）和**因联合训练翻车而丢失根 rotation 的层 15**；
  唯一 net 正是层 22，其 gate 改善也最小（+0.035%，即该层部署的变换最接近不动）。
- 被 gate 拒绝的层不全对应 delta=0：层 8 两侧同 identity → 全 0；层 15 因与根 arm 不同而有非零 delta
  （7 正 5 负，净 −0.0040，量级与根该层 gate 收益 +2.43% 相符）。
- 分组：test split −0.00706 vs validation −0.00353；按长度 len1024 −0.00939 最差，
  len128 −0.00489，len512 −0.00278，len10 −0.00148。校准窗口本身含长窗口，但长评测窗口恶化更系统性。

## Q3 gate 校准改善与 eval 恶化的幅度关系

接受层上 gate 相对改善与 eval mean delta **完全反序**（n=4，Spearman = −1）：

  gate +8.81%(层1) → −0.0214；+4.99%(层0) → −0.0023；+3.41%(层5) → −0.0049；+0.035%(层22) → +0.0008。

gate 认为改善越大的层，eval 恶化越大。这不是"gate 改善接近噪声"，而是**校准窗口上的改善与
未见窗口上的效果系统性反号**：s（和与之耦合的 rotation）拟合的是 4+1 个校准窗口的量化舍入边界细节，
这些边界移动对新窗口不迁移。对照：根的纯 rotation 用同一单窗口 gate 定价，官方 +21，
说明 rotation 自由度的校准收益是窗口稳定的，而 scale 自由度的校准收益是窗口特异的。

## Q4 s 的形态

- 接受层 `learned_scale` 形状 [4 groups, 256]，**1024/1024 全部非零**（稠密微调，不是稀疏修正）。
- max|s| = 0.2095~0.2385，**远低于 log(2)=0.693 截断，clamp 命中 0%**——截断不是约束因素。
- 零均值投影精确（每行 |mean| ≤ 1.6e-9）；每通道行内 std ≈ 0.056~0.066，
  p05/p95 ≈ ∓0.09~0.11，即逐通道 ±10% 量级的 exp(s) 抖动。
- 训练收敛：audit 只有 final train_loss（无逐步曲线）；与根同层 train_loss 对比（上表）显示
  4/6 层联合训练终点损失反而更高 → 在固定 32 步、共享 loss 的耦合目标下，s 的 STE 梯度
  （穿过 `_dense_to_hif4` 的 straight-through）把噪声注入 rotation 优化。

## 机制性解释：A-G1 为什么本地净负

三个叠加因素，按贡献排序：

1. **scale 自由度的校准收益不可迁移（主因）**。Q·diag(e^s)、K·diag(e^-s) 在精确算术下是 gauge，
   收益只能来自量化非线性——即逐通道移动舍入边界去贴合 4+1 个校准窗口的特定取整模式。
   这在未见窗口上系统性反噬（gate 改善与 eval delta 完全反序）。
2. **联合优化拖垮 rotation 本身（次因）**。固定 32 步内多 1024 个 s 参数共享同一目标，
   4/6 层终点 train loss 与 gate loss 都比纯 rotation 差；层 15 直接翻车被拒，
   丢掉根已有的 rotation 收益（−0.0040）。
3. **不是 gate 过拟合单 fold 的锅（在 rotation 上成立的 gate 对 scale 失灵）**。
   根用同一单窗口 gate 定价 rotation 拿到官方 +21；问题在于 scale 类自由度的收益结构
   本身就是窗口特异的，任何 fold 数的 gate 都难以为其定价（除非多折聚合压掉边界噪声）。

## 对下一版 Attention 机制设计的约束

属于**已禁止邻域/同实现重扫**（不应注册）：
- 调小 scale 幅度、加 s 正则、改 LR/步数、多训几步、分步先 rotation 后 scale、
  对 gate 阈值/窗口选择做邻域扫描后重试同一"joint affine gauge"——全部是 v227 这一具体实现的
  参数邻域，规则明确"失败换机制，不扫 threshold/seed/alpha/步数邻域"。
- v223/v224/v225 已关闭 Q/K 正交坐标变换族的 rotation/event/group/seed/block 直接邻域；
  rotation+scale 联合是该族的 affine 扩展，此形态（训练循环内生成长 scale）已随 v227 关闭。

仍开放的**新自由度**：
- 计划已指定的方向：**Q/K 合法 mantissa 舍入边界自由度**（不动连续变换，直接在码本舍入决策上做文章），
  与 v227 的连续 gauge 路径不同族。
- 若未来再引入任何"其收益仅来自量化非线性"的自由度（per-channel scale、码偏移等），
  前置要求：gate 必须多折固定聚合（校准/选择/验证分离规则的既有要求），且先用离线探针证明
  该校准收益跨窗口稳定（正控），否则单窗口 gate 会再次反定价。
- 层 15 现象提示：任何新机制若在根已接受 rotation 的层上改变 rotation 训练轨迹，
  需要保持"rotation 不接受时回退到根的 rotation"而非"回退到 identity"——即新自由度应叠加在
  根的已验证臂之上，而不是替换其训练过程。
