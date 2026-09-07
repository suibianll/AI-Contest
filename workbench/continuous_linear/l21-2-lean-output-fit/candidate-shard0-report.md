# L21-2 低维 A@W 拟合完整候选：shard0 本地结果

> run_id：`l21-2-lowdim-candidate`。日期：2026-09-07。侧：Linear。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。

## 1. 候选实现

在 L4 上复制为候选，静态 actorder 校准返回前插入 `_lowdim_aw_fit_weight`：
- 每校准 fold 经真实 dynamic activation 得 Xh；teacher = X·W_origᵀ；
- 每 64 块：块内 Gram `G_B=XhᵀXh` eigh top-8 特征方向 U；闭式
  `ΔB=solve(AᵀA+λI, AᵀR)`；`W_D=W_B+ΔB·Uᵀ` → snap 回父 scale/lv2/lv3 格点
  （15 codes，只改 sign/mant）；块两臂真实校准目标 L 严格改善才替换；
- 写回合法五字段（固定 scale/lv2/lv3，sign/mant round(clamp 7)*0.25）。
- `_LOWDIM_AW_FIT=True, _LOWDIM_RB=8`。

修复历史：mant 写回 `.round()` 需先 `*4`（银行家舍入 0.5→0 曾导致 -60 灾难；修正后合法）。

## 2. shard0（56 case，default panel）vs L4

| 度量 | 值 |
|---|---|
| cand mean | 0.610599 |
| L4 mean | 0.628479 |
| Δgain | **−0.017880** |
| 正向/负向 case | 8 / 48 |
| 最大 +Δ | +0.0078 |
| 最坏 −Δ | −0.0573 |

role 分组 Δmean：q −0.0085、k −0.0057、v −0.0068、o −0.0122、fc_gate −0.0256、
fc_up −0.0289、proj −0.0375 —— 全部负向，无正向 role。

## 3. 解读与对照

- 与全维 L21（rel 1.3~1.8，即 Δ 约 −0.1~−0.2 量级）相比，低维 rb=8 幅度温和
  （−0.018），方向正确但收敛不足；o 层单卡探针曾 −1.3% 改善，但 shard 层面
  全 role 平均仍负。
- v2（完整重量化）、v3（权重 SVD 全局低秩）均更差；v1（逐块激活 Gram + snap
  父格点 + rb=8）是最优形式。
- 官方 50 样例与校准同源是未证实假设（纠偏记录 §1）；本地 eval-v3 holdout 独立窗口
  不等同 OOD。本地负向只覆盖此具体实现。
- **负向门核算纠偏**（见 [21071纠偏记录 §3](../../logs/execution/2026-09-07-21071-next-cycle-evidence-correction.md)）：
  原报告"最坏 case −0.0573 证明平均负向损失>0.02"口径错误。按报告四舍五入值边界复核：
  `mean(max(-delta,0)) ≤ 0.017880 + 8*0.0078/56 = 0.018994286 < 0.02`。
  这是边界上限复核，不是六 shard 通过证明；原 shard 负向事实不变（8+/48−），
  L1_negative<0.02 边界未超，不据此保留或撤销任何卡，也不自动重跑旧候选。

## 4. 裁决

- 候选**本地负向**（shard0 8+/48−）；按纠偏 §3，平均负向损失边界
  ≤0.01899 未超 0.02，但单个 shard 的负向事实仍成立，**不构成可晋级本地证据**；
  不推进完整六 shard、不提交官方；父 L4 不变。
- 按工作包，旧逐块激活 Gram / 块一次 / 权重 SVD 等具体实现负向只关闭对应实现，
  不否定 A@W 低维拟合整族；L23 改为残差交叉子空间构造（白化 H 的 rank-8 SVD），
  并非本报告的激活 Gram 方向实现，需按新卡另做去重与实现。
- 不关闭低维拟合族，不扫 rank/基/teacher/正则邻域；若用户提供 21071 源码或
  明确公式，优先绑定复现。

## 5. 产物

- `workbench/continuous_linear/anchor21-l1/candidate/solution.py`（含低维拟合模块）
- `artifacts/proxy_v3/continuous/linear/anchor21-l1/smoke-check/`（shard0 JSON）
- `workbench/continuous_linear/l21-2-lean-output-fit/lowdim-analysis.md`
- `probe_lowdim*.py`（三种形式探针）