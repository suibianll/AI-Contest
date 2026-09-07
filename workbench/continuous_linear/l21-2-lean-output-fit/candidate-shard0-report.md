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
- 官方 50 样例与校准同分布（用户 21071 证据），本地 eval-v3 holdout 为分布外
  ——本地负向不等同官方负向，但**本地负向损失门未达**（L1_negative 显著
  >0.02：最坏 −0.057），按工作包 §5 有限探索通道不满足自动官方探索条件。

## 4. 裁决

- 候选**本地负向**（shard0 8/48），未达官方探索负向损失门；不推进完整六
  shard，不提交官方；父 L4 不变。
- 不关闭低维拟合族：rb/方向/teacher 变体未穷尽，但按工作包不可现场扫参；
  若用户提供 21071 源码或明确公式，优先绑定复现，否则本地不再重复此类探针。

## 5. 产物

- `workbench/continuous_linear/anchor21-l1/candidate/solution.py`（含低维拟合模块）
- `artifacts/proxy_v3/continuous/linear/anchor21-l1/smoke-check/`（shard0 JSON）
- `workbench/continuous_linear/l21-2-lean-output-fit/lowdim-analysis.md`
- `probe_lowdim*.py`（三种形式探针）