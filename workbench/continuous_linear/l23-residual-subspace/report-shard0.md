# L23 残差交叉子空间 A@W 拟合：4B shard0 本地结果

> run_id：`l23-residual-subspace`。日期：2026-09-07。侧：Linear（4B 面板）。
> 契约：[21071工作包 §4](docs/superpowers/archive/plans/21071-evidence-driven-research.md)。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。本地诊断，不提交官方。

## 1. 机制与实现

- 每 64 块：G = Σω Xh_B^T Xh_B + λI；H = Σω Xh_B^T R（残差交叉）；
  Cholesky 白化 H → rank-8 截断 SVD → 解回 ΔW（`_l23_block_solve`）。
- 固定父 scale/lv2/lv3（sf·lv2·lv3 格点），只改 sign/mant；写回仅已接受块。
- fit/select：fold 内 token 奇偶拆；fit（偶数行）加权拼（ω_f 按 fit 行 Y 范数），
  select（奇数行）完整提案 vs 父比较各 fold 归一化误差差值，median<0 严格接受。
- λ 继承父块 ridge（窄 0.2 / 宽 0.3，w=2048）；rank 固定 8；不扫参。

## 2. 数学检查（全部 PASS，`math_check.py`）

- [A] 白化 ΔW 形状 [64,o]，目标比零显著下降（−22.9%）。
- [B] rank→64 收敛到全维 LS（r=32 max 差 5.8e-16）。
- [C] **残差主导方向例子：白化残差子空间目标下降 85.8% vs 旧激活 Gram top-8
  仅 0.01%**——证明 L23 子空间确实依赖残差交叉，数学上不等价于旧实现。

## 3. 4B shard0 结果（56 case，vs L4 paired）

- L23 mean 0.4898 vs L4 0.5089 → Δmean **−0.0191**。
- 按 (layer, role) 配对（候选与父 case 排序可能不同，zip 直对比不可靠）：
  34+ case 改变，**全部 role 负向**：

| role | n_chg | mean Δ gain |
|---|---|---|
| o | 8 | **−0.130** (−0.50 ~ −0.02) |
| proj | 6 | −0.043 |
| k | 4 | −0.027 |
| q | 4 | −0.022 |
| v | 4 | −0.018 |
| fc_gate | 4 | −0.009 |
| fc_up | 4 | −0.004 |

- smem 打印的 select_deltas 全为正样本（median 5e-5~3e-4）：块级 select 个别
  接受（accepted_blocks>0，机制可达），但接受块整体伤 holdout。

## 4. 解读与裁决

- 数学新颖性成立（残差交叉 vs 旧激活 Gram 差异显著），但 **4B 本地 holdout
  全部 role 负向**：低维残差 ~ 格点投影的拟合，在校准 select 上不迁移到
  holdout——与旧 L21/L21-2 结论一致，属"A@W 输出拟合本地各实现序列"的
  再次具体负向。
- 按工作包：**失败只关闭此残差监督子空间实现；不扫 rank/基/teacher/ridge 邻域；
  不否定 A@W 低维拟合整族**。父 L4 不变；未提交官方。
- 时间：smoke CUBLAS 一次偶发失败（父 `_adaround_mantissa`，重跑后正常），非机制。

## 5. 产物

- `workbench/continuous_linear/l23-residual-subspace/math_check.py`
- `workbench/continuous_linear/l23-residual-subspace/dedup-r0.md`
- `workbench/continuous_linear/l23-residual-subspace/candidate/solution.py`
- `artifacts/proxy_v3/continuous/linear/l23-residual-subspace/smoke-shard0/`、
  `.../v189-control-shard0/`
- 本报告