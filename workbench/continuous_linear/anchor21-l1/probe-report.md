# L21-1 真实闭环探针：逐列条件求解方向证据

> run_id：`anchor21-l1`。侧：Linear。日期：2026-09-07。
> 契约：[linear-output-followthrough.md](../../docs/superpowers/plans/workpackages/linear-output-followthrough.md) L21-1。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。本探针为本地诊断，不提交官方。

## 1. 数学最小检查（全部 PASS）

`math_check.py`：
- [A] 逐列 OBQ 条件补偿与"固定列后直接条件重解"**逐位一致**（非对角 H，
  lam=0.1 max 差 0.00e+00）。
- [B] L21-1 条件补偿与旧 JDRQ ±一格局部下降**给出不同格点**
  （max|W_l21 − W_jdrq| = 0.35）：公式级区别成立。
- [C] 奇异块（Xh=0）：加 ridge 后 H 有限、可逆 PASS。

## 2. 真实闭环探针（`probe_real_loop.py`，维度修正版）

- 输入：官方 nvfp4_encode + 父真实 `hif4_calibration_and_quantize_weight` +
  真实 `hif4_dynamic_quantize_activation`（L-R1 同源，12/12 一致）。
- 权重 [o, in]；块沿 in 维 64 列；固定父 scale/lv2/lv3，只改 sign/mant；
  合法值 scale·{−1.75,…,1.75}，写回五字段；λ=0.2（父窄层 ridge 同语义）。
- 逐块两臂：块提案 vs 保持，用真实校准目标比较，严格改善才替换。
- 过拟合对照：训练=全部 fold 加权（ω_f），独立验证= fold-1 + test holdout。

| state | L_tr 起点→终点 | changed | L_val(父→端点) | test holdout 父→新 | rel |
|---|---|---|---|---|---|
| L0-o | 1.29e-6 → 1.60e-7 | 13/14 | 1.41e-7 → 3.45e-8 | 2.08e-7 → 2.66e-7 | **1.28 DEGRADE** |
| L11-proj | 2.88e-6 → 1.29e-8 | 76/76 | 1.52e-4 → 1.27e-6 | 1.83e-4 → 3.25e-4 | **1.78 DEGRADE** |
| L0-fc_up | 5.73e-7 → 5.27e-8 | 14/14 | 7.16e-4 → 1.32e-4 | 9.44e-4 → 1.47e-3 | **1.56 DEGRADE** |

## 3. 解读

- **可达性成立**：attempted=全块、accepted>0（13~76 块全接受）、合法格点校验通过、
  finite、不扫参。
- **校准 fold 上 L 大幅改善**（过拟合方向，逐列 OBQ 精确求解器天然如此）。
- **独立 test holdout 3/3 统一负向（+28%~+78%）**；fold-1 验证正向 → 现象是
  "校准窗口分布内拟合，test window 分布外退化"——权重吸收了校准激活分布
  的量化误差结构（A@W 输出补偿），跨窗口不复用。
- 与旧 JDRQ 的 112-case 负向一致；与 AGENTS"A@W 拟合掩盖量化误差于分布依赖
  结构中会反向"的历史裁决一致。

## 4. 裁决

- **本地真实闭环负向（3/3 holdout 退化），不构成可直接晋级候选**。
- 按工作包 §4 失败分支：**不闭关闭固定层级一遍求解卡**（仅 3 代表 state，
  不能扩展为全族否定）；不提交官方、不扫列序/步数；下一步选择：
  (a) 完整六 shard 验证（若协调者要求本地稳健证据）；
  (b) 分析固定 activation/层级限制 → L21-2 精简主干（删旧训练/状态依赖，
      不混 mean）；
  (c) 保留卡，等新机制证据。
- 父 L4 不变，未触发提交。

## 5. 产物

- `workbench/continuous_linear/anchor21-l1/math_check.py`
- `workbench/continuous_linear/anchor21-l1/probe_real_loop.py`
- 本报告 `workbench/continuous_linear/anchor21-l1/probe-report.md`