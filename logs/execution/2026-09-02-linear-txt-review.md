# linear.txt（17000+ 版本 Linear 实现）审查记录（2026-09-02）

> 对象：`linear.txt`（1657 行，两个公共 API：`hif4_calibration_and_quantize_weight`、
> `hif4_dynamic_quantize_activation`）。声称提取自"当前 17000+ 版本" solution.py。
> 本记录只做静态审查，未运行（其依赖的 solution.py 不在仓库中，无法独立导入）。

## 1. 结构：非自包含、配套源码不在仓库（阻断性）

- `linear.txt` L23-106 `from solution import (...)`：依赖外部 `solution.py` 提供约 60 个常量
  与函数（含 `_activation_gptq_quantize`、`_loss_capture_ratio`、`_WEIGHT_E2E_REFINE`、
  `_BLOCK_SWAP_ROUNDS`、`_LINEAR_E2E_*`、`_ADAPTIVE_*`、`_ACTIVATION_GPTQ*` 等）。

- 已核对：根 `solution.py` 与归档 v158 均**没有** `_activation_gptq_quantize`、
  `full_sweep_top_k`（`_dense_to_hif4` 无此参数 → 签名不兼容）、`_WEIGHT_E2E_REFINE`、
  `_ACTIVATION_GPTQ`、`_BLOCK_SWAP` 等。→ **linear.txt 不是从本仓库任何现有 solution.py
  提取的**，对应的是另一个（未入库的）17000+ 版 solution.py。

- 含义：单独无法运行也无法评测；按 AGENTS.md 正式提交必须合并成自包含单文件，且需拿到
  配套 solution.py（含 Attention 六个 API）才能做 smoke/评测。

## 2. 逻辑正确性抽查（linear.txt 自身，未见严重 bug）

- 等价变换链（SmoothQuant d → permutation → block-Hadamard → GPTQ/e2e refine）与动态侧
  （smooth\_inv → permutation → block → 量化）顺序一致；动态去平滑用 `smooth_inv` ✓。

- 权重 GPTQ（L764-899）：H = Gram + 岭；Cholesky 逆、块序补偿 `W[rest] -= δ @ H_inv_bb⁻¹ H_inv_bc`
  公式正确，cholesky 失败双倍岭回退 ✓。

- `_weight_e2e_refine`（L560-757）：块输出误差排名 → 逐块试 E6M2 offset → 用增量
  `residuals -= x@δ` 精确评估真实输出 MSE；应用时同步更新 dw/residuals ✓；形状
  scale\_factor\[n\_out,blocks,1,1,1] / lv2..lv3 / mant 布局正确。

- 动态 API 校验完整性：activation\_state 必须 dict、`in_features` 匹配 ✓。

## 3. 风险点（按严重度）

| 级 | 项                                                                                    | 说明                                                                                                                                                                                                          |
| - | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 高 | `activation_state` 携带完整 `h_inv`（in×in 稠密）                                            | Qwen proj 4864² ≈ 95MB×24 层 ≈ GB 级 state，v158 的 state 仅小块（gram64 等）；需核对官方 state 体积/深度上限与 `validate_state` 是否放行                                                                                              |
| 高 | 在线 `hif4_dynamic_quantize_activation` 每 case 执行 `_activation_gptq_quantize`（块序 GPTQ） | 在线成本显著高于 v158 的"一次编码+有限 refine"；AGENTS.md 要求动态只执行编译规则——需确认其确定性/复杂度，并实测单 case 耗时与官方 300s 预算                                                                                                                  |
| 中 | 校准期成本显著上升                                                                            | 每 smooth/perm 候选都做完整权重量化 + 每样本激活量化（约 20-30 次/层）；`_LINEAR_SMOOTH_END_TO_END`(True/hybrid) 每候选做 mm；`_WEIGHT_E2E_REFINE`（topK×offsets×mm）；`_ADAPTIVE_ACT_GPTQ_REG`/`_ADAPTIVE_OFFSETS` 枚举拟合。168 层总时间需 smoke 实测 |
| 中 | `_smooth_scale` 的 soft-clamp 分支依赖未知常量 `_SMOOTH_SOFT_CLAMP*`                          | 配套 solution.py 未入库，行为不可见                                                                                                                                                                                    |

## 4. 相对 v158 的新机制清单

block-swap 8 通道组局部置换（按激活加权 HiF4 精确损失）；权重 e2e refine（E6M2 offset ×
真实输出 MSE）；权重 GPTQ（块序 Hessian 补偿）；激活 GPTQ（ŴᵀŴ Gram + adaptive 岭回归）；
adaptive offsets（校准期选最佳 offset 集；data-driven ratio（按损失捕获比定 refine 上限）；
e2e/hybrid 变换度量。

> 注：若该版本真是官方高分（17000+），其本地 Linear 高分（相对 A1/A2/A3 观测的 v86/v158
> 家族结论）属于不同实现族，上述"变换族伪收益"结论不能直接套用，需拿到配套 solution.py
> 后独立 smoke/配对验证。

## 5. 待办（需用户提供/决策）

1. 提供与 linear.txt 配套的完整 `solution.py`（含 Attention），并确认其版本号/官方成绩；
2. 决定是否合并为单文件自包含候选并分配版本；
3. 合并后按评测流水线：smoke（六 API/合法性）→ `--linear-only --compact-panel` 配对 →
   目标侧 default audit（重点看 h\_inv state 合法性、在线 GPTQ 耗时、官方 300s）。

