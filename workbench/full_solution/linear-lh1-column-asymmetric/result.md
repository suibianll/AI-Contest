# L-H1 逐列非对称权重量化 — 第 0 步预检记录与关闭（2026-09-09）

- run_id: `linear-lh1-column-asymmetric`
- 结果：**预检 B 未通过 → `NO_DISTINCT_RESIDUAL`，按计划关闭，未制作 4B 候选，未运行任何 shard 评测。**
- 预检脚本：`precheck.py`（固定 seed 20260909，CPU，无调参）；结果 JSON：`precheck.json`。
- 根 `solution.py` 未修改；未创建 candidate。

## 预检定义（按计划固定）

- "逐列非对称"：同一输入列内正权重和负权重允许选择不同 magnitude code，但最终只输出合法五字段
  （scale_factor/scale_lv2/scale_lv3/sign/mant），不引入正负双 scale、zero-point 或自定义解码。
- 合成块：含正负权重的 64×64 块（W ~ N(0,1) 混合符号）；校准激活 X 为相关高斯
  （随机混合矩阵 + 逐列 lognormal 尺度），2 窗 × 256 行。

## 预检 A（表达性）：EXPRESSIBLE

- 对根 `_dense_to_hif4` 编码的一列做非对称码赋值（正元素 mant=1.75、负元素 mant=0.75，
  sign 与符号一致、mant=0 时 sign=0 保持 canonical zero），其余字段不变。
- 候选五字段通过 `evaluator/reference_hif4.py::validate_hif4_params`；相对根编码有 56 个元素
  解码值不同且全部合法。该方向不需要五字段无法保存的正负双 scale。

## 预检 B（残余空间）：NO_DISTINCT_RESIDUAL

流程：根完整 `hif4_calibration_and_quantize_weight`（含 smooth/perm/block-smooth/rank-2 残差
变换 + Weight GPTQ）→ 冻结最终部署 activation state → 用根动态 API 得 Q(XR)，重构部署坐标
W′（连乘积不变性 X′W′ᵀ≈XWᵀ 已验证，rel 1.9e-3，系 NVFP4/BF16 精度）→ 完整输出目标
`||XWᵀ − Q(XR)Q(WR^{−T})ᵀ||²`（base loss 32168.96）→ 枚举全部 64 列 × 结构化 (m_pos, m_neg)
∈ {0..7}² 共 4096 个合法状态（冻结根 scale/lv2/lv3 层级）。

证据数字：

- **0/4096 个结构化状态严格改善完整输出目标**（其中非对称 m_pos≠m_neg 也是 0）。
- 最优结构化状态相对根仍 **+5826.49（+18.1%）更劣**（best column 61, m_pos=0.75, m_neg=0.5）。
- 标量公式正确性已独立验证（直接重算 28810.410 vs 公式 28810.414，rel 1e-7 级）。
- 诊断（非裁决项）：逐元素 mant±1 邻码在完整目标下确有 643/7467 个改善状态，但最优仅
  −0.295% 相对改善，且该形态正是 AW8 的"无结构逐码"形态（已关闭：校准集过拟合、
  holdout 方向为负、3.8× 成本）。即：完整目标与根 proxy 目标的差距存在但极小，
  且只能由已关闭的逐码自由度触及；L-H1 的结构化逐列非对称族残余空间为零。

## 关闭依据

1. 计划 L-H1 第 0 步残余空间预检明确："未找到反例则记 `NO_DISTINCT_RESIDUAL` 并关闭，
   不制作 4B 候选"。结构化枚举（该方向实际实现将使用的搜索族：每列正/负各一个 magnitude
   code + 共享层级 scale）在合成块上零改善 → 记 `NO_DISTINCT_RESIDUAL`。
2. 与 AW 族九连败归因一致（`logs/execution/2026-09-09-aw-fitting-family-analysis.md`）：
   根 Weight GPTQ 已是 Hessian 度量下的逐列 A@W 拟合 + 误差反馈，事后残余空间极小；
   唯一能改善完整目标的是逐元素码自由（AW8 形态，已证过拟合）。
3. 本关闭只针对"L-H1 结构化逐列非对称编码"这一具体实现形态，不推广为
   "合法编码空间已饱和"（AGENTS.md §7 cb1/cb2 边界不变）。

## 产物

- `workbench/full_solution/linear-lh1-column-asymmetric/precheck.py`
- `workbench/full_solution/linear-lh1-column-asymmetric/precheck.json`
- 本文件 `result.md`
