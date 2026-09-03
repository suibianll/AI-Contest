# v174 候选：Kronecker 压缩解析 CAT（L4）Linear（未提交/未知官方） + 标准 Attention

> 状态：**ARCHIVED — 官方未提交，`scoreNA / timeNA`**
>
> 工作包：`docs/superpowers/plans/2026-09-03-post-v162-low-complexity-algorithm-expansion-plan.md` §11
> （Linear 工作包 L4：Kronecker 压缩的解析 CAT）。计划执行顺序：A1/A2/A3/A4 已收官或
> 已 REJECTED，L1 审计判 DUPLICATE 概念、L2/L3 在 v172/v173 处理；本包为 L4。
>
> 父版本：v166（rank-1 Linear + standard Attention），SHA
> `9C0EAC6A7CA883A1F8962C11735744271259460F5EBBF23D530A5BBCF12B4646`，官方 `4590 / 226s`。
> L4 从 v166 复制，只改 `hif4_calibration_and_quantize_weight` 与
> `hif4_dynamic_quantize_activation` 及其直接 helper；四个 Attention API 与 v162 standard
> 逐位一致。

## 1. 唯一算法机制（预注册，§11）

把完整 64×64 解析 CAT target 投影成两个 8×8 因子，`R = R_left ⊗ R_right`（每 64-block 固定
8×8 reshape，非 rank 扫描、非 Householder）：

```text
M     = _cat64_blocks(activation_covariance, weight_gram, strength=0.25)   # SPD [blocks,64,64]
G     = logm(M)                                  # 经 eigh 特征幂
A_raw = partial_trace_right(G) / 8 = einsum("ipjp->ij", G4) / 8
B_raw = partial_trace_left(G)  / 8 = einsum("ipiq->pq", G4) / 8
A     = A_raw - trace(A_raw)/8 * I
B     = B_raw - trace(B_raw)/8 * I
R_left  = expm(A)                                 # torch.linalg.matrix_exp
R_right = expm(B)
# det 归一：每 8x8 因子除以 det^(1/8)，使 det(R_left(x)R_right)=1
```

- **应用**：每个连续 64 向量 reshape 为 8×8（flat `i*8+j` 布局），`V' = R_left V R_right`
  （`R_right` 对称，等价于 `R_left ⊗ R_right` 作用于扁平向量），动态成本两次 batched 8×8 乘
  （O(16)/block，相对 dense CAT 的 O(64²)）。**Weight 使用精确逆因子（`inverse=True` =
  `inv(R_left)`, `inv(R_right)`）**，**Activation 使用正因子（`inverse=False`）**。

- **连续域乘积不变**（xF 对称）：`A' = A F`、`W' = F^{-1} W`，故
  `A' W'^T = (A F) ((F^{-1}) W)^T = A W^T`。全局链 smooth/perm/Hadamard + rank-1 + Kron 各自
  乘积不变，叠加后 `deploy(A)·deploy(W)^T = A·W^T` 保持。

- **校准插入点**：v166 的 rank-1 残差重分布之后、权重编码之前（部署坐标系最后一环）。用部署
  样本构造 CAT target `M`（`_KRON64_CALIB_ROWS`=256 激活行、`_KRON64_WEIGHT_GRAM_ROWS`=1024
  权重行），投影得因子后重绑定所有部署 operand：`weight_smooth`（逆）、
  `transformed_activation_samples`（正）、`gram_full`（`F C F^T`）、`weight_group_gram`、
  `h_x_smooth`（`diag(F diag(h) F^T)` 平方因子规则，或从变换后样本重算）；下游 GPTQ 编码与
  activation importance/gram/h_inv 全部在 Kron 坐标系由父流程自身一次性计算。

- **动态路径**：`hif4_dynamic_quantize_activation` 在 permutation/Hadamard + rank-1 之后、
  HiF4 encode 之前调一次 `_apply_kron64_rows(dense, kron_left, kron_right, inverse=False)`。

- **不拟合 strength、不搜索因子形状**（`_KRON64_STRENGTH` 固定 0.25）；AT 状态新增
  `kron_left`/`kron_right`（CPU float32 `[blocks,8,8]`），version 4→5。Kronecker-sum 仅是
  `logm(M)` 的投影（非精确重建），这是 L4 的压缩代价，非正确性缺陷。

## 2. 本地验证（描述性，官方裁决）

| 项目                                   | 结果                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 隔离导入 + 六 API（临时目录，脱离仓库）        | OK，六 API 可调用；`validate_state`/`validate_hif4_params` 通过                                            |
| 连续域乘积不变 `A' W'^T = A W^T`（合成随机 64 向量）| 最坏相对误差 **3.361e-07**（要求 < 1e-3）                                                                    |
| 相同检查（真实校准因子，28 layer-role 扫描）    | 每 state 重扫最坏相对误差 **2.709e-07**                                                                    |
| Kron 因子非恒等比例（真实校准）               | **640/640 = 100%** 64-block 非恒等；28/28 state 均含 ≥1 非恒等块；0 个 no-op 层；mean per-8×8 `\|\|F−I\|\|` = 0.0156 |
| 因子 det 归一（合成）                      | 最大 `\|det-1\|` = 9.5e-07                                                                                |
| linear compact 56（配对 v166 compact smoke）| **0.702896**（父 0.705508）；paired mean **−0.002733**、median **+0.001191**、`34+/22−/0=`（非时间门禁，纯描述）   |
| API total / wall（28 state 校准 + 56 case 打分）| API 63.240s、wall 68.364s（每 state 校准 ~2.3s，在 2–5s 目标内）                                          |
| more details                            | per-role 最坏 case、cross-holdout 28/28 同号、W-only/A-only/interaction 见 `v174-compact-linear.json`         |

时间归因：Kron 每 state 校准相较父 v166 增加约 +0.5s（本地 CUDA，含 CAT target eigh×2 +
因子推导 + 部署 operand 重绑定）。本地 CUDA 时间对官方时间不可预测（AGENTS §1 已记失效），
官方时间由真实提交裁决，不作为本地拒绝依据。

## 3. 判读与 version

- 待官方首批回传后判读：`step_gain = S(v174) − 4590`、`side_contrib = S(v174) − 1001`、
  `Linear ratio = (S(v174) − 4587)/3586`（v160 固定口径）。`S > 4590` 且 `<300s` → RETAINED
  成为新 Linear 父侧；否则 REJECTED；`>300s` → TIMEOUT。
- version：activation_state `4 → 5`；state 新增 `kron_left`/`kron_right`。
- 控制：Attention 四 API 未被触碰；linear compact 内 Attention 臂未运行（0 case）。

## 4. 证据

- 源码 SHA256（归档 `solution.py`）：`2A5B74161A8C606DE0DC05E2735BDC51E3CD11D2A4DA910211049001E6F84A7D`
- 父 v166 SHA256：`9C0EAC6A7CA883A1F8962C11735744271259460F5EBBF23D530A5BBCF12B4646`
- compact 结果：`artifacts/official_eval/v174-compact-linear.json`、`logs/official_eval/v174-compact-linear.md`
- 设计：活动计划 §11