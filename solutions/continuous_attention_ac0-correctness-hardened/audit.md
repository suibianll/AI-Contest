# AC0 Correctness Hardening — 审计文件 (audit.md)

> 日期：2026-09-08。归档：`solutions/continuous_attention_ac0-correctness-hardened/`。
> 父：R3 `solutions/v162_attention_r3-rotation-center_allgates/solution.py`
> （SHA256 `A5C679D7A2B349A879B2019B4A613244F5FEA7050E407B6475B9E29BD1C146DC`，
> 官方 14405 / 238s）。AC0 SHA256：
> `F817E4C24CAAA8D1325A5A0045F8A0F057EE0DC5EB4C0B7167298FE67BB4F5A2`。
> 本文件按「已确认正确 / 已发现并修复 / 保留风险」三节记录，验收证据一律为实测，
> 不使用 "probably correct / should be fine" 作为依据。

## 1. 已确认正确（confirmed correct）

以下各项经 correctness battery（`correctness_battery.py`，30/30 PASS）或真实 4B
校准证据验证：

| 项 | 证据 |
|---|---|
| `_attention_forward` GQA/logits/mask/softmax 语义 | 代码审阅 + T6：logits=QKᵀ/√d，causal=triu(…,1)=−inf，K/V `repeat_interleave(group)`，softmax over keys。与 evaluator `v2._attention` 一致（proxy_v3_eval L448-459 同公式） |
| K-center softmax 不变性 | T4：FP64 `maxerr=5.6e-15 < 1e-8` |
| 正交旋转 Q'K'ᵀ=QKᵀ（FP64 数学） | T2：`rel=~1e-16 < 1e-8`；部署 float32 路径实测 `rel≈2.9e-7`（实现舍入，非数学误差） |
| 逆变换不变性 Q'=QT, K'=KT⁻ᵀ | T3：`rel<1e-8` |
| center compile（先 center 后 transform == transform 后 transformed center） | T5：`rel=3.7e-16 < 1e-12` |
| 训练/部署五字段逐位一致（同一输入+state） | T7：Q/K/V 五字段（scale_factor/scale_lv2/scale_lv3/sign/mant）`torch.equal` 全部成立；训练 hard forward 现在直接调用部署 API |
| 身份 parity（禁用 learned 后 AC0 == R3） | T1：Q/K/V 五字段逐位一致 |
| 真实 4B 六层 FP64 QK-invariance audit | 校准产物（见 §2 表）：全部层 `valid=True`，误差 5.23e-07 ~ 6.11e-07 < 1e-6 |
| 已安装 rotation 正交性 | 全部层 ortho ≈ 7.15e-07 ~ 8.34e-07 << `_A2_ORTHO_TOLERANCE=1e-3` |
| state/params 合法性契约 | T9 + 评测器 `validate_state` / `validate_hif4_params` 全过（CPU、finite、无 grad、合法 dtype、深度/节点上限） |
| 部署 transform 顺序 | `_attention_transform_dense_reference` 与 `_nvfp4_to_hif4` 逐操作一致（T7 逐位证明），顺序：NVFP4 dequant → K center → multiplier → permutation → attention rotation → block smooth → pair transform → learned rotation → learned center(K) → HiF4 encode |

## 2. 已发现并修复（found & fixed）

### F1. Q/K coupled transform 静默失败（最优先）

- **旧行为**：`_nvfp4_to_hif4` 中 `learned_rotation` 与 `learned_center` 各自包在
  独立 `try: … except Exception: pass` 里（R3 L4269-4288）。rotation 失败而 center
  成功（或反之）时，dense 停在半变换状态，且完全无记录。
- **失败模式**：Q'K'ᵀ ≠ QKᵀ 可能被静默接受；Q/K 侧变换不一致。
- **新行为**：单一耦合 try 块；任一失败 → `coupled_attention_transform_valid=False`，
  `dense` 原子回退到 parent（未学习变换前）状态，事件写入模块级
  `_AC0_FALLBACK_EVENTS`（含 `fallback_reason`），不再静默。`_learned_strict=True`
  时（校准/test 语境）直接 raise。
- **测试证据**：T8（Q 坏 rotation → parent 输出；K 坏 rotation → parent 输出；
  K 坏 center → parent 输出；事件记录 `events=2` 且含 reason；好 state 不回退）；
  smoke 中 `STRICT_MODE_RAISED_OK`。

### F2. 训练 hard forward 与部署路径不一致（trainer/deployment mismatch）

- **旧行为**：`_a2_train_rotation` 用 `_dense_to_hif4(q_rot)`（标准编码器）做
  hard forward——没有 multiplier/permutation/rotation/block_smooth/pair_transform
  栈、没有 importance/offsets/refine、没有 learned-order；V 用 `_dense_to_hif4(v)`
  而非部署 `hif4_dynamic_quantize_v`；rotation 施加在原始坐标，部署却在 v189 栈之后。
- **失败模式**：训练目标 ≠ 部署目标（objective mismatch）；训练出的 rotation/center
  针对错误坐标系优化。
- **新行为**：训练 hard forward 改为候选 state（parent state + learned 覆盖）调用
  真实 `hif4_dynamic_quantize_q/k` 与 `hif4_dynamic_quantize_v`，逐位与部署一致；
  rotation 施加位置与部署一致（栈末）；梯度通过统一 reference
  `_attention_state_transform_dense`（= `_attention_transform_dense_reference`，
  include_learned=False）计算 rotation 输入。mse_std 分母改用部署 parent 路径
  （与 gate 分母语义一致）。V 走真实部署路径。
- **测试证据**：T7 五字段逐位一致；T1 身份 parity 未破坏非 learned 路径。
- **行为影响**：训练收敛点改变（各层 rotation 与 R3 不同，maxdiff ~0.4），
  属「原 R3 错误路径被修正」的可解释差异。

### F3. 变换顺序不统一 / 第二套近似 stack

- **旧行为**：`_attention_state_transform_dense`（训练侧 mirror）不含 learned
  transforms，与部署顺序无显式契约。
- **新行为**：新增统一 helper `_attention_transform_dense_reference(...)` 为唯一
  dense-reference（含 learned rotation/center 的部署位置）；`_attention_state_transform_dense`
  委托给该 reference（include_learned=False）。校准/测试全部走同一逻辑。
- **测试证据**：T7（reference+encode == 部署 API 五字段）。

### F4. 缺失 shape 校验（GQA / rotation / center）

- **旧行为**：`_a2_apply_group_rotation` 无 shape 校验（错误 shape 被 F1 的
  try/except 吞掉）；`_check_attention_state` 不校验 learned 字段。
- **新行为**：
  - `_a2_apply_group_rotation`：要求 rotation 3D、`num_heads % groups == 0`、
    `shape == (groups, head_dim, head_dim)`，否则 `ValueError`。
  - `_check_attention_state`：校验 learned_rotation（3D、groups 整除 num_heads、
    K 侧 groups == kv_num_heads）、learned_center `(num_heads, head_dim)`。
  - 校准期 `_validate_learned_qk_pair`：GQA `q_heads % kv_heads == 0`、
    Q/K rotation 相同、CPU/finite/no-grad、shape、正交性 ≤ `_A2_ORTHO_TOLERANCE`。
- **测试证据**：T6（q=16/kv=4、14/2、8/8 GQA 映射正确；非整除 groups 被拒）；
  T8；smoke `FALLBACK_RECORDED` reason 显示 shape 校验触发路径。

### F5. FP64 QK-invariance audit 缺失

- **旧行为**：无任何「学习变换确实保持 QK^T（mod per-query shift）」的校准期检查。
- **新行为**：`_ac0_qk_invariance_audit`——对 gate 窗口 dense Q/K，比较
  include_learned=False vs True 的 GQA logits 矩阵（per-head Q_h K_gᵀ，K repeat）
  减去 key 均值；相对 Frobenius 误差 < 1e-6 才有效（FP64）。失败 →
  `candidate_invalid=True` → 整个 proposal 回退 parent（identity arm），理由写入 state。
- **测试证据**：真实 4B 六层全部 `valid=True`（见 §2 表）；battery 对合成 rotation
  复核通过。

### F6. 校准失败静默回退无记录

- **旧行为**：`hif4_calibration_attention` 的 except 分支只置 `a2_arm="fallback"`，
  无原因记录。
- **新行为**：`a2_fallback_reason = "<ExcType>: <msg>"` 写入 q/k state audit。
- **测试证据**：smoke 中 `FALLBACK_REASON` 字段返回实际异常文本。

### 真实 4B 校准证据表（AC0，来自评测器校准缓存，与评测运行逐位一致）

| layer | AC0 arm | gate id→rot | inv valid | inv err | ortho | R3 arm（4B 缓存） | 备注 |
|---|---|---|---|---|---|---|---|
| 0 | rotation | 1.0000→0.9457 | True | 6.11e-07 | 7.75e-07 | rotation (0.9533) | 一致 |
| 1 | rotation | 1.0000→0.9484 | True | 5.84e-07 | 7.75e-07 | rotation (0.8866) | 一致 |
| 5 | rotation | 1.0000→0.9229 | True | 5.23e-07 | 8.34e-07 | rotation (0.9177) | 一致 |
| 8 | identity | 1.0000→1.0287 | True | 5.27e-07 | 7.15e-07 | identity (1.0008) | 一致 |
| 15 | rotation | 1.0000→0.9973 | True | 5.41e-07 | 7.15e-07 | identity (1.0396) | **边际翻转**（0.997 vs 1.040） |
| 22 | rotation | 1.0000→0.9944 | True | 5.48e-07 | 7.15e-07 | rotation (0.9933) | 一致 |

5/6 层 arm 与 R3 一致；L15 由训练路径修正后从 identity 翻转为 marginal rotation
（gate 0.9973，接近 1.0），是本轮唯一的行为分叉层。

## 3. 保留风险（retained risks，明确标注）

1. **UNVERIFIED — 校准对输入 device 敏感（CPU vs CUDA pairs）**。同一窗口 NVFP4
   pairs 在 CPU 传入与 CUDA 传入时，v189 栈校准（K-center 求解、pair-matrix 拟合等
   的 CPU/CUDA 舍入差异）可导致边际层（如 L15）gate 决策翻转
   （实测 CPU pairs gate=1.0281 → identity；CUDA pairs gate=0.9973 → rotation，
   逐位可复现）。此为 R3 既有特性（非 AC0 引入）；评测器始终传 CUDA pairs，
   官方行为以 CUDA 路径为准。AC0 不改变该行为（§3 禁止改算法数学）。
2. **UNVERIFIED — 边际 gate（≈1.0）的 arm 决策稳定性**。L15 的 gate 比值 0.9973
   接近阈值，任何等价实现层面的数值扰动都可能翻转。已在 audit 中记录，A29 阶段
   若在 L15 观测到对 rotation 参数敏感的收益，需先复核该层 gate 稳健性。
3. **已记录（不阻塞）— 训练收敛点与 R3 不同**。parity 修正后各层 rotation/center
   与 R3 不同（maxdiff ~0.4），这是「错误路径被修正」的直接后果；本地 paired
   评测总体 +0.0035（72 case），无系统性退化证据。
4. **UNVERIFIED — 官方时间**。AC0 校准在 R3 基础上增加：训练期部署路径编码
   （~32 步×4 窗×2 侧）、FP64 audit（≤256 行采样）、pair 校验。shard 校准
   api≈10s/层（4B 单层）。官方 300s 硬限以官方回传为准；本地时间仅记录
   （4B 指引：本地时间不设门禁）。
5. **已确认无风险 — 历史归档未动**。R3/A2/A22/A23 及全部 `solutions/*` 历史
   `solution.py` SHA 保持不变（R3 复核 `A5C679D7…`）；根 `solution.py`（v189）
   未修改。

## 4. 结论

AC0 = R3 数学行为 + implementation hardening。除 L15 边际 gate 翻转外，所有层
arm 决策一致；FP64 QK-invariance 审计在真实 4B 数据上全部通过（< 1e-6），证明
R3 的 rotation/center 机制在修正后的训练路径下确实保持 QK 语义。AC0 可作为后续
Attention 新算法（A29）的唯一实现父。