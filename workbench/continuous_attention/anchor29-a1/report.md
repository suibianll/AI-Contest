# A29 实现归档报告 — v163_attention_a29-final-residual-s

日期：2026-09-08 ｜ 代理：Attention（连续研究循环 §7 A-R1）｜ 状态：**实现+验证完成，待负向 L1 检查**

## 1. 身份与派生

- 父：R3 `solutions/v162_attention_r3-rotation-center_allgates/solution.py`（时间父 14405/238s），
  工作树字节 SHA-256 前缀 `A5C679D7A2B3`（CRLF 保留，`gen_solution.py` 字节级校验）。
- 生成物：`solutions/v163_attention_a29-final-residual-s/solution.py`，
  SHA-256 `EC6DDC66FFBD2AF5AE3C843258FC6E1A510B822F8D293415E755EB23614EDCDF`。
- 派生方式：11513 行入口改名 `_r3_a2_calibration_attention`（绑定当时 9235 主校准），尾部追加
  新入口 `hif4_calibration_attention` = R3 全栈逐位 + `_a29_apply`；六 API 定义计数检查通过
  （calibration_attention 2 个定义 = 主校准 + A29 入口）。
- 冻结卡：`mechanism.md` + `config.json`（commit a045459）。S=0 逐位恢复 R3（smoke S4 证）。

## 2. 实现要点（与卡的对应 + 成本模型纠偏）

- **方向（卡 §数学）**：δℓ_i* = argmin_(1ᵀz=0)‖Rout_i − zᵀM_i‖²，M_i=J_iV̂，欠定 min-norm
  直接构造 z_i = M_i(M_iᵀM_i)⁺r_i（批量实现 tmp=pinvA@r → w=tmp@V̂ → dL = p∘w − p(pᵀw)）。
  法方程 L(S)=G_qSG_ek − G_eSG_k = B，L 自伴但不定 → 投影 CG + breakdown 冻结（卡预注册）。
- **成本模型纠偏（偏离卡 §7 文字成本估计，语义不变）**：
  1. per-query-row batched eigh (T_q,256,256) 实测是瓶颈（T=512 单 (g,h) 61.4s；与卡
     "<15s/层=20 次 eigh" 不相容）→ **Q 行均匀采样 32 行**（`_A2_MAX_Q_TOKENS=32` 先例，
     `_a2_even_indices`），数学对象改为行采样帧上的同一 min-norm/法方程，预注册语义不变。
  2. 12 个 (fold,group) CG 实例 **单次堆叠 batched CG**（tensor-mask 冻结，单次末尾同步），
     逐位等价于单实例串行流程（verify_math V1b/V5 对照）。
  3. causal mask 帧修正：采样 Q 行对**全长 K** 做 causal（rows vs arange(T_k)），非采样行间
     互相比较；dL 在非 causal 位置恒 0（J_i 零列保证），B 聚合自动只计合法对。
- **步长（卡唯一 α）**：输出敏感度（|dQ|·‖K̂_g 列‖、|dK|·Σ_h‖Q̂_gh 列‖）加权的 1/64 分位
  首个正向翻码距离，clamp ‖αS‖₂≤log(2)/2。无调节参数。
- **门禁（双 hard-output）**：fold3 严格选择 + fold4 完全独立 holdout，Lhard 为单样本 causal
  部署 MSE；任一不过 → 整层回父（audit 记录，state 逐位=R3）。fold4 不得改方向/步长/配置。
- **audit**：a29_arm / a29_cg(12 实例 residual+iters) / a29_s_fnorm / a29_alpha(+blocks) /
  a29_lhard_parent/prop_f3/f4 / a29_changed_codes_q/k / a29_error(type@行号)。
  attempted/selected/holdout_pass 语义由 a29_arm 值承载（skip/zero_dir/invalid_alpha/
  parent_fold3/parent_fold4/deployed/fallback）。

## 3. 验证链

| 层级 | 结果 |
|---|---|
| verify_math（14 项）| **14/14 PASS**：V1a FD 线性化 rel 2.4e-10；V1b 批量=逐行 pinv 7e-16；V1c min-norm 反解 1e-15；V2a/b exp 互逆/对称 ~1e-15；V2c 连续 QK 编译平移不变 9.4e-4(scale 5.2)；V3a/b/c 旋转复合+编译 center 5.8e-7；V4 码计数；V5a/b/c CG 下降/不变量/breakdown 安全 |
| smoke（合成 5 折）| **S1–S6 全过**：A29 开销 13.95s（重写前 413s 风险）；S=0 非部署逐位恢复；V 逐位；audit finite。合成随机数据 L 不定 → 12 实例全 breakdown/门禁拒绝，符合设计 |
| gate_check（真实 4B proxy，6 层）| 见 §4 |

## 4. gate_check 4B 结果（qwen3.5-4b-proxy-v2.pt，折长 10/128/512/1024/1024）

| layer | arm | overhead | changed codes (q/k) | Lhard f3 parent→prop | Lhard f4 parent→prop | 部署 |
|---|---|---|---|---|---|---|
| 0 | parent_fold3 | 9.1s | — | 5.3371e-4 → 5.3396e-4 ✗ | — | 否 |
| **1** | **deployed** | 14.9s | **25563 / 3406** | **2.2183e-3 → 2.1993e-3 ✓** | **2.0336e-3 → 2.0292e-3 ✓** | **是**(3 张量) |
| 5 | parent_fold3 | 9.4s | — | 2.0249e-3 → 2.0286e-3 ✗ | — | 否 |
| 8 | parent_fold3 | 6.8s | — | 7.5581e-3 → 7.6179e-3 ✗ | — | 否 |
| 15 | parent_fold3 | 7.7s | — | 2.7959e-3 → 2.8014e-3 ✗ | — | 否 |
| 22 | parent_fold4 | 7.7s | 25989 / 3776 | 1.78941e-2 → 1.78932e-2 ✓ | 1.77251e-2 → 1.77308e-2 ✗ | 否 |

- **机制存活判定**：S* 非零（fnorm≈0.58）、changed codes>0、layer 1 双门禁严格通过 → 四条
  失败分支（δℓ*/S* 零、码零、训练降但 fold3/4 反转）均未触发；layer 22 正是"fold3 改善、
  fold4 反转→回父"的预注册行为，holdout 门禁有效。
- CG 在真实数据上仍 0 收敛 / 5–9 早冻结（L 不定是常态）：方向来自 1–2 步 CG + top-4
  重构 + 折平均，由双 hard 门禁兜底——与卡"投影 CG + breakdown 冻结"一致。
- 非部署层 state_tensor_diffs_vs_r3 = 0（逐位控制 ✓）。

## 5. 风险与下一步

1. **官方时间风险（主要）**：4B 面板 A29 开销 6.8–14.9s/层（6 层共 ~55s，约为 R3 校准的
   ~125%）。官方 40 层 hybrid 仅 10 层 softmax 注意力状态层，但官方折大小未知；投影不确定。
   卡 TIMEOUT 分支：只关闭该校准复杂度实现（不改机制判定）。
2. **负向检查（官方探索前最后一道）**：eval-v3 attention-only v163 vs v162（4B cache），
   判据 L1 < 0.02 且非 no-op（layer 1 已证非 no-op）；72 例 Δmean 不预测官方符号、微负不停交。
3. 通过后固定代表并官方探索；证伪按卡五分支处理。
4. 后续队列：A30（闭式逐通道互逆输出能量平衡）→ A31（三角误差搬运）；A32 需单独授权。

## 6. 工程记录

- 关键修复：causal 帧修正（rows vs arange(T_k)）、apply_batched 改直接 batched matmul
  （旧 einsum "bie,bie->be" 把矩阵乘收缩成 (12,256) 导致 CG 内 256-vs-12 广播错）、
  `_a29_solve_group_S(S)` 签名收敛、a29_error 带 traceback 行号。
- 教训：**同一文件多处 Edit 禁止并行批次**（读-改-写竞态会静默丢失编辑，本日两次踩中，
  已靠 grep 全量核对 + 串行重做恢复）。
- 提交链：a045459（冻结卡）→ b2b8597（v3 实现+smoke+verify_math）→ 本报告 + gate_check 产物。
