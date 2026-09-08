# A29 机制卡：最终输出残差驱动的量化边界 Q/K 互逆补偿

依据：[2026-09-08 持续研究循环 §7 A-R1](../../../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md)。
前置：[F2 深度归因](f2_deep_attribution.json)（P0，零 API）+ [A28 解释纠偏](../../../logs/execution/2026-09-08-a28-interpretation-correction.md)。

## 1. 靶点

F4 = 0.468（最终输出 gain 缺口，主目标）。**不再优化 Q/K operand MSE**——利用 QK 项与 V 项的负交叉补偿直接压最终硬输出 MSE。账本支持：QK 份额 L15 0.317 / L1 0.255 / L8 0.227 / L0 0.166（[F2 归因]）；L22 为 V 主导（QK 0.035），A29 在 L22 预期 no-op、gate 应拒绝（预注册预期，不建路由）。

## 2. 改变什么

在**时间父 R3**（14405/238s，SHA `A5C679D7...146DC`）的完整部署栈（含全部已验证旧训练 rotation+center）最终连续坐标上，附 per-GQA 组对称零迹矩阵 `S_g ∈ R^(256×256)`：

```text
Q⁺ = Q̃·exp(S_g),  K⁺ = K̃·exp(−S_g),  center 同步编译 c' = c·exp(−S_g)
连续 logits：Q⁺(K⁺−c')ᵀ = Q̃exp(S)exp(−S)(K̃−c)ᵀ = Q̃(K̃−c)ᵀ  —— 严格不变
```

部署零新增算子：S 合入 state["learned_rotation"] 输出侧（`new_rot = old_rot @ exp(±S)`，行向量右乘约定）与 learned_center（`c' = c·exp(−S)`）；`S=0` 逐位恢复 R3。

**方向来源（一次闭式，零迭代优化）**：每训练 fold × GQA 组，
```text
Q̂,K̂,V̂ = 真实五字段硬解码（父 state）；P0 = softmax(Q̂K̂ᵀ/√d + mask)；Y0 = P0V̂
Rout = Yref − Y0；J_i = diag(p_i) − p_i p_iᵀ；M_i = J_i V̂
δℓ_i* = argmin_(1ᵀz=0) ||Rout_i − zᵀM_i||²   （minimum-norm 伪逆，rcond=1e-6，RMS 限幅至父 logit 误差 RMS）
A_g(S) = (Q̃ S E_kᵀ − E_q S K̃ᵀ)/√d（E_q=Q̂−Q̃, E_k=K̂−K̃ 为父量化残差）
S* = argmin_(S=Sᵀ,trS=0) Σ_i ||δℓ_i* − A_g(S)_i||²   （法方程 G_q S G_ek − G_e S G_k = Q̃ᵀΔL E_k − E_qᵀΔL K̃；CG 固定容差 1e-8、上限 500 iter，预注册不扫）
```
S* 特征分解取**绝对特征值最大 4 个方向**（固定 rank，不扫）；fold0-2 各自求解后 Frobenius 归一、等权平均为唯一 S_dir。

**步长（真实码边界，唯一步长）**：沿 S_dir 用父五字段合法码相邻中点与 dQ=Q̃S、dK=−K̃S 计算每块首个正向翻码距离，取**输出敏感度加权 1/64 分位**为唯一 α，约束 ||αS||_F ≤ log(2)/2；只生成一个 proposal，经真实编码/解码验证。

## 3. 为何有效

- A23 官方 +13 证明"互逆残余改变码分配"机制有效，但 amax 乘积代理与最终输出脱节（方法审计）；
- A26-A（解析代理反向）与 A27-B（折内迭代，ρ=+0.778 深度单调换 4B 回退）证伪两条旧路线；A29 的结构对应修正：**真实最终输出残差定方向 + 真实码边界定步长 + 零折内迭代 + fold4 独立硬输出 holdout**；
- V 码分配饱和（A28）不构成输出下界（纠偏），QK-V 交叉补偿是 V 码不变下的合法收益通道。

## 4. 如何证伪（写入后不调整）

1. δℓ* 或 S* 数值为零（六层）→ 关闭 Jacobian→冻结误差线性映射，F4 仍 OPEN；
2. S* 非零但 changed codes = 0（六层全零）→ 关闭 1/64 边界步长实现（不改分位数重试）；
3. 训练折 Lhard 降但 fold3 或 fold4 未严格降 → 关闭固定聚合，登记 transfer 失败；
4. fold3/4 通过但 72 例微负 → 记录风险，仍保留一个固定代表的官方探索资格（A23 先例）；
5. 专项负向损失（negative L1）≥ 0.02 或 no-op → 不提交官方；
6. 官方负向 → 关闭 A29 具体机制；TIMEOUT → 只关闭该校准复杂度实现。

## 5. 去重四项比对（目标/变量/插入点/编码）

| 对照 | 目标 | 变量 | 插入点 | 编码/求解 |
|---|---|---|---|---|
| A23 | amax 乘积 ratio | 全矩阵 S（Adam 32 步） | A22-2 父栈 | 迭代训练 |
| A24 | 量化输出 STE | 全维互逆 | A22-2 父栈 | STE 恒零（死路） |
| A26-A | 独立舍入残差代理 | 网格残差 | A23 父栈 | 解析代理（反向） |
| A27-B | 真实读出 FD | Hadamard 8 谱带 | **单主干 v189 基座** | signGD 迭代 |
| A30（后继） | 硬输出 | 逐通道对角 d | R3 父栈 | 闭式能量平衡 |
| A31（后继） | 硬输出 | 三角 nilpotent 搬运 | R3 父栈 | 一次 SVD |
| **A29** | **最终硬输出** | 全矩阵 S，**rank-4 方向** | **R3 父栈 learned_rotation/center** | **Jacobian 伪逆 + 一次 CG + 码边界步长，零迭代** |

变量参数化与 A23 同形（对称零迹 S），但插入点父不同（R3 vs A22 系）、目标不同（最终输出 vs amax 乘积）、方向来源不同（闭式伪逆 vs Adam）、步长机制不同（码边界唯一 α vs 学习率）——四项均不等价。

## 6. 隔离与验证

- fold0-2 只产生 driver；fold3 只在完整父与唯一 proposal 间严格选择；fold4 完全独立 holdout，**最终硬输出 MSE 严格下降才部署，否则整层回父（逐位恢复 R3）**；72 评测窗只作外部诊断；
- 验证链：softmax-Jacobian/伴随/minimum-norm 小形状 vs autograd FD、互逆转置恒等、mask/GQA/K-center 同步编译逐位、S=0 逐位恢复 R3、真实 changed codes 计数、六 API 单文件、合法 CPU state、finite、V/Linear 逐位 control、专项负向损失 <0.02。

## 7. 成本

校准期：每层 5 fold × 4 组 = 20 次（256×256 eigh + CG + 两次硬读出）≈ 预计 <15s/层；部署零增量（矩阵已合入，无新增动态算子）。

## 8. 父与对照

score/time 双父：从 time_parent **R3 (14405/238s)** 构建；score target = **A2 14440/274s**（官方超越才升级侧父）；A23 (14437/276s) 是"保留旧训练 + 互逆残余"官方正向的机制证据参考。本地 4B 只记录风险，不预测官方符号。
