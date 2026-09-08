# L23 与旧实现数学去重表（R0）

> 日期：2026-09-07。侧：Linear。依据：[21071工作包 §3-4](docs/superpowers/archive/plans/21071-evidence-driven-research.md)。

## 1. L23 卡定义（唯一注册目标）

对每 64 输入列块 B（父部署顺序一遍）：
- `R_f = Y_f − Xh_f W_current`，`Y_f = X_f W_original^T`（原 NVFP4 解码 teacher）
- 目标：`Σ_f ω_f ||R_f − Xh_f,B ΔW_B||² + λ||ΔW_B||²`
- `ω_f = 1/(F·max(||Y_f||², 1e-12))`，只用 fit 数据
- 构造：`G = Σ_ω Xh_B^T Xh_B + λI`，`H = Σ_ω Xh_B^T R`（残差交叉）
- **Cholesky 白化 H → rank-8 截断 SVD → 解回 ΔW_B**；不显式求全维逆
- rank 固定 8；λ 继承父同坐标块 ridge 规则（绑定代码与数值，禁调）
- 低维方向**同时依赖输出残差**，不只依赖 Xh 能量
- 块两臂：fit 基/系数 → select 各 fold 归一化误差差值 median 严格负才接受；
  validation/test 不参与选基/rank/参数

## 2. 与旧实现逐项比较（目标/变量/插入点/合法编码/训练调用图）

| 维度 | L23（残差交叉子空间） | 旧逐块激活 Gram top-8 (v1) | 旧块一次 LS (v2) | 旧权重 SVD 全局低秩 (v3) | 旧 JDRQ/GPTAQ |
|---|---|---|---|---|---|
| 拟合目标 | `Σω_f ‖R_f−XhΔW‖²+λ‖ΔW‖²`，R=Y−XhW_current | 同形式（块两臂 L） | 同形式 | `min‖XhΔWᵀ−R‖²`（全局） | λ>0 岭更新到 JDRQ target |
| 低维基 | **Cholesky 白化 H → rank-8 SVD**（依赖残差交叉） | 激活 Gram `XhᵀXh` eigh top-8（只依赖激活能量） | **无低维基**（64 列全自由度 LS） | 原权重 SVD 左右奇异（只依赖权重） | 无显式子空间（逐坐标补偿） |
| 白化 | G 的 Cholesky | 无（直接用 eigh） | 无 | 无 | 无 |
| rank | 8 固定 | 8/16 试过 | 64 | 8 | N/A |
| 插入点 | 校准内权重编码后、返回前 | 同 | 同 | 同 | 同 |
| 合法编码 | 投影父五字段格点，两臂 select | snap 父 scale 格点 | 完整重量化 | 完整重量化 | 逐坐标合法 |
| 隔离 | fit/select 奇偶 fold，median 严格负 | 全部 fold 训练 | fold-0 训练 fold-1 验证 | 全部 fold | 预注册参数 |
| 本地结果 | 待测 | shard0 −0.0179（8+/48−） | 3 state 退化 1.3~1.8× | 3 state 退化 1.4~1.7× | 官方 112 case 负向 |

## 3. 去重结论

- **L23 与旧 v1 数学不同**：v1 基=激活 Gram eigh（只依赖 XhᵀXh），L23 基=
  G 白化后的残差交叉 H（依赖 XhᵀR，即同时看残差与激活）。白化方向、rank-8
  截断 SVD 回解均是新构造；不重扫 rank。
- **与 v2/v3 不同**：后两者无残差相关基（v2 全自由度、v3 只依赖权重奇异向量）。
- **与 JDRQ/GPTAQ 不同**：后两者逐坐标/岭更新到 JDRQ target，无白化子空间。
- 每块 `W_current,B+ΔW_B → 父合法格点投影 → select median<0 才接受`：
  fit/select 分离与旧"全部 fold 训练+holdout"不同，符合工作包隔离要求。
- 结论：**注册 L23，不更名重试旧实现**。

## 4. 记录修正（R0）

- `candidate-shard0-report.md` §3 负向门核算已按纠偏 §3 修正（边界 ≤0.01899<0.02）。
- 本仓库尚未取得 21071 外部源码/SHA/配置；以下实现为独立推导假设，不声称复现。