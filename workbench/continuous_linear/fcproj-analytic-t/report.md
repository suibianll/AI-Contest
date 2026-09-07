# fc/proj 解析求解机制：诊断证据归档

> 日期：2026-09-07。侧：Linear。run_id：`fcproj-analytic-t`。
> 契约：官方 P3（fc/proj 唯一官方增益桶）+ 用户指示"尝试 fc/proj 大形状桶的解析求解机制"。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s，本地最高 0.6368）。

## 1. 动机

官方 P3（2026-09-05）：v160 Linear 官方增益 100% 落 W2(fc_gate/fc_up)=1818 +
W3(proj)=1767 大形状桶；q/k/v/o 官方**零收益**。本地 fc 0.528 / proj 0.564 也
是 L4 最差桶 → 官方敏感桶即本地最差桶，是本侧唯一合理机制目标。

## 2. 尝试 1：正交特征基解析 T（`probe_analytic_T.py`）

方案：对部署坐标校准激活 X 构造块内 Gram 平均
`G = mean_b (X_b^T X_b) ∈ R^{64×64}`，取特征基 `T = eigh(G).eigenvectors`
（正交、闭式、不训练、不改父），比较 T=I vs T=Q 下标准 HiF4 codec 的
最终输出 MSE（方向代理；明确不冒充部署 GPTQ）。

结果（5/5 fc/proj 代表 state **全部退化**）：

| state | eig top2 | std_mse T=I | T=Q | rel |
|---|---|---|---|---|
| L0-fc_up | 0.059 | 1.2989e-3 | 1.3439e-3 | **1.035**（退化） |
| L0-fc_gate | 0.154 | 2.4490e-3 | 2.8534e-3 | **1.165**（退化） |
| L11-proj | 0.041 | 2.4882e-4 | 2.6190e-4 | **1.053**（退化） |
| L0-proj | 0.040 | 1.8763e-4 | 2.0313e-4 | **1.083**（退化） |
| L11-fc_up | 0.071 | 2.3623e-3 | 2.4415e-3 | **1.034**（退化） |

## 3. 尝试 2：Kronecker 解析 T（`probe_analytic_T_kron.py`）

方案：同一 G 上闭式求 `min||T1⊗T2 − G||_F²`（SVD-based，T1/T2 8×8 因子，
FlatQuant 卡部署形式），cond(T) 1.15-10.78 可逆。

结果（4 个 state：2 退化、2 持平，无材料改善）：

| state | cond | rel | 判定 |
|---|---|---|---|
| L0-fc_up | 1.50 | 1.025 | DEGRADE |
| L0-fc_gate | 10.78 | 2.998 | DEGRADE |
| L11-proj | 1.22 | 1.004 | FLAT |
| L0-proj | 1.15 | 0.982 | FLAT |

## 4. 与既有证据互证

- P1 同坐标诊断（2026-09-05）：**v186 的连续变换族已饱和到输出零偏差**
  （`X_tW_t == XWᵀ` 全 336 case 机器精度内）；官方 4166 分差只能靠
  **减少量化扰动**，不是继续找连续坐标等价变换。
- L1 FlatQuant 训练式：COST_HOLD（每步完整硬前向 0.62-3.24s×32 不可承受）。
- 上一轮 fast-path 34/35 退化（撤回，但方向一致）。
- AGENTS §7：Householder 全族、块序族、E6M2 offset/alpha/sweep 局部扫描
  全部关闭；`_WEIGHT_E2E_REFINE`（E6M2 scale offset 精化，默认 False）
  属于被禁的 offset 扫描边界。

## 5. 结论

- **fc/proj 解析旋转族（正交特征基 + Kronecker）无本地可迁移余量**：
  9 个真实 state 均无改善（7 退化 / 2 持平），且与 P1"连续变换饱和"、
  L1"训练式 COST_HOLD"一致。不再在该族持续投本地算力。
- 不创建候选，不注册官方探索；父 L4 不变。
- 该证据只否定"解析旋转"这一类，不扩大到整个 fc/proj 桶；若未来出现
  非旋转的（如表示/编码/重要性）解析机制，仍可单独登记验证。

## 6. 产物

- `workbench/continuous_linear/probe_analytic_T.py`
- `workbench/continuous_linear/probe_analytic_T_kron.py`