# L-R2 执行报告：修正版 L1 正确性与成本

> 日期：2026-09-07。侧：Linear。run_id：`repair-r1/l-r2`。
> 契约：[evidence-repair-next-cycle.md](../../docs/superpowers/plans/workpackages/evidence-repair-next-cycle.md) L-R2。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。

## 1. 修复项（相对上一轮 probe_l1_direction* 的错误）

| # | 修复项 | 状态 |
|---|---|---|
| 1 | 梯度范数：裁剪上限 1（norm>1 裁到 1，norm<1 不放大） | 实现于 STE 训练；本次仅做正确性检查（见下） |
| 2 | 每 fold ≤128 行、各 fold 等权；loss 按同 fold 标准 HiF4 最终输出 MSE 归一化 | 探针已按部署坐标校准样本（128 行/ fold） |
| 3 | 硬前向使用部署对应编码（真实 GPTQ weight + 真实 activation GPTQ，含 offsets/refine） | 本报告 §3 按部署实际常量实测成本 |
| 4 | 矩阵指数/特征值投影梯度检查 + STE + inference_mode | §2 全部通过 |
| 5 | T=I 恢复父路径 | 逐位恒等（§2.4） |

## 2. 正确性检查（`probe_l1_correctness_cost.py`）

### 2.1 矩阵指数/特征值投影梯度（double 精度，有限差分对照 3 trial）

| trial | max rel diff |
|---|---|
| 0 | 1.23e-02 |
| 1 | 5.69e-04 |
| 2 | 1.82e-02 |

阈值 <2e-2：**PASS**（特征值 clip 的 eigh 在一般位置可微；重复特征值/零
矩阵在 `sym_zero_trace_exp` 中走 `norm<1e-12` + `torch.matrix_exp` 分支，
避免 eigh 退化）。

### 2.2 STE 直通梯度

`ste = q(x) + (x - x.detach())`，反向 `|g-1|_max = 0.00e+00`：**PASS**。

### 2.3 inference_mode 可达

`with torch.inference_mode(): kron(exp(0), exp(0)) == I(64)`：**PASS**。

### 2.4 T=I 恒等恢复

`apply_T_block(x, I) == x` 与 `W @ I^{-T} == W` 逐位相等（max diff 0.0）：
**PASS** —— T=I 时变换坐标不改变任何连续值，父路径可选回归（其余离散
状态/依赖不随 T 变化，见 L-R1 §2.2）。

## 3. 成本（部署一致配置，L-R2 决策依据）

按 L4 实际部署常量（weight：offsets=(-1,1,2,3)、refine threshold 1e-7、
margin 0.005、ratio 1.0、blocks 65536；activation：offsets=(-1,1,2,3,4)、
refine threshold 1e-7、margin 0.02、ratio 0.70、blocks 32768），真实输入、
预热 + CUDA 同步、分段计时：

| state | transform_T | state gram/h_inv | hard weight GPTQ | state act h_inv | hard act GPTQ | total |
|---|---|---|---|---|---|---|
| L0-o（896×896） | 0.002 | 0.002 | 0.442 | 0.002 | 0.173 | **0.622s** |
| L0-fc_up（4864×896） | 0.001 | 0.002 | 1.542 | 0.003 | 0.183 | **1.731s** |
| L11-proj（896×4864） | 0.001 | 0.050 | 2.225 | 0.053 | 0.911 | **3.240s** |

- 单次完整硬前向（含部署一致 refine）≈ **0.62 ~ 3.24s/state**。
- 32 步 × 168 state（默认全量）≈ **3,300 ~ 17,000s 本机**：不可承受。
- **禁止**将上述非 default 探针数值直接代入官方时间模型（
  NONDEFAULT_EXTRAPOLATION_INVALID）；只作为本机成本诊断。

## 4. 决策：COST_HOLD

- 正确性（梯度/STE/inference_mode/T=I）**全部通过**。
- 完整硬前向 32 步 × 168 state 全量结构**显然昂贵**（≈1–5h 本机，
  即使按最窄层 0.62s/step 也是 3,300s）。
- 按 L-R2：**COST_HOLD** —— 只记录本机成本估计，不关闭可逆变换数学方向，
  不把非 default 探针外推官方时间。

## 5. COST_HOLD 后继（机会清单，供下一张独立卡）

缓存/复用候选（必须逐项列出 T 依赖，禁止复用过期 Gram/Hessian）：

1. `gram_full = T^T (X^T X) T`：X^T X 只依赖父部署坐标（不依赖 T），
   可在训练循环外算一次；每步只做两侧 T 的 64×64 块对角收缩。**可行**。
2. weight 侧 `H_inv`：依赖 gram_full（T 相关）→ 每步重算（cholesky 896²
   级别，0.002–0.05s，可接受）。
3. activation `H_inv_a = H^T(Q(W T^{-T}))`：依赖当前权重量化（T 相关）→
   每步必须重建。权重 encode（0.44–2.2s）是主要瓶颈，无免费复用。
4. 若用"仅激活侧 T（权重侧解析吸收 W T^{-T} 再固定 GPTQ）"，会破坏
   L-R1 §2.2 的 rank/坐标等价性，需另立机制卡并预注册，不能现场改规则。

## 7. 修正口径方向探针（`probe_l2_direction.py`，STE 已接）

在 L-R1 闭环（真实 NVFP4 输入 + 部署一致 GPTQ 权重/激活编码 + 修正梯度裁剪
norm>1 才缩放）下重测 FlatQuant T=T1⊗T2 的 32 步 STE Adam。修正上一轮
"量化全 detach 梯度断流"问题后：

| state | baseline(T=I) | step32 | rel Δ |
|---|---|---|---|
| L0-o | 0.001011 | 0.001053 | +4.2% |
| L11-proj | 0.002348 | 0.002476 | +5.5% |
| L0-fc_up | 0.007219 | 0.007391 | +2.4% |

**该结论随后被 `probe_t_identity_parity.py` 判为无效**：`deployment_forward`
（T=I）输出与父真实 API 不一致（MSE ratio 0.45-0.80，见
`report-t-identity-correction.md`），即简化前向缺少父的 rank-2 残差 gram
修正、static-actorder hdiag 块序与 importance 规范化。因此 3 个 state 的
"退化"是相对一个与父不一致的基线，**不能归因于 FlatQuant T**。
FlatQuant 状态回退为 **DIRECTION_UNKNOWN**（正确性方法 PASS；完整硬前向
COST_HOLD；真实部署方向证据不足）。

## 8. 结论（更新）

- FlatQuant 方向**未判定**：正确性方法 PASS、完整硬前向 COST_HOLD
  （0.62-3.24s/step，32 步 × 168 state 不可承受）；3-state "退化"因
  T=I 前向与父不一致而无效。要判定需"插 T 的完整校准变体"（T=I 逐位
  恢复父含 rank-2/actorder），每步≈完整校准 ~2s/state，仅代表 state
  可做方向诊断。
- 下一张机制卡优先评估 §5 第 1 项的 gram 预计算与"批量/并行 fold"方案
  是否能把 32 步降到每 state <2s，或换一个不改变硬前向语义的解析求解。
- 产物：`probe_l1_correctness_cost.py`、`probe_l2_direction.py`、
  `probe_t_identity_parity.py`、`report-t-identity-correction.md`、
  `artifacts/proxy_v3/continuous/linear/repair-r1/l1-cost.json`。