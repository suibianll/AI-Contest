# v187 计划：Attention Jacobian 敏感度加权 HiF4（2026-09-04）

> 状态：ACTIVE。
>
> 父实现为 v185 clean-room 六 API，仅用于验证一个新的解析机制；不读取或复制 v182/v186
> 的实现。v185 官方 `8446/165s`，因此本计划首先是机制研究，不因小幅本地正向直接提交。

## 1. 假设

v185 使用等权张量重建误差选择 Q/K 的 HiF4 层级，但最终目标是 Attention 输出：

```text
P = softmax(QK^T / sqrt(d))
O = PV
```

量化误差的一阶输出代价不是 `||E_Q||² + ||E_K||²`，而是 Jacobian 加权误差。对单个
query token `i` 和 key token `j`：

```text
dO_i / dQ_ic = Σ_j P_ij (V_j - O_i) K_jc / sqrt(d)
dO_i / dK_jc = P_ij (V_j - O_i) Q_ic / sqrt(d)
```

因此用 `||dO/dQ_c||²`、`||dO/dK_c||²` 作为每坐标 HiF4 reconstruction importance，
应当比 v185 的等权编码更贴近最终算子误差。

## 2. 唯一机制

1. 在 calibration 阶段，用当前已接受的 K-center/QK-balance/gamma 后的连续 Q/K 计算
   causal 与 non-causal 两种 Attention Jacobian 对角敏感度；
2. Q importance 在同一 KV group 的 query heads 内共享，K importance 按 KV head 共享；
3. 每个 fold 先按 head 内均值归一化，再跨 fold 取 median；
4. 固定 log-space `1/4` 收缩到 identity，并 clamp 到 `[0.5, 2.0]`；
5. leave-one-fold-out 比较最终 Attention output MSE，只有 median 正向且 worst-fold
   不低于固定容差才部署；
6. dynamic Q/K 仅把 state 中固定 importance 传给现有 HiF4 求解器；V 与 Linear 完全不变。

不搜索收缩率、clamp、fold、head sharing、offset、refine ratio 或 gate 阈值。

## 3. 泛化与复杂度约束

- 自由度由 `token × head × dim` 的 Jacobian 统计压缩为 `KV-head × 64`，并向全 1 收缩；
- 参数不按 layer role、token、序列长度或测试 case 路由；
- state 只保存 CPU tensor；在线无 Jacobian、Gram、候选循环或矩阵逆；
- 校准和 gate 使用 leave-one-fold-out，最终 state 才使用全 folds 聚合；
- V 保持等权编码，遵守 V 侧结构性关闭结论。

## 4. 验证顺序

1. 从 v185 单文件生成独立 v187 文件，只加入本机制；
2. `py_compile`、六 API smoke、reference HiF4/state validator；
3. Attention compact，配对 v185，检查 reachability、mean、L1 与四个 case；
4. 若 compact 非正向则 REJECTED；若正向，运行 Attention default 配对 v185；
5. 同时与 v182 immutable default JSON 比较绝对差距；
6. 记录六 API 时间并代入官方时间模型；
7. 只有 `Δmean>0`、`L1<0.02`、control 不变且已接近 v182 才允许注册官方候选。

由于 v185 官方落后 v182 `9152` 分，本轮即使相对 v185 正向，只要仍显著低于 v182，就只保留
为 clean-room 研究父，不消耗官方配额。

## 5. 产物

- 源码：`solutions/20260904_v187_attn-jacobian-sensitivity_scoreNA_timeNA/solution.py`
- 结果：同目录 `result.md`
- 本地原始结果：`artifacts/official_eval/v187-*`
- 本地报告：`logs/official_eval/v187-*`

## 6. 裁决

- 接口、state、finite 或真实调用图失败：修实现，记 ERROR；
- Jacobian gate 不可达或 compact `Δmean≤0`：REJECTED，关闭该机制；
- 相对 v185 正向但仍远低于 v182：RESEARCH RETAINED，不提交；
- 接近或超过 v182 且 R2/时间门禁通过：候选，等待用户决定是否使用配额。
