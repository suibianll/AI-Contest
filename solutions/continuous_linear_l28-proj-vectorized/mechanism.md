# L28：批量 Cholesky + eigh + 投影向量化（时间安全重构，READY_FOR_OFFICIAL）

研究循环卡 L28，L23b 官方 TIMEOUT（2026-09-08）后的**实质复杂度重构**（在 L27 基础上
叠加投影向量化）。机制（残差交叉子空间 A@W 低维拟合）不变。

## 背景与改动（一卡一配置）

L23b 超时根因：逐块循环内 ① 每块 `SVD(Hw)`（[64, o]，o≤9216）② 每块构造
`[15, 64, o]` 网格张量（o=9216 时 8.8M 元素）做 argmin 投影。约 2.4 万块 × 两次
大张量算子，官方 CPU 上超预算。

L28 三项等价替换（均保持增量残差语义）：

1. **批量 Cholesky 预计算**：`G_b = Xb_bᵀXb_b` 只依赖 Xb，全部块一次 batched 调用。
2. **eigh 替换 SVD**：`SVD(Hw)`（[64,o]）→ `eigh(HwHwᵀ)`（[64,64]）取 top-8 特征向量
   = Hw 的 top-8 左奇异向量（投影等价，实测 max diff 2.4e-15）。
3. **投影向量化**：最近邻合法码 = `round(Wd/scale/0.25).clamp(-7,7)*0.25*scale`，
   等价于 15×64×o 网格 argmin（实测 max diff 0.0），省掉每块 15× 中间张量。

## 验证（4B shard0 paired，父 L4）

| 指标 | L23b | L27 | L28 |
|---|---|---|---|
| fit_gain（校准折叠） | 0.9478 | 0.9453 | **0.9453** |
| accepted（宽层） | 144/144 | 144/144 | 144/144 |
| delta_mean（vs L4） | -0.2182 | -0.2179 | **-0.2178** |
| api_total（本地） | 232.1s | 220.7s | **217.9s** |
| calib_w（本地） | 176.1s | 174.3s | **173.1s** |

本地 CUDA 收益有限（GPU 上 SVD/网格快）；官方 CPU 收益来自去掉每块 64×9216 SVD 与
15× 网格分配。三项数学等价均已单独验证。

## 状态

- **READY_FOR_OFFICIAL**（取代 L27；包路径
  `solutions/continuous_linear_l28-proj-vectorized/solution.py` + SHA
  `44D7E964F8264633…`）。
- 官方 300s 为唯一时间裁决；正向且 <300s 则登记新 Linear 侧父；
  再超时则需把逐块循环端到端批量化。
- Linear 侧父 L4 与根 v189 不变。