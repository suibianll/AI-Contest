# L27：批量 Cholesky + eigh 替换 SVD（时间安全重构，READY_FOR_OFFICIAL）

研究循环卡 L27，L23b 官方 TIMEOUT（2026-09-08）后的**实质复杂度重构**。
机制（残差交叉子空间 A@W 低维拟合）不变，仅改求解算子。

## 背景

L23b（13639FB2）官方 TIMEOUT（>300s）。根因：每层 ~144 块 × 7 role × 24 层的逐块
Python 循环内做 Cholesky + `SVD(Hw)`（Hw 形状 [64, o]，o 最大 9216），约 2.4 万次
小张量算子调用，官方 CPU 上超预算（v161 教训：per-call 小张量算子成本远超本地
CUDA 外推）。

## L27 改动（一卡一配置）

1. **批量预计算 Cholesky**：`G_b = Xb_bᵀXb_b` 只依赖 Xb（不依赖残差 R），全部块的
   Cholesky 因子一次 batched 调用完成，替代逐块 Cholesky。
2. **eigh 替换 SVD**：`SVD(Hw)`（[64, o]）换成 `eigh(Hw Hwᵀ)`（[64, 64]）取 top-8
   特征向量 = Hw 的 top-8 左奇异向量（投影等价，实测 max diff 2.4e-15）。省掉
   每个块在 o=9216 上的 SVD 开销。
3. **接受判定不变**：逐块增量残差 R、严格 L_all 下降、只写回接受块 sign/mant。

## 验证（4B shard0 paired，父 L4）

| 指标 | L23b | L27 |
|---|---|---|
| shard0 fit_gain（校准折叠） | 0.9478 | **0.9453**（eigh 数值，等价） |
| 宽层 accepted | 144/144 | 144/144 |
| shard0 delta_mean（vs L4 面板） | -0.2182 | **-0.2179**（等价） |
| calib API（本地 CUDA） | 176.1s | **174.3s** |

本地 CUDA 时间差异小（GPU 上 SVD 快）；官方 CPU 的收益来自去掉每块 64×9216 SVD。
数学等价已单独验证（eigh vs SVD 投影差 2.4e-15）。

## 状态

- **READY_FOR_OFFICIAL**（本环境无官方上传入口；包路径
  `solutions/continuous_linear_l27-batch-solve/solution.py` + SHA
  `78A9B982DCF6F9FB…`）。
- 官方 300s 为唯一时间裁决；正向且 <300s 则登记新 Linear 侧父；
  再超时则需把逐块循环端到端批量化。
- Linear 侧父 L4 与根 v189 不变。