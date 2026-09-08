# L23b 官方 TIMEOUT 处置与 L28 时间安全重构（2026-09-08）

执行者：Linear 侧执行代理。依据 AGENTS §3（官方 TIMEOUT 关闭具体复杂度实现、
机制族 OPEN、仅实质复杂度变化允许再探索）与
[持续研究循环](../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md)。

## 回传登记

用户回传 "l23b也超时了" → L23b（SHA `13639FB2…10FE0`）官方 TIMEOUT（>300s，
分数/时间未知）。已更新：
- `solutions/continuous_linear_l23b-residual-subspace/official-result.json` → TIMEOUT
- manifest/mechanism.md/state.json 同步
- 回传日志 `logs/execution/2026-09-08-l23b-official-timeout.md`

## 超时根因分析

L23b 本地六 shard API 1399.15s（父 1055.71s，+343s）；旧 L23（33D1DA51）亦超时。
两次超时说明：**逐块 rank-8 求解 + 逐块合法投影循环**（每层 ~144 块 × 7 role ×
24 层 ≈ 2.4 万次小张量算子调用）在官方 CPU 上超 300s。增量残差重构不足以把
校准期成本降到官方预算——与 v161 "per-call 小张量算子成本远超本地 CUDA 外推"
教训一致。

## L28 时间安全重构（READY_FOR_OFFICIAL）

机制（残差交叉子空间 A@W 低维拟合 + 增量残差接受判定）不变，三项等价算子替换：

| 项 | L23b | L28 | 等价性 |
|---|---|---|---|
| Cholesky | 逐块 144 次 | 批量一次（只依赖 Xb） | 精确 |
| 低维基 | `SVD(Hw)` [64,o] | `eigh(HwHwᵀ)` [64,64] | 投影差 2.4e-15 |
| 合法投影 | 15×64×o 网格 argmin | round/clamp 直接计算 | max diff 0.0 |

4B shard0 paired（父 L4）：fit_gain 0.9453（L23b 0.9478）、accepted 144/144、
delta_mean -0.2178（L23b -0.2182，等价）、api_total 217.9s（L23b 232.1s）。

## 状态

- **L28 READY_FOR_OFFICIAL**（SHA `44D7E964F8264633…`，取代 L27 `78A9B982…`；
  包路径 `solutions/continuous_linear_l28-proj-vectorized/solution.py`；本环境
  无官方上传入口，交付包 + SHA）。
- **fit_gain 0.9 研究目标仍有效**（L23b 0.9486 达成；L28 0.9453 保持）。
- Linear 侧父 L4（4607/247s）与根 v189 不变。
- 若 L28 官方再超时：剩余方向为逐块循环端到端批量化（需改变接受判定语义，
  属新的机制卡，须先证明不破坏 fit_gain ≥ 0.9）或换机制。