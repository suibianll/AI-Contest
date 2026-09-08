# Linear 时间约束系统性评估：L29/L30/L32 全部不可行（2026-09-08）

依据 [持续研究循环](../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md) §7。

## 背景

用户核心诉求：L28 官方 286s（+4分/+39s vs L4 247s）时间太长，无法与 Attention 组合。
需时间安全的 Linear 改进。组合时间由 Linear 校准主导（v189 完整 shard1 中 Linear 占 98%：
Linear 校准 142.8s + Linear 动态 47.4s ≈ 190s/193.5s）。

## 候选系统性评估

| 卡 | 机制 | 结果 | 原因 |
|---|---|---|---|
| L29-Q | 薄 QR 压缩 | REJECTED_BEFORE_IMPLEMENTATION | N=138≤d（≥1024），rank(A)=N，无压缩 |
| L29-G | 充分统计量/Gram | REJECTED_BEFORE_IMPLEMENTATION | 每块 H 更新 [d,o]，d>N 时比 L28 [N,o] 慢 17.6×（实测 87.8ms vs 4.98ms） |
| L30 | 全局 reduced-rank | REJECTED | fit_gain 0.707≈L4（无收益），api 322.9s>300s（d×o 大矩阵） |
| L32 | 每块 rank-8 联合互逆 R=I+UVᵀ | **时间不可行** | Woodbury 权重写回 [o,64]@[64,64] 每块 99ms，144 块=14.3s/宽层；24×7 state 必超时（L30 先例） |

## 结构性结论

- **N=138<<d** 使所有"压缩校准计算"的数学等价方案（QR/Gram）都更慢而非更快——
  L28 的逐块 [N,o] 已是 N 很小时的最优表示。
- **联合坐标/复杂机制**（L30 全局、L32 每块 Woodbury）都因 d×o 或权重写回的大矩阵
  超 300s。
- **在时间约束下，L28（4611/286s）是 Linear 官方最优**；组合 Linear 只能用 L4（247s 时间安全）。
- L28 官方 RETAINED 保持 Linear 侧父；组合 Linear 侧用 L4。

## 下一步

Linear 时间侧已系统性穷尽（L29-Q/G/L30/L32 全不可行）。剩余选择：
1. 接受 L28 为独立侧父 + 组合用 L4；
2. 未来若官方时间预算放宽或评测机更快，再评估 L32/L30；
3. 等待 Attention 侧 A29/A30/A31 收益（Attention 是主收益线，计划 §7 明确）。

此评估不撤销 L28 官方正向结果；L4 继续作为组合时间预算父。