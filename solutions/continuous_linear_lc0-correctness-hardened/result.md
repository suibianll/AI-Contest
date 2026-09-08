# LC0 Linear result（官方回传 2026-09-08）

| 项目 | 结果 |
|---|---|
| 候选 | `continuous_linear_lc0-correctness-hardened` |
| 父版本 | `continuous_linear_l28-proj-vectorized`（SHA `44D7E964…`） |
| 源码 SHA256 | `E702D7C1F087097906E752DEFA5CBFF05C2E41585883B0EC8D5E4F937AA4AF3D` |
| **官方分数 / 时间** | **4614 / 294s（+3 vs L28 4611/286s；294s < 300s）** |
| 官方状态 | **PASS（RETAINED_AS_CANDIDATE）** |
| 本地协议 | `proxy-v3`, 4B, Linear-only, shard0 |
| 本地 case 数 | 56 |
| 本地候选 mean | `0.2884021593` |
| 本地父 mean（L28） | `0.2910968103` |
| paired delta | `-0.002695`（本地负，官方正——确认 4B proxy 不排序官方） |

## 官方判定

- **4614/294s**，相对 L28（4611/286s）**+3 分 / +8s**；294s < 300s 通过。
- 核心正向改动：**MSE_STD 归一化 A@W 校准目标**（`ω_f = 1/(F·numel·MSE_STD_f)`），
  与纠偏指令建议的 `ω_f = 1/max(MSE_STD,f)` 一致。
- **时间余量仅 6s**（294→300），无法与 Attention 组合；L28 维持组合时间父。
- 本地 shard0 微负（-0.0027）但官方 +3——再次确认本地 proxy 与官方排序反转。

## 后续

- LC0 登记为**已确认官方正向的候选**（+3 分机制：MSE_STD 归一化目标）。
- L28 仍是 Linear 时间父（286s）；组合 Linear 侧用 L28/L4 级时间安全版本。
- 该 +3 机制可直接继承到 L-A0/A1 Full-64 Direct A@W 路线的目标归一化。