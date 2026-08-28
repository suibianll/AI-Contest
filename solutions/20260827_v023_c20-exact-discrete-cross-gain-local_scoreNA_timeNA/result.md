# v023 — C20 Exact Discrete Cross-Gain Selection

- Date: 2026-08-27
- Candidate ID: `C20`
- Parent: `C17`
- Unique mechanism: rank activation 8×8 candidates by the exact best single-coordinate objective decrease achievable with the current scale hierarchy and 15-value signed HiF4 code grid; apply the same block-local cross objective during updates.
- Source SHA256: `148C344177DCCB734F930F919322F23C0FF2CEA3FAB1263426D04B07C4336FB4`
- Parent SHA256: `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- Local status: `local-accepted-not-promoted`
- Official status: **已提交，16081 分 / 152s**（2026-08-28 回填）

## 官方提交结果（2026-08-28 回填）：16081 / 152s

| 版本 | Linear mean | 官方分数 | 官方时间 |
|---|---:|---:|---:|
| **C20 / v023（本归档）** | 0.5931 | **16081** | **152s** |
| C21 / v024（门控精确交叉） | 0.5930 | 16043 | 173.8s |
| C18 / v021（8×8 cross 首版） | 0.5897 | — | — |

- C20 是 cross（最终输出监督）系列的中间锚点，官方分**高于**其后的
  C21/v024（16081 vs 16043）且更快（152s vs 173.8s）。
- **合规警示**：C20 属于"Linear 输出监督路径"家族（cross-gain
  选择），该路径已被官方**后来明确禁止**（见 26000 计划 Phase 0 /
  README）。本官方分是**历史非合规锚点**，仅为完整性记录，**不
  构成合规先例**，也不作为后续候选的父版本或分数基数。

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.31pp | +0.27pp | +0.32pp | +0.70pp | +0.46pp | +0.42pp | +0.413pp |

- Attention remains exactly C17 (`0.4497/0.4942`).
- CUDA algorithm-stage `25.19s`, versus C17 `24.63s`, ratio `1.023`.
- Eleven release tests passed.

## Fixed local matrix

| Case | C17 Linear mean | C20 Linear mean | Delta | Safety result |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5890 | 0.5931 | +0.413pp | pass |
| amax6 offset 97 | 0.5696 | 0.5813 | +1.163pp | pass; proj +4.39pp |
| amax6 offset 193 | 0.5888 | 0.5932 | +0.433pp | pass |
| amax6 offset 389 | 0.5867 | 0.5911 | +0.445pp | pass |
| amax4 offset 0 | 0.4927 | 0.4978 | +0.503pp | pass |
| pow2 offset 0 | 0.5524 | 0.5475 | **-0.490pp** | **fail: proj -5.87pp** |

- GQA offset 193 Attention remains exactly C17 (`0.4169/0.4928`).
- CPU timing was skipped after the fixed-matrix safety failure.

## Decision

`local-accepted-not-promoted`. Exact discrete selection delivers large gains in five configurations but exposes an unacceptable pow2 wide-projection regression (`proj 0.4890→0.4303`). C17 remains Champion; C20 is retained as the parent evidence for an all-width final-output safety gate.

Next candidate: keep C17's pure 8×8 result as the per-layer fallback, and store the exact cross state only when it improves final output MSE over that fallback on calibration samples. The new gate applies to wide layers as well as narrow layers.
