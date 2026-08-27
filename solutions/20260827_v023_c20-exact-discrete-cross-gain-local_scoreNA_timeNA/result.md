# v023 — C20 Exact Discrete Cross-Gain Selection

- Date: 2026-08-27
- Candidate ID: `C20`
- Parent: `C17`
- Unique mechanism: rank activation 8×8 candidates by the exact best single-coordinate objective decrease achievable with the current scale hierarchy and 15-value signed HiF4 code grid; apply the same block-local cross objective during updates.
- Source SHA256: `148C344177DCCB734F930F919322F23C0FF2CEA3FAB1263426D04B07C4336FB4`
- Parent SHA256: `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

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
