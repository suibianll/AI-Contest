# v016 — C13 All-Width Activation 8×8 Residual

- Date: 2026-08-27
- Candidate ID: `C13`
- Parent: `C11`
- Unique mechanism: lower the activation 8×8 eligibility threshold from 1025 to 64 while retaining C11's 2% coverage, one sweep and cap 4096.
- Source SHA256: `205F09124FE87DEFE500FB78B7849C8CFC4D476067BA9AFC8374EC64DE264F8F`
- Parent SHA256: `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.56pp | +1.12pp | +0.08pp | +0.66pp | +0.36pp | 0.00pp | +0.463pp |

- Attention remains exactly C11 (`0.4497/0.4942`).
- CUDA algorithm-stage `23.34s`, versus C11 `22.32s`, ratio `1.046`.
- Nine release tests passed.

## Fixed local matrix

| Case | C11 Linear mean | C13 Linear mean | Delta | Safety result |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5816 | 0.5862 | +0.463pp | pass |
| amax6 offset 97 | 0.5628 | 0.5669 | +0.412pp | pass |
| amax6 offset 193 | 0.5815 | 0.5860 | +0.457pp | pass |
| amax6 offset 389 | 0.5796 | 0.5839 | +0.427pp | pass |
| amax4 offset 0 | 0.4828 | 0.4880 | +0.517pp | **fail: o -0.91pp** |
| pow2 offset 0 | 0.5451 | 0.5494 | +0.423pp | pass |

- GQA offset 193 Attention remains exactly C11 (`0.4169/0.4928`).
- Every aggregate configuration improves substantially, but amax4 `o` changes `0.4208→0.4117` (`-0.91pp`), violating the preregistered no-component-regression gate.
- CPU timing was skipped after the safety failure.

## Decision

`local-accepted-not-promoted`. This is a high-value branch with broad, repeatable aggregate gains, but it cannot replace C11 while the amax4 `o` regression is unguarded. C11 remains Champion.

Next candidate: make all-width 8×8 eligibility a calibration decision. Compare base 4×4 versus 8×8 final output MSE per layer on calibration samples, and store the 8×8 Gram only when mean improves and no calibration sample regresses beyond tolerance. Wide C11 behavior remains unconditional.
