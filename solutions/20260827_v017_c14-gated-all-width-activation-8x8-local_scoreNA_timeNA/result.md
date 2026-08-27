# v017 — C14 Calibration-Gated All-Width Activation 8×8

- Date: 2026-08-27
- Candidate ID: `C14`
- Parent: `C11`
- Unique mechanism: for `in_features <= 1024`, enable the all-width 8×8 activation residual only when sampled final-output MSE improves by at least 0.05% on average and no calibration sample regresses by more than 0.1%; retain C11's wide path unconditionally.
- Source SHA256: `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- Parent SHA256: `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- Local status: `local-champion`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.56pp | +1.12pp | +0.06pp | +0.60pp | +0.36pp | 0.00pp | +0.450pp |

- Candidate Linear mean: `0.5861`, versus C11 `0.5816`.
- Attention remains exactly C11 (`0.4497/0.4942`).
- CUDA algorithm-stage `24.99s`, versus C11 `22.32s`, ratio `1.120`.

## Fixed local matrix

| Case | C11 Linear mean | C14 Linear mean | Delta | Component safety |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5816 | 0.5861 | +0.450pp | pass |
| amax6 offset 97 | 0.5628 | 0.5671 | +0.427pp | pass |
| amax6 offset 193 | 0.5815 | 0.5860 | +0.453pp | pass |
| amax6 offset 389 | 0.5796 | 0.5839 | +0.423pp | pass |
| amax4 offset 0 | 0.4828 | 0.4900 | +0.723pp | pass; o +0.33pp |
| pow2 offset 0 | 0.5451 | 0.5493 | +0.420pp | pass |

- All six aggregate configurations and all recorded Linear components meet the safety gate.
- The C13 amax4 `o` regression is repaired: `0.4117` in C13, `0.4208` in C11, `0.4241` in C14.
- GQA offset 193 Attention remains exactly C11 (`0.4169/0.4928`).
- Same-environment CPU pair: C14 `58.05s`, C11 `60.30s`, ratio `0.963`; treated as timing parity rather than a speed claim.
- Ten release tests pass, including direct execution of the calibration-gate decision and state validation.

## Decision

`accepted as local Champion`. C14 converts C13's broad aggregate gain into a component-safe candidate across the full fixed matrix while remaining within the time budget.

Next candidate: replace the activation quadratic Gram source with the actually deployed quantized weight (`W_hat^T W_hat`) while retaining C14's gate. This aligns the activation-error quadratic term with the final Linear operator.
