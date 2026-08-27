# v021 — C18 Block-Local Activation/Weight-Error Cross Term

- Date: 2026-08-27
- Candidate ID: `C18`
- Parent: `C17`
- Unique mechanism: add a block-local linear cross term from `(W_hat-W_smooth)^T W_hat` to the activation 8×8 coordinate objective; retain C17's gate, coverage, sweep and cap.
- Source SHA256: `92627EE321D434FB1D62E54185276B8D431CCA90F661AF6CA118F6BE14C2EBAC`
- Parent SHA256: `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.02pp | +0.06pp | +0.05pp | +0.25pp | +0.06pp | +0.02pp | +0.077pp |

- All six Linear components improve; the largest effect is on `o`.
- Attention remains exactly C17 (`0.4497/0.4942`).
- CUDA algorithm-stage `25.19s`, versus C17 `24.63s`, ratio `1.023`.
- Eleven release tests passed, including a synthetic proof that the cross-term objective is non-increasing and validation of the additional CPU state tensor.

## Decision

The cross term is directionally correct across every component, but the mean gain is below the preregistered `+0.2pp` gate. Fixed regression and CPU timing were skipped. C17 remains Champion; the positive branch is retained for a cross-aware selection experiment.

Next candidate: use the same block-local cross objective both for coordinate updates and for ranking the bounded candidate groups. C18 still selected groups using C17's old pure-quadratic loss, which can spend the 8% budget on groups with little cross-aware improvement potential.
