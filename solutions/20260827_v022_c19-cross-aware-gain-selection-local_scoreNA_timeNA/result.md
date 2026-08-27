# v022 — C19 Cross-Aware Gain Selection

- Date: 2026-08-27
- Candidate ID: `C19`
- Parent: `C17`
- Unique mechanism: rank the bounded 8×8 activation candidates by the cross-aware coordinate gain upper bound `max_i (H·e+b)_i²/H_ii`, and use the same block-local cross objective for updates.
- Source SHA256: `D58DD0C0D09E84C680B26E124B8610CB4CC4A6F9AA45AD5ECFB2441677C00F68`
- Parent SHA256: `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.09pp | +0.13pp | +0.08pp | +0.38pp | +0.16pp | +0.07pp | +0.152pp |

- All six components improve and the gain is approximately twice C18's `+0.077pp`.
- Attention remains exactly C17 (`0.4497/0.4942`).
- CUDA algorithm-stage `25.25s`, versus C17 `24.63s`, ratio `1.025`.
- Eleven release tests passed.

## Decision

Cross-aware ranking materially improves budget use, but the gain remains below the preregistered `+0.2pp` promotion gate. Fixed regression and CPU timing were skipped. C17 remains Champion.

Next candidate: replace the continuous Newton upper bound with the exact best single-coordinate decrease achievable on the current HiF4 code grid. This preserves the same cross objective and 8% budget while removing continuous-versus-discrete ranking error.
