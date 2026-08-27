# v019 — C16 Gated Activation 8×8 Coverage 4%

- Date: 2026-08-27
- Candidate ID: `C16`
- Parent: `C14`
- Unique mechanism: raise only the activation 8×8 residual coverage from 2% to 4%; retain C14's calibration gate, one sweep, cap 4096 and dense-weight Gram.
- Source SHA256: `4EDCB571250F498F63613848AE8E73527BF1F7FB9F603EF12C886083705FB07B`
- Parent SHA256: `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.13pp | +0.19pp | +0.05pp | +0.25pp | +0.12pp | +0.15pp | +0.148pp |

- Candidate Linear mean `0.5876`, versus C14 `0.5861`.
- All six Linear components improve; Attention remains exactly C14 (`0.4497/0.4942`).
- CUDA algorithm-stage `24.78s`, versus C14 `24.99s` on its development run.
- Ten release tests passed.

## Decision

The effect is broad and unambiguously positive, but it remains below the preregistered `+0.2pp` promotion gate. Fixed regression and CPU timing were skipped. C14 remains Champion and C16 is retained as evidence that activation coverage has not fully saturated.

Next candidate: one final bounded 8% coverage check from C14. It must clear `+0.2pp`; after that, activation 8×8 coverage tuning closes regardless of outcome.
