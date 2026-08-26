# v012 — C9 16×16 Second Sweep

- Date: 2026-08-27
- Candidate ID: `C9`
- Parent: `C5`
- Unique mechanism: retain C5's 2% 16×16 coverage and increase only the 16×16 coordinate sweeps from one to two.
- Source SHA256: `9B70C7825364A4734C67BD895A626850025B32E8BD32CB296CECB00960EA181F`
- Parent SHA256: `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.01pp | +0.02pp | +0.01pp | -0.01pp | +0.03pp | +0.09pp | +0.025pp |

- Candidate Linear scores: `0.6348/0.6957/0.5964/0.5396/0.4960/0.5201`.
- Attention remains exactly C5 (`0.4497/0.4942`).
- CUDA algorithm-stage `22.35s`, versus C5 `20.81s` on its development run.
- Eight release tests passed in the project virtual environment.

## Decision

The second sweep has a deterministic but negligible effect and is far below the preregistered `+0.2pp` gate. Fixed regression and CPU timing were skipped. C5 remains Champion; further 8×8/16×16 sweep and coverage tuning is closed.

The next candidate changes mechanism: extend the already proven 4×4 activation quadratic objective to the currently excluded 3072-wide FFN down-projection inputs, with a target-specific proj gate and explicit state/timing checks.
