# v009 — C6 16×16 Coverage 4%

- Date: 2026-08-27
- Candidate ID: `C6`
- Parent: `C5`
- Unique change: increase 16×16 top-loss coverage from 2% to 4%; all other C5 mechanisms remain fixed.
- Source SHA256: `A0CA63BDB301978E34ABB9501193FF8FB61CF6B291BBE19A1B37A1792D55E362`
- Parent SHA256: `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.07pp | +0.10pp | +0.04pp | +0.04pp | +0.05pp | +0.08pp | +0.063pp |

- Attention remains exactly C5 (`0.4497/0.4942`).
- CUDA algorithm-stage `20.64s`.
- Eight tests passed.

## Decision

Positive but below the preregistered `+0.2pp` promotion threshold. `local-accepted-not-promoted`; fixed regression and CPU runs were skipped. C5 remains Champion and 16×16 coverage expansion stops at 2%.

The next candidate must test a new correlation scale rather than more coverage. A bounded top-K 32×32 experiment is the next planned step; it will retain C5 and operate on at most 1% of groups.
