# v011 — C8 Strictly Bounded 64×64 Linear Quadratic

- Date: 2026-08-27
- Candidate ID: `C8`
- Parent: `C5`
- Unique mechanism: retain C5, then refine the highest-loss 0.5% of complete 64-channel blocks with one `H·e` coordinate sweep, capped at 1024 blocks.
- Source SHA256: `E9E1A77B0A4427DAC616C2B00500CAF295211019741B8D6249C4ABC54DFA158D`
- Parent SHA256: `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.07pp | +0.06pp | +0.05pp | +0.04pp | +0.09pp | +0.23pp | +0.090pp |

- Attention remains exactly C5 (`0.4497/0.4942`).
- CUDA algorithm-stage `23.55s`, versus C5 `20.81s` on its development run.
- Nine tests passed, including the 64×64 quadratic non-increase test.

## Decision

The effect is positive but below the preregistered `+0.2pp` gate, while calibration cost is higher than the smaller-group candidates. `local-accepted-not-promoted`; fixed regression and CPU runs were skipped. C5 remains Champion and group-size expansion ends at 16×16.

Next candidate: keep C5's 2% 16×16 coverage but increase its coordinate sweeps from one to two. This isolates convergence depth from coverage and group size.
