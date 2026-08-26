# v010 — C7 top-K 32×32 Linear Quadratic

- Date: 2026-08-27
- Candidate ID: `C7`
- Parent: `C5`
- Unique mechanism: retain C5, then refine the highest-loss 1% of contiguous 32-channel weight groups with one `H·e` coordinate sweep, capped at 2048 groups.
- Source SHA256: `430383CD370D80A8F842BE5DDB38A44E05AC83B13D88950D99C0806322214C3E`
- Parent SHA256: `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.10pp | +0.11pp | +0.05pp | +0.16pp | +0.10pp | +0.22pp | +0.123pp |

- Attention remains exactly C5 (`0.4497/0.4942`).
- CUDA algorithm-stage `21.99s`.
- Nine tests passed, including the synthetic 32×32 quadratic non-increase check.

## Decision

The new correlation scale has a positive effect across all six Linear components, but the mean gain is below the preregistered `+0.2pp` threshold. `local-accepted-not-promoted`; fixed regression and CPU runs were skipped. C5 remains Champion.

The next scale experiment, if executed, is a strictly bounded 64×64 candidate on at most 0.5% of blocks. It must clear the same development gate; no full-block or full-coverage GPTQ is authorized.
