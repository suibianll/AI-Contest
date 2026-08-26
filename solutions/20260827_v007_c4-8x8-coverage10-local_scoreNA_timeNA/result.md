# v007 — C4 8×8 Coverage 10%

- Date: 2026-08-27
- Candidate ID: `C4`
- Parent: `C3`
- Unique change: increase top-K 8×8 weight quadratic refinement coverage from 5% to 10%; all other algorithms, caps and sweeps remain unchanged.
- Source SHA256: `7E868911F03679FFBEDE7BA5B15A1650B7EA01453B62CA459EF284E51E4D9F2F`
- Parent SHA256: `413B1C8F4FEE342F2E2A2AD73DE80D4E55237828BB56D4D89E647B5C6DF59AA2`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.15pp | +0.15pp | +0.11pp | +0.14pp | 0.00pp | 0.00pp | +0.092pp |

- Attention remains exactly C3 (`0.4497/0.4942`).
- CUDA algorithm-stage `20.02s`, with no measurable time growth.
- Seven tests passed.

## Decision

The effect is positive and reproducible on the development case, but below the preregistered `+0.2pp` promotion threshold. `local-accepted-not-promoted`; fixed regression windows and CPU timing were not run. C3 remains Champion.

The result shows 8×8 coverage saturation: fc/proj receive no additional gain beyond the highest-loss 5%. Further coverage expansion is stopped. The next isolated direction is limited top-K 16×16 refinement to test correlations across adjacent 8-channel groups.
