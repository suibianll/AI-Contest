# v006 — C3 top-K 8×8 Linear Quadratic

- Date: 2026-08-27
- Candidate ID: `C3`
- Parent: `C1`
- Unique mechanism: after the existing exact 4×4 hierarchy solve, refine the highest quadratic-loss 5% of contiguous 8-channel weight groups with two coordinate sweeps and incremental `H·e` updates.
- Source SHA256: `413B1C8F4FEE342F2E2A2AD73DE80D4E55237828BB56D4D89E647B5C6DF59AA2`
- Parent SHA256: `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- Local status: `local-champion`
- Official status: `unavailable`
- Official score/runtime: `NA / NA`

## Mechanism

- Calibration-only refinement; dynamic activation and Attention paths are unchanged.
- Extract contiguous 8×8 blocks from the transformed activation covariance.
- Rank weight groups by current exact quadratic loss and refine at most 5%, capped at 8192 groups per matrix.
- For each selected group, test the 15 legal signed mantissa values coordinate-wise and update cached `H·e` after each accepted coordinate.
- Keep scale_factor/lv2/lv3 fixed and update only legal sign/mant fields; accept only a strict quadratic loss reduction.

## Fixed local matrix

| Case | Parent Linear mean | C3 Linear mean | Delta | Attention result |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5668 | 0.5779 | +1.10pp | identical to C1 |
| amax6 offset 97 | 0.5489 | 0.5599 | +1.10pp | identical to C1 |
| amax6 offset 193 | 0.5633 | 0.5720 | +0.87pp | identical to C1 |
| amax6 offset 389 | 0.5631 | 0.5714 | +0.83pp | identical to C1 |
| amax4 offset 0 | 0.4669 | 0.4792 | +1.23pp | identical to C1 |
| pow2 offset 0 | 0.5318 | 0.5410 | +0.92pp | identical to C1 |

Offset 0 amax6 component deltas:

| q | k | v | o | fc | proj |
|---:|---:|---:|---:|---:|---:|
| +0.58pp | +1.11pp | +0.24pp | +0.96pp | +0.49pp | +3.24pp |

- Across the six recorded configurations, 35 of 36 Linear component means improve (`97.2%` win rate). The only negative component is pow2 proj at `-0.17pp`, within the preregistered non-target tolerance; its total Linear mean still improves by `+0.92pp`.
- GQA offset 193 Attention is exactly C1 (`0.4169/0.4928`); C3 neither fixes nor worsens the known C1 Attention tail.

## Time and validation

- CUDA offset 0 algorithm-stage: `20.14s` versus C1 `20.57s`.
- CPU offset 0 algorithm-stage: `54.29s` versus C1 `54.72s`, ratio `0.992`.
- Dynamic time remains approximately `2.0s`; the new work is calibration-only.
- Seven tests passed, including exact field legality and a synthetic proof that the 8×8 refinement does not increase its quadratic objective.
- Root and archived source SHA256 match exactly.

## Decision

`accepted as local Champion`. C3 provides a repeated `+0.83pp` to `+1.23pp` Linear-mean gain across fixed windows and scale modes, preserves C1 Attention, and stays inside the time budget.

Next candidate should build on C3. Because the 5% refinement produced broad gains without measurable time growth, the next isolated experiment is a preregistered coverage ladder (10% only); it must beat C3 rather than B0/C1 and will be archived independently.
