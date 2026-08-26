# v008 — C5 top-K 16×16 Linear Quadratic

- Date: 2026-08-27
- Candidate ID: `C5`
- Parent: `C3`
- Unique mechanism: keep C3's 5% 8×8 refinement, then refine the highest-loss 2% of contiguous 16-channel weight groups with one coordinate sweep and incremental `H·e` updates.
- Source SHA256: `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- Parent SHA256: `413B1C8F4FEE342F2E2A2AD73DE80D4E55237828BB56D4D89E647B5C6DF59AA2`
- Local status: `local-champion`
- Official status: `unavailable`
- Official score/runtime: `NA / NA`

## Fixed local matrix

| Case | C3 Linear mean | C5 Linear mean | Delta | Attention result |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5779 | 0.5802 | +0.23pp | identical to C3 |
| amax6 offset 97 | 0.5599 | 0.5617 | +0.18pp | identical to C3 |
| amax6 offset 193 | 0.5720 | 0.5766 | +0.46pp | identical to C3 |
| amax6 offset 389 | 0.5714 | 0.5743 | +0.28pp | identical to C3 |
| amax4 offset 0 | 0.4792 | 0.4818 | +0.27pp | identical to C3 |
| pow2 offset 0 | 0.5410 | 0.5436 | +0.27pp | identical to C3 |

Offset 0 component deltas versus C3:

| q | k | v | o | fc | proj |
|---:|---:|---:|---:|---:|---:|
| +0.16pp | +0.20pp | +0.09pp | +0.35pp | +0.15pp | +0.45pp |

- All 36 component means across the six fixed configurations improve (`100%` win rate).
- GQA offset 193 Attention remains exactly C3 (`0.4169/0.4928`).
- Relative to the earlier C1 parent, offset 0 Linear mean is now approximately `+1.34pp`, while retaining C1's Attention gain.

## Time and validation

- CUDA offset 0 algorithm-stage: `20.81s` versus C3 `20.14s`, ratio `1.033`.
- CPU offset 0 algorithm-stage: `55.92s` versus C3 `54.29s`, ratio `1.030`.
- Dynamic time is unchanged; 16×16 refinement is weight-calibration-only.
- Eight tests passed, including synthetic non-increase proofs for both 8×8 and 16×16 quadratic objectives.
- Root and archived source SHA256 match exactly.

## Decision

`accepted as local Champion`. C5 adds a consistent cross-configuration Linear gain over C3, improves every recorded component mean, preserves Attention exactly and stays far below the 1.15 time ratio gate.

The next isolated experiment is a 16×16 coverage ladder from 2% to 4%. It must improve Linear mean by at least `+0.2pp` over C5; otherwise 16×16 coverage is considered saturated and C5 remains Champion.
